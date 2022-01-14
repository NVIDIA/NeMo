# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
Conversion script to convert Megatron_LM checkpoints into nemo checkpoint.
  Example to run this conversion script:
    python -m torch.distributed.launch --nproc_per_node=<tensor_model_parallel_size> megatron_lm_ckpt_to_nemo.py \
     --checkpoint_folder <path_to_PTL_checkpoints_folder> \
     --checkpoint_name <checkpoint_name> \
     --nemo_file_path <path_to_output_nemo_file> \
     --model_type <megatron model type> \
     --tensor_model_parallel_size <tensor_model_parallel_size>
"""

import os
from argparse import ArgumentParser
from collections import OrderedDict
from typing import Any, Optional

import torch
from pytorch_lightning.core.saving import load_hparams_from_tags_csv, load_hparams_from_yaml
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.migration import pl_legacy_patch

from nemo.collections.nlp.models.language_modeling.megatron_bert_model import MegatronBertModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.utils import AppState, logging


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint_folder",
        type=str,
        default=None,
        required=True,
        help="Path to Megatron-LM checkpoints saved during training. Ex: /raid/Megatron_LM/checkpoints",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default='model_optim_rng.pt',
        required=True,
        help="Name of checkpoint to be used. Ex: model_optim_rng.pt",
    )

    parser.add_argument(
        "--hparams_file",
        type=str,
        default=None,
        required=False,
        help="Path config for restoring. It's created during training and may need to be modified during restore if restore environment is different than training. Ex: /raid/nemo_experiments/megatron_gpt/hparams.yaml",
    )
    parser.add_argument("--nemo_file_path", type=str, default=None, required=True, help="Path to output .nemo file.")

    parser.add_argument("--tensor_model_parallel_size", type=int, required=True, default=None)

    parser.add_argument("--local_rank", type=int, required=False, default=os.getenv('LOCAL_RANK', -1))

    parser.add_argument("--model_type", type=str, required=True, default="gpt", choices=["gpt", "t5", "bert"])

    args = parser.parse_args()
    return args


def parse_weights(weight_dict: OrderedDict, parent_key: str, total: list, converted: OrderedDict, translator: dict):
    for key in weight_dict:
        new_key = key
        name_translate = translator

        for replace_key in name_translate:
            if key.find(replace_key) >= 0:
                new_key = key.replace(replace_key, name_translate[replace_key])
        if isinstance(weight_dict[key], OrderedDict) or isinstance(weight_dict[key], dict):
            parse_weights(weight_dict[key], parent_key + '.' + new_key, total, converted, translator)
        else:
            num_parameters = torch.prod(torch.tensor(weight_dict[key].cpu().size())).item()
            total[0] += num_parameters
            final_key = 'model' + parent_key + '.' + new_key
            converted[final_key] = weight_dict[key]


def load_from_checkpoint(
    cls,
    checkpoint_path: str,
    map_location: Any = None,
    hparams_file: Optional[str] = None,
    strict: bool = True,
    **kwargs,
):
    """
        Loads Megatron_LM checkpoints, convert it, with some maintenance of restoration.
        For documentation, please refer to LightningModule.load_from_checkpoin() documentation.
        """
    checkpoint = None
    try:
        cls._set_model_restore_state(is_being_restored=True)
        # TODO: replace with proper PTL API

        with pl_legacy_patch():
            if map_location is not None:
                old_checkpoint = pl_load(checkpoint_path, map_location=map_location)
            else:
                old_checkpoint = pl_load(checkpoint_path, map_location=lambda storage, loc: storage)

        total_params = [0]
        checkpoint = OrderedDict()
        checkpoint['state_dict'] = OrderedDict()
        parse_weights(
            old_checkpoint['model'], "", total_params, checkpoint['state_dict'], translator=kwargs['translator']
        )
        print('converted {:.2f}M parameters'.format(total_params[0] / 1e6))

        if hparams_file is not None:
            extension = hparams_file.split(".")[-1]
            if extension.lower() == "csv":
                hparams = load_hparams_from_tags_csv(hparams_file)
            elif extension.lower() in ("yml", "yaml"):
                hparams = load_hparams_from_yaml(hparams_file)
            else:
                raise ValueError(".csv, .yml or .yaml is required for `hparams_file`")

            hparams["on_gpu"] = False

            # overwrite hparams by the given file
            checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY] = hparams

        check_point_version = old_checkpoint.get('checkpoint_version', 0)
        if check_point_version < 3:
            # need to do the transpose of query_key_value variables
            if hparams_file is not None:
                np = hparams['cfg']['num_attention_heads']
            elif 'config' in old_checkpoint and 'num-attention-heads' in old_checkpoint['config']:
                np = old_checkpoint['config']['num-attention-heads']

            else:
                logging.warning("cannot determine the number attention heads")
                raise ValueError('need to know number of attention heads')

            if check_point_version == 0:
                # 3, np, hn -> np, 3, hn
                for key in checkpoint['state_dict']:
                    if key.find('query_key_value') >= 0:
                        weight = checkpoint['state_dict'][key]
                        if len(weight.size()) == 2:
                            # weight
                            weight = weight.view(3, np, -1, weight.size()[-1])
                            weight = weight.transpose(0, 1).contiguous()
                            checkpoint['state_dict'][key] = weight.view(-1, weight.size()[-1])
                        else:
                            # biase
                            weight = weight.view(3, np, -1)
                            weight = weight.transpose(0, 1).contiguous()
                            checkpoint['state_dict'][key] = weight.view(-1)
            elif check_point_version == 1:
                # np, hn, 3 -> np, 3, hn
                for key in checkpoint['state_dict']:
                    if key.find('query_key_value') >= 0:
                        weight = checkpoint['state_dict'][key]
                        if len(weight.size()) == 2:
                            # weight
                            weight = weight.view(np, -1, 3, weight.size()[-1])
                            weight = weight.transpose(1, 2).contiguous()
                            checkpoint['state_dict'][key] = weight
                        else:
                            # biase
                            weight = weight.view(np, -1, 3)
                            weight = weight.transpose(1, 2).contiguous()
                            checkpoint['state_dict'][key] = weight

        # for past checkpoint need to add the new key
        if cls.CHECKPOINT_HYPER_PARAMS_KEY not in checkpoint:
            checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY] = {}
        # override the hparams with values that were passed in
        # TODO: can we do this without overriding?
        config_kwargs = kwargs.copy()
        if 'trainer' in config_kwargs:
            config_kwargs.pop('trainer')
        checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY].update(config_kwargs)

        if 'cfg' in kwargs:
            model = cls._load_model_state(checkpoint, strict=strict, **kwargs)
        else:
            model = cls._load_model_state(
                checkpoint, strict=strict, cfg=checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY].cfg, **kwargs
            )
            # register the artifacts
            cfg = checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY].cfg
            if cfg.tokenizer.model is not None:
                model.register_artifact("tokenizer.tokenizer_model", cfg.tokenizer.model)
            if cfg.tokenizer.vocab_file is not None:
                model.register_artifact("tokenizer.vocab_file", cfg.tokenizer.vocab_file)
            if cfg.tokenizer.merge_file is not None:
                model.register_artifact("tokenizer.merge_file", cfg.tokenizer.merge_file)
        checkpoint = model

    finally:
        cls._set_model_restore_state(is_being_restored=False)
    return checkpoint


def convert(rank, world_size, args):

    app_state = AppState()
    app_state.data_parallel_rank = 0
    trainer = Trainer(gpus=args.tensor_model_parallel_size)
    # TODO: reach out to PTL For an API-safe local rank override
    trainer.accelerator.training_type_plugin._local_rank = rank

    if args.tensor_model_parallel_size is not None and args.tensor_model_parallel_size > 1:
        # inject model parallel rank
        checkpoint_path = os.path.join(args.checkpoint_folder, f'mp_rank_{rank:02d}', args.checkpoint_name)
    else:
        checkpoint_path = os.path.join(args.checkpoint_folder, args.checkpoint_name)

    if args.model_type == 'gpt':
        ## this dictionary is used to rename the model parameters
        name_translate = {}
        name_translate['transformer'] = 'encoder'
        model = load_from_checkpoint(
            MegatronGPTModel,
            checkpoint_path,
            hparams_file=args.hparams_file,
            trainer=trainer,
            translator=name_translate,
            strict=False,
        )
    elif args.model_type == 'bert':
        ## this dictionary is used to rename the model parameters
        name_translate = {}
        name_translate['transformer'] = 'encoder'
        name_translate['attention.'] = 'self_attention.'
        model = load_from_checkpoint(
            MegatronBertModel,
            checkpoint_path,
            hparams_file=args.hparams_file,
            trainer=trainer,
            translator=name_translate,
            strict=False,
        )
    else:
        raise NotImplemented("{} is not supported".format(args.model_type))

    model._save_restore_connector = NLPSaveRestoreConnector()

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    model.save_to(args.nemo_file_path)

    logging.info(f'NeMo model saved to: {args.nemo_file_path}')


if __name__ == '__main__':
    args = get_args()
    world_size = args.tensor_model_parallel_size

    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://', rank=args.local_rank, world_size=world_size
        )

    convert(args.local_rank, world_size, args)

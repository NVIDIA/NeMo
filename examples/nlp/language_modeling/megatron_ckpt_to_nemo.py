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
Conversion script to convert PTL checkpoints into nemo checkpoint.
  Example to run this conversion script:
     python3 megatron_ckpt_to_nemo.py \
     --checkpoint_folder <path_to_PTL_checkpoints_folder> \
     --checkpoint_name <checkpoint_name> \
     --nemo_file_path <path_to_output_nemo_file> \
     --tensor_model_parallel_size <tensor_model_parallel_size>\
     --pipeline_model_parallel_size <pipeline_model_parallel_size> \
     --model_type <model type>
"""

import os
import tempfile
import itertools
import shutil
from argparse import ArgumentParser

import torch
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_bart_model import MegatronBARTModel
from nemo.collections.nlp.models.language_modeling.megatron_bert_model import MegatronBertModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.models.language_modeling.megatron_retrieval_model import MegatronRetrievalModel
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model
from nemo.collections.nlp.models.machine_translation.megatron_nmt_model import MegatronNMTModel
from nemo.utils import AppState
from nemo.utils.model_utils import inject_model_parallel_rank

from nemo.collections.nlp.parts.nlp_overrides import (
    NEMO_MEGATRON_MODEL_PARALLEL_APPSTATE_OVERRIDE,
    NLPDDPStrategy,
    NLPSaveRestoreConnector
)

def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint_folder",
        type=str,
        default=None,
        required=True,
        help="Path to PTL checkpoints saved during training. Ex: /raid/nemo_experiments/megatron_gpt/checkpoints",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default=None,
        required=True,
        help="Name of checkpoint to be used. Ex: megatron_gpt--val_loss=6.34-step=649-last.ckpt",
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
    parser.add_argument("--pipeline_model_parallel_size", type=int, required=True, default=None)
    parser.add_argument(
        "--pipeline_model_parallel_split_rank",
        type=int,
        required=False,
        default=None,
        help="If pipeline parallel size > 1, this is the rank at which the encoder ends and the decoder begins.",
    )
    parser.add_argument(
        "--model_type", type=str, required=True, default="gpt", choices=["gpt", "t5", "bert", "nmt", "bart", "retro"]
    )

    args = parser.parse_args()
    return args


def convert(args):
    os.environ[NEMO_MEGATRON_MODEL_PARALLEL_APPSTATE_OVERRIDE] = "true"

    app_state = AppState()
    app_state.data_parallel_rank = 0
    app_state.local_rank = 0
    app_state.global_rank = 0
    app_state.pipeline_model_parallel_size = args.pipeline_model_parallel_size
    app_state.tensor_model_parallel_size = args.tensor_model_parallel_size
    app_state.model_parallel_size = app_state.tensor_model_parallel_size * app_state.pipeline_model_parallel_size
    app_state.world_size = args.tensor_model_parallel_size * args.pipeline_model_parallel_size
    app_state.tensor_model_parallel_rank = 0
    app_state.pipeline_model_parallel_rank = 0


    save_connector = NLPSaveRestoreConnector()
    trainer = Trainer(devices=1, strategy=NLPDDPStrategy(), accelerator="cpu")

    checkpoint_path = inject_model_parallel_rank(os.path.join(args.checkpoint_folder, args.checkpoint_name))
    if args.model_type == 'gpt':
        model = MegatronGPTModel.load_from_checkpoint(checkpoint_path, hparams_file=args.hparams_file, trainer=trainer).to("cpu")
    elif args.model_type == 'bert':
        model = MegatronBertModel.load_from_checkpoint(
            checkpoint_path, hparams_file=args.hparams_file, trainer=trainer
        ).to("cpu")
    elif args.model_type == 't5':
        model = MegatronT5Model.load_from_checkpoint(checkpoint_path, hparams_file=args.hparams_file, trainer=trainer).to("cpu")
    elif args.model_type == 'bart':
        model = MegatronBARTModel.load_from_checkpoint(
            checkpoint_path, hparams_file=args.hparams_file, trainer=trainer
        ).to("cpu")
    elif args.model_type == 'nmt':
        model = MegatronNMTModel.load_from_checkpoint(checkpoint_path, hparams_file=args.hparams_file, trainer=trainer).to("cpu")
    elif args.model_type == 'retro':
        model = MegatronRetrievalModel.load_from_checkpoint(
            checkpoint_path, hparams_file=args.hparams_file, trainer=trainer
        ).to("cpu")
    

    checkpoint_dir = args.checkpoint_folder
    checkpoint_name = args.checkpoint_name
    save_path = os.path.dirname(args.nemo_file_path)

    with tempfile.TemporaryDirectory() as tmpdir:

        if app_state.pipeline_model_parallel_size == 1:
            # move weights to the tmpdir
            for tp_rank in range(app_state.tensor_model_parallel_size):
                os.makedirs(os.path.join(tmpdir, f'mp_rank_{tp_rank:02d}'))
                mp_model_weights = os.path.join(
                    checkpoint_dir, f'mp_rank_{tp_rank:02d}', checkpoint_name
                )
                shutil.copy(
                    mp_model_weights,
                    os.path.join(tmpdir, f'mp_rank_{tp_rank:02d}', save_connector.model_weights_ckpt),
                )
        else:
            # move weights to the tmpdir
            for tp_rank, pp_rank in itertools.product(
                range(app_state.tensor_model_parallel_size),
                range(app_state.pipeline_model_parallel_size),
            ):
                os.makedirs(os.path.join(tmpdir, f'tp_rank_{tp_rank:02d}_pp_rank_{pp_rank:03d}'))
                mp_model_weights = os.path.join(
                    checkpoint_dir, f'tp_rank_{tp_rank:02d}_pp_rank_{pp_rank:03d}',checkpoint_name
                )
                shutil.copy(
                    mp_model_weights,
                    os.path.join(
                        tmpdir, f'tp_rank_{tp_rank:02d}_pp_rank_{pp_rank:03d}', save_connector.model_weights_ckpt
                    ),
                )

        # create config and artifacts in tmpdir
        config_yaml = os.path.join(tmpdir, save_connector.model_config_yaml)
        model.to_config_file(path2yaml_file=config_yaml)
        if hasattr(model, 'artifacts') and model.artifacts is not None:
            save_connector._handle_artifacts(model, nemo_file_folder=tmpdir)
            save_connector._update_artifact_paths(model, path2yaml_file=config_yaml)

        # create tar file
        save_connector._make_nemo_file_from_folder(filename=save_path,source_dir=tmpdir)
        
    os.environ.pop(NEMO_MEGATRON_MODEL_PARALLEL_APPSTATE_OVERRIDE, None)




if __name__ == '__main__':
    args = get_args()
    convert(args)

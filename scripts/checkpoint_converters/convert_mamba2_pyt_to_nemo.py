# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import os
import re
from argparse import ArgumentParser
from collections import defaultdict
from datetime import timedelta

import megatron.core.parallel_state as ps
import torch
import torch.distributed as dist
from megatron.core.dist_checkpointing.serialization import load_plain_tensors
from megatron.core.models.mamba import MambaModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.spec_utils import import_module
from megatron.training.arguments import core_transformer_config_from_args
from omegaconf.omegaconf import OmegaConf
from torch._C._distributed_c10d import PrefixStore
from torch.distributed import rendezvous
from megatron.core.transformer.enums import AttnBackend
from nemo.collections.nlp.models.language_modeling.megatron_mamba_model import MegatronMambaModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronLMPPTrainerBuilder
from nemo.collections.nlp.parts.utils_funcs import torch_dtype_from_precision
from nemo.utils import logging

'''
Examples

torchrun --nproc-per-node=1 /opt/NeMo/scripts/checkpoint_converters/convert_mamba2_pyt_to_nemo.py \
                                --input_name_or_path /lustre/fsw/portfolios/llmservice/users/kezhik/nemotron-5.1-hybrid/nemotron5/8b_hybrid/checkpoints/1t-hybrid-phase3/iter_2324000 \
                                --output_path /lustre/fsw/portfolios/coreai/users/ataghibakhsh/final_nm5/nm5_8b_base_8k1.nemo \
                                --mamba_ssm_ngroups 8 \
                                --precision bf16 \
                                --source_dist_ckpt \
                                --tokenizer_type tiktoken \
                                --tokenizer_library tiktoken \
                                --tokenizer_model_dir None \
                                --tokenizer_vocab_file /lustre/fsw/portfolios/coreai/users/ataghibakhsh/nm5_56b_final/multiMixV8.gpt4o_nc_sd.500000.128k.vocab.json \
                                --dist_ckpt_format torch_dist \
                                --check_fwd_pass
'''


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--hparams_file",
        type=str,
        default=f"{os.path.dirname(__file__)}/../../examples/nlp/language_modeling/conf/megatron_mamba_config.yaml",
        required=False,
        help="Path config for restoring. It's created during training and may need to be modified during restore if restore environment is different than training. Ex: /raid/nemo_experiments/megatron_gpt/hparams.yaml",
    )
    parser.add_argument("--output_path", type=str, default=None, required=True, help="Path to output .nemo file.")
    parser.add_argument(
        "--input_name_or_path",
        type=str,
        required=True,
    )
    parser.add_argument("--mamba_ssm_ngroups", type=int, default=8, help="ngroups for Mamba model")
    parser.add_argument(
        "--precision", type=str, default="bf16", choices=["bf16", "32"], help="Precision for checkpoint weights saved"
    )
    parser.add_argument(
        "--source_dist_ckpt", action="store_true", help="Set if the source checkpoint is a distributed checkpoint"
    )
    parser.add_argument("--tokenizer_type", type=str, default=None, help="tokenizer type (tiktoken, megatron, etc...)")
    parser.add_argument(
        "--tokenizer_library",
        type=str,
        default=None,
        help="tokenizer library (tiktoken, megatron, huggingface, etc...)",
    )
    parser.add_argument(
        "--tokenizer_model_dir", type=str, default=None, help="Path to the tokenizer.model, required for 8b models"
    )
    parser.add_argument(
        "--tokenizer_vocab_file", type=str, default=None, help="Path to the tokenize vocab, required for tiktokenizer"
    )
    parser.add_argument(
        "--cpu_only",
        action="store_true",
        help="If set, only CPU is used for conversion, to check fwd pass accuracy, don't set this flag",
    )
    parser.add_argument(
        "--dist_ckpt_format",
        type=str, 
        default="torch_dist",
        help="Distributed checkpointing format, torch_dist or zarr. Default is torch_dist",
    )
    parser.add_argument("--check_fwd_pass", action="store_true", help="Set if you want to check fwd pass accuracy")
    args = parser.parse_args()
    return args


try:

    class Utils:

        world_size = torch.cuda.device_count()
        rank = int(os.environ['LOCAL_RANK'])
        inited = False
        store = None

        @staticmethod
        def initialize_distributed():
            if not torch.distributed.is_initialized() and Utils.rank >= 0:
                print(f'Initializing torch.distributed with rank: {Utils.rank}, ' f'world_size: {Utils.world_size}')
                torch.cuda.set_device(Utils.rank % torch.cuda.device_count())
                init_method = 'tcp://'
                master_ip = os.getenv('MASTER_ADDR', 'localhost')
                master_port = os.getenv('MASTER_PORT', '6000')
                init_method += master_ip + ':' + master_port
                rendezvous_iterator = rendezvous(
                    init_method, Utils.rank, Utils.world_size, timeout=timedelta(minutes=1)
                )
                store, rank, world_size = next(rendezvous_iterator)
                store.set_timeout(timedelta(minutes=1))

                # Use a PrefixStore to avoid accidental overrides of keys used by
                # different systems (e.g. RPC) in case the store is multi-tenant.
                store = PrefixStore("default_pg", store)
                Utils.store = store

                torch.distributed.init_process_group(
                    backend='nccl', world_size=Utils.world_size, rank=Utils.rank, store=store
                )

                torch.distributed.barrier()
            Utils.inited = True

        @staticmethod
        def set_world_size(world_size=None, rank=None):
            Utils.world_size = torch.cuda.device_count() if world_size is None else world_size
            if torch.distributed.is_initialized() and Utils.world_size != torch.distributed.get_world_size():
                torch.distributed.destroy_process_group()

            if rank is None:
                Utils.rank = int(os.environ['LOCAL_RANK'])
                if Utils.rank >= Utils.world_size:
                    Utils.rank = -1
            else:
                Utils.rank = rank

        @staticmethod
        def destroy_model_parallel():
            if not Utils.inited:
                return
            torch.distributed.barrier()
            ps.destroy_model_parallel()
            Utils.inited = False

        @staticmethod
        def initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            **kwargs,
        ):
            ps.destroy_model_parallel()
            Utils.initialize_distributed()
            ps.initialize_model_parallel(
                tensor_model_parallel_size,
                pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size,
                **kwargs,
            )
            Utils.inited = True

except:
    pass


def dist_ckpt_handler(checkpoint_dir, cpu_only):

    if cpu_only:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'  # Ensure this port is available
        world_size = 1
        rank = 0
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)  # ckpt conversion done on CPU
    else:
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    state_dict = load_plain_tensors(checkpoint_dir)

    key_list = list(state_dict.keys())
    for k in key_list:
        if "optimizer" in k:
            state_dict.pop(k)
    dist_ckpt_args = state_dict['args']
    dist_ckpt_args.cp_comm_type = [None]
    state_dict.pop('args')
    state_dict.pop('checkpoint_version')
    state_dict.pop('iteration')
    state_dict.pop('opt_param_scheduler')
    state_dict.pop('num_floating_point_operations_so_far')

    for i, symbol in enumerate(dist_ckpt_args.hybrid_override_pattern):
        if symbol == 'M':
            state_dict[f'decoder.layers.{i}.mixer.in_proj.weight'] = torch.cat(
                [
                    state_dict[f'decoder.layers.{i}.mixer.in_proj.weight.z'],
                    state_dict[f'decoder.layers.{i}.mixer.in_proj.weight.x'],
                    state_dict[f'decoder.layers.{i}.mixer.in_proj.weight.B'],
                    state_dict[f'decoder.layers.{i}.mixer.in_proj.weight.C'],
                    state_dict[f'decoder.layers.{i}.mixer.in_proj.weight.dt'],
                ],
                dim=0,
            )

            state_dict.pop(f'decoder.layers.{i}.mixer.in_proj.weight.z')
            state_dict.pop(f'decoder.layers.{i}.mixer.in_proj.weight.x')
            state_dict.pop(f'decoder.layers.{i}.mixer.in_proj.weight.B')
            state_dict.pop(f'decoder.layers.{i}.mixer.in_proj.weight.C')
            state_dict.pop(f'decoder.layers.{i}.mixer.in_proj.weight.dt')

            state_dict[f'decoder.layers.{i}.mixer.conv1d.weight'] = torch.cat(
                [
                    state_dict[f'decoder.layers.{i}.mixer.conv1d.weight.x'],
                    state_dict[f'decoder.layers.{i}.mixer.conv1d.weight.B'],
                    state_dict[f'decoder.layers.{i}.mixer.conv1d.weight.C'],
                ],
                dim=0,
            )
            state_dict.pop(f'decoder.layers.{i}.mixer.conv1d.weight.x')
            state_dict.pop(f'decoder.layers.{i}.mixer.conv1d.weight.B')
            state_dict.pop(f'decoder.layers.{i}.mixer.conv1d.weight.C')

            state_dict[f'decoder.layers.{i}.mixer.conv1d.bias'] = torch.cat(
                [
                    state_dict[f'decoder.layers.{i}.mixer.conv1d.bias.x'],
                    state_dict[f'decoder.layers.{i}.mixer.conv1d.bias.B'],
                    state_dict[f'decoder.layers.{i}.mixer.conv1d.bias.C'],
                ],
                dim=0,
            )
            state_dict.pop(f'decoder.layers.{i}.mixer.conv1d.bias.x')
            state_dict.pop(f'decoder.layers.{i}.mixer.conv1d.bias.B')
            state_dict.pop(f'decoder.layers.{i}.mixer.conv1d.bias.C')
    if cpu_only:
        dist.destroy_process_group()
        model = None
        mcore_out =None 
        dist_ckpt_args.multi_latent_attention = False
        config = core_transformer_config_from_args(dist_ckpt_args)
        config.attention_backend=AttnBackend.flash

    else:
        dist_ckpt_args.multi_latent_attention = False
        config = core_transformer_config_from_args(dist_ckpt_args)
        config.attention_backend=AttnBackend.flash
        config.tensor_model_parallel_size = 1
        config.pipeline_model_parallel_size = 1
        config.sequence_parallel=False
        config.gradient_accumulation_fusion=False
        config.tp_comm_overlap=False
        config.use_cpu_initialization = False  # TE needs CUDA so there is no need to load the model on CPU
        assert dist_ckpt_args.use_legacy_models == False, "Mamba only supported in Mcore!"

        if dist_ckpt_args.spec is not None:
            mamba_stack_spec = import_module(dist_ckpt_args.spec)
        else:
            raise ("You must provide a valid Mamba layer spec!")

        model = MambaModel(
            config=config,
            mamba_stack_spec=mamba_stack_spec,
            vocab_size=dist_ckpt_args.padded_vocab_size,
            max_sequence_length=dist_ckpt_args.max_position_embeddings,
            pre_process=True,
            hybrid_attention_ratio=dist_ckpt_args.hybrid_attention_ratio,
            hybrid_mlp_ratio=dist_ckpt_args.hybrid_mlp_ratio,
            hybrid_override_pattern=dist_ckpt_args.hybrid_override_pattern,
            post_process=True,
            fp16_lm_cross_entropy=dist_ckpt_args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not dist_ckpt_args.untie_embeddings_and_output_weights,
            position_embedding_type=dist_ckpt_args.position_embedding_type,
            rotary_percent=dist_ckpt_args.rotary_percent,
            rotary_base=dist_ckpt_args.rotary_base,
        )

        for k, v in model.state_dict().items():
            if "_extra" in k:
                state_dict[k] = v
        model.load_state_dict(state_dict, strict=True)
        sequence_length = 128
        micro_batch_size = 2
        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = None
        mcore_out = model.forward(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)

        ps.destroy_model_parallel()
        dist.destroy_process_group()

    return state_dict, dist_ckpt_args, mcore_out, model, config


def convert(args):

    if args.source_dist_ckpt:
        checkpoint_weights, dist_ckpt_args, mcore_out, mlm_model, mlm_config = dist_ckpt_handler(
            args.input_name_or_path, cpu_only=args.cpu_only
        )
    else:
        checkpoint_weights = torch.load(args.input_name_or_path, map_location='cpu')
        if 'model' in checkpoint_weights:
            checkpoint_weights = checkpoint_weights['model']
    new_state_dict = {}

    if 'backbone' in list(checkpoint_weights.keys())[0]:
        if 'model' in list(checkpoint_weights.keys())[0]:
            checkpoint_weights = {key.replace('model.', '', 1): value for key, value in checkpoint_weights.items()}

            # Codestral Mamba Model Tokenizer Settings
            tokenizer_library = 'megatron'
            tokenizer_type = 'GPTSentencePieceTokenizer'
            tokenizer_model = args.tokenizer_model_dir

        else:

            # Tri Dao and Albert Gu Mamba Model Tokenizer Settings
            tokenizer_library = 'huggingface'
            tokenizer_type = 'EleutherAI/gpt-neox-20b'
            tokenizer_model = None

        layer_keys = [key for key in checkpoint_weights.keys() if re.match(r'backbone\.layers\.\d+\.', key)]
        layer_numbers = set(int(re.search(r'backbone\.layers\.(\d+)\.', key).group(1)) for key in layer_keys)
        num_layers = max(layer_numbers) + 1

        direct_mappings = {
            'model.embedding.word_embeddings.weight': 'backbone.embedding.weight',
            'model.decoder.final_norm.weight': 'backbone.norm_f.weight',
            'model.output_layer.weight': 'lm_head.weight',
        }

        for new_key, old_key in direct_mappings.items():
            new_state_dict[new_key] = checkpoint_weights[old_key]

        layer_attributes = [
            'mixer.A_log',
            'mixer.D',
            'mixer.conv1d.weight',
            'mixer.conv1d.bias',
            'mixer.in_proj.weight',
            'mixer.dt_bias',
            'mixer.out_proj.weight',
            'mixer.norm.weight',
            'norm.weight',
        ]

        for i in range(num_layers):
            for attr in layer_attributes:
                if attr == 'norm.weight':
                    new_key = f'model.decoder.layers.{i}.mixer.in_proj.layer_norm_weight'
                    old_key = f'backbone.layers.{i}.norm.weight'
                else:
                    new_key = f'model.decoder.layers.{i}.{attr}'
                    old_key = f'backbone.layers.{i}.{attr}'
                new_state_dict[new_key] = checkpoint_weights[old_key]

    else:

        layer_keys = [key for key in checkpoint_weights.keys() if re.match(r'decoder\.layers\.\d+\.', key)]
        layer_numbers = set(int(re.search(r'decoder\.layers\.(\d+)\.', key).group(1)) for key in layer_keys)
        num_layers = max(layer_numbers) + 1

        for key, value in checkpoint_weights.items():
            if '.norm.weight' in key and 'mixer' not in key:
                key = key[:-11] + 'mixer.in_proj.layer_norm_weight'
            new_state_dict["model." + key] = value

        # NVIDIA Mamba Model Tokenizer Settings
        tokenizer_library = args.tokenizer_library
        tokenizer_type = args.tokenizer_type
        tokenizer_model = args.tokenizer_model_dir
        tokenizer_vocab = args.tokenizer_vocab_file

    layers = defaultdict(list)

    for key in new_state_dict.keys():
        match = re.match(r'model\.decoder\.layers\.(\d+)\.(\w+)', key)
        if match:
            index, layer_type = match.groups()
            layers[index].append(layer_type)

    layer_pattern = ''
    for i in range(max(map(int, layers.keys())) + 1):
        index_str = str(i)
        layer_types = layers.get(index_str, [])
        if 'mixer' in layer_types:
            layer_pattern += 'M'
        elif 'self_attention' in layer_types:
            layer_pattern += '*'
        elif 'mlp' in layer_types:
            layer_pattern += '-'
        else:
            raise AssertionError("Layer not found. Each layer must be eiher MLP, Mamba, or Attention")

    nemo_config = OmegaConf.load(args.hparams_file)
    nemo_config.trainer["precision"] = args.precision
    nemo_config.model.vocab_size, nemo_config.model.hidden_size = new_state_dict[
        'model.embedding.word_embeddings.weight'
    ].shape
    nemo_config.model.num_layers = num_layers
    nemo_config.model.hybrid_override_pattern = layer_pattern
    if mlm_config:
        nemo_config.model.mamba_state_dim = dist_ckpt_args.mamba_state_dim
        nemo_config.model.mamba_head_dim = dist_ckpt_args.mamba_head_dim
        nemo_config.model.mamba_num_groups = dist_ckpt_args.mamba_num_groups
        nemo_config.model.num_attention_heads = dist_ckpt_args.num_attention_heads
        nemo_config.model.dist_ckpt_format=args.dist_ckpt_format
    nemo_config.model.tokenizer.library = tokenizer_library
    nemo_config.model.tokenizer.type = tokenizer_type
    nemo_config.model.tokenizer.model = tokenizer_model
    nemo_config.model.tokenizer.vocab_file = tokenizer_vocab
    if args.source_dist_ckpt:
        nemo_config.model.kv_channels = dist_ckpt_args.kv_channels
    if "-" in layer_pattern:
        nemo_config.model.ffn_hidden_size = new_state_dict[
            f'model.decoder.layers.{layer_pattern.index("-")}.mlp.linear_fc1.weight'
        ].shape[0]
    else:
        nemo_config.model.ffn_hidden_size = nemo_config.model.hidden_size

    if args.cpu_only:
        nemo_config.model.use_cpu_initialization = True
    else:
        nemo_config.model.use_cpu_initialization = False

    logging.info(f"Loading Mamba2 Pytorch checkpoint : `{args.input_name_or_path}`")

    trainer = MegatronLMPPTrainerBuilder(nemo_config).create_trainer()
    nemo_model = MegatronMambaModel(nemo_config.model, trainer)

    for k, v in nemo_model.state_dict().items():
        if "_extra" in k:
            new_state_dict[k] = v

    nemo_model.load_state_dict(new_state_dict, strict=True)
    dtype = torch_dtype_from_precision(args.precision)
    nemo_model = nemo_model.to(dtype=dtype)
    if args.check_fwd_pass:
        assert mcore_out is not None
        assert args.cpu_only is False
        sequence_length = 128
        micro_batch_size = 2
        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = None
        nemo_out = nemo_model.forward(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)
        if torch.allclose(nemo_out, mcore_out, rtol=1e-5, atol=1e-8):
            logging.info("The outputs of nemo and mcore models are close!")
        else:
            logging.info("The outputs of nemo and mcore models differ significantly!")

    nemo_model.save_to(args.output_path)
    logging.info(f'Mamba2 NeMo model saved to: {args.output_path}')


if __name__ == '__main__':
    args = get_args()
    convert(args)

# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from argparse import ArgumentParser
from omegaconf import OmegaConf

from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.utils import logging

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

tokeniser_dict = {
    'library': 'huggingface',
    'type': 'EleutherAI/gpt-neox-20b',
    'use_fast': True,
}

override_model_dict = {
    'core_attention_bias_type': 'alibi', #'no_bias',
    'position_embedding_type': 'none',
    'mcore_gpt': True,
    'transformer_engine': True,
    'micro_batch_size': 1,
    'global_batch_size': 4,
    'rampup_batch_size': None,
    'tensor_model_parallel_size': 1,
    'pipeline_model_parallel_size': 1,
    'virtual_pipeline_model_parallel_size': None,
    'megatron_amp_O2': True,
    'use_cpu_initialization': False,
    'hidden_size': 4096,
    'encoder_seq_length': 2048,
    'max_position_embeddings': 2048,
    'num_layers': 32,
    'num_attention_heads': 32,
    'ffn_hidden_size': 4 * 4096,
    'precision': 'bf16',
    'layernorm_epsilon': 1e-5,
    'pre_process': True,
    'post_process': True,
    'num_tokentypes': 0,
    'apply_query_key_layer_scaling': False,
    'parallel_output': False,
    'bias': False,
    'bias_dropout_add_fusion': False,
    'bias_activation_fusion': False,
    'transformer_block_type': 'pre_ln',
    'normalization': 'layernorm',
    'fp32_residual_connection': False,
    'hidden_dropout': 0,
    'attention_dropout': 0,
    'ffn_dropout': 0,
    'share_embeddings_and_output_weights': True,
    'sequence_parallel': False,
    'normalize_attention_scores': True,
    'use_flash_attention': False,
    'override_vocab_size': 50432,
}

def rel_err(a, b):
    return 2 * (a - b).abs() / (a.abs() + b.abs() + 1e-8)


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def convert_state_dict(state_dict, amp=False):
    def get_new_key(old_key):
        return (
            old_key.replace('transformer.', '')
            .replace('blocks.', 'decoder.layers.')
            .replace('norm_1.weight', 'self_attention.linear_qkv.layer_norm_weight') # QK norm 
            .replace('attn.Wqkv', 'self_attention.linear_qkv')
            .replace('attn.out_proj','self_attention.linear_proj')
            .replace('norm_2.weight', 'mlp.linear_fc1.layer_norm_weight')
            .replace('ffn.up_proj', 'mlp.linear_fc1')
            .replace('ffn.down_proj', 'mlp.linear_fc2')
            .replace('wte.weight', 'embedding.word_embeddings.weight')
            .replace('norm_f.weight', 'decoder.final_layernorm.weight')
        )

    new_dict = {}

    for old_key, val in state_dict.items():
        new_key = get_new_key(old_key)

        if 'linear_qkv.weight' in new_key:
            new_dict[new_key] = val.view(3, override_model_dict['num_attention_heads'], val.shape[0] // 3 // override_model_dict['num_attention_heads'], val.shape[1]).transpose(0, 1).reshape(val.shape[0], val.shape[1])
        else:
            new_dict[new_key] = val

    return new_dict


def adjust_nemo_config(model_config, ref_config):
    model_config.update(override_model_dict)
    model_config['tokenizer'] = tokeniser_dict

    return model_config


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--name_or_path", type=str,
                        default="mosaicml/mpt-7b")
    parser.add_argument(
        "--hparams_file",
        type=str,
        default=f"{os.path.dirname(__file__)}/../../examples/nlp/language_modeling/conf/megatron_gpt_config.yaml",
        required=False,
        help="Path config for restoring. It's created during training and may need to be modified during restore if restore environment is different than training. Ex: /raid/nemo_experiments/megatron_gpt/hparams.yaml",
    )
    parser.add_argument("--save_path", type=str, default='mpt_7b_mcore.nemo',
                        required=False, help="Path to output .nemo file.")
    parser.add_argument(
        "--precision", type=str, default="bf16", choices=["bf16", "32"], help="Precision for checkpoint weights saved"
    )
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = get_args()
    logging.info(f"Loading checkpoint from HF: `{args.name_or_path}`")
    hf_tokenizer = AutoTokenizer.from_pretrained(args.name_or_path)
    hf_model = AutoModelForCausalLM.from_pretrained(args.name_or_path,
                                                    torch_dtype=torch.bfloat16,
                                                    trust_remote_code=True)

    nemo_config = OmegaConf.load(args.hparams_file)
    nemo_config.model = adjust_nemo_config(
        nemo_config.model, hf_model.config.to_dict())

    nemo_config.trainer["precision"] = args.precision
    trainer = MegatronTrainerBuilder(nemo_config).create_trainer()
    model = MegatronGPTModel(nemo_config.model, trainer)

    old_state_dict = hf_model.state_dict()
    nemo_state_dict = convert_state_dict(
        old_state_dict, amp=nemo_config.model.megatron_amp_O2)

    model.model.load_state_dict(nemo_state_dict, strict=False)

    logging.info(f'=' * 50)
    # Verifications
    input_texts = [
        'query: how much protein should a female eat',
    ]

    # Tokenize the input texts
    hf_tokenizer.pad_token = hf_tokenizer.eos_token
    batch_dict = hf_tokenizer(
        input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
    batch_dict_cuda = {k: v.cuda() for k, v in batch_dict.items()}
    hf_model = hf_model.cuda().eval()
    model = model.eval()

    hf_outputs = hf_model(**batch_dict_cuda, output_hidden_states=True)
    ids = batch_dict_cuda['input_ids']

    id_tensors = [torch.unsqueeze(torch.LongTensor(
        id_list), dim=0) for id_list in ids.cpu()]

    masks_and_position_ids = [
        get_ltor_masks_and_position_ids(
            id_tensor, hf_tokenizer.eos_token, False, False, False)
        for id_tensor in id_tensors
    ]    
    for tokens, attn_mask_and_pos_ids in zip(id_tensors, masks_and_position_ids):
        attn_mask, _, pos_ids = attn_mask_and_pos_ids

        outputs = model(tokens=tokens, text_position_ids=pos_ids.cuda(
        ), attention_mask=attn_mask.cuda(), labels=None)

    hf_next_token = hf_outputs.logits[0, -1].argmax()
    next_token = outputs.squeeze()[-1].argmax()
    
    assert hf_next_token == next_token, f'prediction mismatch: {hf_tokenizer.decode(hf_next_token)} != {hf_tokenizer.decode(next_token)}'

    model.save_to(args.save_path)
    logging.info(f'NeMo model saved to: {args.save_path}')

if __name__ == '__main__':
    main()
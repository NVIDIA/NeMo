import os
from argparse import ArgumentParser

import torch
from omegaconf import OmegaConf

from nemo.collections.nlp.models.language_modeling.megatron import GPTModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.utils import logging

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def create_rename_keys(num_hidden_layers):
    rename_keys = []
    for i in range(num_hidden_layers):
        # encoder layers: attention mechanism, 2 feedforward neural networks, and 2 layernorms
        rename_keys.extend(
            [
                (
                    f"encoder.layer.{i}.attention.self.query.weight",
                    f"model.language_model.encoder.layers.{i}.self_attention.query.weight",
                ),
                (
                    f"encoder.layer.{i}.attention.self.query.bias",
                    f"model.language_model.encoder.layers.{i}.self_attention.query.bias",
                ),
                (
                    f"encoder.layer.{i}.attention.self.key.weight",
                    f"model.language_model.encoder.layers.{i}.self_attention.key.weight",
                ),
                (
                    f"encoder.layer.{i}.attention.self.key.bias",
                    f"model.language_model.encoder.layers.{i}.self_attention.key.bias",
                ),
                (
                    f"encoder.layer.{i}.attention.self.value.weight",
                    f"model.language_model.encoder.layers.{i}.self_attention.value.weight",
                ),
                (
                    f"encoder.layer.{i}.attention.self.value.bias",
                    f"model.language_model.encoder.layers.{i}.self_attention.value.bias",
                ),
                (
                    f"encoder.layer.{i}.attention.output.dense.weight",
                    f"model.language_model.encoder.layers.{i}.self_attention.dense.weight",
                ),
                (
                    f"encoder.layer.{i}.attention.output.dense.bias",
                    f"model.language_model.encoder.layers.{i}.self_attention.dense.bias",
                ),
                (
                    f"encoder.layer.{i}.attention.output.LayerNorm.weight",
                    f"model.language_model.encoder.layers.{i}.input_layernorm.weight",
                ),
                (
                    f"encoder.layer.{i}.attention.output.LayerNorm.bias",
                    f"model.language_model.encoder.layers.{i}.input_layernorm.bias",
                ),
                (
                    f"encoder.layer.{i}.intermediate.dense.weight",
                    f"model.language_model.encoder.layers.{i}.mlp.dense_h_to_4h.weight",
                ),
                (
                    f"encoder.layer.{i}.intermediate.dense.bias",
                    f"model.language_model.encoder.layers.{i}.mlp.dense_h_to_4h.bias",
                ),
                (
                    f"encoder.layer.{i}.output.dense.weight",
                    f"model.language_model.encoder.layers.{i}.mlp.dense_4h_to_h.weight",
                ),
                (
                    f"encoder.layer.{i}.output.dense.bias",
                    f"model.language_model.encoder.layers.{i}.mlp.dense_4h_to_h.bias",
                ),
                (
                    f"encoder.layer.{i}.output.LayerNorm.weight",
                    f"model.language_model.encoder.layers.{i}.post_attention_layernorm.weight",
                ),
                (
                    f"encoder.layer.{i}.output.LayerNorm.bias",
                    f"model.language_model.encoder.layers.{i}.post_attention_layernorm.bias",
                ),
            ]
        )

    # Non-layer dependent keys
    rename_keys.extend(
        [
            ("embeddings.word_embeddings.weight", "model.language_model.embedding.word_embeddings.weight"),
            ("embeddings.position_embeddings.weight", "model.language_model.embedding.position_embeddings.weight"),
            ("embeddings.token_type_embeddings.weight", "model.language_model.embedding.tokentype_embeddings.weight"),
            ("embeddings.LayerNorm.weight", "model.language_model.encoder.initial_layernorm.weight"),
            ("embeddings.LayerNorm.bias", "model.language_model.encoder.initial_layernorm.bias"),
            ("pooler.dense.weight", "model.language_model.pooler.dense.weight"),
            ("pooler.dense.bias", "model.language_model.pooler.dense.bias"),
        ]
    )

    return rename_keys


def convert_state_dict(state_dict, amp=False):
    def get_new_key(old_key):
        if old_key == 'wte.weight':
            return 'language_model.embedding.word_embeddings.weight'
        elif old_key == 'norm_f.weight':
            return 'language_model.encoder.final_layernorm.weight'
        else:
            p1 = old_key.replace('blocks.', 'language_model.encoder.layers.')
            p2 = p1.replace('norm_1.weight', 'input_layernorm.weight')
            p3 = p2.replace('attn.Wqkv.weight', 'self_attention.query_key_value.weight')
            p4 = p3.replace('attn.out_proj.weight', 'self_attention.dense.weight')
            p5 = p4.replace('norm_2.weight', 'post_attention_layernorm.weight')
            p6 = p5.replace('ffn.up_proj.weight', 'mlp.dense_h_to_4h.weight')
            p7 = p6.replace('ffn.down_proj.weight', 'mlp.dense_4h_to_h.weight')

            return p7

    new_dict = {}

    for old_key, val in state_dict.items():
        new_key = get_new_key(old_key)
        if amp:
            new_key = 'module.' + new_key

        new_dict[new_key] = val

    return new_dict


def adjust_tensor_shapes(model, nemo_state_dict):
    """
    Adapt tensor shapes in the state dictionary to ensure compatibility with a different model structure.

    Parameters:
    nemo_state_dict (dict): The state dictionary of the model.

    Returns:
    dict: The updated state dictionary with modified tensor shapes for compatibility.
    """

    # Note: For 'key' and 'value' weights and biases, NeMo uses a consolidated tensor 'query_key_value'.
    for key_ in list(nemo_state_dict.keys()):
        if "self_attention.query" in key_:
            key_q = key_
            key_k = key_.replace('self_attention.query', 'self_attention.key')
            key_v = key_.replace('self_attention.query', 'self_attention.value')
            key_new = key_.replace('self_attention.query', 'self_attention.query_key_value')
            value_new = torch.concat((nemo_state_dict[key_q], nemo_state_dict[key_k], nemo_state_dict[key_v]), dim=0)
            nemo_state_dict[key_new] = value_new
            del nemo_state_dict[key_q], nemo_state_dict[key_k], nemo_state_dict[key_v]

    # Padding to new vocab size
    original_embedding = nemo_state_dict['model.language_model.embedding.word_embeddings.weight']
    vocab_size = original_embedding.size(0)
    if model.padded_vocab_size > vocab_size:
        zeros_to_add = torch.zeros(
            model.padded_vocab_size - vocab_size,
            original_embedding.size(1),
            dtype=original_embedding.dtype,
            device=original_embedding.device,
        )
        # Concatenate the two tensors along rows
        padded_embedding = torch.cat([original_embedding, zeros_to_add], dim=0)
        nemo_state_dict['model.language_model.embedding.word_embeddings.weight'] = padded_embedding

    return nemo_state_dict


def adjust_nemo_config(model_config, ref_config):
    tokeniser_dict = {
        'library': 'huggingface',
        'type': 'EleutherAI/gpt-neox-20b',
        'use_fast': True,
    }

    override_model_dict = {
        'micro_batch_size': 1,
        'global_batch_size': 4,
        'rampup_batch_size': None,
        'tensor_model_parallel_size': 1,
        'pipeline_model_parallel_size': 1,
        'virtual_pipeline_model_parallel_size': None,
        'megatron_amp_O2': True,
        'transformer_engine': False,
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
        'normalization': 'layernorm',  # TODO (Ethan He): verify low_precision_layernorm vs layernorm
        'fp32_residual_connection': False,
        'hidden_dropout': 0,
        'attention_dropout': 0,
        'ffn_dropout': 0,
        'megatron_legacy': True,
        'share_embeddings_and_output_weights': True,
        'sequence_parallel': False,
        'position_embedding_type': 'alibi',
        'normalize_attention_scores': True,
        'use_flash_attention': False,
        'override_vocab_size': 50432,
    }
    model_config.update(override_model_dict)
    model_config['tokenizer'] = tokeniser_dict

    return model_config


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--name_or_path", type=str, default="intfloat/e5-large-unsupervised")
    parser.add_argument(
        "--hparams_file",
        type=str,
        default=f"{os.path.dirname(__file__)}/../../examples/nlp/language_modeling/conf/megatron_gpt_config.yaml",
        required=False,
        help="Path config for restoring. It's created during training and may need to be modified during restore if restore environment is different than training. Ex: /raid/nemo_experiments/megatron_gpt/hparams.yaml",
    )
    parser.add_argument("--save_path", type=str, default=None, required=True, help="Path to output .nemo file.")
    parser.add_argument(
        "--precision", type=str, default="32", choices=["bf16", "32"], help="Precision for checkpoint weights saved"
    )

    args = parser.parse_args()
    return args


@torch.no_grad()
def convert(args):
    logging.info(f"Loading checkpoint from HF: `{args.name_or_path}`")
    hf_tokenizer = AutoTokenizer.from_pretrained(args.name_or_path)
    hf_model = AutoModelForCausalLM.from_pretrained(args.name_or_path)

    nemo_config = OmegaConf.load(args.hparams_file)
    nemo_config.model = adjust_nemo_config(nemo_config.model, hf_model.config.to_dict())

    nemo_config.trainer["precision"] = args.precision
    trainer = MegatronTrainerBuilder(nemo_config).create_trainer()
    model = MegatronGPTModel(nemo_config.model, trainer)

    old_state_dict = hf_model.state_dict()
    nemo_state_dict = convert_state_dict(old_state_dict, amp=nemo_config.model.megatron_amp_O2)
    # nemo_state_dict = adjust_tensor_shapes(model, new_state_dict)
    if nemo_config.model.megatron_amp_O2:
        missing_keys, unexpected_keys = model.model.load_state_dict(nemo_state_dict, strict=False)
    else:
        missing_keys, unexpected_keys = super(GPTModel, model.model).load_state_dict(nemo_state_dict, strict=True)
    # model.load_state_dict(nemo_state_dict, strict=True)
    logging.info(f'=' * 50)
    # Verifications
    input_texts = [
        'query: how much protein should a female eat',
        'query: summit define',
        "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
        "passage: Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
    ]

    # Tokenize the input texts
    hf_tokenizer.pad_token = hf_tokenizer.eos_token
    batch_dict = hf_tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
    batch_dict_cuda = {k: v.cuda() for k, v in batch_dict.items()}
    hf_model = hf_model.cuda().eval()
    model = model.eval()

    hf_outputs = hf_model(**batch_dict_cuda, output_hidden_states=True)
    embeddings_hf = average_pool(hf_outputs.logits, batch_dict_cuda['attention_mask'])
    embeddings_hf = F.normalize(embeddings_hf, p=2, dim=1)

    ids = batch_dict_cuda['input_ids']

    id_tensors = [torch.unsqueeze(torch.LongTensor(id_list), dim=0) for id_list in ids.cpu()]

    masks_and_position_ids = [
        get_ltor_masks_and_position_ids(id_tensor, hf_tokenizer.eos_token, False, False, False)
        for id_tensor in id_tensors
    ]

    output_tensors = []
    for tokens, attn_mask_and_pos_ids in zip(id_tensors, masks_and_position_ids):
        attn_mask, _, pos_ids = attn_mask_and_pos_ids

        outputs = model(tokens=tokens, text_position_ids=pos_ids.cuda(), attention_mask=attn_mask.cuda(), labels=None)
        output_tensors.append(outputs)

    output_tensors = torch.concat(output_tensors, dim=0)
    embeddings = average_pool(outputs, batch_dict_cuda['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    # print(embeddings)
    print(embeddings - embeddings_hf)

    model.save_to(args.save_path)
    logging.info(f'NeMo model saved to: {args.save_path}')


if __name__ == '__main__':
    args = get_args()
    convert(args)

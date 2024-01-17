#%%
import os

os.environ['NVTE_FLASH_ATTN'] = '0'
os.environ['NVTE_FUSED_ATTN'] = '0'

from argparse import ArgumentParser

import torch
from omegaconf import OmegaConf

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
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
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
    args = parser.parse_args([])
    return args


with torch.no_grad():
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
    # nemo_state_dict = adjust_tensor_shapes(model, new_state_dict)
    # from IPython import embed; embed(header="here")
    # for param in model.model.parameters():
    #     param.zero_()

    model.model.load_state_dict(nemo_state_dict, strict=False)

    logging.info(f'=' * 50)
    # Verifications
    input_texts = [
        'query: how much protein should a female eat',
        # 'query: summit define',
        # "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
        # "passage: Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
    ]

    # Tokenize the input texts
    hf_tokenizer.pad_token = hf_tokenizer.eos_token
    batch_dict = hf_tokenizer(
        input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
    batch_dict_cuda = {k: v.cuda() for k, v in batch_dict.items()}
    hf_model = hf_model.cuda().eval()
    model = model.eval()

    model.save_to(args.save_path)
    logging.info(f'NeMo model saved to: {args.save_path}')

    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                activations[name] = output[0].detach()
            else:
                activations[name] = output.detach()
            if len(input):
                if isinstance(input, tuple):
                    activations[name + '_inp'] = input[0].detach()
                else:
                    activations[name + '_inp'] = input.detach()
            # for i, o in enumerate(output):
            #     if o is not None:
            #         activations[name + str(i)] = o.detach()
        return hook

    # for name, layer in model.model.named_modules():
    #     print(name)
    #     if 'module.decoder.layers.0' in name:
    #         layer.register_forward_hook(get_activation(name))

    model.model.module.decoder.layers[0].self_attention.register_forward_hook(get_activation('self_attention'))
    model.model.module.decoder.layers[0].self_attention.core_attention.register_forward_hook(get_activation('DotProductAttention'))
    model.model.module.decoder.layers[0].self_attention.linear_qkv.register_forward_hook(get_activation('Wqkv'))
    model.model.module.decoder.layers[0].self_attention.linear_proj.register_forward_hook(get_activation('out_proj'))
    model.model.module.decoder.layers[0].mlp.register_forward_hook(get_activation('mlp'))
    model.model.module.decoder.layers[0].mlp.linear_fc1.register_forward_hook(get_activation('mlp_fc1'))
    model.model.module.decoder.layers[0].mlp.linear_fc2.register_forward_hook(get_activation('mlp_fc2'))
    model.model.module.decoder.final_layernorm.register_forward_hook(get_activation('norm_f'))
    # model.model.module.decoder.layers[1].register_forward_hook(get_activation('block1'))
    hf_model.transformer.blocks[0].attn.register_forward_hook(get_activation('hf_self_attention'))
    hf_model.transformer.blocks[0].attn.Wqkv.register_forward_hook(get_activation('hf_Wqkv'))
    hf_model.transformer.blocks[0].attn.out_proj.register_forward_hook(get_activation('hf_out_proj'))
    hf_model.transformer.blocks[0].ffn.register_forward_hook(get_activation('hf_mlp'))
    hf_model.transformer.blocks[0].ffn.up_proj.register_forward_hook(get_activation('hf_mlp_fc1'))
    hf_model.transformer.blocks[0].ffn.down_proj.register_forward_hook(get_activation('hf_mlp_fc2'))
    hf_model.transformer.norm_f.register_forward_hook(get_activation('hf_norm_f'))
    # hf_model.transformer.blocks[1].register_forward_hook(get_activation('hf_block1'))

    hf_outputs = hf_model(**batch_dict_cuda, output_hidden_states=True)
    embeddings_hf = average_pool(
        hf_outputs.hidden_states[-1], batch_dict_cuda['attention_mask'])
    embeddings_hf = F.normalize(embeddings_hf, p=2, dim=1)

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

    hf_output_token = hf_outputs.logits[0, -1].argmax()
    print(hf_output_token, hf_tokenizer.decode(hf_output_token))
    output_token = outputs[0, -1].argmax()
    print(output_token, hf_tokenizer.decode(output_token))

    model.model.module.post_process = False

    output_tensors = []
    for tokens, attn_mask_and_pos_ids in zip(id_tensors, masks_and_position_ids):
        attn_mask, _, pos_ids = attn_mask_and_pos_ids

        outputs = model(tokens=tokens, text_position_ids=pos_ids.cuda(
        ), attention_mask=attn_mask.cuda(), labels=None)
        output_tensors.append(outputs.squeeze())

    output_tensors = torch.stack(output_tensors, dim=0)
    embeddings = average_pool(output_tensors, batch_dict_cuda['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)

    for k, v in activations.items():
        activations[k].squeeze_()

    for k, v in activations.items():
        if k.startswith('hf_'):
            print(k[3:])
            diff = activations[k[3:]] - activations[k]
            print('mean', activations[k[3:]].mean(), activations[k].mean())
            print('std', activations[k[3:]].std(), activations[k].std())
            print('max diff', diff.abs().max())
            relative_diff = rel_err(activations[k[3:]], activations[k])
            print('relative diff mean&max', relative_diff.mean(), relative_diff.max())
            # print(diff)

        else:
            pass

    print(embeddings - embeddings_hf)
    print('std', torch.std(embeddings), torch.std(embeddings_hf))
    print('mean', torch.mean(embeddings), torch.mean(embeddings_hf))
    print('max abs err:', (embeddings - embeddings_hf).abs().max())
    print('rel_err:', rel_err(embeddings, embeddings_hf).mean())
    # from IPython import embed; embed()



# %%
inp = activations['norm_f_inp'].clone()
hf_inp = activations['hf_norm_f_inp'].clone()
out = activations['norm_f'].clone()
hf_out = activations['hf_norm_f'].clone()

# %%
print(rel_err(inp, hf_inp).mean())
print(rel_err(out, hf_out).mean())
print(rel_err(hf_out, hf_model.transformer.norm_f(inp)).mean())
print(rel_err(hf_out, model.model.module.decoder.final_layernorm(hf_inp)).mean())

hf_model.transformer.lm_head
# %%

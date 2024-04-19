import os
from argparse import ArgumentParser
import torch
from omegaconf.omegaconf import OmegaConf
from nemo.collections.nlp.models.language_modeling.megatron_griffin_model import MegatronGriffinModel
from nemo.utils import logging
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronLMPPTrainerBuilder
from nemo.collections.nlp.parts.utils_funcs import torch_dtype_from_precision
from transformers import AutoModelForCausalLM

'''
CUDA_VISIBLE_DEVICES="0" python /home/ataghibakhsh/NeMo/scripts/checkpoint_converters/convert_griffin_to_mcore.py --output_path /home/ataghibakhsh/griffin_pretrained_from_hf.nemo --hparams_file /home/ataghibakhsh/NeMo/examples/nlp/language_modeling/conf/megatron_griffin_config.yaml 
'''

def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--hparams_file",
        type=str,
        default=f"{os.path.dirname(__file__)}/../../examples/nlp/language_modeling/conf/megatron_griffin_config.yaml",
        required=False,
        help="Path config for restoring. It's created during training and may need to be modified during restore if restore environment is different than training. Ex: /raid/nemo_experiments/megatron_gpt/hparams.yaml",
    )
    parser.add_argument("--output_path", type=str, default=None, required=True, help="Path to output .nemo file.")
    parser.add_argument("--input_name_or_path", type=str, default="google/recurrentgemma-2b")  
    parser.add_argument(
        "--precision", type=str, default="32", choices=["bf16", "32"], help="Precision for checkpoint weights saved"
    )  
    args = parser.parse_args()
    return args


def convert(args):
    
    logging.info(f"Loading checkpoint from HF: `{args.input_name_or_path}`")
    model_hf = AutoModelForCausalLM.from_pretrained(args.input_name_or_path, device_map="auto")

    nemo_config = OmegaConf.load(args.hparams_file)
    nemo_config.trainer["precision"] = args.precision
    trainer = MegatronLMPPTrainerBuilder(nemo_config).create_trainer()

    model = MegatronGriffinModel(nemo_config.model, trainer)

    new_state_dict = {}

    new_state_dict['model.embedding.word_embeddings.weight'] = model_hf.state_dict()['model.embed_tokens.weight']
    new_state_dict['model.decoder.final_layernorm.weight'] = model_hf.state_dict()['model.final_norm.weight']+1
    for l in range(nemo_config.model.num_layers):
        print(f"Converting Layer {l}")
        print("********************")
        # new_state_dict[f'model.layers.{l}.input_layernorm.weight'] = model_hf.state_dict()[f'model.layers..{l}.temporal_pre_norm.scale']

        new_state_dict[f'model.decoder.layers.{l}.mlp.linear_fc1.weight'] = torch.cat([model_hf.state_dict()[f'model.layers.{l}.mlp_block.gate_proj.weight'], model_hf.state_dict()[f'model.layers.{l}.mlp_block.up_proj.weight']])
        new_state_dict[f'model.decoder.layers.{l}.mlp.linear_fc1.bias'] = torch.cat([model_hf.state_dict()[f'model.layers.{l}.mlp_block.gate_proj.bias'], model_hf.state_dict()[f'model.layers.{l}.mlp_block.up_proj.bias']]).flatten()
        new_state_dict[f'model.decoder.layers.{l}.mlp.linear_fc2.weight'] = model_hf.state_dict()[f'model.layers.{l}.mlp_block.down_proj.weight']
        new_state_dict[f'model.decoder.layers.{l}.mlp.linear_fc2.bias'] = model_hf.state_dict()[f'model.layers.{l}.mlp_block.down_proj.bias']
        new_state_dict[f'model.decoder.layers.{l}.mlp.linear_fc1._extra_state'] = model.state_dict()[f'model.decoder.layers.{l}.mlp.linear_fc1._extra_state']
        new_state_dict[f'model.decoder.layers.{l}.mlp.linear_fc2._extra_state'] = model.state_dict()[f'model.decoder.layers.{l}.mlp.linear_fc2._extra_state']

        new_state_dict[f'model.decoder.layers.{l}.mlp.linear_fc1.layer_norm_weight'] = model_hf.state_dict()[f'model.layers.{l}.channel_pre_norm.weight']+1

        if l % 3 == 2:
            new_state_dict[f'model.decoder.layers.{l}.self_attention.linear_proj.weight'] = model_hf.state_dict()[f'model.layers.{l}.temporal_block.o_proj.weight']
            new_state_dict[f'model.decoder.layers.{l}.self_attention.linear_proj.bias'] = model_hf.state_dict()[f'model.layers.{l}.temporal_block.o_proj.bias']
            new_state_dict[f'model.decoder.layers.{l}.self_attention.linear_qkv.layer_norm_weight'] = model_hf.state_dict()[f'model.layers.{l}.temporal_pre_norm.weight']+1
            new_state_dict[f'model.decoder.layers.{l}.self_attention.linear_qkv.weight'] = torch.cat([model_hf.state_dict()[f'model.layers.{l}.temporal_block.q_proj.weight'], 
                                                            model_hf.state_dict()[f'model.layers.{l}.temporal_block.k_proj.weight'],
                                                            model_hf.state_dict()[f'model.layers.{l}.temporal_block.v_proj.weight']])
            new_state_dict[f'model.decoder.layers.{l}.self_attention.linear_qkv.bias'] = torch.zeros(new_state_dict[f'model.decoder.layers.{l}.self_attention.linear_qkv.weight'].shape[0])
            new_state_dict[f'model.decoder.layers.{l}.self_attention.linear_proj._extra_state'] = model.state_dict()[f'model.decoder.layers.{l}.self_attention.linear_proj._extra_state']
            new_state_dict[f'model.decoder.layers.{l}.self_attention.linear_qkv._extra_state'] = model.state_dict()[f'model.decoder.layers.{l}.self_attention.linear_qkv._extra_state']


        else:
            
            new_state_dict[f'model.decoder.layers.{l}.recurrent_layer.linear_y.layer_norm_weight'] = model_hf.state_dict()[f'model.layers.{l}.temporal_pre_norm.weight']+1
            new_state_dict[f'model.decoder.layers.{l}.recurrent_layer.linear_x.layer_norm_weight'] = model_hf.state_dict()[f'model.layers.{l}.temporal_pre_norm.weight']+1
            new_state_dict[f'model.decoder.layers.{l}.recurrent_layer.linear_y.weight'] = model_hf.state_dict()[f'model.layers.{l}.temporal_block.linear_y.weight']
            new_state_dict[f'model.decoder.layers.{l}.recurrent_layer.linear_y.bias'] = model_hf.state_dict()[f'model.layers.{l}.temporal_block.linear_y.bias']
            new_state_dict[f'model.decoder.layers.{l}.recurrent_layer.linear_x.weight'] = model_hf.state_dict()[f'model.layers.{l}.temporal_block.linear_x.weight']
            new_state_dict[f'model.decoder.layers.{l}.recurrent_layer.linear_x.bias'] = model_hf.state_dict()[f'model.layers.{l}.temporal_block.linear_x.bias']
            new_state_dict[f'model.decoder.layers.{l}.recurrent_layer.linear_out.weight'] = model_hf.state_dict()[f'model.layers.{l}.temporal_block.linear_out.weight']
            new_state_dict[f'model.decoder.layers.{l}.recurrent_layer.linear_out.bias'] = model_hf.state_dict()[f'model.layers.{l}.temporal_block.linear_out.bias']

            new_state_dict[f'model.decoder.layers.{l}.recurrent_layer.conv_1d.conv_1d.weight'] = model_hf.state_dict()[f'model.layers.{l}.temporal_block.conv_1d.weight']
            new_state_dict[f'model.decoder.layers.{l}.recurrent_layer.conv_1d.conv_1d.bias'] = model_hf.state_dict()[f'model.layers.{l}.temporal_block.conv_1d.bias']

            new_state_dict[f'model.decoder.layers.{l}.recurrent_layer.rg_lru.a_param'] = model_hf.state_dict()[f'model.layers.{l}.temporal_block.rg_lru.recurrent_param']
            new_state_dict[f'model.decoder.layers.{l}.recurrent_layer.rg_lru.input_gate.w'] = model_hf.state_dict()[f'model.layers.{l}.temporal_block.rg_lru.input_gate_weight']
            new_state_dict[f'model.decoder.layers.{l}.recurrent_layer.rg_lru.input_gate.b'] = model_hf.state_dict()[f'model.layers.{l}.temporal_block.rg_lru.input_gate_bias']
            new_state_dict[f'model.decoder.layers.{l}.recurrent_layer.rg_lru.a_gate.w'] = model_hf.state_dict()[f'model.layers.{l}.temporal_block.rg_lru.recurrent_gate_weight']
            new_state_dict[f'model.decoder.layers.{l}.recurrent_layer.rg_lru.a_gate.b'] = model_hf.state_dict()[f'model.layers.{l}.temporal_block.rg_lru.recurrent_gate_bias']
            
            new_state_dict[f'model.decoder.layers.{l}.recurrent_layer.linear_y._extra_state'] = model.state_dict()[f'model.decoder.layers.{l}.recurrent_layer.linear_y._extra_state']
            new_state_dict[f'model.decoder.layers.{l}.recurrent_layer.linear_x._extra_state'] = model.state_dict()[f'model.decoder.layers.{l}.recurrent_layer.linear_x._extra_state']
            new_state_dict[f'model.decoder.layers.{l}.recurrent_layer.linear_out._extra_state'] = model.state_dict()[f'model.decoder.layers.{l}.recurrent_layer.linear_out._extra_state']

    model.load_state_dict(new_state_dict, strict=True)
    dtype = torch_dtype_from_precision(args.precision)
    model = model.to(dtype=dtype)

    print("Restored!")

    model.save_to(args.output_path)
    logging.info(f'Griffin NeMo model saved to: {args.output_path}')


if __name__ == '__main__':
    args = get_args()
    convert(args)
import os
from argparse import ArgumentParser
import torch
from omegaconf.omegaconf import OmegaConf
import json
from nemo.collections.nlp.models.language_modeling.megatron_jamba_model import MegatronJambaModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronLMPPTrainerBuilder
from nemo.collections.nlp.parts.utils_funcs import torch_dtype_from_precision
from nemo.utils import logging
# python /home/ataghibakhsh/NeMo/scripts/checkpoint_converters/convert_mamba2_pyt_to_nemo.py --input_name_or_path /home/ataghibakhsh/mamba2_ckpts/mamba2-780m --output_path /home/ataghibakhsh/forks/mamba2_780m.nemo
# python /home/ataghibakhsh/NeMo/scripts/checkpoint_converters/convert_mamba2_pyt_to_nemo.py --input_name_or_path /home/ataghibakhsh/rogers/mamba_share --output_path /home/ataghibakhsh/forks/mmm_mamba2.nemo

def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--hparams_file",
        type=str,
        default=f"{os.path.dirname(__file__)}/../../examples/nlp/language_modeling/conf/megatron_jamba_config.yaml",
        required=False,
        help="Path config for restoring. It's created during training and may need to be modified during restore if restore environment is different than training. Ex: /raid/nemo_experiments/megatron_gpt/hparams.yaml",
    )
    parser.add_argument("--output_path", type=str, default=None, required=True, help="Path to output .nemo file.")
    parser.add_argument("--input_name_or_path", type=str, required=True,)
    parser.add_argument(
        "--precision", type=str, default="32", choices=["bf16", "32"], help="Precision for checkpoint weights saved"
    )
    args = parser.parse_args()
    return args


def convert(args):

    mmm = True

    with open(args.input_name_or_path+'/config.json', 'r') as config_file:
        pytorch_config = json.load(config_file)

    nemo_config = OmegaConf.load(args.hparams_file)
    nemo_config.trainer["precision"] = args.precision
    nemo_config.model.hidden_size=pytorch_config['d_model']
    nemo_config.model.num_layers=pytorch_config['n_layer']
    if mmm:
        nemo_config.model.vocab_size=pytorch_config['vocab_size']
    else:

        nemo_config.model.vocab_size=pytorch_config['vocab_size']+11
    nemo_config.model.make_vocab_size_divisible_by=pytorch_config['pad_vocab_size_multiple']
    nemo_config.model.hybrid_override_pattern="M"*nemo_config.model.num_layers

    nemo_config.model.use_cpu_initialization = True

    logging.info(f"Loading Mamba2 Pytorch checkpoint : `{args.input_name_or_path}`")
    

    if mmm:
        from safetensors.torch import load_file
        pytorch_model = load_file(args.input_name_or_path+'/pytorch_model.bin')
        trainer = MegatronLMPPTrainerBuilder(nemo_config).create_trainer()
    else:
        pytorch_model = torch.load(args.input_name_or_path+'/pytorch_model.bin', map_location='cpu')

    nemo_model_from_pyt = MegatronJambaModel(nemo_config.model, trainer)

    new_state_dict = {}

    if mmm:
        new_state_dict['model.embedding.word_embeddings.weight'] = pytorch_model['tok_embeddings.weight']
        new_state_dict['model.decoder.final_norm.weight'] = pytorch_model['norm.weight']
        new_state_dict['model.output_layer.weight'] = pytorch_model['output.weight']
        for i in range(nemo_config.model.num_layers):
            print(f'layer {i}')
            new_state_dict[f'model.decoder.layers.{i}.mixer.A_log'] = pytorch_model[f'layers.{i}.mixer.A_log']
            new_state_dict[f'model.decoder.layers.{i}.mixer.D'] = pytorch_model[f'layers.{i}.mixer.D']
            new_state_dict[f'model.decoder.layers.{i}.mixer.conv1d.weight'] = pytorch_model[f'layers.{i}.mixer.conv1d.weight']
            new_state_dict[f'model.decoder.layers.{i}.mixer.conv1d.bias'] = pytorch_model[f'layers.{i}.mixer.conv1d.bias']
            new_state_dict[f'model.decoder.layers.{i}.mixer.in_proj.weight'] = pytorch_model[f'layers.{i}.mixer.in_proj.weight']
            new_state_dict[f'model.decoder.layers.{i}.mixer.dt_bias'] = pytorch_model[f'layers.{i}.mixer.dt_bias']
            new_state_dict[f'model.decoder.layers.{i}.mixer.out_proj.weight'] = pytorch_model[f'layers.{i}.mixer.out_proj.weight']
            new_state_dict[f'model.decoder.layers.{i}.mixer.norm.weight'] = pytorch_model[f'layers.{i}.mixer.norm.weight']
            new_state_dict[f'model.decoder.layers.{i}.norm.weight'] = pytorch_model[f'layers.{i}.norm.weight']

    else:

        new_state_dict['model.embedding.word_embeddings.weight'] = pytorch_model['backbone.embedding.weight']
        new_state_dict['model.decoder.final_norm.weight'] = pytorch_model['backbone.norm_f.weight']
        new_state_dict['model.output_layer.weight'] = pytorch_model['lm_head.weight']
        for i in range(nemo_config.model.num_layers):

            new_state_dict[f'model.decoder.layers.{i}.mixer.A_log'] = pytorch_model[f'backbone.layers.{i}.mixer.A_log']
            new_state_dict[f'model.decoder.layers.{i}.mixer.D'] = pytorch_model[f'backbone.layers.{i}.mixer.D']
            new_state_dict[f'model.decoder.layers.{i}.mixer.conv1d.weight'] = pytorch_model[f'backbone.layers.{i}.mixer.conv1d.weight']
            new_state_dict[f'model.decoder.layers.{i}.mixer.conv1d.bias'] = pytorch_model[f'backbone.layers.{i}.mixer.conv1d.bias']
            new_state_dict[f'model.decoder.layers.{i}.mixer.in_proj.weight'] = pytorch_model[f'backbone.layers.{i}.mixer.in_proj.weight']
            new_state_dict[f'model.decoder.layers.{i}.mixer.dt_bias'] = pytorch_model[f'backbone.layers.{i}.mixer.dt_bias']
            new_state_dict[f'model.decoder.layers.{i}.mixer.out_proj.weight'] = pytorch_model[f'backbone.layers.{i}.mixer.out_proj.weight']
            new_state_dict[f'model.decoder.layers.{i}.mixer.norm.weight'] = pytorch_model[f'backbone.layers.{i}.mixer.norm.weight']
            new_state_dict[f'model.decoder.layers.{i}.norm.weight'] = pytorch_model[f'backbone.layers.{i}.norm.weight']
        
    nemo_model_from_pyt.load_state_dict(new_state_dict, strict=True)
    dtype = torch_dtype_from_precision(args.precision)
    nemo_model_from_pyt = nemo_model_from_pyt.to(dtype=dtype)
    nemo_model_from_pyt.save_to(args.output_path)
    logging.info(f'Mamba2 NeMo model saved to: {args.output_path}')


if __name__ == '__main__':
    args = get_args()
    convert(args)

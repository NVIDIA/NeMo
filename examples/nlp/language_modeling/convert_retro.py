import importlib
import os
import pathlib
import sys
from collections import OrderedDict
import yaml

import torch

enum_code = '''
import enum

class ModelType(enum.Enum):
    encoder_or_decoder = 1
    encoder_and_decoder = 2


class LayerType(enum.Enum):
    encoder = 1
    decoder = 2


class AttnType(enum.Enum):
    self_attn = 1
    cross_attn = 2


class AttnMaskType(enum.Enum):
    padding = 1
    causal = 2
'''


def install_megatron_dependence():
    # this is a hack to install required modules for MegatronLM checkpoints
    # run the following so we don't have to install Megatron_LM code
    megatron_name = 'megatron'
    megatron_spec = importlib.util.spec_from_loader(megatron_name, loader=None, is_package=True)

    megatron_module = importlib.util.module_from_spec(megatron_spec)
    sys.modules[megatron_name] = megatron_module

    model_name = 'model'
    model_spec = importlib.util.spec_from_loader(model_name, loader=None, is_package=True)

    model_module = importlib.util.module_from_spec(model_spec)

    megatron_module.__dict__['model'] = model_module

    sys.modules[megatron_name + '.' + model_name] = model_module

    enums_name = 'enums'
    enums_spec = importlib.util.spec_from_loader(enums_name, loader=None, is_package=True)
    enums_module = importlib.util.module_from_spec(enums_spec)

    model_module.__dict__['enums'] = enums_module

    sys.modules[megatron_name + '.' + model_name + '.' + enums_name] = enums_module

    exec(enum_code, enums_module.__dict__)


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


def set_weights(P, lm_checkpoint, new_checkpoint):
    pre_layer = min(P)
    count = 0
    for name in lm_checkpoint['state_dict'].keys():
        name = name.strip()
        parts = name.split('.')
        if parts[2] == 'embedding':
            # embedding layer
            parts[2] = 'encoder_embedding'
            del parts[1]
            new_checkpoint['.'.join(parts)] = lm_checkpoint['state_dict'][name]
        else:
            if len(parts) > 5 and parts[5] == 'retriever':
                parts[3] = 'model'
                del parts[4], parts[5]
                del parts[1]
                parts[3] = "layers"
                new_checkpoint['.'.join(parts)] = lm_checkpoint['state_dict'][name]
            elif parts[3] == 'final_layernorm':
                parts[1] = 'post_decoder.model'
                del parts[2]
                # parts[1] = 'encoder'
                # parts[2] = 'model'
                new_checkpoint['.'.join(parts)] = lm_checkpoint['state_dict'][name]
            elif parts[2] == "output_layer":
                del parts[1]
                new_checkpoint['.'.join(parts)] = lm_checkpoint['state_dict'][name]
            else:
                layer_number = int(parts[4])
                if layer_number < pre_layer:
                    parts[2] = 'pre_decoder.model'
                    del parts[1]
                    new_checkpoint['.'.join(parts)] = lm_checkpoint['state_dict'][name]
                elif layer_number == pre_layer:
                    if parts[5] == 'retriever':
                        # skip the extra retriever
                        continue
                    if count < 6: # used to be 8
                        parts[2] = 'pre_decoder.model'
                        del parts[1]
                        new_checkpoint['.'.join(parts)] = lm_checkpoint['state_dict'][name]
                    else:
                        parts[2] = 'post_decoder.model'
                        parts[4] = str(layer_number - pre_layer)
                        if parts[5] == 'inter_attention':
                            parts[5] = 'inter_attention.cross_attention'
                        del parts[1]
                        new_checkpoint['.'.join(parts)] = lm_checkpoint['state_dict'][name]
                    count += 1
                else:
                    parts[2] = 'post_decoder.model'
                    parts[4] = str(layer_number - pre_layer)
                    if parts[5] == 'inter_attention':
                        parts[5] = 'inter_attention.cross_attention'
                    del parts[1]
                    new_checkpoint['.'.join(parts)] = lm_checkpoint['state_dict'][name]


if __name__ == "__main__":
    # result_path = '/raid/users/aficek/gpt/gpt3-2b-pretraining-retro-fitting/converted'
    # lm_ckpt_path = '/raid/users/aficek/gpt/gpt3-2b-pretraining-retro-fitting/iter_0097656'
    # result_path = '/home/aficek/software/playground/retro_v2_800m_dir_2/'
    # result_path = '/gpt/megatron_retro_converted_9B/'
    # lm_ckpt_path = '/raid/users/aficek/gpt/retro_v2/gpt3-800m-pretraining-retro-fitting/iter_0195312'
    # lm_ckpt_path = '/gpt/megatron_lm_9B_retro_checkpoint/'
    # 800m retro
    lm_ckpt_path = '/home/aficek/software/playground/retro_convert/gpt3-800m-pretraining-retro-fitting/iter_0195312'
    result_path = '/home/aficek/software/playground/retro_convert/gpt3-800m-pretraining-retro-fitting/converted/'
    # 2b retro 
    # lm_ckpt_path = '/home/aficek/software/playground/retro_convert/gpt3-2b-pretraining-retro-fitting/iter_0097656'
    # result_path = '/home/aficek/software/playground/retro_convert/gpt3-2b-pretraining-retro-fitting/converted/'
    all_files = pathlib.Path(lm_ckpt_path).glob('**/model_optim_rng.pt')
    for lm_ckpt in all_files:
        subdir = pathlib.Path(lm_ckpt).parts[-2]
        # P = [8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 39]
        P = [8, 11, 14, 17, 20, 23]

        # install_megatron_dependence()
        old_checkpoint = torch.load(lm_ckpt, 'cuda:3')
        args_dict = vars(old_checkpoint["args"])
        with open(result_path + "adlr_model_config.yaml", 'w') as file:
            for key, value in args_dict.items():
                yaml.dump({key: str(value)}, file, default_flow_style=False)
        checkpoint = OrderedDict()
        checkpoint['state_dict'] = OrderedDict()
        total_params = [0]
        name_translate = {}
        name_translate['transformer'] = 'encoder'
        name_translate['.attention.'] = '.self_attention.'
        name_translate['word_embeddings_for_head'] = 'word_embeddings'
        parse_weights(old_checkpoint['model'], "", total_params, checkpoint['state_dict'], translator=name_translate)
        print('\n'.join(checkpoint['state_dict'].keys()))
        new_checkpoint = OrderedDict()
        set_weights(P, checkpoint, new_checkpoint)
        os.makedirs(result_path + subdir, exist_ok=True)
        vocab_size = new_checkpoint['model.encoder_embedding.word_embeddings.weight'].shape[0]
        new_checkpoint['model.tokens_head.bias'] = torch.zeros(256000, dtype=torch.float32, device='cuda:3')
        new_checkpoint['model.encoder.rotary_pos_emb.inv_freq'] = torch.zeros(16, dtype=torch.float32, device='cuda:3')
        new_checkpoint['model.pre_decoder.rotary_pos_emb.inv_freq'] = torch.zeros(16, dtype=torch.float32, device='cuda:3')
        new_checkpoint['model.post_decoder.rotary_pos_emb.inv_freq'] =torch.zeros(16, dtype=torch.float32, device='cuda:3')
        output_path = result_path + subdir + '/model_weights.ckpt'
        torch.save(new_checkpoint, output_path)
        print("Output to path: ", output_path)
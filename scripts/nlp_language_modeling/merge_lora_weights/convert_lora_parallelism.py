import tempfile
import re
from argparse import ArgumentParser

import torch
from omegaconf import OmegaConf

from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector

r"""
Script to convert TPx PPx LoRA checkpoint to TP1 PP1

python convert_lora_parallelism.py \
 --input_name_or_path <path to extracted, TP1 PP1 legacy checkpoint folder> \
 --output_path <path to output nemo file>
"""


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_path", type=str, default=None, required=True, help="Path to TPx PPx LoRA checkpoint",
    )
    parser.add_argument("--output_path", type=str, default=None, required=True, help="Path to output TP1 PP1 LoRA checkpoint")
    args = parser.parse_args()
    return args

def replace_number_add_offset(key, offset_value):
    # This function uses regular expression to find layer number in the state dict key
    # and replaces it with its value plus an offset value

    if offset_value == 0:
        return key

    # Define the pattern to match numbers in the string
    pattern = r'layers.(\d+)'

    # Function to be used as replacement
    # It converts the found number to integer, adds offset, and returns as string
    def add_offset(match):
        return "layers." + str(int(match.group(1)) + offset_value)

    # Use re.sub() to replace all occurrences of the pattern with the result of add_offset
    result_string = re.sub(pattern, add_offset, key)

    return result_string

def convert_lora_ckpt(lora_nemo, tp_size=None, pp_size=None):
    """
    Convert a lora checkpoint to TP1 PP1
    TODO (Only PP is supported for now)
    """
    lora_state_dict = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        NLPSaveRestoreConnector._unpack_nemo_file(lora_nemo, tmpdir)
        config_file = f"{tmpdir}/model_config.yaml"
        config = OmegaConf.load(config_file)
        if tp_size is None:
            # todo TP not supported yet
            tp_size = config.tensor_model_parallel_size
        if pp_size is None:
            pp_size =config.pipeline_model_parallel_size

        num_layers_per_pp_rank = config.num_layers // pp_size

        for i in range(pp_size):
            ckpt_file = f"{tmpdir}/tp_rank_00_pp_rank_{i:03d}/model_weights.ckpt"
            cur_state_dict = torch.load(ckpt_file, map_location=torch.device('cpu'))
            layer_offset = num_layers_per_pp_rank * i
            for key, value in cur_state_dict.items():
                new_key = replace_number_add_offset(key, layer_offset)
                lora_state_dict[new_key] = value

        return lora_state_dict

if __name__ == '__main__':
    args = get_args()
    ckpt = convert_lora_ckpt(args.input_path)
    torch.save(ckpt, args.output_path)
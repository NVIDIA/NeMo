import re

import torch
from ammo.torch.quantization.nn import QuantLinear, QuantLinearConvBase

from nemo.collections.multimodal.modules.stable_diffusion.attention import LinearWrapper

from .plugin_calib import PercentileCalibrator


def filter_func(name):
    pattern = re.compile(r".*(emb_layers.1|time_embed|input_blocks.0.0|^out\.1$|skip_connection|label_emb).*")
    return pattern.match(name) is not None


class _QuantNeMoLinearWrapper(QuantLinearConvBase):
    default_quant_desc_weight = QuantLinear.default_quant_desc_weight


AXES_NAME = {
    "nemo": {
        "x": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
        "timesteps": {0: "steps"},
        "y": {0: "batch_size"},
        "context": {0: "batch_size", 1: "sequence_length"},
    }
}


def generate_dummy_inputs(sd_version, device):
    dummy_input = {}
    if sd_version == "nemo":
        dummy_input["x"] = torch.ones(2, 4, 32, 32).to(device)  ## inference resolution is 256 for this example
        dummy_input["y"] = torch.ones(2, 1280).to(device)
        dummy_input["timesteps"] = torch.ones(2).to(device)
        dummy_input["context"] = torch.ones(2, 80, 2048).to(device)
    else:
        raise NotImplementedError(f"Unsupported sd_version: {sd_version}")

    return dummy_input


def load_calib_prompts(batch_size, calib_data_path="./calib_prompts.txt"):
    with open(calib_data_path, "r") as file:
        lst = [line.rstrip("\n") for line in file]
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]


def get_int8_config(model, quant_level=3, alpha=0.8, percentile=1.0, num_inference_steps=20, global_min=False):
    quant_config = {
        "quant_cfg": {
            "*lm_head*": {"enable": False},
            "*output_layer*": {"enable": False},
            "default": {"num_bits": 8, "axis": None},
        },
        "algorithm": {"method": "smoothquant", "alpha": alpha},
    }
    for name, module in model.named_modules():
        w_name = f"{name}*weight_quantizer"
        i_name = f"{name}*input_quantizer"

        if w_name in quant_config["quant_cfg"].keys() or i_name in quant_config["quant_cfg"].keys():
            continue
        if filter_func(name):
            continue
        if isinstance(module, (torch.nn.Linear, LinearWrapper)):
            if (
                (quant_level >= 2 and "ff.net" in name)
                or (quant_level >= 2.5 and ("to_q" in name or "to_k" in name or "to_v" in name))
                or quant_level == 3
            ):
                quant_config["quant_cfg"][w_name] = {"num_bits": 8, "axis": 0}
                quant_config["quant_cfg"][i_name] = {"num_bits": 8, "axis": -1}
        elif isinstance(module, torch.nn.Conv2d):
            quant_config["quant_cfg"][w_name] = {"num_bits": 8, "axis": 0}
            quant_config["quant_cfg"][i_name] = {
                "num_bits": 8,
                "axis": None,
                "calibrator": (
                    PercentileCalibrator,
                    (),
                    {
                        "num_bits": 8,
                        "axis": None,
                        "percentile": percentile,
                        "total_step": num_inference_steps,
                        "global_min": global_min,
                    },
                ),
            }
    return quant_config


def quantize_lvl(unet, quant_level=2.5):
    """
    We should disable the unwanted quantizer when exporting the onnx
    Because in the current ammo setting, it will load the quantizer amax for all the layers even
    if we didn't add that unwanted layer into the config during the calibration
    """
    for name, module in unet.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            module.input_quantizer.enable()
            module.weight_quantizer.enable()
        elif isinstance(module, (torch.nn.Linear, LinearWrapper)):
            if (
                (quant_level >= 2 and "ff.net" in name)
                or (quant_level >= 2.5 and ("to_q" in name or "to_k" in name or "to_v" in name))
                or quant_level == 3
            ):
                module.input_quantizer.enable()
                module.weight_quantizer.enable()
            else:
                module.input_quantizer.disable()
                module.weight_quantizer.disable()

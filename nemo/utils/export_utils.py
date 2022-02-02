# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Callable, Dict, Optional, Type
from enum import Enum

import onnx
import torch
import torch.nn as nn

from nemo.utils import logging
from nemo.utils.neural_type_utils import to_onnxrt_input

class ExportFormat(Enum):
    """Which format to use when exporting a Neural Module for deployment"""
    ONNX = (1,)
    TORCHSCRIPT = (2,)


_EXT_DICT = {
    ".pt": ExportFormat.TORCHSCRIPT,
    ".ts": ExportFormat.TORCHSCRIPT,
    ".onnx": ExportFormat.ONNX,
}


class BatchNorm1dNoAutoCast(nn.BatchNorm1d):
    def __init__(self, num_features, **kwargs):
        nn.BatchNorm1d.__init__(self, num_features, **kwargs)

    def forward(self, x):
        if torch.onnx.is_in_onnx_export():
            with torch.cuda.amp.autocast(enabled=False):
                ret = nn.BatchNorm1d.forward(self, x.to(torch.float))
        else:
            ret = nn.BatchNorm1d.forward(self, x)
        return ret

class LayerNormExp(nn.LayerNorm):
    def forward(self, input):
        if True: # torch.onnx.is_in_onnx_export():
            axes = tuple([-i for i in range(len(self.normalized_shape), 0, -1)])
            input.sub_(input.mean(axes, keepdim=True))
            with torch.cuda.amp.autocast(enabled=False):
                # variance = e((x - e(x))^2), and (x - e(x)) is the numerator in the layer_norm formula
                numerator=input.to(dtype=torch.float)
                variance = numerator.mul(numerator).mean(axes, keepdim=True)
                denominator = variance.add(self.eps).sqrt()
                return numerator.div(denominator).mul(self.weight).add(self.bias).to(dtype=input.dtype)
        else:    
            return nn.LayerNorm.forward(input)

def get_export_format(filename: str):
    _, ext = os.path.splitext(filename)
    try:
        return _EXT_DICT[ext]
    except KeyError:
        raise ValueError(f"Export file {filename} extension does not correspond to any export format!")

def augment_filename(output: str, prepend: str):
    path, filename = os.path.split(output)
    filename = f"{prepend}-{filename}"
    return os.path.join(path, filename)

def forward_method(self):
    if hasattr(self, "forward_for_export"):
        return self.forward_for_export
    else:
        return self.forward
    
        
def wrap_forward_method(self):
    tp = type(self)
    old_forward_method = None
    if hasattr(tp, "forward_for_export"):
        forward_method = tp.forward_for_export
        old_forward_method = tp.forward
        tp.forward = forward_method
    else:
        forward_method = None
    return forward_method, old_forward_method

def parse_input_example(input_example):
    input_list = list(input_example)
    input_dict = {}
    # process possible kwargs
    if isinstance(input_list[-1], dict):
        input_dict = input_list[-1]
        input_list = input_list[:-1]
    return input_list, input_dict

def verify_runtime(
    output, input_list, input_dict, input_names, output_names, output_example, check_tolerance=0.01,
):
    # Verify the model can be read, and is valid
    onnx_model = onnx.load(output)
    try:
        import onnxruntime
    except (ImportError, ModuleNotFoundError):
        logging.warning(f"ONNX generated at {output}, not verified - please install onnxruntime.\n")
        onnx.checker.check_model(onnx_model, full_check=True)
        return

    sess = onnxruntime.InferenceSession(onnx_model.SerializeToString(), providers=['CUDAExecutionProvider'])
    ort_out = sess.run(output_names, to_onnxrt_input(input_names, input_list, input_dict))
    all_good = True

    for i, out in enumerate(ort_out[0]):
        expected = output_example[i]
        if torch.is_tensor(expected):
            if not torch.allclose(
                    torch.from_numpy(out), expected.cpu(), rtol=check_tolerance, atol=100 * check_tolerance
            ):
                all_good = False
                logging.info(f"onnxruntime results mismatch! PyTorch(expected):\n{expected}\nONNXruntime:\n{out}")
    status = "SUCCESS" if all_good else "FAIL"
    logging.info(f"ONNX generated at {output} verified with onnxruntime : " + status)


apex_available = True

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm

    def replace_FusedLayerNorm(n: nn.Module) -> Optional[nn.BatchNorm2d]:
        """
        Replaces Apex's FusedLayerNorm with nn.LayerNorm. This is required for ONNX export.
        Args:
           n: the FusedLayerNorm pytorch module to replace
        Returns:
           Equivalent LayerNorm module
        """
        if not apex_available or not isinstance(n, FusedLayerNorm):
            return None

        dev = next(n.parameters()).device
        mod = nn.LayerNorm(n.normalized_shape, eps=n.eps, elementwise_affine=n.elementwise_affine,).to(dev)

        n_state = n.state_dict()
        mod.load_state_dict(n_state)
        return mod

    default_Apex_replacements = {"FusedLayerNorm": replace_FusedLayerNorm}

except Exception as e:
    default_Apex_replacements = {}
    apex_available = False
    
def expand_Conv1D(conv1d: nn.Module) -> Optional[nn.Conv2d]:
    """
    Expands a Conv1D into a Conv2D. This is required for many (closed source) commercial tools with poor support for 1D Convolutions in Onnx.
    Args:
        conv1d: the Conv1D pytorch module to expand
    Returns:
        conv2d: Conv2D module with identical weights and params
    """
    if not isinstance(conv1d, nn.Conv1d):
        return None
    conv2d = nn.Conv2d(
        conv1d.in_channels,
        conv1d.out_channels,
        kernel_size=(conv1d.kernel_size[0], 1),
        stride=(conv1d.stride[0], 1),
        padding=(conv1d.padding[0], 0),
        dilation=(conv1d.dilation[0], 1),
        groups=conv1d.groups,
        padding_mode=conv1d.padding_mode,
    ).to(device=conv1d.weight.device, dtype=conv1d.weight.dtype)
    conv2d.bias = conv1d.bias
    conv2d.weight = nn.Parameter(conv1d.weight.unsqueeze(-1))
    # check that expansion is valid
    for _ in range(2):
        sample_input = torch.rand(1, conv1d.in_channels, 256).to(
            device=conv1d.weight.device, dtype=conv1d.weight.dtype
        )
        close = conv1d(sample_input).mean() - conv2d(sample_input.unsqueeze(-1)).squeeze().mean()
        if close.abs() > 1.0:
            raise ValueError("Unable to expand Conv1D to Conv2D")
    return conv2d


def expand_BatchNorm1d(bn1d: nn.Module) -> Optional[nn.BatchNorm2d]:
    """
    Expands a BatchNorm1d into a BatchNorm2d. This is required for many (closed source) commercial tools with poor support for BatchNorm1d in Onnx.
    Args:
        bn1d: the BatchNorm1d pytorch module to expand
    Returns:
        bn2d: BatchNorm2d module with identical weights and params
    """
    if not isinstance(bn1d, nn.BatchNorm1d):
        return None
    mod = torch.nn.BatchNorm2d(
        bn1d.num_features,
        eps=bn1d.eps,
        momentum=bn1d.momentum,
        affine=bn1d.affine,
        track_running_stats=bn1d.track_running_stats,
    ).to(device=conv1d.weight.device, dtype=conv1d.weight.dtype)
    bn_state = bn1d.state_dict()
    mod.load_state_dict(bn_state)
    return mod


def expand_ConvTranspose1D(conv1d: nn.Module) -> Optional[nn.ConvTranspose2d]:
    """
    Expands a Conv1D into a Conv2D. This is required for many (closed source) commercial tools with poor support for 1D Convolutions in Onnx.
    Args:
        conv1d: the Conv1D pytorch module to expand
    Returns:
        conv2d: Conv2D module with identical weights and params
    """
    if not isinstance(conv1d, nn.ConvTranspose1d):
        return None
    conv2d = nn.ConvTranspose2d(
        conv1d.in_channels,
        conv1d.out_channels,
        kernel_size=(conv1d.kernel_size[0], 1),
        stride=(conv1d.stride[0], 1),
        padding=(conv1d.padding[0], 0),
        dilation=(conv1d.dilation[0], 1),
        groups=conv1d.groups,
        padding_mode=conv1d.padding_mode,
    ).to(device=conv1d.weight.device, dtype=conv1d.weight.dtype)
    conv2d.bias = conv1d.bias
    conv2d.weight = nn.Parameter(conv1d.weight.unsqueeze(-1))
    # check that expansion is valid
    for _ in range(2):
        sample_input = torch.rand(1, conv1d.in_channels, 256).to(
            device=conv1d.weight.device, dtype=conv1d.weight.dtype
        )
        close = conv1d(sample_input).mean() - conv2d(sample_input.unsqueeze(-1)).squeeze().mean()
        if close.abs() > 1.0:
            raise ValueError("Unable to expand Conv1D to Conv2D")
    return conv2d


def simple_replace(BaseT: Type[nn.Module], DestT: Type[nn.Module]) -> Callable[[nn.Module], Optional[nn.Module]]:
    """
    Generic function generator to replace BaseT module with DestT. BaseT and DestT should have same atrributes. No weights are copied.
    Args:
        BaseT : module type to replace
        DestT : destination module type
    Returns:
        swap function to replace BaseT module with DestT
    """

    def expansion_fn(mod: nn.Module) -> Optional[nn.Module]:
        if not isinstance(mod, BaseT):
            return None
        args = [getattr(mod, name, None) for name in mod.__constants__]
        out = DestT(*args)
        return out

    return expansion_fn


def swap_modules(model: nn.Module, mapping: Dict[str, nn.Module]):
    """
    This function swaps nested modules as specified by "dot paths" in mod with a desired replacement. This allows
    for swapping nested modules through arbitrary levels if children

    NOTE: This occurs in place, if you want to preserve model then make sure to copy it first.

    """
    for path, new_mod in mapping.items():
        expanded_path = path.split(".")
        parent_mod = model
        for sub_path in expanded_path[:-1]:
            parent_mod = parent_mod._modules[sub_path]  # noqa
        parent_mod._modules[expanded_path[-1]] = new_mod  # noqa

    return model


def replace_modules(
    model: nn.Module, expansions: Dict[str, Callable[[nn.Module], Optional[nn.Module]]] = None
) -> nn.Module:
    """
    Top-level function to replace modules in model, specified by class name with a desired replacement.
    NOTE: This occurs in place, if you want to preserve model then make sure to copy it first.
    Args:
        model : top level module
        expansions : replacement dictionary: module class name -> replacement function generator
    Returns:
        model, possibly modified in-place
    """
    mapping: Dict[str, nn.Module] = {}
    for name, m in model.named_modules():
        m_type = type(m).__name__
        if m_type in expansions:
            swapped = expansions[m_type](m)
            if swapped:
                mapping[name] = swapped
    logging.warning(f"Swapped {len(mapping)} modules")
    swap_modules(model, mapping)
    return model


default_1D_2D_replacements = {
    "Conv1d": expand_Conv1D,
    "ConvTranspose1d": expand_ConvTranspose1D,
    "BatchNorm1d": expand_BatchNorm1d,
    "AdaptiveAvgPool1d": simple_replace(nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d),
    "AvgPool1d": simple_replace(nn.AvgPool1d, nn.AvgPool2d),
}

def replace_for_export(model: nn.Module, replace_1D_2D: bool = False) -> nn.Module:
    """
    Top-level function to replace default set of modules in model
    NOTE: This occurs in place, if you want to preserve model then make sure to copy it first.
    Args:
        model : top level module
        replace_1D_2D : include 1D -> 2D replacements
    Returns:
        model, possibly modified in-place
    """
    replace_modules(model, default_Apex_replacements)
    if replace_1D_2D:
        # TODO: add squeeze/unsqueeze
        replace_modules(model, default_1D_2D_replacements)

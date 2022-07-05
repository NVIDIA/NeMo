# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from torch.nn import Linear

from nemo.collections.nlp.modules.common.megatron.module import MegatronModule

try:
    from apex.transformer import tensor_parallel

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


class MuReadout(Linear):
    '''Drop-in replacement for all output linear layers.

    An "output" linear layer is one that maps from a width dimension (e.g.,
    `d_model` in a Transformer) to a non-width dimension (e.g., vocab size).

    This layer implements the version of μP with a 1/width multiplier and a
    constant variance initialization for both weights and biases.
    '''

    def __init__(self, *args, readout_zero_init=False, output_mult=1.0, **kwargs):
        self.output_mult = output_mult
        self.readout_zero_init = readout_zero_init
        super().__init__(*args, **kwargs)

    def reset_parameters(self) -> None:
        if self.readout_zero_init:
            self.weight.data[:] = 0
            if self.bias is not None:
                self.bias.data[:] = 0
        else:
            super().reset_parameters()

    def width_mult(self):
        assert hasattr(self.weight, 'infshape'), (
            'Please call set_base_shapes(...). If using torch.nn.DataParallel, '
            'switch to distributed training with '
            'torch.nn.parallel.DistributedDataParallel instead'
        )
        return self.weight.infshape.width_mult()

    def _rescale_parameters(self):
        '''Rescale parameters to convert SP initialization to μP initialization.

        Warning: This method is NOT idempotent and should be called only once
        unless you know what you are doing.
        '''
        if hasattr(self, '_has_rescaled_params') and self._has_rescaled_params:
            raise RuntimeError(
                "`_rescale_parameters` has been called once before already. "
                "Unless you know what you are doing, usually you should not be calling `_rescale_parameters` more than once.\n"
                "If you called `set_base_shapes` on a model loaded from a checkpoint, "
                "or just want to re-set the base shapes of an existing model, "
                "make sure to set the flag `rescale_params=False`.\n"
                "To bypass this error and *still rescale parameters*, set `self._has_rescaled_params=False` before this call."
            )
        if self.bias is not None:
            self.bias.data *= self.width_mult() ** 0.5
        self.weight.data *= self.width_mult() ** 0.5
        self._has_rescaled_params = True

    def forward(self, x):
        return super().forward(self.output_mult * x / self.width_mult())


class MuSharedReadout(MuReadout):
    '''`MuReadout` with weights shared with an `nn.Embedding` layer.
    
    Inputs:
        weight: should be weight of an `nn.Embedding` layer
        other inputs are fed to `MuReadout`
    '''

    def __init__(self, weight, bias=True, **kwargs):
        super().__init__(*weight.shape, bias=bias, **kwargs)
        self.weight = weight


class MegatronTokenLevelHead(MegatronModule):
    """Masked LM head for token-based encoder-decoder models (e.g., T5)

    Arguments:
        parallel_output: wether output logits being distributed or not.
    """

    def __init__(self, weight, parallel_output):
        super(MegatronTokenLevelHead, self).__init__()
        self.linear = MuSharedReadout(weight)
        self.linear.bias.model_parallel = True
        self.linear.bias.partition_dim = 0
        self.linear.bias.stride = 1
        self.parallel_output = parallel_output

    def forward(self, hidden_states):
        # Parallel logits.
        input_parallel = tensor_parallel.copy_to_tensor_model_parallel_region(hidden_states)
        logits_parallel = self.linear(input_parallel)
        # Gather if needed.
        if self.parallel_output:
            return logits_parallel
        return tensor_parallel.gather_from_tensor_model_parallel_region(logits_parallel)


def rescale_linear_bias(linear):
    '''Rescale bias in nn.Linear layers to convert SP initialization to μP initialization.

    Warning: This method is NOT idempotent and should be called only once
    unless you know what you are doing.
    '''
    if hasattr(linear, '_has_rescaled_params') and linear._has_rescaled_params:
        raise RuntimeError(
            "`rescale_linear_bias` has been called once before already. Unless you know what you are doing, usually you should not be calling `rescale_linear_bias` more than once.\n"
            "If you called `set_base_shapes` on a model loaded from a checkpoint, or just want to re-set the base shapes of an existing model, make sure to set the flag `rescale_params=False`.\n"
            "To bypass this error and *still rescale biases*, set `linear._has_rescaled_params=False` before this call."
        )
    if linear.bias is None:
        return
    fanin_mult = linear.weight.infshape[1].width_mult()
    linear.bias.data *= fanin_mult ** 0.5
    linear._has_rescaled_params = True

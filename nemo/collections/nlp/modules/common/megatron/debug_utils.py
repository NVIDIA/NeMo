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

import torch


def get_forward_hook(name, trainer, rank):
    fp = open(f'debug_info/forward_{name}_rank{rank}.txt', 'w')
    header = False

    def forward_hook(module, inputs, outputs):
        nonlocal header
        nonlocal fp
        if trainer.training:
            values = []
            headers = []
            for i in inputs:
                if isinstance(i, torch.Tensor) and (
                    i.dtype == torch.float or i.dtype == torch.half or i.dtype == torch.bfloat16
                ):
                    if not header:
                        headers.append('input')
                    values.append(f'{i.data.norm()}')
            if isinstance(outputs, tuple):
                for i in outputs:
                    if isinstance(i, torch.Tensor) and (
                        i.dtype == torch.float or i.dtype == torch.half or i.dtype == torch.bfloat16
                    ):
                        if not header:
                            headers.append('output')
                        values.append(f'{i.data.norm()}')
            else:
                headers.append('output')
                values.append(f'{outputs.data.norm()}')
            values.append(f'{trainer.global_step}')
            if not header:
                headers.append('step')
                fp.write(','.join(headers) + '\n')
                header = True
            fp.write(','.join(values) + '\n')
        fp.flush()

    return forward_hook


def get_backward_hook(name, trainer, rank):
    fp = open(f'debug_info/backward_{name}_rank{rank}.txt', 'w')
    header = False

    def backward_hook(module, inputs, outputs):
        nonlocal header
        nonlocal fp
        if trainer.training:
            values = []
            headers = []
            for i in inputs:
                if isinstance(i, torch.Tensor) and (
                    i.dtype == torch.float or i.dtype == torch.half or i.dtype == torch.bfloat16
                ):
                    if not header:
                        headers.append('input')
                    values.append(f'{i.data.norm()}')
            if isinstance(outputs, tuple):
                for i in outputs:
                    if isinstance(i, torch.Tensor) and (
                        i.dtype == torch.float or i.dtype == torch.half or i.dtype == torch.bfloat16
                    ):
                        if not header:
                            headers.append('output')
                        values.append(f'{i.data.norm()}')
            else:
                headers.append('output')
                values.append(f'{outputs.data.norm()}')
            values.append(f'{trainer.global_step}')
            if not header:
                headers.append('step')
                fp.write(','.join(headers) + '\n')
                header = True
            fp.write(','.join(values) + '\n')
        fp.flush()

    return backward_hook


def get_tensor_hook(name, trainer, rank, weight):
    fp = open(f'debug_info/tensor_{name}_rank{rank}.txt', 'w')
    header = False

    def backward_hook(grad):
        nonlocal header
        nonlocal fp
        values = []
        values.append(f'{weight.data.norm()}')
        values.append(f'{grad.data.norm()}')
        values.append(f'{trainer.global_step}')
        fp.write(','.join(values) + '\n')
        fp.flush()
        return grad

    return backward_hook


def tensor_clip(name, trainer, rank, weight):
    def backward_hook(grad):
        norm = grad.data.norm()
        if norm > 1.0:
            grad = grad / norm
        return grad

    return backward_hook


def register_hooks(module, trainer):
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        for name, tensor in module.named_parameters():
            if name != '':
                tensor.register_hook(get_tensor_hook(name, trainer, rank, tensor))

        for name, layer in module.named_modules():
            if name != '':
                layer.register_forward_hook(get_forward_hook(name, trainer, rank))
                layer.register_backward_hook(get_backward_hook(name, trainer, rank))


def register_aggressive_gradient_clip_hooks(module, trainer):
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        for name, tensor in module.named_parameters():
            tensor.register_hook(tensor_clip(name, trainer, rank, tensor))

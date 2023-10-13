import _freqencoder as _backend
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd


class _freq_encoder(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)  # force float32 for better precision
    def forward(ctx, inputs, degree, output_dim):
        # inputs: [B, input_dim], float
        # RETURN: [B, F], float

        if not inputs.is_cuda:
            inputs = inputs.cuda()
        inputs = inputs.contiguous()

        B, input_dim = inputs.shape  # batch size, coord dim

        outputs = torch.empty(B, output_dim, dtype=inputs.dtype, device=inputs.device)

        _backend.freq_encode_forward(inputs, B, input_dim, degree, output_dim, outputs)

        ctx.save_for_backward(inputs, outputs)
        ctx.dims = [B, input_dim, degree, output_dim]

        return outputs

    @staticmethod
    # @once_differentiable
    @custom_bwd
    def backward(ctx, grad):
        # grad: [B, C * C]

        grad = grad.contiguous()
        inputs, outputs = ctx.saved_tensors
        B, input_dim, degree, output_dim = ctx.dims

        grad_inputs = torch.zeros_like(inputs)
        _backend.freq_encode_backward(grad, outputs, B, input_dim, degree, output_dim, grad_inputs)

        return grad_inputs, None, None


freq_encode = _freq_encoder.apply


class FreqEncoder(nn.Module):
    def __init__(self, input_dim=3, degree=4):
        super().__init__()

        self.input_dim = input_dim
        self.degree = degree
        self.output_dim = input_dim + input_dim * 2 * degree

    def __repr__(self):
        return f"FreqEncoder: input_dim={self.input_dim} degree={self.degree} output_dim={self.output_dim}"

    def forward(self, inputs, **kwargs):
        # inputs: [..., input_dim]
        # return: [..., ]

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.reshape(-1, self.input_dim)

        outputs = freq_encode(inputs, self.degree, self.output_dim)

        outputs = outputs.reshape(prefix_shape + [self.output_dim])

        return outputs

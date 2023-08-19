# This implementation is based on the TRT-LLM Linear and RowLinear layers.
# https://gitlab-master.nvidia.com/ftp/tekit/-/blob/main/tensorrt_llm/layers/linear.py

import tensorrt as trt
from tensorrt_llm._utils import int32_array
from tensorrt_llm.functional import allgather, allreduce, cast, concat, constant, matmul, mul, shape, slice
from tensorrt_llm.layers import Linear, RowLinear
from tensorrt_llm.parameter import Parameter
from tensorrt_llm.quantization.functional import dequantize, quantize


class Int8SmoothQuantLinear(Linear):
    """
    Quantized Linear layer with smooth quantization.

    Args:
        in_features: size of input features
        out_features: size of output features
        bias: If set to ``False``, the layer will not learn an additive bias.
        dtype: the dtype of the layer, default is ``trt.float32``
        tp_group: the tensor parallel group
        tp_size: the tensor parallel size
        gather_output: whether to gather the output of the layer

    Attributes:
        activation_scaling_factor: the scaling factor for the activation, expected to be a scalar
        weights_scaling_factor: the scaling factor for the weights, expected to be a 1D tensor
        prequant_scaling_factor: prequantization scales to be multiplied with activations, expected to be a 1D tensor
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        dtype=None,
        tp_group=None,
        tp_size=1,
        gather_output=True,
    ):
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            dtype=dtype,
            tp_group=tp_group,
            tp_size=tp_size,
            gather_output=gather_output,
        )
        self.activation_scaling_factor = Parameter(shape=(1,), dtype=trt.float32)
        self.weights_scaling_factor = Parameter(shape=(self.out_features,), dtype=trt.float32)
        self.prequant_scaling_factor = Parameter(shape=(self.in_features,), dtype=trt.float32)

    def forward(self, x):
        act_cast_out = cast(x, "float32")
        # Do this multiplication only if prequant_scaling_factor is not None
        scaled_act_cast_out = mul(act_cast_out, self.prequant_scaling_factor.value)

        quantized_out = quantize(scaled_act_cast_out, self.activation_scaling_factor.value, "int8")
        dequantized_out = dequantize(quantized_out, self.activation_scaling_factor.value)

        w_cast_out = cast(self.weight.value, "float32")
        w_quant_out = quantize(w_cast_out, self.weights_scaling_factor.value, "int8", axis=0)
        w_deq_out = dequantize(w_quant_out, self.weights_scaling_factor.value, axis=0)

        x = matmul(dequantized_out, w_deq_out, transb=True)

        if self.bias is not None:
            x = x + self.bias.value

        if self.gather_output and self.tp_size > 1 and self.tp_group is not None:
            # 1. [dim0, local_dim] -> [dim0 * tp_size, local_dim]
            x = allgather(x, self.tp_group)

            # 2. [dim0 * tp_size, local_dim] -> [dim0, local_dim * tp_size]
            # 2.1 split
            split_size = shape(x, dim=0) / self.tp_size
            ndim = x.ndim()
            starts = [constant(int32_array([0])) for _ in range(ndim)]
            sizes = [shape(x, dim=d) for d in range(ndim)]
            sizes[0] = split_size
            sections = []
            for i in range(self.tp_size):
                starts[0] = split_size * i
                sections.append(slice(x, concat(starts), concat(sizes)))
            # 2.2 concat
            x = concat(sections, dim=1)

        return x


class Int8SmoothQuantRowLinear(RowLinear):
    """
    Quantized RowLinear layer with smooth quantization.

    Args:
        in_features: size of input features
        out_features: size of output features
        bias: If set to ``False``, the layer will not learn an additive bias.
        dtype: the dtype of the layer, default is ``trt.float32``
        tp_group: the tensor parallel group
        tp_size: the tensor parallel size

    Attributes:
        activation_scaling_factor: the scaling factor for the activation, expected to be a scalar
        weights_scaling_factor: the scaling factor for the weights, expected to be a 1D tensor
        prequant_scaling_factor: prequantization scales to be multiplied with activations, expected to be a 1D tensor
    """

    def __init__(self, in_features, out_features, bias=True, dtype=None, tp_group=None, tp_size=1):
        super().__init__(in_features, out_features, bias=bias, dtype=dtype, tp_group=tp_group, tp_size=tp_size)
        self.activation_scaling_factor = Parameter(shape=(1,), dtype=trt.float32)

        self.weights_scaling_factor = Parameter(shape=(self.out_features,), dtype=trt.float32)
        self.prequant_scaling_factor = Parameter(shape=(self.in_features,), dtype=trt.float32)

    def forward(self, x):
        act_cast_out = cast(x, "float32")

        # Do this multiplication only if prequant_scaling_factor is not None
        scaled_act_cast_out = mul(act_cast_out, self.prequant_scaling_factor.value)

        quantized_out = quantize(scaled_act_cast_out, self.activation_scaling_factor.value, "int8")

        dequantized_out = dequantize(quantized_out, self.activation_scaling_factor.value)

        w_cast_out = cast(self.weight.value, "float32")

        w_quant_out = quantize(w_cast_out, self.weights_scaling_factor.value, "int8", axis=0)
        w_deq_out = dequantize(w_quant_out, self.weights_scaling_factor.value, axis=0)

        x = matmul(dequantized_out, w_deq_out, transb=True)

        if self.tp_size > 1 and self.tp_group is not None:
            x = allreduce(x, self.tp_group)

        if self.bias is not None:
            x = x + self.bias.value

        return x

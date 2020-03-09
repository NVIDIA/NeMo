import torch
from torch.utils.data import Dataset

from ....core.neural_types import *
from ...pytorch.nm import DataLayerNM


def neuralType2TensorShape(neural_type: NeuralType, default_dim=32, skip_batch_axis=True) -> torch.Size:
    """
    Converts Neural Type to torch tensor shape.
    Args:
      neural_type: input Neural Type
      default_dim: default dimension to use if not specified by Neural Type
      skip_batch_axis: (default: True) whether to skip batch axis is resulting
      shape.

    Returns:
      torch.Size
    """
    dims = []
    for axis in neural_type.axes:
        if axis.kind == AxisKind.Batch and skip_batch_axis:
            continue
        if axis.size is not None:
            dims.append(axis.size)
        else:
            dims.append(default_dim)
    return torch.Size(dims)


class _ZeroDS(Dataset):
    def __init__(self, size, shapes, dtype):
        Dataset.__init__(self)
        self._size = size
        self._tensor_shapes = shapes
        self._type = dtype

    def __getitem__(self, index):
        if self._type is not None:
            if not isinstance(self._type, list):
                types = [self._type] * len(self._tensor_shapes)
            else:
                types = self._type
        else:
            types = [torch.FloatTensor] * len(self._tensor_shapes)
        res = []
        for ts, tp in zip(self._tensor_shapes, types):
            res.append(torch.zeros(ts).type(tp))
        return tuple(res)

    def __len__(self):
        return self._size


class ZerosDataLayer(DataLayerNM):
    """
    DataLayer Neural Module which emits zeros.
    This module should be used for debugging/benchmarking purposes.

    Args:
        size: (int) size of the underlying dataset
        output_ports: which output ports it should have
        dtype: Dtype of the output tensors.
        batch_size (int): Size of batches to output.
        shapes: If None, this will be inferred from output_ports. Else,
            specifies the shape of the output tensors.
            Defaults to None.
    """

    def __init__(self, size, output_ports, dtype, batch_size, shapes=None):
        self._output_ports = output_ports
        DataLayerNM.__init__(self)
        self._size = size
        self._type = dtype
        self._batch_size = batch_size
        self._shapes = shapes
        if self._shapes is None:
            self._shapes = [neuralType2TensorShape(pval) for pname, pval in self._output_ports.items()]

        self._dataset = _ZeroDS(size=self._size, shapes=self._shapes, dtype=self._type)

    @property
    def input_ports(self):
        return {}

    @property
    def output_ports(self):
        return self._output_ports

    def __len__(self):
        return len(self._dataset)

    @property
    def data_iterator(self):
        return None

    @property
    def dataset(self):
        return self._dataset

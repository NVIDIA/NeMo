# Copyright (c) 2019 NVIDIA Corporation
import torch as t
import torch.nn as nn

from ...core import NeuralModule, DeviceType
from ...utils.helpers import rgetattr, rsetattr


class TrainableNeuralModuleWrapper(NeuralModule, nn.Module):
    """This class wraps an instance of Pytorch's nn.Module and
    returns NeuralModule's instance."""

    @staticmethod
    def create_ports():
        """
        Ports are passed in the constructor and handled there
        """
        return {}, {}

    def __init__(self, pt_nn_module, input_ports_dict, output_ports_dict,
                 **kwargs):
        NeuralModule.__init__(self, **kwargs)
        nn.Module.__init__(self)
        self._input_ports = input_ports_dict
        self._output_ports = output_ports_dict
        self._device = t.device(
            "cuda" if self.placement in [DeviceType.GPU, DeviceType.AllGpu]
            else "cpu"
        )
        self._pt_module = pt_nn_module
        self._pt_module.to(self._device)

    # def forward(self, *input):
    #  return self._pt_module(input)

    def eval(self):
        return self._pt_module.eval()

    def train(self):
        return self._pt_module.train()

    def __call__(self, force_pt=False, *input, **kwargs):
        pt_call = len(input) > 0 or force_pt
        if pt_call:
            return self._pt_module.__call__(*input, **kwargs)
        else:
            return NeuralModule.__call__(self, **kwargs)

    def get_weights(self):
        result = dict()
        for name, parameter in self.named_parameters():
            result[name] = (parameter, parameter.requires_grad)
        return result

    def save_to(self, path):
        t.save(self._pt_module.state_dict(), path)

    def restore_from(self, path):
        self._pt_module.load_state_dict(t.load(path))

    def parameters(self):
        return self._pt_module.parameters()

    def named_parameters(self):
        return self._pt_module.named_parameters()

    def freeze(self, weights=None):
        for name, param in self._pt_module.named_parameters():
            if weights is None or name in weights:
                param.requires_grad = False

    def unfreeze(self, weights=None):
        for name, param in self._pt_module.named_parameters():
            if weights is None or name in weights:
                param.requires_grad = True

    def get_weights(self):
        result = dict()
        for name, parameter in self._pt_module.named_parameters():
            result[name] = (parameter, parameter.requires_grad)
        return result

    def set_weights(self, name2weight, name2name_and_transform=None):
        self._pt_module.load_state_dict(
            {key: name2weight[key][0] for key in name2weight.keys()}
        )

    def tie_weights_with(self, module, weight_names):
        for name in weight_names:
            rsetattr(self._pt_module, name, rgetattr(module, name))

    @property
    def num_weights(self):
        return sum(
            p.numel() for p in self._pt_module.parameters() if p.requires_grad)

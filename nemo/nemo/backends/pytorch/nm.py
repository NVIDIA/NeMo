# Copyright (c) 2019 NVIDIA Corporation
import logging
from abc import abstractmethod
from typing import Dict, Set, Tuple, Optional, List

import torch as t
import torch.nn as nn

from ...core import NeuralModule, DeviceType, WeightShareTransform
from ...utils.helpers import rgetattr, rsetattr, get_cuda_device


class TrainableNM(NeuralModule, nn.Module):
    """A helper Base class for NeuralModule's based on Pytorch's nn.Module.

    If you have a Pytorch class which derives from nn.Module you can
    covert it into a NeuralModule, by replacing inheriting from this class
    instead

    Your constructor then should look like this:

    .. code-block:: python

      def __init__(self, **kwargs):
        TrainableNM.__init__(self, **kwargs)
        self._input_ports = {..}
        self._output_ports = {..}
        .... # you code

    Then make sure that your forward(..) method accepts arguments named like
    input ports.
    """

    def __init__(self, **kwargs):
        NeuralModule.__init__(self, **kwargs)  # For NeuralModule API
        nn.Module.__init__(self)  # For PyTorch API
        self._device = get_cuda_device(self.placement)

    def __call__(self, force_pt=False, *input, **kwargs):
        pt_call = len(input) > 0 or force_pt
        if pt_call:
            return nn.Module.__call__(self, *input, **kwargs)
        else:
            return NeuralModule.__call__(self, **kwargs)

    def get_weights(self):
        result = dict()
        for name, parameter in self.named_parameters():
            result[name] = (parameter, parameter.requires_grad)
        return result

    def set_weights(self, name2weight, name2name_and_transform=None):
        if name2weight is not None and len(name2weight) > 0:
            if name2name_and_transform is None:
                self.load_state_dict(
                    {key: name2weight[key][0] for key in name2weight.keys()}
                )
            else:
                self.load_state_dict(
                    {key: name2weight[key][0] for key in name2weight.keys()}
                )

    def tie_weights_with(self, module, weight_names,
                         name2name_and_transform=None):
        if module is None:
            raise ValueError("Module to tie weights can't be None")
        if weight_names is None or len(weight_names) == 0:
            raise ValueError("Please provide weight names to tie")

        if name2name_and_transform is None:
            for name in weight_names:
                rsetattr(self, name, rgetattr(module, name))
        else:
            for self_w_name in weight_names:
                if self_w_name in name2name_and_transform:
                    if (
                            name2name_and_transform[self_w_name][1]
                            == WeightShareTransform.SAME
                    ):
                        rsetattr(
                            self,
                            self_w_name,
                            rgetattr(module,
                                     name2name_and_transform[self_w_name][0]),
                        )
                    elif (
                            name2name_and_transform[self_w_name][1]
                            == WeightShareTransform.TRANSPOSE
                    ):
                        raise NotImplementedError(
                            "Sorry, currently this is not implemented."
                        )
                else:
                    rsetattr(self, self_w_name, rgetattr(module, self_w_name))

    def save_to(self, path):
        # t.save(self._pt_module.state_dict(), path)
        t.save(self.state_dict(), path)

    def restore_from(self, path, local_rank=0):
        # self._pt_module.load_state_dict(t.load(path))
        if self.placement == DeviceType.AllGpu:
            load_device = f"cuda:{local_rank}"
        else:
            load_device = self._device
        self.load_state_dict(t.load(path, map_location=load_device))

    def freeze(self, weights=None):
        if hasattr(self, "_pt_module"):
            for name, param in self._pt_module.named_parameters():
                if weights is None or name in weights:
                    param.requires_grad = False
        else:
            for name, param in self.named_parameters():
                if weights is None or name in weights:
                    param.requires_grad = False

    def unfreeze(self, weights=None):
        if hasattr(self, "_pt_module"):
            for name, param in self._pt_module.named_parameters():
                if weights is None or name in weights:
                    param.requires_grad = True
        else:
            for name, param in self.named_parameters():
                if weights is None or name in weights:
                    param.requires_grad = True

    @property
    def num_weights(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class NonTrainableNM(NeuralModule):
    def __init__(self, **kwargs):
        NeuralModule.__init__(self, **kwargs)  # For NeuralModule API
        self._device = get_cuda_device(self.placement)

    def __call__(self, force_pt=False, *input, **kwargs):
        pt_call = len(input) > 0 or force_pt
        if pt_call:
            with t.no_grad():
                return self.forward(*input, **kwargs)
        else:
            return NeuralModule.__call__(self, **kwargs)

    def forward(self, *input):
        """Defines the computation performed at every call.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def get_weights(self) -> Optional[Dict[(str, bool)]]:
        None

    def set_weights(self, name2weight: Dict[(str, Tuple[str, bool])],
                    name2name_and_transform: Dict[
                        (str, Tuple[str, WeightShareTransform])] = None):
        pass

    def tie_weights_with(self, module, weight_names=List[str],
                         name2name_and_transform: Dict[
                             (str, Tuple[str, WeightShareTransform])] = None):
        pass

    def save_to(self, path: str):
        pass

    def restore_from(self, path: str):
        pass

    def freeze(self, weights: Set[str] = None):
        pass

    def unfreeze(self, weights: Set[str] = None):
        pass

    @property
    def num_weights(self):
        return 0


class DataLayerNM(NeuralModule):
    """A helper Base class for creating Pytorch-based data layers.
    You must implement __len__ method to return dataset size and
    data_iterator property to return iterator over the dataset.
    """

    def __init__(self, **kwargs):
        # if 'batch_size' not in kwargs:
        #    logging.warning("No batch_size specified in the data layer. "
        #                    "Setting batch_size to 1.")
        #    kwargs['batch_size'] = 1
        NeuralModule.__init__(self, **kwargs)  # For NeuralModule API
        self._device = get_cuda_device(self.placement)

    def get_weights(self):
        logging.warning(
            "Data Layer does not have any weights to return. "
            "This get_weights call returns None."
        )
        return None

    def set_weights(self, name2weight: Dict[(str, bool)],
                    name2name_and_transform):
        logging.warning(
            "Data Layer does not have any weights to set. "
            "This set_weights call is ignored."
        )
        return None

    def tie_weights_with(self, module, weight_names):
        logging.warning(
            "Data Layer does not have any weights to tie. "
            "This tie_weights_with call is ignored."
        )
        return None

    def save_to(self, path):
        logging.warning(
            "Data Layer does not have any state to save. "
            "This save_to call is ignored."
        )
        return None

    def restore_from(self, path):
        raise NotImplementedError(
            "Data Layer could not be restored from any saved " "state."
        )
        return None

    def freeze(self, weights: Set[str] = None):
        logging.warning(
            "Data Layer does not have any weights to freeze. "
            "This freeze call is ignored."
        )
        return None

    def unfreeze(self, weights: Set[str] = None):
        logging.warning(
            "Data Layer does not have any weights to unfreeze. "
            "This unfreeze call is ignored."
        )
        return None

    @property
    def num_weights(self):
        return 0

    @abstractmethod
    def __len__(self):
        """Dataset size"""
        pass

    @property
    @abstractmethod
    def dataset(self):
        """Should return an instance of torch.utils.data.Dataset. Should
        implement
        either this or `data_iterator`. If this is implemented, `data_iterator`
        should return None."""
        pass

    @property
    @abstractmethod
    def data_iterator(self):
        """"Iterator over the dataset. It is a good idea to return
        torch.utils.data.DataLoader here. Should implement either this or
        `dataset`.
        If this is implemented, `dataset` property should return None.
        """


class LossNM(NeuralModule):
    """A helper Base class for creating Pytorch-based loss function modules.
    You must implement _loss_function method.
    """

    def __init__(self, **kwargs):
        NeuralModule.__init__(self, **kwargs)  # For NeuralModule API
        self._device = get_cuda_device(self.placement)

    def get_weights(self):
        # logging.warning("Loss function module does not have any weights to
        # return. "
        #                "This get_weights call returns None.")
        return None

    def set_weights(self, name2weight: Dict[(str, bool)],
                    name2name_and_transform):
        logging.warning(
            "Loss function module does not have any weights to set. "
            "This set_weights call is ignored."
        )
        return None

    def tie_weights_with(self, module, weight_names):
        logging.warning(
            "Loss function module does not have any weights to tie. "
            "This tie_weights_with call is ignored."
        )
        return None

    def save_to(self, path):
        logging.warning(
            "Loss function module does not have any state to save. "
            "This save_to call is ignored."
        )
        return None

    def restore_from(self, path):
        raise NotImplementedError(
            "Loss function module could not be restored from " "any saved "
            "state."
        )
        return None

    def freeze(self, weights: Set[str] = None):
        logging.warning(
            "Loss function module does not have any weights to freeze. "
            "This freeze call is ignored."
        )
        return None

    def unfreeze(self, weights: Set[str] = None):
        logging.warning(
            "Loss function module does not have any weights to "
            "unfreeze. This unfreeze call is ignored."
        )
        return None

    @property
    def num_weights(self):
        return 0

    @abstractmethod
    def _loss_function(self, **kwargs):
        pass

    def __call__(self, force_pt=False, *input, **kwargs):
        if force_pt:
            return self._loss_function(**kwargs)
        else:
            return NeuralModule.__call__(self, **kwargs)

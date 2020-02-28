from typing import Set, Dict, Tuple, Optional, List
import uuid
from nemo.core import NeuralModule, WeightShareTransform, NeuralType


class CompositeNeuralModule(NeuralModule):
    def __init__(self, modules, pipeline):
        self.modules = modules
        self.pipeline = pipeline
        self.__input_ports = modules[0].input_ports
        self.__output_ports = modules[-1].output_ports

    def __call__(self, *args, **kwargs):
        return self.pipeline(*args, **kwargs)

    # def __enter__(self):
    #     return self
    #
    # def __exit__(self, type, value, traceback):
    #     return None

    @property
    def input_ports(self) -> Optional[Dict[str, NeuralType]]:
        return self.__input_ports

    @property
    def output_ports(self) -> Optional[Dict[str, NeuralType]]:
        return self.__output_ports

    def get_weights(self) -> Optional[Dict[(str, bool)]]:
        pass

    def set_weights(self, name2weight: Dict[(str, Tuple[str, bool])],
                    name2name_and_transform: Dict[(str, Tuple[str, WeightShareTransform])] = None):
        pass

    def tie_weights_with(self, module, weight_names=List[str],
                         name2name_and_transform: Dict[(str, Tuple[str, WeightShareTransform])] = None):
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
        pass
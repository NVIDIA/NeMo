from abc import ABC
from re import L
from typing import Dict, List, Optional

from nemo.core.classes import NeuralModule
from nemo.core.neural_types import ChannelType, MaskType, NeuralType

__all__ = ['MegatronEncoderModule']


class MegatronEncoderModule(NeuralModule, ABC):
    """ Base class for encoder neural module to be used in NLP models. """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "encoder_mask": NeuralType(('B', 'T'), MaskType()),
        }

    @property
    def input_names(self) -> List[str]:
        return ['input_ids', 'encoder_mask']

    @property
    def output_names(self) -> List[str]:
        return ['encoder_output']

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"encoder_output": NeuralType(('B', 'T', 'D'), ChannelType())}

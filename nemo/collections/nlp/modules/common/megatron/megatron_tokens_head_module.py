from abc import ABC
from typing import Dict, List, Optional

from nemo.core.classes import NeuralModule
from nemo.core.neural_types import ChannelType, MaskType, NeuralType

__all__ = ['MegatronTokensHeadModule']


class MegatronTokensHeadModule(NeuralModule, ABC):
    """ Base class for encoder neural module to be used in NLP models. """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "dec_output": NeuralType(('B', 'T', 'D'), ChannelType()),
            "embeddings_weights": NeuralType(('T', 'D'), MaskType()),
        }

    @property
    def input_names(self) -> List[str]:
        return ['dec_output', 'embeddings_weights']

    @property
    def output_names(self) -> List[str]:
        return ['logits']

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"logits": NeuralType(('B', 'T', 'D'), ChannelType())}

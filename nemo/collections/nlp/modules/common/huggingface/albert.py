from transformers import (
    AlbertModel,
)

from nemo.utils.decorators import experimental

from nemo.core.classes import NeuralModule, typecheck
from nemo.collections.nlp.modules.common.huggingface.bert import BertModule


__all__ = ['AlbertEncoder']

@experimental
class AlbertEncoder(AlbertModel, BertModule):
    """
    ALBERT wraps around the Huggingface implementation of ALBERT from their
    transformers repository for easy use within NeMo.
    Args:
        pretrained_model_name (str): If using a pretrained model, this should
            be the model's name. Otherwise, should be left as None.
        config_filename (str): path to model configuration file. Optional.
        vocab_size (int): Size of the vocabulary file, if not using a
            pretrained model.
        hidden_size (int): Size of the encoder and pooler layers.
        num_hidden_layers (int): Number of hidden layers in the encoder.
        num_attention_heads (int): Number of attention heads for each layer.
        intermediate_size (int): Size of intermediate layers in the encoder.
        hidden_act (str): Activation function for encoder and pooler layers;
            "gelu", "relu", and "swish" are supported.
        max_position_embeddings (int): The maximum number of tokens in a
        sequence.
    """


    def save_to(self, save_path: str):
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        pass

    @typecheck()
    def forward(self, **kwargs):
        res = super().forward(**kwargs)
        return res
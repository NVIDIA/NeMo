from typing import Dict, Optional
from nemo import logging

from transformers import BertModel
from nemo.core.classes import NeuralModule, typecheck
from nemo.utils.decorators import experimental

from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types import ChannelType, NeuralType
from nemo.utils.decorators import experimental
import json

from nemo.collections.nlp.modules.common.megatron.megatron_utils import get_megatron_lm_models_list, get_megatron_config_file
from nemo.collections.nlp.modules.common.huggingface.huggingface_utils import get_huggingface_lm_models_list, get_huggingface_lm_model

__megatron_utils_satisfied = False

try:
    __megatron_utils_satisfied = True
    from nemo.collections.nlp.modules.common.megatron.megatron_bert import MegatronBERT
    from nemo.collections.nlp.modules.common.megatron.megatron_utils import *

except Exception as e:
    logging.error('Failed to import Megatron Neural Module and utils: `{}` ({})'.format(str(e), type(e)))
    __megatron_utils_satisfied = False

__all__ = ['BERTEncoder']


@experimental
class BERTEncoder(NeuralModule):
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
    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "token_type_ids": NeuralType(('B', 'T'), ChannelType()),
            "attention_mask": NeuralType(('B', 'T'), ChannelType(), optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "last_hidden_states": NeuralType(('B', 'T', 'D'), ChannelType()),
            "pooler_output ": NeuralType(('B', 'D'), ChannelType()),
            "hidden_states ": NeuralType(('B', 'T', 'D'), ChannelType()),
            "attentions ": NeuralType(('B', 'H', 'T', 'D'), ChannelType()),
        }

    def __init__(
        self,
        pretrained_model_name,
        config_file=None,
        params={}
    ):
        super().__init__()
        self._model = None

        if pretrained_model_name in get_huggingface_lm_models_list():
            self._model_type = "HFBERT"

            total = 0
            if pretrained_model_name is not None:
                total += 1
            if config_file is not None:
                total += 1

            if total != 1:
                raise ValueError(
                    "Only one of pretrained_model_name "
                    + "or config_filename should be passed into the "
                    + "BERTEncoder constructor."
                )

            self._model = get_huggingface_lm_model(bert_config=config_file, pretrained_model_name=pretrained_model_name)
        elif __megatron_utils_satisfied and pretrained_model_name in get_megatron_lm_models_list():
            self._model_type = "MEGATRON"
            if pretrained_model_name == 'megatron-bert-cased' or pretrained_model_name == 'megatron-bert-uncased':
                if not (config_file and "checkpoint" in params):
                    raise ValueError(f'Both the config file and the pretrained checkpoint are required for Megatron model of {pretrained_model_name}')
            if not config_file:
                config_file = get_megatron_config_file(pretrained_model_name)
            if isinstance(config_file, str):
                with open(config_file) as f:
                    config = json.load(f)
            if "vocab" not in params:
                vocab = get_megatron_vocab_file(pretrained_model_name)
            if "checkpoint" not in params:
                checkpoint = get_megatron_checkpoint(pretrained_model_name)
            self._model = MegatronBERT(
                model_name=pretrained_model_name,
                vocab_file=vocab,
                hidden_size=config['hidden-size'],
                num_attention_heads=config['num-attention-heads'],
                num_layers=config['num-layers'],
                max_seq_length=config['max-seq-length'],
            )
        else:
            raise ValueError(f'{pretrained_model_name} is not supported')

        if checkpoint:
            self._model.restore_from(checkpoint)
            logging.info(f"{pretrained_model_name} model restored from {checkpoint}")

        # TODO: what happens to device?
        # self.to(self._device) # sometimes this is necessary

    @typecheck()
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if self._model_type in ["bert", "roberta"]:
            out = self._model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=None)
        elif self._model_type in ["megataron"]:
            out = self._model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=None)

        return out

    def resize_token_embeddings(self, new_vocab_size):
        """ Resize input token embeddings matrix of the model if new_num_tokens != config.vocab_size.
        Take care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.
        Args:
            new_vocab_size: (`optional`) int:
                New number of tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end.
                If not provided or None: does nothing and just returns a pointer to the input tokens
                ``torch.nn.Embeddings`` Module of the model.
        Return: ``torch.nn.Embeddings``
            Pointer to the input tokens Embeddings Module of the model
        """
        return self.bert.resize_token_embeddings(new_vocab_size)

    @property
    def hidden_size(self):
        """
            Property returning hidden size.
            Returns:
                Hidden size.
        """
        return self._hidden_size

    @classmethod
    def save_to(self, save_path: str):
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        pass


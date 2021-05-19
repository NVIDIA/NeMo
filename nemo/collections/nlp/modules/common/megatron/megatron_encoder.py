# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional


from nemo.collections.nlp.modules.common.encoder_module import EncoderModule
from nemo.collections.nlp.modules.common.megatron.megatron_utils import get_megatron_lm_model
from nemo.core.classes.common import typecheck
from nemo.utils import logging


class MegatronEncoderModule(EncoderModule):
    """ Class for using Megatron encoders in NeMo NLP."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        pretrained: bool = True,
        config_dict: Optional[dict] = None,
        checkpoint_file: Optional[str] = None,
    ):
        """Gets Megatron BERT based model to be used as an Encoder in NeMo NLP.
        Use the model_name arg to get a named model architecture. 
        Available model names can be found with get_megatron_lm_models_list() or
        Use the pretrained arg to get the named model architecture with or without pretrained weights.

        If not using a pretrained model, then model_name is None, then we can pass in a custom configuration via the config_dict.
        For example, to instantiate a Megatron BERT model with custom configuration we would do:
            config_dict={
                'hidden_size': 1536,
                ...
            } 


        Args:
            model_name (Optional[str]): Named model Megatron architecture from NeMo. Defaults to None.
            pretrained (bool): Use True to get pretrained weights. 
                                        False will use the same architecture but with randomly initialized weights.
                                        Defaults to False.
            config_dict (Optional[dict], optional): Use for custom configuration of the HuggingFace model. Defaults to None.
            checkpoint_file (Optional[str], optional): Provide weights for the transformer from a local checkpoint.
                                                       If using model parallel then this should be a directory. Defaults to None.
        """
        super().__init__()

        if not pretrained:
            raise ValueError('We currently only support pretrained Megatron models. Please set pretrained=True')

        if not checkpoint_file and not model_name:
            raise ValueError(
                'Currently Megatron models must be loaded from a pretrained model name or a pretrained checkpoint.'
            )

        if model_name or checkpoint_file:
            model = get_megatron_lm_model(
                pretrained_model_name=model_name, config_dict=config_dict, checkpoint_file=checkpoint_file
            )[0]

        self._checkpoint_file = checkpoint_file
        self._hidden_size = model.hidden_size
        self._vocab_size = model.vocab_size

        self._encoder = model

    @typecheck()
    def forward(self, input_ids, encoder_mask):
        encoder_hidden_states = self._encoder.forward(
            input_ids=input_ids, attention_mask=encoder_mask, token_type_ids=None
        )
        return encoder_hidden_states

    @property
    def checkpoint_file(self) -> Optional[str]:
        return self._checkpoint_file

    @property
    def hidden_size(self) -> Optional[int]:
        return self._hidden_size

    @property
    def vocab_size(self) -> Optional[int]:
        return self._vocab_size

# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from typing import List, Optional

from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.parts.k2.classes import ASRK2Mixin
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging

# use k2 import guard
# fmt: off
from nemo.core.utils.k2_utils import k2_import_guard # isort:skip
k2_import_guard()
# fmt: on


class EncDecK2SeqModel(ASRK2Mixin, EncDecCTCModel):
    """Encoder decoder models with various lattice losses."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        pass

    def change_vocabulary(self, new_vocabulary: List[str]):
        """
        Changes vocabulary used during CTC decoding process. Use this method when fine-tuning on from pre-trained model.
        This method changes only decoder and leaves encoder and pre-processing modules unchanged. For example, you would
        use it if you want to use pretrained encoder when fine-tuning on a data in another language, or when you'd need
        model to learn capitalization, punctuation and/or special characters.

        If new_vocabulary == self.decoder.vocabulary then nothing will be changed.

        Args:
            new_vocabulary: list with new vocabulary. Must contain at least 2 elements. Typically, \
            this is target alphabet.

        Returns: None

        """
        super().change_vocabulary(new_vocabulary)

        if self.use_graph_lm:
            self.token_lm = None
            self.token_lm_cache_dict = None
            self.token_lm_path = None
            logging.warning(
                f"""With .change_vocabulary() call for a model with criterion_type=`{self.loss.criterion_type}`,
                            either a new token_lm or a token_lm_path has to be set manually."""
            )

        self._update_k2_modules(self.graph_module_cfg)


class EncDecK2SeqModelBPE(ASRK2Mixin, EncDecCTCModelBPE):
    """Encoder decoder models with Byte Pair Encoding and various lattice losses."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        pass

    def change_vocabulary(self, new_tokenizer_dir: str, new_tokenizer_type: str):
        """
        Changes vocabulary of the tokenizer used during CTC decoding process.
        Use this method when fine-tuning on from pre-trained model.
        This method changes only decoder and leaves encoder and pre-processing modules unchanged. For example, you would
        use it if you want to use pretrained encoder when fine-tuning on a data in another language, or when you'd need
        model to learn capitalization, punctuation and/or special characters.

        Args:
            new_tokenizer_dir: Path to the new tokenizer directory.
            new_tokenizer_type: Either `bpe` or `wpe`. `bpe` is used for SentencePiece tokenizers,
                whereas `wpe` is used for `BertTokenizer`.

        Returns: None

        """
        super().change_vocabulary(new_tokenizer_dir, new_tokenizer_type)

        if self.use_graph_lm:
            self.token_lm = None
            self.token_lm_cache_dict = None
            self.token_lm_path = None
            logging.warning(
                f"""With .change_vocabulary() call for a model with criterion_type=`{self.loss.criterion_type}`,
                            either a new token_lm or a token_lm_path has to be set manually."""
            )

        self._update_k2_modules(self.graph_module_cfg)

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


from typing import Dict, List

import torch

from nemo.collections.asr.inference.pnc.token_classification.punctuation_capitalization_model import (
    PunctuationCapitalizationModel,
)
from nemo.collections.common.inference.utils.device_utils import setup_device
from nemo.collections.common.inference.utils.word import Word, join_words


class PunctuationCapitalizer:

    def __init__(
        self,
        model_name: str,
        device: str = 'cuda',
        device_id: int = 0,
        compute_dtype: str = "float32",
        use_amp: bool = False,
    ):
        """
        A model for restoring punctuation and capitalization in text.
        Args:
            model_name: (str) path to the model checkpoint or a model name from the NGC cloud.
            device: (str) device to run the model on.
            device_id: (int) device ID to run the model on.
            compute_dtype: (str) data type to use for computation.
            use_amp: (bool) whether to use automatic mixed precision.
        """
        self.model_name = model_name
        self.device_str, self.device_id, self.compute_dtype = setup_device(device, device_id, compute_dtype)
        self.use_amp = use_amp
        self.pnc_model = self.load_model()

    def load_model(self) -> PunctuationCapitalizationModel:
        """
        Load the punctuation and capitalization model.

        Returns:
            Loaded PunctuationCapitalizationModel instance.

        Raises:
            RuntimeError: If model loading fails.
        """
        try:
            model = PunctuationCapitalizationModel.from_pretrained(
                model_name=self.model_name,
                refresh_cache=False,
                override_config_path=None,
                map_location=torch.device(self.device_str),
                strict=True,
                return_config=False,
                trainer=None,
                save_restore_connector=None,
                return_model_file=False,
            )
            return model
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def add_punctuation_capitalization_list(self, transcriptions: List[str], params: Dict) -> List[str]:
        """
        Args:
            transcriptions: (List[str]) list of input strings.
            params: (Dict) dictionary of runtime parameters.
        Returns:
            (List[str]) list of punctuated and capitalized transcriptions.
        """
        with (
            torch.amp.autocast(device_type=self.device_str, dtype=self.compute_dtype, enabled=self.use_amp),
            torch.inference_mode(),
            torch.no_grad(),
        ):
            transcriptions = self.pnc_model.add_punctuation_capitalization(
                transcriptions,
                batch_size=params.get('batch_size', None),
                max_seq_length=params.get('max_seq_length', 45),
                step=params.get('step', 8),
                margin=params.get('margin', 16),
                return_labels=params.get('return_labels', False),
                dataloader_kwargs=params.get('dataloader_kwargs', None),
            )
            return transcriptions

    def add_punctuation_capitalization_to_words(
        self, words: List[List[Word]], params: Dict, sep: str = ' '
    ) -> List[List[Word]]:
        """
        Args:
            words: (List[List[Word]]) list of input words.
            params: (Dict) dictionary of runtime parameters.
            sep: (str) separator to join the words.
        Returns:
            (List[List[Word]]) list of punctuated and capitalized words.
        """
        if len(words) == 0:
            return words

        pnc_words_list = [[w.copy() for w in sample_words] for sample_words in words]
        pnc_transcriptions = join_words(pnc_words_list, sep)
        pnc_transcriptions = self.add_punctuation_capitalization_list(pnc_transcriptions, params)
        for i, text in enumerate(pnc_transcriptions):
            if text == "":
                continue
            text_words = text.split(sep)
            for j, word in enumerate(text_words):
                pnc_words_list[i][j].text = word
        return pnc_words_list

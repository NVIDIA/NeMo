# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from functools import lru_cache

from nemo.collections.llm.gpt.data.core import create_sft_dataset
from nemo.collections.llm.gpt.data.fine_tuning import FineTuningDataModule


class ChatDataModule(FineTuningDataModule):
    """
    Base class for fine-tuning an LLM on chat datasets.
    This class calls `GPTSFTChatDataset` for chat template processing

    See base class `FineTuningDataModule` for more details.
    """

    def __init__(
        self,
        use_hf_tokenizer_chat_template: bool = False,
        *args,
        **kwargs,
    ):
        """Data module for finetuning on chat datasets.
        See base class `FineTuningDataModule` for more details of the arguments.

        Args:
            use_hf_tokenizer_chat_template: Whether to use the chat template from the HuggingFace tokenizer. If True, uses the tokenizer's
                built-in chat template. If False, uses default chat template from GPTSFTChatDataset. Defaults to False.
        """

        super().__init__(
            *args,
            **kwargs,
        )
        self.use_hf_tokenizer_chat_template = use_hf_tokenizer_chat_template

    @lru_cache
    def _create_dataset(self, path, pack_metadata_path=None, is_test=False, **kwargs):
        # pylint: disable=C0115,C0116
        return create_sft_dataset(
            path,
            tokenizer=self.tokenizer,
            seq_length=(self.seq_length if is_test or self.packed_sequence_size <= 0 else self.packed_sequence_size),
            memmap_workers=self.memmap_workers,
            seed=self.seed,
            chat=True,
            is_test=is_test,
            pack_metadata_file_path=None,  # packing is not supported
            pad_cu_seqlens=False,
            use_hf_tokenizer_chat_template=self.use_hf_tokenizer_chat_template,
            **kwargs,
        )

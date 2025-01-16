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

from typing import Optional, Union

from lhotse.cut import Cut

from nemo.collections.common.data.prompt_fn import get_prompt_format_fn
from nemo.collections.common.prompts import PromptFormatter
from nemo.collections.multimodal.speech_llm.parts.utils.data_utils import PromptFormatterTextProcessing


class MockCut:
    """
    A mock class to wrap the input data for the prompt formatter.
    """

    def __init__(self, sample: Union[str, dict]):
        """
        Args:
            sample: The input data to be formatted.
        """
        if isinstance(sample, str):
            setattr(self, 'context', sample)
        elif isinstance(sample, dict):
            for k, v in sample.items():
                setattr(self, k, v)


class AutoPromptFormatter:
    """
    A class to automatically format the prompt based on the input data.
    """

    def __init__(self, tokenizer, prompt_format: Optional[str] = None):
        """
        Args:
            tokenizer: The tokenizer to use.
            prompt_format: The prompt format string.
        """
        self.prompt_format = prompt_format
        self.prompt = PromptFormatter.resolve(prompt_format)(tokenizer)
        self.prompt_formatter = get_prompt_format_fn(Cut, self.prompt)

    def __call__(self, inputs, *args, **kwargs):
        wrapped_inputs = self.maybe_wrap_inputs(inputs)
        return self.prompt_formatter(wrapped_inputs, *args, **kwargs)

    def maybe_wrap_inputs(self, inputs):
        if isinstance(inputs, Cut):
            return [inputs]
        elif isinstance(inputs, list) and all(isinstance(i, Cut) for i in inputs):
            return inputs
        elif isinstance(inputs, list):
            return [MockCut(i) for i in inputs]
        else:
            return [MockCut(inputs)]


class TextProcesserWithPromptFormatter(PromptFormatterTextProcessing):
    """
    Wrapper class that processes text and uses the prompt formatter.
    """

    def __init__(
        self,
        tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        prompt_format: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        audio_locator: Optional[str] = None,
    ):
        """
        Args:
            tokenizer: The tokenizer to use.
            prompt_format: The prompt format string.
            max_seq_length: The maximum sequence length.
            audio_locator: The audio locator for inserting audio.
        """
        super().__init__(tokenizer, prompt_format, max_seq_length, audio_locator)
        self.prompt_format_fn = AutoPromptFormatter(tokenizer=tokenizer, prompt_format=prompt_format)

    def __call__(self, *args, **kwds):
        return self.process_sample(*args, **kwds)

    def process_sample(self, sample):
        return self._process_example(sample)

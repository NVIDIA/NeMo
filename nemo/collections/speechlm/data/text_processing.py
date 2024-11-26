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

from nemo.collections.common.prompts import PromptFormatter, get_prompt_format_fn
from nemo.collections.multimodal.speech_llm.parts.utils.data_utils import PromptFormatterTextProcessing


class MockCut:
    def __init__(self, sample: Union[str, dict]):
        if isinstance(sample, str):
            setattr(self, 'context', sample)
        elif isinstance(sample, dict):
            for k, v in sample.items():
                setattr(self, k, v)


class AutoPromptFormatter:
    def __init__(self, prompt_format: Optional[str] = None):
        self.prompt_format = prompt_format
        self.prompt_formatter = get_prompt_format_fn(prompt_format)

    def __call__(self, inputs, tokenizer):
        wrapped_inputs = self.maybe_wrap_inputs(inputs)
        return self.prompt_formatter(wrapped_inputs, tokenizer)

    def maybe_wrap_inputs(self, inputs):
        if isinstance(inputs, Cut):
            return [inputs]
        elif isinstance(inputs, list) and all(isinstance(i, Cut) for i in inputs):
            return inputs
        elif isinstance(inputs, list):
            return [MockCut(i) for i in inputs]
        else:
            return [MockCut(inputs)]


class TextProcessingWithPromptFormatter(PromptFormatterTextProcessing):
    def __init__(
        self,
        tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        prompt_format: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        audio_locator: Optional[str] = None,
    ):
        super().__init__(tokenizer, prompt_format, max_seq_length, audio_locator)
        self.prompt_format_fn = AutoPromptFormatter(prompt_format)

    def __call__(self, *args, **kwds):
        return self.process_sample(*args, **kwds)

    def process_sample(self, sample):
        return self._process_example(sample)

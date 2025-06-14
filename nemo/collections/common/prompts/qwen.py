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
# pylint: disable=C0115
from nemo.collections.common.prompts.formatter import Modality, PromptFormatter

QWEN_BOT = "<|im_start|>"
QWEN_EOT = "<|im_end|>"


class QwenPromptFormatter(PromptFormatter):
    NAME = "qwen"
    OUTPUT_ROLE = "assistant"
    INFERENCE_PREFIX = f"{QWEN_BOT}assistant\n"
    TEMPLATE = {
        "user": {
            "template": f"{QWEN_BOT}user\n|message|{QWEN_EOT}\n",
            "slots": {
                "message": Modality.Text,
            },
        },
        OUTPUT_ROLE: {
            "template": f"{INFERENCE_PREFIX}|message|{QWEN_EOT}\n",
            "slots": {
                "message": Modality.Text,
            },
        },
    }

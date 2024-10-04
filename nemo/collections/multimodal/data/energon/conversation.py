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
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BaseConversationTemplateConfig:
    """Conversation template config related parameters"""

    system: Optional[str] = (
        "".format()
    )  # fmt: off
    roles: List[str] = field(default_factory=lambda: ['user', 'assistant'])
    stop_string: Optional[str] = None
    chat_template = None


class LLaVATemplateConfig(BaseConversationTemplateConfig):
    """LLava specific template configuration which extends the base config"""

    system: Optional[str] = (
        "A chat between a curious user and artificial assistant agent. The assistant gives helpful, detailed and polite answers to user's questions.".format()
    )  # fmt: off
    roles: List[str] = field(default_factory=lambda: ['user', 'assistant'])
    stop_string: str = "</s>"
    chat_template = """
    {%- for message in messages %}
        {%- if message['role'] == 'system' %}
            {{- message['content'].strip() + ' ' -}}
        {%- elif message['role'] == 'user' %}
            {{- 'USER: ' -}} {{- message['content'].strip() + ' ' -}}
        {%- elif message['role'] == 'assistant' %}
            {{- 'ASSISTANT: ' -}} {{- message['content'].strip() -}}
            {{- '</s>' -}}
        {%- endif %}
    {%- endfor -%}
    """
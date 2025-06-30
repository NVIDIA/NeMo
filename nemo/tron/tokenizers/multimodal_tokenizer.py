# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Multimodal tokenizer."""
from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer
# Mark tokens that will be ignored in the loss function with this value.
# Same ignore_index in https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
from megatron.core.models.multimodal.llava_model import (IGNORE_INDEX,
                                                         IMAGE_TOKEN)

IMAGE_TAGS = {
    "nvlm": ("<Image>", "</Image>"),
    "internvl": ("<img>", "</img>"),
    "": None,  # Image tag not used.
}


# The default mistral template raises exceptions so we use a custom one.
mistral_custom_template = """
{{- bos_token }}
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{- '[INST] ' + message['content'] + '[/INST]' }}
    {%- elif message['role'] == 'assistant' %}
        {{- ' ' + message['content'] + eos_token}}
    {%- endif %}
{%- endfor %}
{% if add_generation_prompt %}{{ ' ' }}{% endif %}
"""


nvlm_yi_34b_template = "{{- bos_token }}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"


qwen2p0_custom_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"


# Note: this is the same template as https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/tokenizer_config.json#L2053
# but we removed the forced system message.
llama3p1_chat_template = """{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- set date_string = \"26 Jul 2024\" %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = none %}\n{%- endif %}\n\n{%- if system_message is not none %}{#- System message + builtin tools #}\n{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n{%- if builtin_tools is defined or tools is not none %}\n    {{- \"Environment: ipython\\n\" }}\n{%- endif %}\n{%- if builtin_tools is defined %}\n    {{- \"Tools: \" + builtin_tools | reject('equalto', 'code_interpreter') | join(\", \") + \"\\n\\n\"}}\n{%- endif %}{%- if tools is not none and not tools_in_user_message %}\n    {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- \"<|eot_id|>\" }}\n\n{%-endif %}{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n{%- endif %}\n    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n    {{- first_user_message + \"<|eot_id|>\"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- \"<|python_tag|>\" + tool_call.name + \".call(\" }}\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\n                {{- arg_name + '=\"' + arg_val + '\"' }}\n                {%- if not loop.last %}\n                    {{- \", \" }}\n                {%- endif %}\n                {%- endfor %}\n            {{- \")\" }}\n        {%- else  %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n            {{- '\"parameters\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- \"}\" }}\n        {%- endif %}\n        {%- if builtin_tools is defined %}\n            {#- This means we're in ipython mode #}\n            {{- \"<|eom_id|>\" }}\n        {%- else %}\n            {{- \"<|eot_id|>\" }}\n        {%- endif %}\n    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- \"<|eot_id|>\" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n"""


@dataclass
class PromptConfig:
    """Config options for different prompt formats."""

    # How many tokens are used for the assistant prefix, e.g. "<|im_start|>assistant\n".
    # Used for masking the assistant prefix.
    assistant_prefix_len: int
    # Padding token ID.
    pad_token_id: int
    # For overriding the default chat format template.
    custom_chat_template: str
    # If the tokenizer inserts BOS token by default.
    has_bos: bool
    # If the tokenizer supports a separate role for system messages.
    has_system_role: bool


class MultimodalTokenizer(MegatronTokenizer):
    """Multimodal Tokenizer."""

    def __init__(
        self,
        tokenizer: MegatronTokenizer,
        prompt_format: str,
        special_tokens: List[str],
        image_tag_type: str,
    ):
        """Tokenizer with a support for non-text inputs.

        Note: Currently, only HuggingFaceTokenizer is supported as the underlying text tokenizer.

        Args:
            tokenizer (MegatronTokenizer): Underlying tokenizer.
            prompt_format (str): Prompt format for the tokenizer.
            special_tokens (List[str]): Non-text tokens.
            image_tag_type (str): Image tag to apply, if any. For example <img><image></img>.
        """
        self._vocab_size = len(tokenizer)

        num_added_tokens = tokenizer.add_tokens(special_tokens, special_tokens=True)
        assert num_added_tokens == len(
            special_tokens
        ), f"failed to add {len(special_tokens)} special tokens; only added {num_added_tokens}"

        self._tokenizer = tokenizer

        if prompt_format == "mistral":
            # Mistral format doesn't have prefix for the assistant message.
            self._prompt_config = PromptConfig(
                assistant_prefix_len=0,
                pad_token_id=tokenizer.unk_token_id,
                custom_chat_template=mistral_custom_template,
                has_bos=True,
                has_system_role=False,
            )
        elif prompt_format == "llama3":
            # "<|start_header_id|>assistant<|end_header|>\n\n" is the prefix for assistant messages.
            self._prompt_config = PromptConfig(
                assistant_prefix_len=4,
                pad_token_id=tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
                custom_chat_template=None,
                has_bos=True,
                has_system_role=True,
            )
        elif prompt_format in ("llama3p1", "llama3p2"):
            # "<|start_header_id|>assistant<|end_header|>\n\n" is the prefix for assistant messages.
            # That occupies 4 tokens and can be masked in the target.
            self._prompt_config = PromptConfig(
                assistant_prefix_len=4,
                pad_token_id=tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>"),
                custom_chat_template=llama3p1_chat_template,
                has_bos=True,
                has_system_role=True,
            )
        elif prompt_format == "nvlm-yi-34b":
            self._prompt_config = PromptConfig(
                assistant_prefix_len=4,
                pad_token_id=tokenizer.pad_token_id,
                custom_chat_template=nvlm_yi_34b_template,
                has_bos=True,
                has_system_role=True,
            )
        elif prompt_format == "chatml":
            # "<|im_start|>assistant\n" is the prefix for assistant messages
            self._prompt_config = PromptConfig(
                assistant_prefix_len=3,
                pad_token_id=tokenizer.pad_token_id,
                custom_chat_template=None,
                has_bos=False,
                has_system_role=True,
            )
        elif prompt_format in ("qwen2p0", "qwen2p5"):
            # "<|im_start|>assistant\n" is the prefix for assistant messages
            self._prompt_config = PromptConfig(
                assistant_prefix_len=3,
                pad_token_id=tokenizer.pad_token_id,
                custom_chat_template=qwen2p0_custom_template,
                has_bos=False,
                has_system_role=True,
            )
        else:
            raise NotImplementedError("unknown multimodal tokenizer type", prompt_format)

        self._image_tag = IMAGE_TAGS[image_tag_type]

    def _apply_image_tag(self, text: Union[str, List[Dict]]):
        """Surround <image> with image tags such as <img> and </img>."""
        if self._image_tag is None:
            return text

        replacement = f"{self._image_tag[0]}{IMAGE_TOKEN}{self._image_tag[1]}"

        if isinstance(text, list):
            for turn in text:
                turn["content"] = turn["content"].replace(IMAGE_TOKEN, replacement)
        else:
            text = text.replace(IMAGE_TOKEN, replacement)

        return text

    def tokenize(self, text: Union[str, List[Dict]]):
        """Tokenize conversation or string input."""
        if isinstance(text, list):
            # This code path is used by the inference code currently.
            return self.tokenize_conversation(text, False, True).tolist()

        return self._encode(text)

    def _encode(self, text: str):
        """Tokenize text input."""
        text = self._apply_image_tag(text)
        return self._tokenizer.encode(text)

    def tokenize_conversation(self, conversation: List[Dict], return_target: bool, add_generation_prompt: bool):
        """Convert a conversation to tokens.

        Args:
            conversation (List[Dict]): Sequence of system/user/assistant messages.
                Must be in the following format:
                [
                    {"role": "user", "content": "something"},
                    {"role": "assistant", "content": "something2"},
                ]
            return_target (bool): Return target tokens with system and assistant masked.
            add_generation_prompt (bool): Add assistant prefix to the end.
        """
        # Skip system message if the tokenizer doesn't have a system role.
        if not self._prompt_config.has_system_role and conversation[0]["role"] == "system":
            conversation = conversation[1:]

        # Apply possible image tag.
        conversation = self._apply_image_tag(conversation)

        tokens = self._tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_assistant_token_mask=False,
            return_tensors="np",
            chat_template=self._prompt_config.custom_chat_template,
        )[0]

        if not return_target:
            return tokens

        target = tokens.copy()

        # Mask system and user tokens in the target.
        idx = 0
        for turn_idx, turn in enumerate(conversation):
            if len(turn["content"]) == 0:
                raise ValueError(f"empty turn in conversation: {conversation}. Skipping.")

            turn_tokens = self._tokenizer.apply_chat_template(
                [turn], tokenize=True, chat_template=self._prompt_config.custom_chat_template
            )

            # There should be only one BOS at the very beginning.
            # After the first turn, skip BOS token.
            if self._prompt_config.has_bos and turn_idx > 0:
                turn_tokens = turn_tokens[1:]

            turn_len = len(turn_tokens)

            role = turn["role"]
            if role in ("system", "user"):
                target[idx : idx + turn_len] = IGNORE_INDEX
            elif role == "assistant":
                if IMAGE_TOKEN in turn["content"]:
                    raise RuntimeError(f"{IMAGE_TOKEN} not allowed in assistant content!")

                if self._prompt_config.assistant_prefix_len > 0:
                    target[idx : idx + self._prompt_config.assistant_prefix_len] = IGNORE_INDEX

            assert np.allclose(
                tokens[idx : idx + turn_len], turn_tokens
            ), f"expected turn tokens to match tokens in conversation {conversation}"

            idx += turn_len

        assert idx == len(tokens), f"mismatch in target masking the conversation {conversation}"

        return tokens, target

    def convert_tokens_to_ids(self, tokens: List[str]):
        """Convert tokens to IDs."""
        return self._tokenizer.convert_tokens_to_ids(tokens)

    def detokenize(self, tokens: List[int]):
        """Detokenize tokens."""
        return self._tokenizer.decode(tokens)

    def get_special_tokens(self):
        """Get special tokens."""
        return self._tokenizer.get_added_vocab()

    @property
    def pad(self):
        """Pad token ID."""
        return self._prompt_config.pad_token_id

    @property
    def eod(self):
        """End of sentence token ID."""
        return self._tokenizer.eos_token_id

    @property
    def vocab(self):
        """Vocab."""
        return NotImplementedError("not used")

    @property
    def inv_vocab(self):
        """Inverse vocab."""
        return NotImplementedError("not used")

    @property
    def vocab_size(self):
        """Vocabulary size."""
        return self._vocab_size

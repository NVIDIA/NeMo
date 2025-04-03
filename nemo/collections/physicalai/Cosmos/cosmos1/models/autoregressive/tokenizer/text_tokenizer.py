# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import AutoTokenizer

from cosmos1.utils import log


def get_tokenizer_path(model_family: str, is_instruct_model: bool = False):
    """
    Get the tokenizer path from the model family and instruct model flag.
    Args:
        model_family (str): The model family.
        is_instruct_model (bool): Whether the model is an instruct model.
    Returns:
        str: The tokenizer path in s3.
    """
    model_family = model_family.lower()
    if model_family == "mistral":
        return "mistralai/Mistral-Nemo-Instruct-2407"
    else:
        assert model_family in ["llama3", "llama3.1"]
        if model_family == "llama3":
            model_path = "meta-llama/Meta-Llama-3-8B"
        elif model_family == "llama3.1":
            model_path = "meta-llama/Llama-3.1-8B"
        else:
            raise ValueError(f"Unsupported model family: {model_family}")
        suffix = "-Instruct" if is_instruct_model else ""
        model_path = f"{model_path}{suffix}"
        return model_path


class TextTokenizer:
    """
    Text tokenizer class built on HuggingFace's Fast Tokenizer (Rust based).
    """

    def __init__(
        self,
        model_family: str,
        is_instruct_model: bool,
        local_path: Optional[str] = None,
    ):
        """
        Initialize the TextTokenizer.
        Args:
            model_family (str): The model family.
            is_instruct_model (bool): Whether the model is an instruct model.
            local_path (Optional[str]): The local path to the tokenizer. If not provided, the tokenizer will be downloaded from the remote path.
        """
        if local_path is None:
            tokenizer_path = get_tokenizer_path(model_family, is_instruct_model)
        else:
            tokenizer_path = local_path

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        self.stop_tokens = {
            self.tokenizer.eos_token_id,
        }
        self.model_family = model_family
        self.is_instruct_model = is_instruct_model
        self.eos_id = self.tokenizer.eos_token_id
        if self.tokenizer.pad_token is None:
            if model_family.startswith("llama"):
                self.pad_id = 128004  # "<|finetune_right_pad_id|>"
            elif model_family == "mistral":
                self.pad_id = 10  # "<pad>"
            elif model_family == "pixtral":
                self.pad_id = 11  # "<pad>"
            else:
                raise ValueError(f"pad_id not defined for model_family {model_family}")
        else:
            self.pad_id = self.tokenizer.pad_token_id

    def tokenize(self, text: str, *, add_special_tokens: bool = False, **kwargs) -> List[str]:
        """
        Converts a string into a sequence of tokens, replacing unknown tokens with the `unk_token`.

        Args:
            text (`str`):
                The sequence to be encoded.
            add_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to add the special tokens associated with the corresponding model.
        Returns:
            `List[str]`: The list of tokens.
        """
        return self.tokenizer.tokenize(text, add_special_tokens=add_special_tokens, **kwargs)

    def encode(
        self,
        text: Union[str, List[str], List[int]],
        *,  # Enforce keyword-only arguments
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> List[int]:
        """
        Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

        Args:
            text (`str`, `List[str]` or `List[int]`):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
                `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
            add_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to add special tokens when encoding the sequences. This will use the underlying
                `PretrainedTokenizerBase.build_inputs_with_special_tokens` function, which defines which tokens are
                automatically added to the input ids. This is usefull if you want to add `bos` or `eos` tokens
                automatically.
            padding (`bool`, `str`, *optional*, defaults to `False`):
                Activates and controls padding. Accepts the following values:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            truncation (`bool`, `str`, *optional*, defaults to `False`):
                Activates and controls truncation. Accepts the following values:

                - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
                  to the maximum acceptable input length for the model if that argument is not provided. This will
                  truncate token by token, removing a token from the longest sequence in the pair if a pair of
                  sequences (or a batch of pairs) is provided.
                - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
                  greater than the model maximum admissible input size).
            max_length (`int`, *optional*):
                Controls the maximum length to use by one of the truncation/padding parameters.

                If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
                is required by one of the truncation/padding parameters. If the model has no specific maximum input
                length (like XLNet) truncation/padding to a maximum length will be deactivated.
            stride (`int`, *optional*, defaults to 0):
                If set to a number along with `max_length`, the overflowing tokens returned when
                `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence
                returned to provide some overlap between truncated and overflowing sequences. The value of this
                argument defines the number of overlapping tokens.
            is_split_into_words (`bool`, *optional*, defaults to `False`):
                Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
                tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
                which it will tokenize. This is useful for NER or token classification.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated.
                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
        """
        return self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            return_tensors=return_tensors,
        )

    def decode(
        self,
        token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor"],
        *,  # Enforce keyword-only arguments
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces`.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `str`: The decoded sentence.
        """
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def apply_chat_template(
        self,
        conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
        *,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
        generation_prefix: str = "",
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Converts a list of dictionaries with `"role"` and `"content"` keys to a list of token
        ids. This method is intended for use with chat models, and will read the tokenizer's chat_template attribute to determine the format and control tokens to use when converting.

        More details can be found at https://huggingface.co/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template

        Args:
            conversation (Union[List[Dict[str, str]], List[List[Dict[str, str]]]]): A list of dicts
                with "role" and "content" keys, representing the chat history so far.
            add_generation_prompt (bool, *optional*):
                If this is set, a prompt with the token(s) that indicate
                the start of an assistant message will be appended to the formatted output. This is useful when you want to generate a response from the model.
                Note that this argument will be passed to the chat template, and so it must be supported in the
                template for this argument to have any effect.
            continue_final_message (bool, *optional*):
                If this is set, the chat will be formatted so that the final
                message in the chat is open-ended, without any EOS tokens. The model will continue this message
                rather than starting a new one. This allows you to "prefill" part of
                the model's response for it. Cannot be used at the same time as `add_generation_prompt`.
            tokenize (`bool`, defaults to `True`):
                Whether to tokenize the output. If `False`, the output will be a string.
            padding (`bool`, defaults to `False`):
                Whether to pad sequences to the maximum length. Has no effect if tokenize is `False`.
            truncation (`bool`, defaults to `False`):
                Whether to truncate sequences at the maximum length. Has no effect if tokenize is `False`.
            max_length (`int`, *optional*):
                Maximum length (in tokens) to use for padding or truncation. Has no effect if tokenize is `False`. If
                not specified, the tokenizer's `max_length` attribute will be used as a default.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Has no effect if tokenize is `False`. Acceptable
                values are:
                - `'tf'`: Return TensorFlow `tf.Tensor` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.
            return_dict (`bool`, defaults to `False`):
                Whether to return a dictionary with named outputs. Has no effect if tokenize is `False`.
            generation_prefix (str): Prefix to add before asking model to generate. Helpful to guide the generation. Defaults to "".
            tokenizer_kwargs (`Dict[str: Any]`, *optional*): Additional kwargs to pass to the tokenizer.
            return_assistant_tokens_mask (`bool`, defaults to `False`):
                Whether to return a mask of the assistant generated tokens. For tokens generated by the assistant,
                the mask will contain 1. For user and system tokens, the mask will contain 0.
                This functionality is only available for chat templates that support it via the `{% generation %}` keyword.
            **kwargs: Additional kwargs to pass to the template renderer. Will be accessible by the chat template.

        Returns:
            `Union[List[int], Dict]`: A list of token ids representing the tokenized chat so far, including control tokens. This
            output is ready to pass to the model, either directly or via methods like `generate()`. If `return_dict` is
            set, will return a dict of tokenizer outputs instead.
        """
        if not self.is_instruct_model:
            raise ValueError(
                "apply_chat_template is only supported for instruct models. You should pass argument is_instruct_model=True to the TextTokenizer constructor."
            )
        # Since generation_prefix is added to the text in the end, ensure that the setting is correct
        if generation_prefix:
            assert not tokenize, "tokenize must be False when generation_prefix is provided."
            assert add_generation_prompt, "add_generation_prompt must be set when generation_prefix is provided."
        formatted_text: Union[str, List[int]] = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            return_dict=return_dict,
            return_assistant_tokens_mask=return_assistant_tokens_mask,
            tokenizer_kwargs=tokenizer_kwargs,
            **kwargs,
        )
        if generation_prefix:
            formatted_text: str = formatted_text + generation_prefix
            log.debug(
                f"Adding generation prefix: {generation_prefix} to the formatted text\n"
                f"Formatted text: {formatted_text}"
            )
        return formatted_text

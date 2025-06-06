# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from transformers import AutoTokenizer as AUTOTOKENIZER

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.utils import logging

__all__ = [
    'AutoTokenizer',
]


class AutoTokenizer(TokenizerSpec):
    """
    Wrapper of HuggingFace AutoTokenizer
    https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer.

    """

    def __init__(
        self,
        pretrained_model_name: str,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        mask_token: Optional[str] = None,
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        pad_token: Optional[str] = None,
        sep_token: Optional[str] = None,
        cls_token: Optional[str] = None,
        unk_token: Optional[str] = None,
        additional_special_tokens: Optional[List] = [],
        use_fast: Optional[bool] = True,
        trust_remote_code: Optional[bool] = False,
        include_special_tokens: bool = False,
        chat_template: Optional[str] = None,
    ):
        """
        Args:
            pretrained_model_name: corresponds to HuggingFace-AutoTokenizer's 'pretrained_model_name_or_path' input
                argument. For more details please refer to the documentation of the `from_pretrained` method here:
                https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer.
                The list of all supported models can be found here: https://huggingface.co/models
            vocab_file: path to file with vocabulary which consists
                of characters separated by newlines.
            mask_token: mask token
            bos_token: the beginning of sequence token
            eos_token: the end of sequence token. Usually equal to sep_token
            pad_token: token to use for padding
            sep_token: token used for separating sequences
            cls_token: class token. Usually equal to bos_token
            unk_token: token to use for unknown tokens
            additional_special_tokens: list of other tokens beside standard special tokens (bos, eos, pad, etc.). For
                example, sentinel tokens for T5 (<extra_id_0>, <extra_id_1>, etc.)
            use_fast: whether to use fast HuggingFace tokenizer
            include_special_tokens: when True, converting text to ids will include special tokens / prompt tokens (if
                any), yielding self.tokenizer(text).input_ids
            chat_template: The chat template string to format "messages" with against the underlying HF tokneizer with
                apply_chat_template function
        """
        try:
            self._initialize_tokenizer(
                pretrained_model_name, vocab_file, merges_file, use_fast, trust_remote_code, chat_template
            )
            assert self.tokenizer, "tokenizer not initialized"
        except Exception:
            try:
                self._initialize_tokenizer(
                    pretrained_model_name, vocab_file, merges_file, not use_fast, trust_remote_code, chat_template
                )
                assert self.tokenizer, "tokenizer not initialized"
            except Exception as e:
                raise ValueError(
                    f'Unable to instantiate HuggingFace AUTOTOKENIZER for {pretrained_model_name}. Exception: {e}'
                )

        self.include_special_tokens = include_special_tokens
        self.original_vocab_size = len(self.tokenizer)
        special_tokens_dict = {}

        # # setting special tokens, by default the default model's special tokens will be preserved
        # # unless passes new values to the special tokens
        if unk_token is not None:
            special_tokens_dict["unk_token"] = unk_token
        if mask_token is not None:
            special_tokens_dict["mask_token"] = mask_token
        if pad_token is not None:
            special_tokens_dict["pad_token"] = pad_token

        # if the model does not have eos_token but has sep_token,
        # set eos_token = sep_token, and vice versa
        if sep_token is not None:
            special_tokens_dict["sep_token"] = sep_token
        elif self.tokenizer.sep_token is None and self.tokenizer.eos_token:
            special_tokens_dict["sep_token"] = self.tokenizer.eos_token
        if eos_token is not None:
            special_tokens_dict["eos_token"] = eos_token
        elif self.tokenizer.eos_token is None and self.tokenizer.sep_token:
            special_tokens_dict["eos_token"] = self.tokenizer.sep_token

        # if the model does not have bos_token but has cls_token,
        # set bos_token = cls_token, and vice versa
        if bos_token is not None:
            special_tokens_dict["bos_token"] = bos_token
        elif self.tokenizer.bos_token is None and self.tokenizer.cls_token:
            special_tokens_dict["bos_token"] = self.tokenizer.cls_token
        if cls_token is not None:
            special_tokens_dict["cls_token"] = cls_token
        elif self.tokenizer.cls_token is None and self.tokenizer.bos_token:
            special_tokens_dict["cls_token"] = self.tokenizer.bos_token

        # add additional special tokens (not standard special tokens such as bos, eod, sep)
        if additional_special_tokens is not None:
            special_tokens_dict["additional_special_tokens"] = additional_special_tokens

        new_tokens_in_vocab = []
        for token in [mask_token, bos_token, eos_token, pad_token, sep_token, cls_token, unk_token]:
            if token is not None and token not in self.tokenizer.get_vocab():
                new_tokens_in_vocab.append(token)
        for token in additional_special_tokens:
            if token is not None and token not in self.tokenizer.get_vocab():
                new_tokens_in_vocab.append(token)

        if len(new_tokens_in_vocab) > 0:
            """
            Special tokens that were not previously included in the tokenizer's vocabulary file will be added to
            the vocabulary and, as a result, the model should be resized, for example:

            # define your model
            pretrained_model_name = 'roberta-base'
            model = nemo_nlp.modules.get_lm_model(pretrained_model_name=pretrained_model_name)

            # define pretrained tokenizer
            tokenizer_default = nemo_nlp.modules.get_tokenizer(tokenizer_name=pretrained_model_name)

            special_tokens = {'bos_token': '<BOS>',
                              'cls_token': '<CSL>',
                              'additional_special_tokens': ['<MY_NER_TOKEN>', '<ANOTHER_TOKEN>']}
            tokenizer_default.add_special_tokens(special_tokens_dict=special_tokens)

            # resize your model so that the embeddings for newly added tokens are updated during training/finetuning
            model.resize_token_embeddings(tokenizer_default.vocab_size)

            See NLP_Tokenizers.ipynb for more details.
            """
            logging.warning(
                f'{new_tokens_in_vocab} \n will be added to the vocabulary.\n'
                f'Please resize your model accordingly, '
                f'see NLP_Tokenizers.ipynb for more details.'
            )
        self.add_special_tokens(special_tokens_dict)
        self.space_sensitive = self.text_to_tokens('x y') != self.text_to_tokens('x') + self.text_to_tokens('y')
        self._inv_vocab_dict = {}

    def _initialize_tokenizer(
        self,
        pretrained_model_name: str,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        use_fast: Optional[bool] = False,
        trust_remote_code: Optional[bool] = False,
        chat_template: Optional[str] = None,
    ):
        # this logic deals with different huggingface tokenizers having different positional args
        if vocab_file is None:
            self.tokenizer = AUTOTOKENIZER.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name,
                use_fast=use_fast,
                trust_remote_code=trust_remote_code,
            )
        elif merges_file is None:
            self.tokenizer = AUTOTOKENIZER.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name,
                vocab_file=vocab_file,
                use_fast=use_fast,
                trust_remote_code=trust_remote_code,
            )
        else:
            self.tokenizer = AUTOTOKENIZER.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name,
                vocab_file=vocab_file,
                merges_file=merges_file,
                use_fast=use_fast,
                trust_remote_code=trust_remote_code,
            )

        if chat_template is not None:
            if getattr(self.tokenizer, 'chat_template', None) is not None:
                logging.info("You are overwriting tokenizer's chat template, confirm this is intended.")
            self.tokenizer.chat_template = chat_template
            self.tokenizer.chat_template_format = "jinja"

    @property
    def vocab_size(self):
        """
        Returns the size of the tokenizer's vocabulary.

        Returns:
            int: The number of tokens in the vocabulary.
        """
        return len(self.tokenizer)

    def add_special_tokens(self, special_tokens_dict: dict) -> int:
        """
        Adds a dictionary of special tokens (eos, pad, cls...). If special tokens are NOT in the vocabulary, they are
        added to it (indexed starting from the last index of the current vocabulary).

        Args:
            special_tokens_dict: dict of string. Keys should be in the list of predefined special attributes:
                [``bos_token``, ``eos_token``, ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``,
                ``mask_token``, ``additional_special_tokens``].
                Tokens are only added if they are not already in the vocabulary.

        Returns:
            Number of tokens added to the vocabulary.
        """
        num_tokens_added = self.tokenizer.add_special_tokens(special_tokens_dict)

        if num_tokens_added > 0:
            logging.info(f'{num_tokens_added} special tokens added, resize your model accordingly.')
        for k in self.tokenizer.SPECIAL_TOKENS_ATTRIBUTES:
            setattr(self, k, getattr(self.tokenizer, k, None))
        return num_tokens_added

    @property
    def additional_special_tokens_ids(self):
        """
        Returns a list of the additional special tokens' IDs (excluding bos, eos, pad, unk).

        Returns:
            List[int]: List of token IDs for additional special tokens, such as sentinel tokens for T5.
        """
        return [self.token_to_id(token) for token in self.additional_special_tokens]

    def text_to_tokens(self, text):
        """
        Converts text into a list of tokens.

        Args:
            text (str): Input text to be tokenized.

        Returns:
            List[str]: List of tokens.
        """
        tokens = self.tokenizer.tokenize(text)
        return tokens

    def tokens_to_text(self, tokens):
        """
        Converts a list of tokens back into text.

        Args:
            tokens (List[str]): List of tokens to be converted.

        Returns:
            str: The reconstructed text.
        """
        text = self.tokenizer.convert_tokens_to_string(tokens)
        return text

    def token_to_id(self, token):
        """
        Converts a single token to its corresponding ID.

        Args:
            token (str): The token to convert.

        Returns:
            int: The ID corresponding to the token.
        """
        return self.tokens_to_ids([token])[0]

    def tokens_to_ids(self, tokens):
        """
        Converts a list of tokens to their corresponding IDs.

        Args:
            tokens (List[str]): List of tokens to convert.

        Returns:
            List[int]: List of token IDs.
        """
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return ids

    def ids_to_tokens(self, ids):
        """
        Converts a list of token IDs back to tokens.

        Args:
            ids (List[int]): List of token IDs to convert.

        Returns:
            List[str]: List of tokens.
        """
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return tokens

    def text_to_ids(self, text):
        """
        Converts text directly to token IDs.

        Args:
            text (str): Input text to be converted to IDs.

        Returns:
            List[int]: List of token IDs. If include_special_tokens is True, will include special tokens from the
            tokenizer's configuration.
        """
        if self.include_special_tokens:
            return self.tokenizer(text).input_ids
        tokens = self.text_to_tokens(text)
        ids = self.tokens_to_ids(tokens)
        return ids

    def apply_chat_template(self, *args, **kwargs):
        """Appies chat template and tokenizes results"""
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def ids_to_text(self, ids, remove_special_tokens=True):
        """
        Converts token IDs back to text.

        Args:
            ids (List[int]): List of token IDs to convert to text.
            remove_special_tokens (bool): Whether to remove special tokens (like [PAD], [CLS], etc.) from the output
            text.

        Returns:
            str: The reconstructed text.
        """
        tokens = self.ids_to_tokens(ids)
        if remove_special_tokens:
            tokens_clean = [t for t in tokens if t not in self.tokenizer.all_special_tokens]
        else:
            tokens_clean = tokens
        text = self.tokens_to_text(tokens_clean)
        return text

    @property
    def vocab(self):
        """
        Returns the vocabulary as a list where the index corresponds to the token ID.

        Returns:
            List[str]: List of tokens in the vocabulary.
        """
        id2vocab = {v: k for k, v in self.tokenizer.vocab.items()}
        return [id2vocab[i] for i in range(len(id2vocab))]

    @property
    def inv_vocab(self):
        """
        Returns the inverse vocabulary mapping (token to ID).

        Returns:
            Dict[str, int]: Dictionary mapping tokens to their IDs.
        """
        if self._inv_vocab_dict == {}:
            self._inv_vocab_dict = {v: k for k, v in self.tokenizer.vocab.items()}
        return self._inv_vocab_dict

    @property
    def pad_id(self):
        """
        Gets the ID of the padding token.

        Returns:
            int or None: The ID of the padding token if it exists, None otherwise.
        """
        if getattr(self, 'pad_token') is None:
            return None
        return self.tokens_to_ids([getattr(self, 'pad_token')])[0]

    @property
    def bos_id(self):
        """
        Gets the ID of the beginning-of-sequence token.

        Returns:
            int or None: The ID of the BOS token if it exists, None otherwise.
        """
        if getattr(self, 'bos_token') is None:
            return None
        return self.tokens_to_ids([getattr(self, 'bos_token')])[0]

    @property
    def eos_id(self):
        """
        Gets the ID of the end-of-sequence token.

        Returns:
            int or None: The ID of the EOS token if it exists, None otherwise.
        """
        if getattr(self, 'eos_token') is None:
            return None
        return self.tokens_to_ids([getattr(self, 'eos_token')])[0]

    @property
    def eod(self):
        """
        Gets the ID of the end-of-document token (same as EOS token). Required for megatron-core compatibility.

        Returns:
            int: The ID of the EOD/EOS token.
        """
        return self.tokens_to_ids([getattr(self, 'eos_token')])[0]

    @property
    def sep_id(self):
        """
        Gets the ID of the separator token.

        Returns:
            int or None: The ID of the separator token if it exists, None otherwise.
        """
        if getattr(self, 'sep_token') is None:
            return None
        return self.tokens_to_ids([getattr(self, 'sep_token')])[0]

    @property
    def cls_id(self):
        """
        Gets the ID of the classifier token.

        Returns:
            int or None: The ID of the classifier token if it exists, None otherwise.
        """
        if getattr(self, 'cls_token') is None:
            return None
        return self.tokens_to_ids([getattr(self, 'cls_token')])[0]

    @property
    def unk_id(self):
        """
        Gets the ID of the unknown token.

        Returns:
            int or None: The ID of the unknown token if it exists, None otherwise.
        """
        if getattr(self, 'unk_token') is None:
            return None
        return self.tokens_to_ids([getattr(self, 'unk_token')])[0]

    @property
    def mask_id(self):
        """
        Gets the ID of the mask token.

        Returns:
            int or None: The ID of the mask token if it exists, None otherwise.
        """
        if getattr(self, 'mask_token') is None:
            return None
        return self.tokens_to_ids([getattr(self, 'mask_token')])[0]

    @property
    def name(self):
        """
        Returns the name of the underlying HuggingFace tokenizer class.

        Returns:
            str: Name of the tokenizer class.
        """
        return type(self.tokenizer).__name__

    def save_vocabulary(self, save_directory: str, filename_prefix: str = None):
        """Saves tokenizer's vocabulary and other artifacts to the specified directory"""
        return self.tokenizer.save_vocabulary(save_directory=save_directory, filename_prefix=filename_prefix)

    def save_pretrained(self, save_directory: str):
        """Saves tokenizer's vocabulary and other artifacts to the specified directory"""
        return self.tokenizer.save_pretrained(save_directory)

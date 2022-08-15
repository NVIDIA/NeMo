
import abc
import re
from typing import Optional, Dict, List, Tuple

import torch
import numpy as np

from nemo.core import Dataset, typecheck
from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.core.neural_types import NeuralType, TokenIndex, LengthsType


class PunctCapSegDataset(Dataset):
    """Base class for a dataset that produces examples for punctuation restoration, true casing, and sentence-boundary
    detection.

    Args:
        language: The language code for this dataset. E.g., 'en', 'es', 'zh'. Used for logging and inferring whether
            this dataset is for a continuous-script language.
        is_continuous: Whether this language is continuous. Determines whether spaces are inserted between concatenated
            sentences, etc. If not set, the language code will be compared against a list of known continuous-script
            language codes and this value will be inferred.
        tokenizer: Text tokenizer. Can be set later, e.g., after an NLP model initializes its BertModule, but must be
            set before producing examples.
        target_pad_value: Pad targets with this value, and use it to indicate ignored tokens (e.g. uncased tokens for
            true casing). Should be the same value used in the loss function to ignore.

    """
    def __init__(
            self,
            language: str = "unk",
            is_continuous: bool = None,
            tokenizer: Optional[TokenizerSpec] = None,
            target_pad_value: int = -100,
            rng_seed: Optional[int] = None
    ) -> None:
        super().__init__()
        self._language = language
        self._tokenizer = tokenizer
        self._target_pad_value = target_pad_value
        # If not explicitly set, make the inference.
        self._is_continuous = is_continuous if (is_continuous is not None) else (language in {"zh", "ja", "my"})
        self._rng_seed = rng_seed

    @property
    def tokenizer(self) -> TokenizerSpec:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: TokenizerSpec):
        self._tokenizer = tokenizer
        self._on_tokenizer_set()

    @abc.abstractmethod
    def _on_tokenizer_set(self):
        """Will be called when tokenizer is set. Can be used to initialize anything dependent on the tokenizer."""
        pass

    @property
    def language(self) -> str:
        return self._language

    def __getitem__(self, index):
        """Implemented by derived classes """
        raise NotImplementedError()

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "punc_input_ids": NeuralType(("B", "T"), TokenIndex()),
            "cap_input_ids": NeuralType(("B", "T"), TokenIndex()),
            "seg_input_ids": NeuralType(("B", "T"), TokenIndex()),
            "punc_pre_target_ids": NeuralType(("B", "T"), TokenIndex()),
            "punc_post_target_ids": NeuralType(("B", "T"), TokenIndex()),
            "cap_target_ids": NeuralType(("B", "T"), TokenIndex()),
            "seg_target_ids": NeuralType(("B", "T"), TokenIndex()),
            "punct_lengths": NeuralType(("B",), LengthsType()),
            "cap_lengths": NeuralType(("B",), LengthsType()),
            "seg_lengths": NeuralType(("B",), LengthsType()),
        }

    @typecheck()
    def collate_fn(self, batch):
        punct_inputs = [x[0] for x in batch]
        cap_inputs = [x[1] for x in batch]
        seg_inputs = [x[2] for x in batch]
        punct_pre_targets_list = [x[3] for x in batch]
        punct_post_targets_list = [x[4] for x in batch]
        cap_targets_list = [x[5] for x in batch]
        seg_targets_list = [x[6] for x in batch]
        punct_lengths = torch.tensor([x.shape[-1] for x in punct_inputs])
        cap_lengths = torch.tensor([x.shape[-1] for x in cap_inputs])
        seg_lengths = torch.tensor([x.shape[-1] for x in seg_inputs])
        batch_size = len(punct_inputs)  # should be all the same size

        # Create empty input ID tensors and fill non-padded regions
        # TODO currently all implementations have tokenizer but tarred dataset may not. Need to know pad id
        punct_input_ids = torch.full(
            size=(batch_size, punct_lengths.max()),
            fill_value=self._tokenizer.pad_id
        )
        cap_input_ids = torch.full(
            size=(batch_size, cap_lengths.max()),
            fill_value=self._tokenizer.pad_id
        )
        seg_input_ids = torch.full(
            size=(batch_size, seg_lengths.max()),
            fill_value=self._tokenizer.pad_id
        )
        for i in range(batch_size):
            punct_input_ids[i, :punct_lengths[i]] = punct_inputs[i]
            cap_input_ids[i, :cap_lengths[i]] = cap_inputs[i]
            seg_input_ids[i, :seg_lengths[i]] = seg_inputs[i]

        # Create empty target tensors and fill non-padded regions
        punct_pre_targets = torch.full(size=[batch_size, punct_lengths.max()], fill_value=self._target_pad_value)
        punct_post_targets = torch.full(size=[batch_size, punct_lengths.max()], fill_value=self._target_pad_value)
        cap_targets = torch.full(size=[batch_size, cap_lengths.max()], fill_value=self._target_pad_value)
        seg_targets = torch.full(size=[batch_size, seg_lengths.max()], fill_value=self._target_pad_value)
        for i in range(batch_size):
            punct_pre_targets[i, :punct_lengths[i]] = punct_pre_targets_list[i]
            punct_post_targets[i, :punct_lengths[i]] = punct_post_targets_list[i]
            cap_targets[i, :cap_lengths[i]] = cap_targets_list[i]
            seg_targets[i, :seg_lengths[i]] = seg_targets_list[i]

        return (
            punct_input_ids, cap_input_ids, seg_input_ids,
            punct_pre_targets, punct_post_targets, cap_targets, seg_targets,
            punct_lengths, cap_lengths, seg_lengths
        )


class TextCleaner(abc.ABC):
    """Base class for language-specific text cleaners.

    The classes derived from this base class will be applied to each input sentence before we generate examples. The
    main idea is that these classes normalize the data specific to our task.

    """

    @abc.abstractmethod
    def clean(self, text: str) -> str:
        raise NotImplementedError()


class StandardPunctNormalizer(TextCleaner):
    """Class for normalizing punctuation in most languages.

    Intended to be run on plain text data before generating examples.

    First, removes all spaces that appear before punctuation tokens. e.g.,
    "foo ." -> "foo.", "foo. . ." -> "foo...", etc.

    Then replaces all instances of 2+ consecutive punctuation tokens by the first punctuation token in the sequence.
    E.g.,
    "foo..." -> "foo."

    Note on the latter that this primarily deals with 1) ellipsis and 2) messy data. In the former case, we replace
    ellipsis with a period, in that latter we do the best we can with messy data in a simple way.

    Args:
        punct_tokens: List of punctuation tokens.

    """
    def __init__(self, punct_tokens: List[str]) -> None:
        punct_tokens = [x for x in punct_tokens if x != "<NULL>"]  # TODO don't assume null token
        # This assumes all punctuation tokens are single characters, which should be true
        escaped_tokens = [re.escape(x) for x in punct_tokens]
        punct_str = "".join(escaped_tokens)
        # Match a punct token, followed immediately by more. Capture the first token.
        self._multi_punct_ptn = re.compile(rf"([{punct_str}])[{punct_str}]+")
        # Match whitespace followed by a punct token. Capture the token.
        self._whitespace_ptn = re.compile(rf"\s+([{punct_str}])")
        # Match punctuation at the beginning of a sentence (not valid except in Spanish)
        self._punct_at_bos_ptn = re.compile(rf"^[{punct_str}\s]+")

    def clean(self, text: str) -> str:
        # Remove punctuation/space at beginning of sentence
        text = self._punct_at_bos_ptn.sub("", text)
        # Remove whitespace before any punctuation token
        text = self._whitespace_ptn.sub(r"\g<1>", text)
        # Replace consecutive punctuation tokens with the first tokens
        text = self._multi_punct_ptn.sub(r"\g<1>", text)
        return text


class SpanishPunctNormalizer(TextCleaner):
    """Class for normalizing punctuation Spanish.

    Similar to a :class:``StandardPunctNormalizer`` but has special rules for dealing with "¡" and "¿".

    For non-inverted punctuation, we follow the same rules as :class:``StandardPunctNormalizer``.

    For inverted punctuation, we allow them to appear at the beginning of a string and allow space before them (but not
    after).

    Args:
        pre_punct_tokens: List of punctuation tokens that can appear before a subword. Basically, inverted punctuation.
        post_punct_tokens: List of punctuation tokens that can appear after a subword.

    """
    def __init__(self, pre_punct_tokens: List[str], post_punct_tokens: List[str]) -> None:
        pre_punct_tokens = [x for x in pre_punct_tokens if x != "<NULL>"]  # TODO don't assume null token
        post_punct_tokens = [x for x in post_punct_tokens if x != "<NULL>"]
        # make char classes e.g. '[\.,?]'
        post_punct_char_str = "".join([re.escape(x) for x in post_punct_tokens])
        pre_punct_char_str = "".join([re.escape(x) for x in pre_punct_tokens])
        all_char_str = "".join([re.escape(x) for x in pre_punct_tokens + post_punct_tokens])

        # Match whitespace followed by a non-inverted token. Capture the token.
        self._whitespace_ptn1 = re.compile(rf"\s+([{post_punct_char_str}])")
        # Match whitespace after inverted token. Capture the token.
        self._whitespace_ptn2 = re.compile(rf"([{pre_punct_char_str}])\s+")

        # Catch inverted punctuation at eos
        self._pre_punct_at_eos_ptn = re.compile(rf"[{pre_punct_char_str}\s]+$")
        # Catch non-inverted at bos
        self._post_punct_at_bos_ptn = re.compile(rf"^[{post_punct_char_str}\s]+")
        # Catch inverted followed by any punctuation, replace with inverted
        self._multi_punct_ptn1 = re.compile(rf"([{pre_punct_char_str}])[{all_char_str}]+")
        # Catch non-inverted followed by any tokens without space
        self._multi_punct_ptn2 = re.compile(rf"([{post_punct_char_str}])[{all_char_str}]+")

    def clean(self, text: str) -> str:
        # Remove punctuation/space at beginning of sentence
        text = self._post_punct_at_bos_ptn.sub("", text)
        text = self._pre_punct_at_eos_ptn.sub("", text)
        # Remove whitespace before any punctuation token
        text = self._whitespace_ptn1.sub(r"\g<1>", text)
        text = self._whitespace_ptn2.sub(r"\g<1>", text)
        # Replace consecutive punctuation tokens with the first tokens
        text = self._multi_punct_ptn1.sub(r"\g<1>", text)
        text = self._multi_punct_ptn2.sub(r"\g<1>", text)
        return text


class ChineseTextCleaner(TextCleaner):
    """Text cleaner for Chinese.

    Args:
        remove_spaces: If True, remove all spaces from the text.
        replace_latin: If true, replace all instances of latin punctuation with the analogous Chinese token. E.g.,
            replace all instances of '.' with '。'
        no_enum_comma: If true, replace all instances of the Chinese enumeration comma "、" with the comma ",". Most
            datasets use these commas interchangeably, so unless you are sure your data correctly and consistently uses
            the enumeration comma correctly, you should set this to True. Otherwise the model will be penalized for
            the messy data.

    """
    def __init__(
            self,
            remove_spaces: bool = True,
            replace_latin: bool = True,
            no_enum_comma: bool = True
    ) -> None:
        self._remove_spaces = remove_spaces
        self._replace_latin = replace_latin
        self._no_enum_comma = no_enum_comma

    def clean(self, text: str) -> str:
        if self._remove_spaces:
            text = re.sub(r"\s+", "", text)
        if self._replace_latin:
            # Replace latin punctuation with Chinese.
            text = re.sub(r"\?", "？", text)
            text = re.sub(r",", "，", text)
            text = re.sub(r"[\\.．]", "。", text)
            text = re.sub(r"!", "！", text)
        if self._no_enum_comma:
            # Replace the enumeration comma with regular comma. The former and latter are often used interchangeably in
            # raw data, so it is difficult or impossible to learn the enumeration comma.
            text = re.sub(r"、", "，", text)
        return text


class PuncTargetsGenerator(abc.ABC):
    """Base class for a punctuation targets generator.

    Base class for generating punctuation targets. Implementations may be language-specific, notably Spanish which uses
    inverted tokens.

    Args:
        post_labels: Punctuation labels that can appear after subwords.
        pre_labels: Punctuation labels that can appear before subwords.
        null_label: The string value of the "null" label, or the label that means "no punctuation here".
        p_drop: The probability of dropping an individual punctuation token in the examples. Should be a high number to
            generate a lot of examples, but by leaving this value < 1.0, we can train the model to not barf when it sees
            properly-punctuated text at inference time.
        rng_seed: Seed for the PRNG, used for choosing whether to drop a punctuation token. Can help with consistency
            in the validation datasets.
    """
    def __init__(
            self,
            tokenizer: Optional[TokenizerSpec],
            post_labels: List[str],
            pre_labels: List[str],
            null_label: str = "<NULL>",
            p_drop: float = 0.9,
            rng_seed: Optional[int] = None
    ) -> None:
        self._p_drop = p_drop
        self._null_label = null_label
        self._rng_seed = rng_seed

        self._pre_label_to_index = {label: i for i, label in enumerate(pre_labels)}
        self._post_label_to_index = {label: i for i, label in enumerate(post_labels)}
        self._pre_null_index = self._pre_label_to_index[null_label]
        self._post_null_index = self._post_label_to_index[null_label]
        # Save as set for quick membership check
        self._pre_labels = set(pre_labels)
        self._post_labels = set(post_labels)
        self._joint_labels = self._pre_labels | self._post_labels
        self._rng = np.random.default_rng(seed=rng_seed)
        self._tokenizer = tokenizer

    @property
    def tokenizer(self) -> TokenizerSpec:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: TokenizerSpec):
        self._tokenizer = tokenizer

    def reseed_rng(self) -> None:
        self._rng = np.random.default_rng(seed=self._rng_seed)

    @abc.abstractmethod
    def generate_targets(self, input_text: str) -> Tuple[List[str], List[int], List[int]]:
        """Applies punctuation dropout and generates an example.

        Args:
            input_text: Text to process.

        Returns:
            (input_tokens, pre_targets, post_targets) where `input_tokens` is a sequence of subword tokens
            and `pre_targets` and `post_targets` are the pre- and post-token targets.
        """
        raise NotImplementedError()

    @classmethod
    def from_lang_code(
            cls,
            lang_code: str,
            tokenizer: Optional[TokenizerSpec],
            pre_labels: List[str],
            post_labels: List[str],
            p_drop: float,
            rng_seed: int
    ):
        """Instantiates a derived class which is applicable to the given language.

        This is a convenience function for instantiating a derived class for a particular language.

        Args:
            lang_code: The language code to use to determine which class to instantiate.
            tokenizer: An optional :class:`TokenizerSpec`. If not set on initialization, needs to be set before creating
                examples.
            pre_labels: Punctuation tokens that can appear before a subword.
            post_labels: Punctuation tokens that can appear after a subword.
            p_drop: The probability of dropping each punctuation token in the examples.
            rng_seed: Seed for any PRNGs

        """
        lang_code = lang_code.lower()
        if len(lang_code) < 2 or len(lang_code) > 3:
            raise ValueError(f"Only 2- or 3-char lang codes recognized. Got '{lang_code}'.")
        # Catch all the special languages, and default to the English-like punctuation processor.
        if lang_code in {"es", "ast"}:
            # Spanish and Asturian use inverted ?!
            return SpanishPuncTargetsGenerator(
                tokenizer=tokenizer, pre_labels=pre_labels, post_labels=post_labels, p_drop=p_drop, rng_seed=rng_seed
            )
        elif lang_code in {"zh", "ja", "my"}:
            # Continuous-script languages. The "basic" class seems to work, so nothing special is implemented yet.
            return BasicPuncTargetsGenerator(
                tokenizer=tokenizer, pre_labels=pre_labels, post_labels=post_labels, p_drop=p_drop, rng_seed=rng_seed
            )
        elif lang_code in {"th"}:
            # Thai -- uses space as punctuation. Don't have a solution, yet.
            raise ValueError(f"Language not supported: {lang_code}")
        else:
            # Assume all other languages use English-like punctuation rules.
            return BasicPuncTargetsGenerator(
                tokenizer=tokenizer, pre_labels=pre_labels, post_labels=post_labels, p_drop=p_drop, rng_seed=rng_seed
            )


class BasicPuncTargetsGenerator(PuncTargetsGenerator):
    """Punctuation example generator suitable for most languages, including English.

    This class assumes that punctuation tokens appear only after subwords, and will work for most languages.

    """

    def generate_targets(self, input_text: str) -> Tuple[List[str], List[int], List[int]]:
        tokens: List[str] = self._tokenizer.text_to_tokens(input_text)
        # Find all punctuation indices. The [-1] index is for subwords that may be preceeded by _ or ##
        # Don't care about pre-tokens because not expecting them with this language.
        punc_indices = [i for i in range(len(tokens)) if tokens[i] in self._post_labels]
        # Don't allow 2 consecutive tokens to be dropped, since we predict whether the current subword is punctuated
        # and this situation would cause errors (although this is probably avoided by proper cleaning)
        ignore_indices = []
        for idx in punc_indices:
            if idx + 1 in punc_indices:
                ignore_indices.append(idx)
        # todo squash the above nested loop into a one liner here
        punc_indices = [x for x in punc_indices if x not in ignore_indices]
        # Randomly select which ones to drop
        indices_to_drop = [i for i in punc_indices if self._rng.random() <= self._p_drop]
        output_length = len(tokens) - len(indices_to_drop)
        pre_targets = [self._pre_null_index] * output_length
        post_targets = [self._post_null_index] * output_length
        num_dropped_so_far = 0
        for i, drop_idx in enumerate(indices_to_drop):
            post_targets[drop_idx - num_dropped_so_far - 1] = self._post_label_to_index[tokens[drop_idx]]
            num_dropped_so_far += 1
        out_tokens = [x for i, x in enumerate(tokens) if i not in indices_to_drop]
        return out_tokens, pre_targets, post_targets


class SpanishPuncTargetsGenerator(PuncTargetsGenerator):
    """Punctuation example generator for Spanish and Asturian.

    """
    def generate_targets(self, input_text: str) -> Tuple[List[str], List[int], List[int]]:
        tokens: List[str] = self._tokenizer.text_to_tokens(input_text)
        # Find all punctuation indices. Indexing -1 allows for HF-style tokenization, e.g., '##.'.
        punc_indices = [i for i in range(len(tokens)) if tokens[i] in self._joint_labels]
        # Randomly select which ones to drop
        indices_to_drop = [i for i in punc_indices if self._rng.random() <= self._p_drop]
        output_length = len(tokens) - len(indices_to_drop)
        pre_targets = [self._pre_null_index] * output_length
        post_targets = [self._post_null_index] * output_length
        num_dropped_so_far = 0
        for i, drop_idx in enumerate(indices_to_drop):
            token = tokens[drop_idx]
            if token in self._pre_labels:
                pre_targets[drop_idx - num_dropped_so_far] = self._pre_label_to_index[token]
            else:
                post_targets[drop_idx - num_dropped_so_far - 1] = self._post_label_to_index[token[-1]]
            num_dropped_so_far += 1
        out_tokens = [x for i, x in enumerate(tokens) if i not in indices_to_drop]
        return out_tokens, pre_targets, post_targets


class CapTargetsGenerator:
    """Generator of true-casing examples.

    Args:
        tokenizer: TokenizerSpec. Can be null during init, but must be set before processing text.
        prob_all_lower: With this probability, lower-case the entire input example. For practical purposes, we want most
            examples all lower-cased.
        prob_all_upper: With this probability, upper-case the entire input example.
        prob_char_lower: Lower-case characters on a char-by-char basis with this probability, if the entire input
            example was not upper- or lower-cased already.
        prob_char_upper: Upper-case characters on a char-by-char basis with this probability, if the entire input
            example was not upper- or lower-cased already.
        rng_seed: Seed for the PRNG, used when choosing whether to upper- or lower-case the inputs.
    """
    def __init__(
            self,
            tokenizer: Optional[TokenizerSpec],
            prob_all_lower: float = 1.0,
            prob_all_upper: float = 0.05,
            prob_char_lower: float = 0.9,
            prob_char_upper: float = 0.05,
            ignore_idx: int = -100,
            rng_seed: Optional[int] = None
    ) -> None:
        self._prob_all_lower = prob_all_lower
        self._prob_all_upper = prob_all_upper
        self._prob_char_lower = prob_char_lower
        self._prob_char_upper = prob_char_upper
        self._ignore_idx = ignore_idx
        self._whitespace_regex = re.compile(r"\s+")
        self._rng_seed = rng_seed
        self._rng = np.random.default_rng(seed=rng_seed) if rng_seed is not None else None
        self._tokenizer = tokenizer

    @property
    def tokenizer(self) -> TokenizerSpec:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: TokenizerSpec):
        self._tokenizer = tokenizer

    def reseed_rng(self):
        self._rng = np.random.default_rng(seed=self._rng_seed)

    def _char_is_uncased(self, char: str):
        return char.lower() == char.upper()

    def _recase_str(self, input_text: str):
        """Randomly re-cases the case of the input text"""
        rng = self._rng if self._rng is not None else np.random.default_rng()
        val = rng.random()
        lower_all = val <= self._prob_all_lower
        upper_all = self._prob_all_lower <= val <= self._prob_all_lower + self._prob_all_upper
        recased_chars = []
        for char in list(input_text):
            # Some chars convert to 2 when re-cased. Ignore these, for now, to maintain alignment.
            if len(char.lower()) != len(char.upper()):
                new_char = char
            elif lower_all:
                new_char = char.lower()
            elif upper_all:
                new_char = char.upper()
            else:
                # In this branch, re-case each char randomly
                val = rng.random()
                if val < self._prob_char_lower:
                    new_char = char.lower()
                elif val < self._prob_char_lower + self._prob_char_upper:
                    new_char = char.upper()
                else:
                    new_char = char
            recased_chars.append(new_char)
        recased_text = "".join(recased_chars)
        return recased_text

    def generate_targets(self, input_text: str) -> Tuple[List[int], List[int]]:
        """Randomly re-cased the input text for inputs, and generates targets which matches the input.

        Args:
            input_text: A plain-text string.

        Returns:
            A tuple (input_ids, targets) where input_ids is the tokenized input IDs, and targets contains one list for
            each input id. ``targets[i][j]`` contains the target in {0, 1} for letter j of subtoken i

        """
        uncased_tokens = self._tokenizer.text_to_tokens(input_text)

        # Decide if entire input will be upper- or lower-cased, or neither
        rng = self._rng if self._rng is not None else np.random.default_rng()
        val = rng.random()
        lower_all = val < self._prob_all_lower
        upper_all = self._prob_all_lower <= val <= self._prob_all_lower + self._prob_all_upper

        # Generate targets for each char in input str
        input_ids: List[int] = []
        targets = [self._ignore_idx] * len(uncased_tokens)
        for i, uncased_token in enumerate(uncased_tokens):
            # Get the ID
            uncased_id = self._tokenizer.tokens_to_ids([uncased_token])[0]
            # If OOV or uncased, target is ignore_index, input is raw token
            if uncased_id == self._tokenizer.unk_id or self._char_is_uncased(uncased_token):
                input_ids.append(uncased_id)
                continue

            # Some chars convert to 2 chars when upper- or lower-cased, e.g., "ß".upper() == 'SS'. Skip these for now.
            if len(uncased_token.lower()) != len(uncased_token.upper()):
                # Don't mess with input
                input_ids.append(uncased_id)
                # If this char does have a case, put it in the targets
                if uncased_token.islower():
                    targets[i] = 0
                elif uncased_token.isupper():
                    targets[i] = 1
                continue

            # Maybe upper- or lower-case this char and mark the input
            if lower_all:
                recased_token = uncased_token.lower()
            elif upper_all:
                recased_token = uncased_token.upper()
            else:
                # In this branch, re-case each char randomly
                val = rng.random()
                if val < self._prob_char_lower:
                    recased_token = uncased_token.lower()
                elif val < self._prob_char_lower + self._prob_char_upper:
                    recased_token = uncased_token.upper()
                else:
                    recased_token = uncased_token
            input_ids.append(self._tokenizer.tokens_to_ids([recased_token])[0])
            target_val = 1 if uncased_token.isupper() else 0
            targets[i] = target_val
        return input_ids, targets


class SegTargetsGenerator:
    """Generator of sentence boundary detection examples.

    Args:
        tokenizer: TokenizerSpec. Can be null during init, but must be set before processing text.
        rng_seed: Seed for the PRNG, used when choosing whether to upper- or lower-case the inputs.
    """
    def __init__(
            self,
            tokenizer: Optional[TokenizerSpec],
            is_continuous: bool,
            prob_lower_case: float = 0.9,
            ignore_idx: int = -100,
            rng_seed: Optional[int] = None
    ) -> None:
        self._ignore_idx = ignore_idx
        self._is_continuous = is_continuous
        self._prob_lower_case = prob_lower_case
        self._whitespace_regex = re.compile(r"\s+")
        self._rng_seed = rng_seed
        self._rng = np.random.default_rng(seed=rng_seed) if rng_seed is not None else None
        self._tokenizer = tokenizer
        if self._tokenizer is not None:
            self._max_subtok_len = max(len(x) for x in self._tokenizer.vocab)

    @property
    def tokenizer(self) -> TokenizerSpec:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: TokenizerSpec):
        self._tokenizer = tokenizer

    def reseed_rng(self):
        self._rng = np.random.default_rng(seed=self._rng_seed)

    def generate_targets(self, input_texts: List[str]) -> Tuple[List[int], List[int]]:
        """Generates sentence boundary detection inputs and targets.

        Returns:
            Tuple of (input_ids, targets) where input_ids is the tokenized inputs and targets is the sentence boundary
            targets.
        """
        # Make segmentation targets.
        input_ids: List[int] = []
        targets: List[int] = []
        rng = self._rng if self._rng is not None else np.random.default_rng()
        for i, text in enumerate(input_texts):
            # Maybe lower-case sentence
            if rng.random() < self._prob_lower_case:
                text = text.lower()
            input_tokens = self._tokenizer.text_to_tokens(text)
            if self._is_continuous and i > 0:
                # For continuous languages, if we're concatenating this sentence, hide the information that implies
                # beginning-of-word. Assume SP for now
                if input_tokens[0] != "▁":
                    raise ValueError(f"Trying to hide BOW marker; expected '▁' but got '{input_tokens[0]}'")
                input_tokens = input_tokens[1:]
            next_seg_input_ids = self._tokenizer.tokens_to_ids(input_tokens)
            input_ids.extend(next_seg_input_ids)
            # Final token is a sentence boundary, all else are not.
            targets.extend([0] * len(next_seg_input_ids))
            targets[-1] = 1
        # Use ignore_index for the final token. It can generate misleading stats because it is too easy for the model to
        # predict. Also it's meaningless because there's no next sentence.
        # TODO should we target this or not? On the other hand, it helps the model learn the context in one direction.
        # targets[-1] = self._ignore_idx
        return input_ids, targets


class TarredPunctCapSegDataset(PunctCapSegDataset):
    """Loads a tarred dataset.

    The idea is that a text-based dataset can do preprocessing, save it as a tar, and this class will use webdataset
    to load the data without needing to do preprocessing. But preprocessing is not a bottleneck so not prioritized for
    now.

    """
    def __init__(
            self,
            tarred_dataset_dir: str,
            language: str = "unk",
            is_continuous: bool = None,
            tokenizer: Optional[TokenizerSpec] = None,
            target_pad_value: int = -100,
    ):
        super().__init__(
            language=language,
            is_continuous=is_continuous,
            tokenizer=tokenizer,
            target_pad_value=target_pad_value
        )
        raise NotImplementedError("Implement TextPunctCapSegDataset.create_tarred_dataset() then implement me.")

    def __getitem__(self, index):
        pass

    def _on_tokenizer_set(self):
        pass


class TextPunctCapSegDataset(PunctCapSegDataset):
    """Punctuation, true-casing, and sentence-boundary detection dataset that uses text files for example generation.

    Args:
        text_files: One or more plain-text files with one sentence per line. Each line should be properly true-cased
            and punctuated.
        language: Language code for this dataset.
        punct_pre_labels: List of punctuation tokens that can appear before subwords.
        punct_post_labels: List of punctuation tokens that can appear after subwords.
        tokenizer: TokenizerSpec to use to tokenize the data. Can be set later.
        cleaners: List of one or more implementation of a :class:``TextCleaner``. Will be applied to each input line in
            the order the cleaners are specified.
        null_label: The string value of the "null" token, or the token that means "no punctuation here".
        max_length: Maximum length of any input.
        prob_drop_punct: Drop punctuation tokens with this probability. 1.0 => drop all, 0.0 -> drop none.
        prob_lower_case: Probability of lower-casing the input before generating examples for punctuation and
            segmentation.
        max_lines_per_eg: Uniformly choose between 1 and this many lines to use per example.
        prob_truncate: Truncate examples with this probability.
        truncate_max_tokens: If truncating an example, truncate between 1 and this many tokens.
        target_pad_value: Padding value used in the targets. Should be the ignore_idx of your loss function.
        drop_if_first_char_lower: When reading the input data, discard lines if the first char is lower (presumably not
            a properly true-cased line).
        drop_if_no_end_punct: When reading input data, drop any line if it does not end in punctuation (presumably an
            improperly-punctuated line).
        drop_if_all_caps: When reading input data, drop any line that is all caps. Probably useful for corpora like
            OpenSubtitles.
        min_input_length_words: When reading input data, drop any line that contains fewer than this many words. For
            continuous script language, ensure this is None because each line will be 1 "word".
        max_input_length_words: When reading input data, drop any line that contains more than this many words
            (presumably a multi-sentence line, which will corrupt sentence boundary detection labels).
        min_input_length_chars: When reading input data, drop any line that contains fewer than this many chars. Intended
            as an analogy to ``min_input_length_words`` for continuous-script languages.
        max_input_length_chars: When reading input data, drop any line that contains more than this many chars. Intended
            as an analogy to ``max_input_length_words`` for continuous-script languages.
        max_lines_per_input_file: If not None, read only the first N lines of each input file.
        unused_punctuation: List of legitimate punctuation characters that are not used as targets. Useful when dropping
            examples that do not end in punctuation, but we are omitting some punctuation labels (e.g., '!') so that we
            retain input sentences with this token and let the model see it.
        rng_seed: Seed for the PRNG. For training, keep at None to prevent the data loader works from using the same
            extra indices each step.
    """
    def __init__(
            self,
            text_files: List[str],
            language: str,
            punct_pre_labels: List[str],
            punct_post_labels: List[str],
            is_continuous: Optional[bool] = None,
            tokenizer: Optional[TokenizerSpec] = None,
            cleaners: Optional[List[TextCleaner]] = None,
            null_label: str = "<NULL>",
            max_length: int = 512,
            prob_drop_punct: float = 0.9,
            prob_lower_case: float = 0.9,
            min_lines_per_eg: int = 1,
            max_lines_per_eg: int = 4,
            prob_truncate: float = 0.2,
            truncate_max_tokens: int = 5,
            truncate_percentage: float = 0.25,
            target_pad_value: int = -100,
            drop_if_first_char_lower: bool = True,
            drop_if_no_end_punct: bool = True,
            drop_if_all_caps: bool = True,
            min_input_length_words: int = None,
            max_input_length_words: int = None,
            max_input_length_chars: int = None,
            min_input_length_chars: int = None,
            max_lines_per_input_file: Optional[int] = None,
            unused_punctuation: Optional[List[str]] = None,
            rng_seed: Optional[int] = None
    ):
        super().__init__(
            language=language,
            tokenizer=tokenizer,
            target_pad_value=target_pad_value,
            is_continuous=is_continuous

        )
        self._text_files = text_files
        self._null_label = null_label
        self._max_length = max_length
        self._punct_pre_labels = punct_pre_labels
        self._punct_post_labels = punct_post_labels
        self._cleaners = cleaners
        self._prob_drop_punct = prob_drop_punct
        self._prob_lower_case = prob_lower_case
        self._max_lines_per_eg = max_lines_per_eg
        self._min_lines_per_eg = min_lines_per_eg
        self._prob_truncate = prob_truncate
        self._truncate_max_tokens = truncate_max_tokens
        self._truncate_percentage = truncate_percentage
        self._drop_if_first_char_lower = drop_if_first_char_lower
        self._drop_if_no_end_punct = drop_if_no_end_punct
        self._drop_if_all_caps = drop_if_all_caps
        self._max_input_length_words = max_input_length_words
        self._min_input_length_words = min_input_length_words
        self._max_input_length_chars = max_input_length_chars
        self._min_input_length_chars = min_input_length_chars
        self._max_line_per_input_file = max_lines_per_input_file

        self._rng_seed = rng_seed
        self._rng = np.random.default_rng(seed=self._rng_seed) if self._rng_seed is not None else None
        self._unused_punctuation = unused_punctuation if unused_punctuation is not None else []

        if self._max_lines_per_eg < 2:
            raise ValueError(f"Max lines per e.g. needs to be at least 2 to create meaningful segmentation targets.")

        self._data: List[str] = self._load_data(self._text_files)

        self._punct_targets_gen: PuncTargetsGenerator = PuncTargetsGenerator.from_lang_code(
            tokenizer=self._tokenizer,
            lang_code=self._language,
            pre_labels=self._punct_pre_labels,
            post_labels=self._punct_post_labels,
            p_drop=self._prob_drop_punct,
            rng_seed=self._rng_seed
        )

        # TODO expose options for probability of lower- or upper-casing examples. Currently all lower-cased.
        self._cap_targets_gen: CapTargetsGenerator = CapTargetsGenerator(tokenizer=self._tokenizer)

        self._seg_targets_gen: SegTargetsGenerator = SegTargetsGenerator(
            tokenizer=self._tokenizer,
            is_continuous=self._is_continuous,
            rng_seed=self._rng_seed,
            prob_lower_case=self._prob_lower_case,
            ignore_idx=target_pad_value
        )

    def create_tarred_dataset(self, output_dir: str):
        raise NotImplementedError(
            "Implement me to save this dataset in a tarred format that can be interpreted by TarredPunctCapSegDataset"
        )

    def _on_tokenizer_set(self):
        """Sets tokenizer for all properties that use it."""
        self._cap_targets_gen.tokenizer = self._tokenizer
        self._seg_targets_gen.tokenizer = self._tokenizer
        self._punct_targets_gen.tokenizer = self._tokenizer

    def reseed_rng(self):
        """Used by validation DS to get same evaluation examples each epoch"""
        worker_info = torch.utils.data.get_worker_info()
        seed = self._rng_seed
        if worker_info is not None:
            seed = worker_info.id + (seed if seed is not None else 0)
        self._rng = np.random.default_rng(seed=seed) if seed is not None else None
        self._punct_targets_gen.reseed_rng()
        self._cap_targets_gen.reseed_rng()

    def _load_data(self, text_files) -> List[str]:
        # Create a set of all legitimate punctuation labels, to filter sentences that do not end with punctuation.
        non_null_punct_labels = set(self._punct_pre_labels + self._punct_post_labels + self._unused_punctuation)
        non_null_punct_labels.remove(self._null_label)
        # Make a regex to determine whether some line contains nothing but punctuation.
        joined_punct_tokens = "".join([re.escape(x) for x in non_null_punct_labels])
        all_punct_ptn = re.compile(rf"^[{joined_punct_tokens}\s]*$")

        data: List[str] = []
        for text_file in text_files:
            with open(text_file) as f:
                num_lines_from_file = 0
                for i, line in enumerate(f):
                    if (
                            self._max_line_per_input_file is not None and
                            num_lines_from_file >= self._max_line_per_input_file
                    ):
                        break
                    line = line.strip()
                    if self._cleaners is not None:
                        for cleaner in self._cleaners:
                            line = cleaner.clean(line)
                    # Drop if blank or just punctuation
                    if not line or all_punct_ptn.match(line):
                        continue
                    # Drop if line does not end in punctuation, if specified.
                    if self._drop_if_no_end_punct and line[-1] not in non_null_punct_labels:
                        continue
                    # Drop if line does not start with an upper-case letter.
                    # Note: for uncase chars, islower() == isupper() == False, so no action is taken.
                    if self._drop_if_first_char_lower and line[0].islower():
                        continue
                    num_words = len(line.split())
                    num_chars = len(line)
                    # Drop if line contains too many words, if specified.
                    if self._max_input_length_words is not None:
                        if num_words > self._max_input_length_words:
                            continue
                    if self._min_input_length_words is not None:
                        if num_words < self._min_input_length_words:
                            continue
                    # Drop if line contains too many characters, if specified (for continuous languages).
                    if self._max_input_length_chars is not None:
                        if num_chars > self._max_input_length_chars:
                            continue
                    if self._min_input_length_chars is not None:
                        if num_chars < self._min_input_length_chars:
                            continue
                    # Drop is entire sentence is upper case
                    if self._drop_if_all_caps and line.isupper():
                        continue
                    data.append(line)
                    num_lines_from_file += 1
            print(f"Dataset for '{self.language}' collected {num_lines_from_file} lines from '{text_file}'")
        print(
            f"Dataset for '{self.language}' collected {len(data)} lines from {len(text_files)} "
            f"file{'s' if len(text_files) > 1 else ''}."
        )
        return data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        rng = self._rng if self._rng is not None else np.random.default_rng()
        # Each sequence starts with BOS and targets ignore first index
        bos = self._tokenizer.bos_id
        eos = self._tokenizer.eos_id
        pad = self._target_pad_value

        # Randomly choose how many additional lines to use
        num_lines_to_concat = rng.integers(self._min_lines_per_eg - 1, self._max_lines_per_eg)
        # Randomly select additional indices to use
        indices_to_use = [idx] + list(rng.integers(0, len(self), num_lines_to_concat))
        texts_to_use: List[str] = [self._data[x] for x in indices_to_use]
        # Concat all texts
        full_text = ("" if self._is_continuous else " ").join(texts_to_use)

        # Make cap targets. Input is punctuated, randomly-cased text.
        cap_input_ids, cap_targets = self._cap_targets_gen.generate_targets(input_text=full_text)
        if len(cap_input_ids) + 2 > self._max_length:
            cap_input_ids = cap_input_ids[:self._max_length - 2]
            cap_targets = cap_targets[:self._max_length - 2]
        # Add BOS/EOS and target padding for those tokens.
        cap_input_ids = [bos] + cap_input_ids + [eos]
        cap_targets = [pad] + cap_targets + [pad]

        # For segmentation and punctuation, we use lower-case text for most examples
        if rng.random() < self._prob_lower_case:
            full_text = full_text.lower()

        # Make punctuation targets
        # input_tokens = self._tokenizer.text_to_tokens(full_text)
        (
            punct_input_tokens,
            punct_pre_targets,
            punct_post_targets
        ) = self._punct_targets_gen.generate_targets(full_text)
        # Add BOS/EOS and target padding for those tokens.
        punct_input_ids = self._tokenizer.tokens_to_ids(punct_input_tokens)
        if len(punct_input_ids) + 2 > self._max_length:
            punct_input_ids = punct_input_ids[:self._max_length - 2]
            punct_pre_targets = punct_pre_targets[:self._max_length - 2]
            punct_post_targets = punct_post_targets[:self._max_length - 2]
        punct_input_ids = [bos] + punct_input_ids + [eos]
        punct_pre_targets = [pad] + punct_pre_targets + [pad]
        punct_post_targets = [pad] + punct_post_targets + [pad]

        # Make segmentation targets.
        seg_input_ids, seg_targets = self._seg_targets_gen.generate_targets(input_texts=texts_to_use)
        if len(seg_input_ids) + 2 > self._max_length:
            seg_input_ids = seg_input_ids[:self._max_length - 2]
            seg_targets = seg_targets[:self._max_length - 2]
        # Add BOS/EOS and target padding for those tokens.
        seg_input_ids = [bos] + seg_input_ids + [eos]
        seg_targets = [pad] + seg_targets + [pad]

        # Maybe truncate punctuation example to avoid learning to always predict punctuation at the end of a sequence
        if self._truncate_max_tokens > 0 and rng.random() < self._truncate_percentage:
            # Always leave at least two tokens: BOS and the first true token in the sequence
            max_truncate = min(self._truncate_max_tokens, len(punct_input_ids) - 2)
            # Always truncate at least two tokens. If we truncate one, it will be the punctuation after a complete
            # sentence (and the opposite of what we want).
            if max_truncate > 1:
                truncate_num_tokens = rng.integers(low=2, high=max_truncate)
                punct_input_ids = punct_input_ids[:-(truncate_num_tokens + 1)] + [eos]
                punct_pre_targets = punct_pre_targets[:-(truncate_num_tokens + 1)] + [pad]
                punct_post_targets = punct_post_targets[:-(truncate_num_tokens + 1)] + [pad]

        # Convert to Tensors
        punct_input_tensor = torch.tensor(punct_input_ids, dtype=torch.long)
        cap_input_tensor = torch.tensor(cap_input_ids, dtype=torch.long)
        seg_input_tensor = torch.tensor(seg_input_ids, dtype=torch.long)
        punct_pre_targets_tensor = torch.tensor(punct_pre_targets, dtype=torch.long)
        punct_post_targets_tensor = torch.tensor(punct_post_targets, dtype=torch.long)
        cap_targets_tensor = torch.tensor(cap_targets, dtype=torch.long)
        seg_targets_tensor = torch.tensor(seg_targets, dtype=torch.long)
        return (
            punct_input_tensor, cap_input_tensor, seg_input_tensor,
            punct_pre_targets_tensor, punct_post_targets_tensor, cap_targets_tensor, seg_targets_tensor
        )

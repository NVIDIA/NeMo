import abc
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from nemo.collections.common.tokenizers import AutoTokenizer, TokenizerSpec
from nemo.core import Dataset, typecheck
from nemo.core.neural_types import IntType, LengthsType, NeuralType, StringType, TokenIndex
from nemo.utils import logging


class InferencePunctCapSegDataset(Dataset):
    """

    Args:
        tokenizer: A :class:`TokenizerSpec` for the model being used for inference.
        input_texts: An optional list of one or more strings to run inference on.
        input_file: An optional file to read lines from. Should be mutually exclusive with `input_texts`.
        max_length: The maximum length for inputs. Longer inputs will be split into multiple batch elements.
        fold_overlap: When folding long sequences, repeat this many tokens from the end of the previous split into the
            beginning of the next split.
    """

    def __init__(
        self,
        tokenizer: TokenizerSpec,
        input_texts: Optional[List[str]] = None,
        input_file: Optional[str] = None,
        max_length: int = 512,
        fold_overlap: int = 16,
    ):
        super().__init__()
        if not ((input_texts is None) ^ (input_file is None)):
            raise ValueError(f"Need exactly one of `input_texts` or `input_file`")
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._fold_overlap = fold_overlap

        self._data: List[str]
        if input_texts is not None:
            self._data = input_texts
        else:
            self._data = []
            with open(input_file) as f:
                for line in f:
                    self._data.append(line.strip())
        logging.info(f"Inference dataset instantiated with {len(self._data)} lines of text.")

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "folded_input_ids": NeuralType(("B", "T"), TokenIndex()),
            "folded_batch_ids": NeuralType(("B",), IntType()),
            "lengths": NeuralType(("B",), LengthsType()),
            "input_strings": [NeuralType(("B",), StringType())],
        }

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        input_text = self._data[idx]
        input_ids = self._tokenizer.text_to_ids(input_text)
        return input_ids, input_text

    def _fold_batch(self, input_ids: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Folds inputs to adhere to max length"""
        out_batch_ids: List[int] = []
        out_input_ids: List[List[int]] = []
        out_lengths: List[int] = []
        bos = self._tokenizer.bos_id
        eos = self._tokenizer.eos_id
        for batch_idx, next_input_ids in enumerate(input_ids):
            start = 0
            while True:
                stop = min(start + self._max_length - 2, len(next_input_ids))
                subsegment_ids = [bos] + next_input_ids[start:stop] + [eos]
                out_input_ids.append(subsegment_ids)
                out_lengths.append(len(subsegment_ids))
                out_batch_ids.append(batch_idx)
                if stop >= len(next_input_ids):
                    break
                start = stop - self._fold_overlap

        batch_ids = torch.tensor(out_batch_ids)
        lengths = torch.tensor(out_lengths)
        ids_tensor = torch.full(
            size=[lengths.shape[0], lengths.max()], dtype=torch.long, fill_value=self._tokenizer.pad_id
        )
        for i, ids in enumerate(out_input_ids):
            ids_tensor[i, : len(ids)] = torch.tensor(ids)

        return ids_tensor, lengths, batch_ids

    @typecheck()
    def collate_fn(self, batch):
        """
        Returns:
            A tuple adhering to this class's `input_types` (folded_input_ids, folded_batch_ids, lengths, input_strings)
                where `folded_input_ids` is the tensor of input tokens, `folded_batch_ids` map each batch element back
                to its original input number (for long sentences that were split), `lengths` is the length of each
                element in `folded_batch_ids`, and `input_strings` is the original texts from which the inputs were
                generated. `input_strings` is returns because some tokenizers are non-invertible, so this will preserve
                the original input texts.
        """
        all_ids: List[List[int]] = [x[0] for x in batch]
        all_strs: List[str] = [x[1] for x in batch]
        input_ids, lengths, batch_ids = self._fold_batch(all_ids)
        return input_ids, batch_ids, lengths, all_strs


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
        multipass: Whether this model runs two passes (punctuation followed by truecasing/sentence boundary) or one pass
            (predict all in parallel)
    """

    def __init__(
        self,
        language: str = "unk",
        is_continuous: bool = None,
        tokenizer: Optional[TokenizerSpec] = None,
        target_pad_value: int = -100,
        rng_seed: Optional[int] = None,
        multipass: bool = True,
    ) -> None:
        super().__init__()
        self._language = language
        self._target_pad_value = target_pad_value
        # If not explicitly set, make the inference.
        self._is_continuous = is_continuous if (is_continuous is not None) else (language in {"zh", "ja", "my"})
        self._rng_seed = rng_seed
        self._max_token_len = 0
        self._multipass = multipass
        # Call setter
        self.tokenizer = tokenizer

    @property
    def tokenizer(self) -> TokenizerSpec:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: TokenizerSpec):
        self._tokenizer = tokenizer
        if tokenizer is not None:
            self._max_token_len = max(len(x) for x in self.tokenizer.vocab)

    @property
    def language(self) -> str:
        return self._language

    @property
    def multipass(self) -> bool:
        return self._multipass

    def __getitem__(self, index):
        """Implemented by derived classes """
        raise NotImplementedError()

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        if self._multipass:
            return {
                "punc_input_ids": NeuralType(("B", "T"), TokenIndex()),
                "cap_seg_input_ids": NeuralType(("B", "T"), TokenIndex()),
                "punc_pre_target_ids": NeuralType(("B", "T", "D"), TokenIndex()),  # D == max_subtoken_len
                "punc_post_target_ids": NeuralType(("B", "T", "D"), TokenIndex()),  # D == max_subtoken_len
                "cap_target_ids": NeuralType(("B", "T", "D"), TokenIndex()),  # D == max_subtoken_len
                "seg_target_ids": NeuralType(("B", "T"), TokenIndex()),
                "punct_lengths": NeuralType(("B",), LengthsType()),
                "cap_seg_lengths": NeuralType(("B",), LengthsType()),
            }
        else:
            return {
                "input_ids": NeuralType(("B", "T"), TokenIndex()),
                "punc_pre_target_ids": NeuralType(("B", "T", "D"), TokenIndex()),  # D == max_subtoken_len
                "punc_post_target_ids": NeuralType(("B", "T", "D"), TokenIndex()),  # D == max_subtoken_len
                "cap_target_ids": NeuralType(("B", "T", "D"), TokenIndex()),  # D == max_subtoken_len
                "seg_target_ids": NeuralType(("B", "T"), TokenIndex()),
                "lengths": NeuralType(("B",), LengthsType()),
            }

    def _fold_indices_to_targets(
        self, tokens: List[str], target_indices: List[int], oov_lengths: List[int]
    ) -> List[List[int]]:
        all_targets: List[List[int]] = []
        # For each token, make one output list
        char_index = 0
        oov_index = 0
        for token in tokens:
            token_targets: List[int] = [self._target_pad_value] * self._max_token_len
            if token == self.tokenizer.unk_token:
                char_index += oov_lengths[oov_index]
                oov_index += 1
                all_targets.append(token_targets)
                continue
            start = 2 if token.startswith("##") else 0
            for i in range(start, len(token)):
                char_target = target_indices[char_index]
                token_targets[i] = char_target
                char_index += 1
            all_targets.append(token_targets)
        return all_targets

    @typecheck()
    def collate_fn(self, batch):
        if self._multipass:
            return self._collate_fn_multipass(batch)
        else:
            return self._collate_fn_one_pass(batch)

    def _collate_fn_one_pass(self, batch):
        inputs = [x[0] for x in batch]
        punct_pre_targets_list = [x[1] for x in batch]
        punct_post_targets_list = [x[2] for x in batch]
        cap_targets_list = [x[3] for x in batch]
        seg_targets_list = [x[4] for x in batch]
        lengths = torch.tensor([x.shape[-1] for x in inputs])
        batch_size = len(inputs)  # should be all the same size

        # Create empty input ID tensors and fill non-padded regions
        input_ids = torch.full(size=(batch_size, lengths.max()), fill_value=self._tokenizer.pad_id)
        for i in range(batch_size):
            input_ids[i, : lengths[i]] = inputs[i]

        # Create empty target tensors and fill non-padded regions
        punct_pre_targets = torch.full(
            size=[batch_size, lengths.max(), self._max_token_len], fill_value=self._target_pad_value
        )
        punct_post_targets = torch.full(
            size=[batch_size, lengths.max(), self._max_token_len], fill_value=self._target_pad_value
        )
        cap_targets = torch.full(
            size=[batch_size, lengths.max(), self._max_token_len], fill_value=self._target_pad_value
        )
        seg_targets = torch.full(size=[batch_size, lengths.max()], fill_value=self._target_pad_value)
        for i in range(batch_size):
            cap_targets[i, : lengths[i], :] = cap_targets_list[i]
            seg_targets[i, : lengths[i]] = seg_targets_list[i]
            punct_post_targets[i, : lengths[i], :] = punct_post_targets_list[i]
            punct_pre_targets[i, : lengths[i], :] = punct_pre_targets_list[i]

        return (input_ids, punct_pre_targets, punct_post_targets, cap_targets, seg_targets, lengths)

    def _collate_fn_multipass(self, batch):
        punct_inputs = [x[0] for x in batch]
        cap_seg_inputs = [x[1] for x in batch]
        punct_pre_targets_list = [x[2] for x in batch]
        punct_post_targets_list = [x[3] for x in batch]
        cap_targets_list = [x[4] for x in batch]
        seg_targets_list = [x[5] for x in batch]
        punct_lengths = torch.tensor([x.shape[-1] for x in punct_inputs])
        cap_seg_lengths = torch.tensor([x.shape[-1] for x in cap_seg_inputs])
        batch_size = len(punct_inputs)  # should be all the same size

        # Create empty input ID tensors and fill non-padded regions
        # TODO currently all implementations have tokenizer but tarred dataset may not. Need to know pad id
        punct_input_ids = torch.full(size=(batch_size, punct_lengths.max()), fill_value=self._tokenizer.pad_id)
        cap_seg_input_ids = torch.full(size=(batch_size, cap_seg_lengths.max()), fill_value=self._tokenizer.pad_id)
        for i in range(batch_size):
            punct_input_ids[i, : punct_lengths[i]] = punct_inputs[i]
            cap_seg_input_ids[i, : cap_seg_lengths[i]] = cap_seg_inputs[i]

        # Create empty target tensors and fill non-padded regions
        punct_pre_targets = torch.full(
            size=[batch_size, punct_lengths.max(), self._max_token_len], fill_value=self._target_pad_value
        )
        punct_post_targets = torch.full(
            size=[batch_size, punct_lengths.max(), self._max_token_len], fill_value=self._target_pad_value
        )
        cap_targets = torch.full(
            size=[batch_size, cap_seg_lengths.max(), self._max_token_len], fill_value=self._target_pad_value
        )
        seg_targets = torch.full(size=[batch_size, cap_seg_lengths.max()], fill_value=self._target_pad_value)
        for i in range(batch_size):
            punct_pre_targets[i, : punct_lengths[i], :] = punct_pre_targets_list[i]
            punct_post_targets[i, : punct_lengths[i], :] = punct_post_targets_list[i]
            cap_targets[i, : cap_seg_lengths[i], :] = cap_targets_list[i]
            seg_targets[i, : cap_seg_lengths[i]] = seg_targets_list[i]

        return (
            punct_input_ids,
            cap_seg_input_ids,
            punct_pre_targets,
            punct_post_targets,
            cap_targets,
            seg_targets,
            punct_lengths,
            cap_seg_lengths,
        )


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
        post_labels: List[str],
        pre_labels: List[str],
        null_label: str = "<NULL>",
        ignore_index: int = -100,
        p_drop: float = 0.9,
        rng_seed: Optional[int] = None,
    ) -> None:
        self._p_drop = p_drop
        self._null_label = null_label
        self._rng_seed = rng_seed
        self._ignore_index = ignore_index

        self._pre_label_to_index = {label: i for i, label in enumerate(pre_labels)}
        self._post_label_to_index = {label: i for i, label in enumerate(post_labels)}
        self._pre_null_index = self._pre_label_to_index[null_label]
        self._post_null_index = self._post_label_to_index[null_label]
        # Save as set for quick membership check
        self._pre_labels = set(pre_labels)
        self._post_labels = set(post_labels)
        self._joint_labels = self._pre_labels | self._post_labels
        self._rng = np.random.default_rng(seed=rng_seed)
        self._max_token_len = None

    def reseed_rng(self) -> None:
        self._rng = np.random.default_rng(seed=self._rng_seed)

    @abc.abstractmethod
    def generate_targets(self, input_text: str) -> Tuple[str, List[int], List[int]]:
        """Applies punctuation dropout and generates an example.

        Args:
            input_text: Text to process.

        Returns:
            (out_text, pre_targets, post_targets) where `out_text` is the de-punctuated text, and `pre_targets` and
                each contain the target for each non-whitespace character in `new_text`.
        """
        raise NotImplementedError()

    @classmethod
    def from_lang_code(
        cls, lang_code: str, pre_labels: List[str], post_labels: List[str], p_drop: float, rng_seed: int
    ):
        """Instantiates a derived class which is applicable to the given language.

        This is a convenience function for instantiating a derived class for a particular language.

        Args:
            lang_code: The language code to use to determine which class to instantiate.
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
                pre_labels=pre_labels, post_labels=post_labels, p_drop=p_drop, rng_seed=rng_seed
            )
        elif lang_code in {"zh", "ja", "my"}:
            # Continuous-script languages. The "basic" class seems to work, so nothing special is implemented yet.
            return BasicPuncTargetsGenerator(
                pre_labels=pre_labels, post_labels=post_labels, p_drop=p_drop, rng_seed=rng_seed
            )
        elif lang_code in {"th"}:
            # Thai -- uses space as punctuation. Don't have a solution, yet.
            raise ValueError(f"Language not supported: {lang_code}")
        else:
            # Assume all other languages use English-like punctuation rules.
            return BasicPuncTargetsGenerator(
                pre_labels=pre_labels, post_labels=post_labels, p_drop=p_drop, rng_seed=rng_seed
            )


class BasicPuncTargetsGenerator(PuncTargetsGenerator):
    """Punctuation example generator suitable for most languages, including English.

    This class assumes that punctuation tokens appear only after subwords, and will work for most languages.

    """

    def generate_targets(self, input_text: str) -> Tuple[str, List[int], List[int]]:
        # Normalize whitespaces
        input_text = re.sub(r"\s+", " ", input_text)
        # Empty outputs
        out_chars: List[str] = []
        post_targets: List[int] = []
        # TODO ignore periods that occur between numbers
        for input_char in input_text:
            # No targets for spaces because they are ignored when generating subtokens
            if input_char == " ":
                out_chars.append(" ")
                continue
            # Either create a target, or append to the input
            if post_targets and input_char in self._post_labels and self._rng.random() < self._p_drop:
                post_targets[-1] = self._post_label_to_index[input_char]
            else:
                out_chars.append(input_char)
                post_targets.append(self._post_null_index)
        pre_targets = [self._pre_null_index] * len(post_targets)
        out_text = "".join(out_chars)
        return out_text, pre_targets, post_targets


class SpanishPuncTargetsGenerator(PuncTargetsGenerator):
    """Punctuation example generator for Spanish and Asturian.

    """

    def generate_targets(self, input_text: str) -> Tuple[str, List[int], List[int]]:
        # Normalize whitespaces
        input_text = re.sub(r"\s+", " ", input_text)
        # Empty outputs
        out_chars: List[str] = []
        post_targets: List[int] = []
        pre_targets: List[int] = []
        non_whitespace_idx = 0
        for input_char in input_text:
            # Ignore spaces because they are ignored when generating subtokens
            if input_char == " ":
                out_chars.append(" ")
                continue
            # Either create a target, or append to the input
            if input_char in self._post_labels and self._rng.random() < self._p_drop:
                post_targets[-1] = self._post_label_to_index[input_char]
            elif input_char in self._pre_labels and self._rng.random() < self._p_drop:
                pre_targets.append(self._pre_label_to_index[input_char])
            else:
                non_whitespace_idx += 1
                out_chars.append(input_char)
                post_targets.append(self._post_null_index)
                if len(pre_targets) < non_whitespace_idx:
                    pre_targets.append(self._pre_null_index)
        out_text = "".join(out_chars)
        return out_text, pre_targets, post_targets


class CapTargetsGenerator:
    """Generator of true-casing examples.

    Args:
    """

    def __init__(self, p_lower: float = 0.9, ignore_idx: int = -100, rng_seed: int = 12345) -> None:
        self._ignore_idx = ignore_idx
        self._p_lower = p_lower
        self._rng_seed = rng_seed
        self._rng: Optional[np.random.Generator] = None
        self.reseed_rng()

    def reseed_rng(self):
        """Used by validation DS to get same evaluation examples each epoch"""
        # Note we ignore worker ID, no significant implications for this task.
        self._rng = np.random.default_rng(seed=self._rng_seed)

    def _char_is_uncased(self, char: str):
        return char.lower() == char.upper()

    def generate_targets(self, input_text: str) -> Tuple[str, List[int]]:
        """Randomly re-cased the input text for inputs, and generates targets which matches the input.

        Args:
            input_text: A plain-text string.

        Returns:
            A tuple (new_text, targets)

        """
        # Normalize spaces to allow assumptions
        input_text = re.sub(r"\s+", " ", input_text)
        out_chars: List[str] = []
        targets: List[int] = []
        for input_char in input_text:
            # No targets for space
            if input_char == " ":
                out_chars.append(" ")
                continue
            if self._char_is_uncased(input_char):
                # If uncased, input is unchanged and target is ignore_index
                targets.append(self._ignore_idx)
                out_chars.append(input_char)
            elif input_char.isupper() and self._rng.random() < self._p_lower and (len(input_char.lower()) == 1):
                # If char is upper, maybe lower-case it and make upper target.
                # Some chars lower-case into two chars; for now, deal with it by ignoring them.
                targets.append(1)
                out_chars.append(input_char.lower())
            else:
                # Otherwise, input char is unchanged and target is the input case
                targets.append(1 if input_char.isupper() else 0)
                out_chars.append(input_char)
        return "".join(out_chars), targets


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
            language=language, is_continuous=is_continuous, tokenizer=tokenizer, target_pad_value=target_pad_value
        )
        raise NotImplementedError("Implement TextPunctCapSegDataset.create_tarred_dataset() then implement me.")

    def __getitem__(self, index):
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
        multipass: bool = True,
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
        rng_seed: Optional[int] = None,
    ):
        super().__init__(
            language=language,
            tokenizer=tokenizer,
            target_pad_value=target_pad_value,
            is_continuous=is_continuous,
            multipass=multipass,
        )
        self._text_files = text_files
        self._null_label = null_label
        self._max_length = max_length
        self._punct_pre_labels = punct_pre_labels
        self._punct_post_labels = punct_post_labels
        self._prob_drop_punct = prob_drop_punct
        self._prob_lower_case = prob_lower_case
        self._max_lines_per_eg = max_lines_per_eg
        self._min_lines_per_eg = min_lines_per_eg
        self._prob_truncate = prob_truncate
        self._truncate_max_tokens = truncate_max_tokens
        self._truncate_percentage = truncate_percentage

        self._rng_seed = rng_seed
        self._rng = np.random.default_rng(seed=self._rng_seed) if rng_seed is not None else None

        self._data: List[str] = self._load_data(self._text_files)

        self._punct_targets_gen: PuncTargetsGenerator = PuncTargetsGenerator.from_lang_code(
            lang_code=self._language,
            pre_labels=self._punct_pre_labels,
            post_labels=self._punct_post_labels,
            p_drop=self._prob_drop_punct,
            rng_seed=self._rng_seed,
        )

        # TODO expose options for probability of lower- or upper-casing examples. Currently all lower-cased.
        self._cap_targets_gen: CapTargetsGenerator = CapTargetsGenerator()

    def create_tarred_dataset(self, output_dir: str):
        raise NotImplementedError(
            "Implement me to save this dataset in a tarred format that can be interpreted by TarredPunctCapSegDataset"
        )

    def reseed_rng(self):
        worker_info = torch.utils.data.get_worker_info()
        seed = self._rng_seed
        # If seed is None, just let numpy initialize an RNG
        if worker_info is not None and seed is not None:
            seed += worker_info.id
        self._rng = np.random.default_rng(seed=seed)
        # Reseed modules as well
        self._cap_targets_gen.reseed_rng()
        self._punct_targets_gen.reseed_rng()

    def _load_data(self, text_files) -> List[str]:
        data: List[str] = []
        for text_file in text_files:
            with open(text_file) as f:
                for line in f:
                    data.append(line.strip())
        return data

    def __len__(self):
        return len(self._data)

    def _find_oov_lengths(self, input_text: str) -> List[int]:
        # Need to do mimic tokenizer's behavior
        if isinstance(self.tokenizer, AutoTokenizer) and self.tokenizer.tokenizer.do_basic_tokenize:
            input_text = " ".join(self.tokenizer.tokenizer.basic_tokenizer.tokenize(input_text))
        tokens = self.tokenizer.text_to_tokens(input_text)
        oov_lengths = []
        words = input_text.split()
        word_num = 0
        for token in tokens:
            if token == self.tokenizer.unk_token:
                oov_lengths.append(len(words[word_num]))
            if not token.startswith("##"):
                word_num += 1
        return oov_lengths

    def __getitem__(self, idx):
        # Important not to let every worker use the same RNG because we randomly concat indices, and that would result
        # in the 2nd+ sentences being the same in every worker.
        # TODO belongs in worker_init_fn
        if self._rng is None:
            self.reseed_rng()
        # Each sequence starts with BOS and targets ignore first index
        bos = self._tokenizer.bos_id
        eos = self._tokenizer.eos_id
        pad = self._target_pad_value
        pad_list = [[pad] * self._max_token_len]

        # Randomly choose how many additional lines to use
        num_lines_to_concat = self._rng.integers(self._min_lines_per_eg - 1, self._max_lines_per_eg)
        # Randomly select additional indices to use
        indices_to_use = [idx] + list(self._rng.integers(0, len(self), num_lines_to_concat))
        worker_info = torch.utils.data.get_worker_info()
        punctuated_texts: List[str] = [self._data[x] for x in indices_to_use]

        unpunctuated_texts = []
        punct_pre_target_indices = []
        punct_post_target_indices = []
        for text in punctuated_texts:
            unpunct_text, pre_targets, post_targets = self._punct_targets_gen.generate_targets(text)
            unpunctuated_texts.append(unpunct_text)
            punct_pre_target_indices.extend(pre_targets)
            punct_post_target_indices.extend(post_targets)

        # If this is a one-pass model, use the un-punctuated texts for cap/seg
        cap_seg_texts = punctuated_texts if self._multipass else unpunctuated_texts

        # Concatenate all the texts
        cap_seg_concat_text = ("" if self._is_continuous else " ").join(cap_seg_texts)

        # Generate true-case targets and re-case the text
        recased_cap_seg_text, cap_target_indices = self._cap_targets_gen.generate_targets(cap_seg_concat_text)

        # Generate tokens
        cap_seg_tokens = self.tokenizer.text_to_tokens(recased_cap_seg_text)
        cap_seg_oov_lengths = self._find_oov_lengths(recased_cap_seg_text)

        # Segmentation is predicted once per subword
        seg_targets = []
        boundary_char_indices = []
        for text in cap_seg_texts:
            num_chars_in_text = len(re.sub(r"\s+", "", text))
            # Subsequent boundaries are in addition to previous
            boundary = num_chars_in_text + (0 if not boundary_char_indices else boundary_char_indices[-1])
            boundary_char_indices.append(boundary)
        char_position = 0
        oov_index = 0
        for token in cap_seg_tokens:
            if token == self.tokenizer.unk_id:
                chars_in_token = cap_seg_oov_lengths[oov_index]
                oov_index += 1
            else:
                chars_in_token = len(token) - (2 if token.startswith("##") else 0)
            char_position += chars_in_token
            # If this subword contains the next boundary char, it's a target. Else negative target.
            if boundary_char_indices and char_position >= boundary_char_indices[0]:
                seg_targets.append(1)
                del boundary_char_indices[0]
            else:
                seg_targets.append(0)

        # Finalize the truecase/sentence boundary inputs and targets
        # Fold true-case targets into subword-based
        cap_targets = self._fold_indices_to_targets(cap_seg_tokens, cap_target_indices, cap_seg_oov_lengths)
        # Trim if too long
        cap_seg_ids = self.tokenizer.tokens_to_ids(cap_seg_tokens)
        if len(cap_seg_ids) + 2 > self._max_length:
            cap_seg_ids = cap_seg_ids[: self._max_length - 2]
            seg_targets = seg_targets[: self._max_length - 2]
            cap_targets = cap_targets[: self._max_length - 2]
        # Add BOS/EOS and target padding for those tokens.
        cap_seg_ids = [bos] + cap_seg_ids + [eos]
        seg_targets = [pad] + seg_targets + [pad]
        cap_targets = pad_list + cap_targets + pad_list

        # Finalize punctuation inputs/targets
        if self._multipass:
            # With multi-pass, punctuation will use different inputs than cap/seg
            punct_input_text = ("" if self._is_continuous else " ").join(unpunctuated_texts)
            # If multipass, need to randomly lower-case the punct input text
            if self._rng.random() < self._prob_lower_case:
                # Avoid lower-casing chars that convert to two chars and change the length of the string.
                # punct_input_text = punct_input_text.lower()
                punct_input_text = "".join(c.lower() if len(c.lower()) == 1 else c for c in punct_input_text)
            punct_tokens = self.tokenizer.text_to_tokens(punct_input_text)
            punct_ids = self.tokenizer.tokens_to_ids(punct_tokens)
            if len(punct_ids) + 2 > self._max_length:
                punct_ids = punct_ids[: self._max_length - 2]
                punct_tokens = punct_tokens[: self._max_length - 2]
            punct_ids = [bos] + punct_ids + [eos]
        else:
            # One pass: all use the same inputs
            punct_tokens = cap_seg_tokens
            punct_input_text = recased_cap_seg_text
        punct_oov_lengths = self._find_oov_lengths(punct_input_text)
        punct_pre_targets = self._fold_indices_to_targets(punct_tokens, punct_pre_target_indices, punct_oov_lengths)
        punct_post_targets = self._fold_indices_to_targets(punct_tokens, punct_post_target_indices, punct_oov_lengths)
        punct_pre_targets = pad_list + punct_pre_targets + pad_list
        punct_post_targets = pad_list + punct_post_targets + pad_list

        # Convert to Tensors. Targets are always the same, but inputs vary based on multi-pass mode.
        punct_pre_targets_tensor = torch.tensor(punct_pre_targets, dtype=torch.long)
        punct_post_targets_tensor = torch.tensor(punct_post_targets, dtype=torch.long)
        cap_targets_tensor = torch.tensor(cap_targets, dtype=torch.long)
        seg_targets_tensor = torch.tensor(seg_targets, dtype=torch.long)
        if self._multipass:
            punct_input_tensor = torch.tensor(punct_ids, dtype=torch.long)
            cap_seg_input_tensor = torch.tensor(cap_seg_ids, dtype=torch.long)
            return (
                punct_input_tensor,
                cap_seg_input_tensor,
                punct_pre_targets_tensor,
                punct_post_targets_tensor,
                cap_targets_tensor,
                seg_targets_tensor,
            )
        else:
            input_ids = torch.tensor(cap_seg_ids)
            return (
                input_ids,
                punct_pre_targets_tensor,
                punct_post_targets_tensor,
                cap_targets_tensor,
                seg_targets_tensor,
            )

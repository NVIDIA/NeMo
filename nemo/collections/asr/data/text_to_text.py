from __future__ import annotations

import concurrent.futures
import copy
import gc
import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

import nemo.core.neural_types as ntypes
from nemo.collections.asr.data.audio_to_text import _speech_collate_fn
from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.core.classes import Dataset
from nemo.core.neural_types import NeuralType
from nemo.utils import logging
from nemo_text_processing.text_normalization.normalize import Normalizer

AnyPath = Union[Path, str]


class TextToTextItem(NamedTuple):
    transcript: torch.Tensor
    tts_text: torch.Tensor
    speaker: int


class TextToTextBatch(NamedTuple):
    tts_texts: torch.Tensor
    tts_text_length: torch.Tensor
    transcripts: torch.Tensor
    transcript_length: torch.Tensor
    speakers: torch.Tensor

    @staticmethod
    def collate_fn(batch: List[TextToTextItem], asr_pad_id: int, tts_text_pad_id: int) -> TextToTextBatch:
        return TextToTextBatch(
            tts_texts=pad_sequence([item.tts_text for item in batch], batch_first=True, padding_value=tts_text_pad_id),
            tts_text_length=torch.tensor([item.tts_text.shape[0] for item in batch]).long(),
            transcripts=pad_sequence([item.transcript for item in batch], batch_first=True, padding_value=asr_pad_id),
            transcript_length=torch.tensor([item.transcript.shape[0] for item in batch]).long(),
            speakers=torch.tensor([item.speaker for item in batch]).long(),
        )


class TextOrAudioToTextBatch(NamedTuple):
    audio_signal: torch.Tensor
    a_sig_length: torch.Tensor
    tts_texts: torch.Tensor
    tts_text_length: torch.Tensor
    speakers: torch.Tensor
    transcripts: torch.Tensor
    transcript_length: torch.Tensor

    @staticmethod
    def collate_fn(
        batch: List[tuple], tts_text_pad_id: int, asr_pad_id: int
    ) -> Union[TextToTextBatch, TextOrAudioToTextBatch, tuple]:
        text_items: List[TextToTextItem] = [item for item in batch if isinstance(item, TextToTextItem)]
        if not text_items:
            return _speech_collate_fn(batch=batch, pad_id=asr_pad_id)

        asr_items = [item for item in batch if not isinstance(item, TextToTextItem)]

        if not asr_items:
            return TextToTextBatch.collate_fn(batch=batch, asr_pad_id=asr_pad_id, tts_text_pad_id=tts_text_pad_id)

        audio_signal = pad_sequence([item[0] for item in asr_items], batch_first=True, padding_value=0.0)
        a_sig_length = torch.tensor([item[1] for item in asr_items]).long()

        tts_texts = pad_sequence(
            [item.tts_text for item in text_items], batch_first=True, padding_value=tts_text_pad_id
        )
        tts_text_length = torch.tensor([item.tts_text.shape[0] for item in text_items]).long()
        speakers = torch.tensor([item.speaker for item in text_items]).long()

        transcripts = pad_sequence(
            [item.transcript for item in text_items] + [item[2] for item in asr_items],
            batch_first=True,
            padding_value=asr_pad_id,
        )
        transcript_length = torch.tensor(
            [item.transcript.shape[0] for item in text_items] + [item[3] for item in asr_items]
        ).long()

        return TextOrAudioToTextBatch(
            audio_signal=audio_signal,
            a_sig_length=a_sig_length,
            tts_texts=tts_texts,
            tts_text_length=tts_text_length,
            speakers=speakers,
            transcripts=transcripts,
            transcript_length=transcript_length,
        )


def _asr_text_to_tokens(text: str) -> np.ndarray:
    """
    Helper function for asr tokenization with multiprocessing pool only.
    Must be defined on the top level.
    Expects asr_tokenizer_global, asr_bos_id_global, asr_eos_id_global to exist in the current pool process
    :param text:
    :return: tokenized text
    """
    ids = asr_tokenizer_global.text_to_ids(text)
    if asr_bos_id_global is not None:
        ids = [asr_bos_id_global] + ids
    if asr_eos_id_global is not None:
        ids.append(asr_eos_id_global)
    return np.asarray(ids)


def _tts_text_to_tokens(text: str) -> np.ndarray:
    """
    Helper function for asr tokenization with multiprocessing pool only.
    Must be defined on the top level.
    Expects tts_tokenizer_global to exist in the current pool process
    :param text:
    :return: tokenized text
    """
    return np.asarray(tts_tokenizer_global(text))


def iterate_manifest(filepath: AnyPath) -> Iterable[Dict[str, Any]]:
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            yield record


class TextToTextDataset(Dataset):
    asr_pad_id: int
    tts_text_pad_id: int
    asr_bos_id: Optional[int] = None
    asr_eos_id: Optional[int] = None

    # @property
    # def output_types(self) -> Optional[Dict[str, NeuralType]]:
    #     """Returns definitions of module output ports."""
    #     return {
    #         'transcripts': NeuralType(('B', 'T'), ntypes.LabelsType()),
    #         'transcript_length': NeuralType(tuple('B'), ntypes.LengthsType()),
    #         'tts_texts': NeuralType(('B', 'T'), ntypes.TokenIndex()),
    #         'tts_text_length': NeuralType(tuple('B'), ntypes.LengthsType()),
    #     }

    def __init__(
        self,
        manifest_filepath: Union[AnyPath, List[AnyPath]],
        speakers_filepath: Union[AnyPath, List[AnyPath]],
        asr_tokenizer: TokenizerSpec,
        asr_use_start_end_token: bool,
        tts_text_normalizer: Normalizer,
        tts_text_normalizer_call_kwargs: Dict,
        tts_parser: Callable,
        tts_text_pad_id: int,
        min_words: int = 1,
        max_words: int = 1_000_000,
        tokenizer_workers: int = 1,
    ):
        super().__init__()
        # ASR tokenizer setup
        if asr_use_start_end_token and hasattr(asr_tokenizer, 'bos_token'):
            self.asr_bos_id = asr_tokenizer.bos_id

        if asr_use_start_end_token and hasattr(asr_tokenizer, 'eos_token'):
            self.asr_eos_id = asr_tokenizer.eos_id

        if hasattr(asr_tokenizer, 'pad_token'):
            self.asr_pad_id = asr_tokenizer.pad_id
        else:
            self.asr_pad_id = 0

        self.asr_tokenizer = asr_tokenizer

        # TTS tokenizer setup
        self.tts_parser = tts_parser
        self.tts_normalizer = tts_text_normalizer
        self.tts_normalizer_kwargs = tts_text_normalizer_call_kwargs
        self.tts_text_pad_id = tts_text_pad_id

        # Load manifest
        if isinstance(manifest_filepath, str):
            manifest_filepath = manifest_filepath.split(",")
        elif isinstance(manifest_filepath, Path):
            manifest_filepath = [manifest_filepath]
        self.manifest_paths = [Path(filepath) for filepath in manifest_filepath]

        # Load speakers
        if isinstance(speakers_filepath, str):
            speakers_filepath = speakers_filepath.split(",")
        elif isinstance(speakers_filepath, Path):
            speakers_filepath = [speakers_filepath]
        speakers: Set[int] = set()
        for filepath in speakers_filepath:
            with open(Path(filepath).expanduser(), "r") as f:
                speakers.update(map(int, f.read().split()))
        self.speakers = np.asarray(sorted(speakers))
        logging.info(f"Loaded {len(self.speakers)} speakers")

        num_skipped_words = 0
        num_skipped_utterances = 0
        asr_texts = []
        tts_texts = []

        for manifest_path in self.manifest_paths:
            for tmp_item in tqdm(iterate_manifest(manifest_path)):
                text = tmp_item["text"]
                num_words = len(text.split())
                if not (min_words <= num_words <= max_words):
                    num_skipped_words += num_words
                    num_skipped_utterances += 1
                    continue
                asr_texts.append(tmp_item["text"])
                tts_texts.append(tmp_item["tts_text_normalized"])

        if num_skipped_utterances:
            logging.warning(f"Skipped {num_skipped_utterances} utterances " f"with {num_skipped_words}")

        num_utterances = len(asr_texts)
        self.data: List[Dict[str, Any]] = [dict() for _ in range(num_utterances)]

        if tokenizer_workers == 1:
            for i, tokenized_text in enumerate(
                tqdm((self._asr_text_to_tokens(text) for text in asr_texts), total=len(asr_texts))
            ):
                self.data[i]["asr_text_tokens"] = tokenized_text
        else:
            # Multiprocessing hack: use global variables for every process (not really global in program context)
            def _init_asr_tokenize_process(tokenizer, bos_id, eos_id):
                global asr_tokenizer_global, asr_bos_id_global, asr_eos_id_global  # process-global
                asr_tokenizer_global = copy.deepcopy(tokenizer)
                asr_bos_id_global = copy.deepcopy(bos_id)
                asr_eos_id_global = copy.deepcopy(eos_id)

            with concurrent.futures.ProcessPoolExecutor(
                initializer=_init_asr_tokenize_process,
                initargs=(asr_tokenizer, self.asr_bos_id, self.asr_eos_id),
                max_workers=tokenizer_workers,
            ) as pool:
                for i, tokenized_text in enumerate(
                    tqdm(pool.map(_asr_text_to_tokens, asr_texts), total=len(asr_texts))
                ):
                    self.data[i]["asr_text_tokens"] = tokenized_text
        # force free memory
        del asr_texts
        gc.collect()

        # fixme: normalization
        if tokenizer_workers == 1:
            logging.warning("Preprocessing text with tokenizer_workers=1 may be slow")
            for i, tokenized_text in enumerate(
                tqdm((self._tts_text_to_tokens(text, normalize=False) for text in tts_texts), total=len(tts_texts))
            ):
                self.data[i]["tts_text_tokens"] = tokenized_text
        else:

            def _init_tts_tokenize_process(tokenizer):
                global tts_tokenizer_global  # process-global
                tts_tokenizer_global = copy.deepcopy(tokenizer)

            with concurrent.futures.ProcessPoolExecutor(
                initializer=_init_tts_tokenize_process,
                initargs=(tts_parser, self.asr_bos_id, self.asr_eos_id),
                max_workers=tokenizer_workers,
            ) as pool:
                for i, tokenized_text in enumerate(
                    tqdm(pool.map(_tts_text_to_tokens, tts_texts), total=len(tts_texts))
                ):
                    self.data[i]["tts_text_tokens"] = tokenized_text
        # force free memory
        del tts_texts
        gc.collect()

    def _asr_text_to_tokens(self, text: str) -> np.ndarray:
        ids = self.asr_tokenizer.text_to_ids(text)
        if self.asr_bos_id is not None:
            ids = [self.asr_bos_id] + ids
        if self.asr_eos_id is not None:
            ids.append(self.asr_eos_id)
        return np.asarray(ids)

    def _tts_text_to_tokens(self, text: str, normalize=True) -> np.ndarray:
        if normalize:
            text = self.tts_normalizer.normalize(text, **self.tts_normalizer_kwargs)
        tokens = self.tts_parser(text)
        return np.asarray(tokens)

    def __getitem__(self, index):
        item = self.data[index]
        return TextToTextItem(
            transcript=torch.from_numpy(item["asr_text_tokens"]).long(),
            tts_text=torch.from_numpy(item["tts_text_tokens"]).long(),
            speaker=random.choice(self.speakers),
        )

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch: List[tuple]) -> Union[TextToTextBatch, TextOrAudioToTextBatch, tuple]:
        return TextOrAudioToTextBatch.collate_fn(
            batch, asr_pad_id=self.asr_pad_id, tts_text_pad_id=self.tts_text_pad_id
        )

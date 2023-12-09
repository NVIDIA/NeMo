import sys
from typing import Dict, Optional, Tuple
from contextlib import contextmanager

import torch.utils.data
from lhotse import CutSet
from lhotse.dataset import AudioSamples, CutMix
from lhotse.dataset.collation import collate_vectors

from nemo.utils import logging
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import (
    BaseTokenizer,
    EnglishCharsTokenizer,
    EnglishPhonemesTokenizer,
)

from naturalspeech2_pytorch.utils.cleaner import TextProcessor

class LhotseTextToSpeechDataset(torch.utils.data.Dataset):
    """
    This dataset is based on `nemo.collections.asr.data.audio_to_text_lhotse.py`.
    Unlike native NeMo datasets, Lhotse dataset defines only the mapping from
    a CutSet (meta-data) to a mini-batch with PyTorch tensors.
    Specifically, it performs tokenization, I/O, augmentation, and feature extraction (if any).
    Managing data, sampling, de-duplication across workers/nodes etc. is all handled
    by Lhotse samplers instead.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
            'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def __init__(self, normalizer=None, text_normalizer_call_kwargs=None, tokenizer=None, corpus_dir=None, old_prefix="download/librilight"):
        super().__init__()
        self.tokenizer = tokenizer

        if tokenizer is not None:
            if isinstance(tokenizer, TextProcessor):
                self.normalizer_call = tokenizer.text_cleaner
                self.text_normalizer_call_kwargs = {"language": "en"}

            elif isinstance(tokenizer, BaseTokenizer):
                self.normalizer = normalizer
                if normalizer is not None:
                    self.normalizer_call = (
                        self.normalizer.normalize
                        if isinstance(self.normalizer, Normalizer)
                        else self.normalizer
                    )
                    if text_normalizer_call_kwargs is not None:
                        self.text_normalizer_call_kwargs = text_normalizer_call_kwargs
                    else:
                        self.text_normalizer_call_kwargs = {}

        self.corpus_dir = corpus_dir
        if corpus_dir is not None:
            self.old_prefix = old_prefix
        self.load_audio = AudioSamples(fault_tolerant=True)

    def __getitem__(self, cuts: CutSet) -> Tuple[torch.Tensor, ...]:
        # Some corpus, e.g., LibriHeavy, whose manifest includes given path prefix, which might not match our folder structure.
        # the following lines fix the path prefix
        if self.corpus_dir is not None:
            for c in cuts:
                old_path = c.recording.sources[0].source
                new_path = old_path.replace(self.old_prefix, self.corpus_dir)
                c.recording.sources[0].source = new_path

        batch = {}

        cuts = cuts.sort_by_duration()
        audio, audio_lens, cuts = self.load_audio(cuts)
        texts = [c.supervisions[0].custom["texts"][0] for c in cuts]

        audio_22050, audio_lens_22050, _ = self.load_audio(cuts.resample(22050))
        audio_24k, audio_lens_24k, _ = self.load_audio(cuts.resample(24000))
        batch.update({
            "audio": audio,
            "audio_lens": audio_lens,
            "audio_22050": audio_22050,
            "audio_lens_22050": audio_lens_22050,
            "audio_24k": audio_24k,
            "audio_lens_24k": audio_lens_24k,
            "texts": texts,
        })

        if self.tokenizer is not None:
            if isinstance(self.tokenizer, TextProcessor):
                with redirect_stdout_to_logger(logging):
                    tokens = [self.tokenizer.text_to_ids(text)[0] for text in texts]
                padding_value = 0
            elif isinstance(self.tokenizer, BaseTokenizer):
                _texts = [c.supervisions[0].custom["texts"][1] for c in cuts]
                texts = [self.normalizer_call(text, **self.text_normalizer_call_kwargs) for text in _texts]
                tokens = [self.tokenizer(text) for text in texts]
                padding_value = self.tokenizer.pad


            tokens = [torch.as_tensor(token_ids).long() for token_ids in tokens]
            token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
            # tokens = collate_vectors(tokens, padding_value=0)
            tokens = collate_vectors(tokens, padding_value=padding_value)
            batch.update({
                "texts": texts,
                "tokens": tokens,
                "token_lens": token_lens
            })

        return batch


class StdoutRedirector:
    def __init__(self, logger):
        self.logger = logger

    def write(self, message):
        if message != '\n':
            self.logger.warning(message)

    def flush(self):
        pass


@contextmanager
def redirect_stdout_to_logger(logger):
    original_stdout = sys.stdout  # 保存原始的 stdout
    sys.stdout = StdoutRedirector(logger)  # 將 stdout 重定向到自定義的類
    try:
        yield
    finally:
        sys.stdout = original_stdout  # 恢復原始的 stdout
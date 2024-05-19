import sys
import os
from glob import glob
from pathlib import Path
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

from nemo.collections.tts.modules.voicebox_modules import MFAEnglishPhonemeTokenizer

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

    def __init__(self, normalizer=None, text_normalizer_call_kwargs=None, tokenizer=None,
                 ds_name="", corpus_dir=None, old_prefix=None, textgrid_dir=None,
                 use_word_postfix=False, use_word_ghost_silence=False, num_workers=0, load_audio=True, sampling_rate=24000):
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

            elif isinstance(self.tokenizer, MFAEnglishPhonemeTokenizer):
                self.normalizer_call = None
                self.text_normalizer_call_kwargs = {}
                self.textgrid_dir = textgrid_dir

        self.ds_name = ds_name
        self.corpus_dir = corpus_dir
        self.old_prefix = old_prefix

        if load_audio:
            self.load_audio = AudioSamples(num_workers=num_workers, fault_tolerant=True)

        self.use_word_postfix = use_word_postfix
        self.use_word_ghost_silence = use_word_ghost_silence
        self.sampling_rate = sampling_rate

    def change_prefix(self, cut):
        # Some corpus, e.g., LibriHeavy, whose manifest includes given path prefix, which might not match our folder structure.
        # the following lines fix the path prefix
        if self.corpus_dir is not None:
            old_path = cut.recording.sources[0].source
            new_path = old_path.replace(self.old_prefix, self.corpus_dir)
            cut.recording.sources[0].source = new_path
        if self.ds_name == "gigaspeech" and not os.path.exists(cut.recording.sources[0].source):
            # HF random path
            old_path = Path(cut.recording.sources[0].source)
            new_path = glob(str(old_path.parents[2] / "*" / old_path.parts[-2] / old_path.parts[-1]))[0]
            # print(str(old_path), str(new_path))
            cut.recording.sources[0].source = new_path
        return cut

    def get_cut_alignment(self, cut):
        phn_alis = cut.supervisions[0].alignment["phones"]
        phn_dur = []
        for ali in phn_alis:
            phn_dur.append((ali.symbol, ali.duration))

        if "words" not in cut.supervisions[0].alignment or (not self.use_word_postfix and not self.use_word_ghost_silence):
            return phn_dur
        
        word_alis = cut.supervisions[0].alignment["words"]
        w2pids = []
        phn_id = 0
        for ali in word_alis:
            wrd = ali.symbol
            if ali.symbol in ["", "sil", "<eps>"]:
                wrd = "<eps>"
            if ali.symbol in ["spn", "<unk>"]:
                wrd = "<unk>"

            w2pids.append([wrd, []])
            wrd_st = ali.start
            wrd_ed = wrd_st + ali.duration

            phn_st = phn_alis[phn_id].start
            phn_ed = phn_st + phn_alis[phn_id].duration
            # while phn_st >= wrd_st and phn_ed <= wrd_ed:
            while phn_st + phn_ed >= 2 * wrd_st and phn_st + phn_ed <= 2 * wrd_ed:
                w2pids[-1][-1].append(phn_id)
                phn_id += 1
                if phn_id < len(phn_alis):
                    phn_st = phn_alis[phn_id].start
                    phn_ed = phn_st + phn_alis[phn_id].duration
                else:
                    break

        new_phn_dur = []
        _wrd = "<eps>"
        for wrd, phn_ids in w2pids:
            if self.use_word_ghost_silence and len(new_phn_dur) > 0:
                if wrd != "<eps>" and _wrd != "<eps>":
                    if self.use_word_postfix:
                        new_phn_dur.append(("sil_S", 0))
                    else:
                        new_phn_dur.append(("sil", 0))
                _wrd = wrd

            postfixs = [""] * len(phn_ids)
            if self.use_word_postfix:
                if len(phn_ids) == 1:
                    postfixs = ["_S"]
                else:
                    postfixs = ["_B"] + ["_I"] * (len(phn_ids)-2) + ["_E"]

            _dur = []
            for phn_id, postfix in zip(phn_ids, postfixs):
                _dur.append((phn_dur[phn_id][0] + postfix, phn_dur[phn_id][1]))
            new_phn_dur += _dur

        return new_phn_dur

    def __getitem__(self, cuts: CutSet) -> Tuple[torch.Tensor, ...]:
        batch = {}

        cuts = cuts.sort_by_duration()
        cuts = cuts.map(self.change_prefix)
        batch.update({
            "cuts": cuts,
            "audio_paths": [cut.recording.sources[0].source for cut in cuts],
        })

        if hasattr(self, 'load_audio'):
            audio, audio_lens, _cuts = self.load_audio(cuts.resample(self.sampling_rate))
            # audio_22050, audio_lens_22050, _cuts = self.load_audio(cuts.resample(22050))
            # audio_24k, audio_lens_24k, _cuts = self.load_audio(cuts.resample(24000))
            batch.update({
                "audio": audio,
                "audio_lens": audio_lens,
                # "audio_22050": audio_22050,
                # "audio_lens_22050": audio_lens_22050,
                # "audio_24k": audio_24k,
                # "audio_lens_24k": audio_lens_24k,
            })
        else:
            _cuts = cuts

        if _cuts[0].supervisions[0].custom is not None and "texts" in _cuts[0].supervisions[0].custom:
            texts = [c.supervisions[0].custom["texts"][0] for c in _cuts]
        else:
            texts = [c.supervisions[0].text for c in _cuts]

        batch.update({
            "texts": texts,
        })
        if not hasattr(self, "load_audio"):
            return batch

        if self.tokenizer is not None:
            if isinstance(self.tokenizer, TextProcessor):
                with redirect_stdout_to_logger(logging):
                    tokens = [self.tokenizer.text_to_ids(text)[0] for text in texts]
                padding_value = 0
            elif isinstance(self.tokenizer, BaseTokenizer):
                if "texts" in _cuts[0].supervisions[0].custom:
                    _texts = [c.supervisions[0].custom["texts"][1] for c in _cuts]
                else:
                    _texts = [c.supervisions[0].text for c in _cuts]
                texts = [self.normalizer_call(text, **self.text_normalizer_call_kwargs) for text in _texts]
                tokens = [self.tokenizer(text) for text in texts]
                padding_value = self.tokenizer.pad
            elif isinstance(self.tokenizer, MFAEnglishPhonemeTokenizer):
                phn_durs = [self.get_cut_alignment(cut) for cut in cuts]
                durs = [[dur for _, dur in phn_dur] for phn_dur in phn_durs]
                durs = [torch.as_tensor(ds) for ds in durs]
                durs = collate_vectors(durs, padding_value=0)
                batch["durations"] = durs

                phns = [[phn for phn, _ in phn_dur] for phn_dur in phn_durs]
                tokens = [self.tokenizer.text_to_ids(phn)[0] for phn in phns]
                padding_value = self.tokenizer.pad_id

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
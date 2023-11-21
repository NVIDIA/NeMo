from typing import Dict, Optional, Tuple

import torch.utils.data
from lhotse import CutSet
from lhotse.dataset import AudioSamples, CutMix
from lhotse.dataset.collation import collate_vectors

from nemo.utils import logging
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType


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

    def __init__(self, tokenizer, corpus_dir=None, old_prefix="download/librilight"):
        super().__init__()
        self.tokenizer = tokenizer
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

        cuts = cuts.sort_by_duration()
        try:
            audio, audio_lens, cuts = self.load_audio(cuts)
            tokens = [torch.as_tensor(self.tokenizer.text_to_ids(c.supervisions[0].text)) for c in cuts]
            token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
            tokens = collate_vectors(tokens, padding_value=0)
            return audio, audio_lens, tokens, token_lens
        except Exception as e:
            # typically when failed to load all of the audios
            logging.error("Failed to load audio batch.", exc_info=e)
            raise e


def _identity(x):
    return x

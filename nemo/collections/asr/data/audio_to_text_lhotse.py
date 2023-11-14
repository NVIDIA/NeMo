from typing import Dict, Optional, Tuple

import torch.utils.data
from lhotse import CutSet
from lhotse.dataset import AudioSamples, CutMix
from lhotse.dataset.collation import collate_vectors

from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType


class LhotseSpeechToTextBpeDataset(torch.utils.data.Dataset):
    """
    This dataset is based on BPE datasets from audio_to_text.py.
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

    def __init__(self, tokenizer, noise_cuts: Optional[CutSet] = None):
        super().__init__()
        self.tokenizer = tokenizer
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.maybe_mix_noise = (
            _identity if noise_cuts is None else CutMix(noise_cuts, pad_to_longest=False, random_mix_offset=True)
        )

    def __getitem__(self, cuts: CutSet) -> Tuple[torch.Tensor, ...]:
        cuts = cuts.sort_by_duration()
        cuts = self.maybe_mix_noise(cuts)
        audio, audio_lens, cuts = self.load_audio(cuts)
        tokens = [torch.as_tensor(self.tokenizer.text_to_ids(c.supervisions[0].text)) for c in cuts]
        token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
        tokens = collate_vectors(tokens, padding_value=0)
        return audio, audio_lens, tokens, token_lens


def _identity(x):
    return x

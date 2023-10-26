from typing import Optional, Dict, Tuple

import torch.utils.data
from lhotse import CutSet
from lhotse.dataset import DynamicBucketingSampler, AudioSamples, PerturbSpeed
from lhotse.dataset.collation import collate_vectors
from omegaconf import DictConfig

from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType, LabelsType


def get_lhotse_audio_to_text_char_dataloader_from_config(
    config,
    local_rank: int,
    global_rank: int,
    world_size: int,
    tokenizer,
    preprocessor_cfg: Optional[DictConfig] = None,
):
    dataset = LhotseSpeechToTextBpeDataset(tokenizer=tokenizer)

    cuts = (
        CutSet.from_file(config.lhotse.cuts_path)
        .filter(lambda c: config.min_duration <= c.duration <= config.max_duration)
        .resample(16000)
    )
    # TODO: support: cuts = CutSet.from_shar(...)

    sampler = DynamicBucketingSampler(
        cuts,
        max_duration=config.lhotse.batch_duration,
        num_buckets=config.lhotse.num_buckets,
        shuffle=True,
        drop_last=True,
        num_cuts_for_bins_estimate=config.lhotse.num_cuts_for_bins_estimate,
        buffer_size=config.lhotse.buffer_size,
        shuffle_buffer_size=config.lhotse.shuffle_buffer_size,
        quadratic_duration=config.lhotse.quadratic_duration,
        rank=global_rank,
        world_size=world_size,
    )

    dloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=None,
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', False),
    )

    return dloader


class LhotseSpeechToTextBpeDataset(torch.utils.data.Dataset):
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
            'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.cut_transforms = [PerturbSpeed(factors=[0.9, 1.1], p=2 / 3)]  # can add others later

    def __getitem__(self, cuts: CutSet) -> Tuple[torch.Tensor, ...]:
        cuts = cuts.sort_by_duration()
        for t in self.cut_transforms:
            cuts = t(cuts)
        audio, audio_lens, cuts = self.load_audio(cuts)
        tokens = [torch.as_tensor(self.tokenizer.text_to_ids(c.supervisions[0].text)) for c in cuts]
        token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
        tokens = collate_vectors(tokens, padding_value=0)
        return audio, audio_lens, tokens, token_lens

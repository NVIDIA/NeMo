import warnings
from typing import Dict, Optional, Tuple

import torch.utils.data
from lhotse import CutSet
from lhotse.dataset import (
    AudioSamples,
    DynamicBucketingSampler,
    IterableDatasetWrapper,
    make_worker_init_fn,
    CutMix,
    DynamicCutSampler,
)
from lhotse.dataset.collation import collate_vectors
from omegaconf import DictConfig

from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType


def get_lhotse_audio_to_text_char_dataloader_from_config(
    config,
    local_rank: int,
    global_rank: int,
    world_size: int,
    tokenizer,
    preprocessor_cfg: Optional[DictConfig] = None,
):
    """
    Setup a Lhotse training dataloder.

    Expects a typical NeMo dataset configuration format, with additional fields: "use_lhotse=True" and "lhotse: <dict>".
    Some fields in the original NeMo configuration are ignored (e.g. ``batch_size``).
    To learn about lhotse specific parameters, search this code for ``config.lhotse``.
    """
    use_shar = config.lhotse.get("shar_path") is not None

    # 1. Load Lhotse manifest.
    if use_shar:
        # Lhotse Shar is the equivalent of NeMo's native "tarred" dataset.
        # The combination of shuffle_shards, and repeat causes this to
        # be an infinite manifest that is internally reshuffled on each epoch.
        # seed="randomized" means we'll defer setting the seed until the iteration
        # is triggered, so we can obtain node+worker specific seed thanks to worker_init_fn.
        # This results in every dataloading worker using full data but in a completely different order.
        if config.lhotse.get("cuts_path") is not None:
            warnings.warn("Note: lhotse.cuts_path will be ignored because lhotse.shar_path was provided.")
        cuts = CutSet.from_shar(in_dir=config.lhotse.shar_path, shuffle_shards=True, seed="randomized").repeat()
    else:
        # Regular Lhotse manifest points to individual audio files (like native NeMo manifest).
        cuts = CutSet.from_file(config.lhotse.cuts_path)

    # Duration filtering, same as native NeMo dataloaders.
    cuts = cuts.filter(lambda c: config.min_duration <= c.duration <= config.max_duration)

    # 2. Optional on-the-fly speed perturbation,
    #    mux here ensures it's uniformly distributed throughout sampling,
    #    and applying it here (before sampler/dataset) ensures optimal
    #    bucket allocation.
    if config.lhotse.get("perturb_speed", False):
        cuts = CutSet.mux(cuts, cuts.perturb_speed(0.9), cuts.perturb_speed(1.1),)

    # 3. The sampler.
    if config.lhotse.get("use_bucketing", True):
        # Bucketing. Some differences from NeMo's native bucketing:
        #    - we can tweak the number of buckets without triggering a full data copy
        #    - batch size is dynamic and configurable via a single param: max_duration (config: batch_duration)
        #    - quadratic_duraion introduces a penalty useful to balance batch sizes for quadratic time complexity models
        sampler = DynamicBucketingSampler(
            cuts,
            max_duration=config.lhotse.batch_duration,
            num_buckets=config.lhotse.get("num_buckets", 10),
            shuffle=config.get("shuffle", False),
            drop_last=config.lhotse.get("drop_last", True),
            num_cuts_for_bins_estimate=config.lhotse.get("num_cuts_for_bins_estimate", 10000),
            buffer_size=config.lhotse.get("buffer_size", 10000),
            shuffle_buffer_size=config.lhotse.get("shuffle_buffer_size", 10000),
            quadratic_duration=config.lhotse.get("quadratic_duration", None),
            rank=0 if use_shar else global_rank,
            world_size=1 if use_shar else world_size,
        )
    else:
        # Non-bucketing, similar to NeMo's regular non-tarred manifests,
        # but we also use batch_duration instead of batch_size here.
        # Recommended for dev/test.
        sampler = DynamicCutSampler(
            cuts,
            max_duration=config.lhotse.batch_duration,
            shuffle=config.get("shuffle", False),
            drop_last=config.lhotse.get("drop_last", False),
            shuffle_buffer_size=config.lhotse.get("shuffle_buffer_size", 10000),
            rank=0 if use_shar else global_rank,
            world_size=1 if use_shar else world_size,
        )

    # 4. Dataset only maps CutSet -> batch of tensors.
    #    For non-shar data, I/O happens inside dataset __getitem__.
    #    For shar data, I/O happens in sampler iteration, so we put it together with the dataset
    #    into an iterable dataset based wrapper (see the next step).
    dataset = LhotseSpeechToTextBpeDataset(tokenizer=tokenizer, noise_cuts=config.lhotse.get("noise_cuts"))

    # 5. Creating dataloader (wrapper is explained in 4. and worker_init_fn in 1.).
    if use_shar:
        dloader_kwargs = dict(
            dataset=IterableDatasetWrapper(dataset=dataset, sampler=sampler,),
            worker_init_fn=make_worker_init_fn(rank=global_rank, world_size=world_size),
            persistent_workers=True,  # helps Lhotse Shar maintain shuffling state
        )
    else:
        dloader_kwargs = dict(dataset=dataset, sampler=sampler)
    dloader = torch.utils.data.DataLoader(
        **dloader_kwargs,
        batch_size=None,
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', False),
    )

    return dloader


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

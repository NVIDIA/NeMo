import random
import secrets
import tarfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import lhotse.lazy
import torch.utils.data
from lhotse import CutSet
from lhotse.dataset import (
    AudioSamples,
    CutMix,
    DynamicBucketingSampler,
    DynamicCutSampler,
    IterableDatasetWrapper,
    make_worker_init_fn,
)
from lhotse.dataset.collation import collate_vectors
from omegaconf import DictConfig

from nemo.collections.asr.data.audio_to_text import expand_sharded_filepaths
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

    # 1. Load a manifest as a Lhotse CutSet.
    #    TODO: support mixing data from multiple sources via CutSet.mux()
    cuts, is_tarred = read_as_cutset(config)

    # Duration filtering, same as native NeMo dataloaders.
    cuts = cuts.filter(
        lambda c: config.get("min_duration", -1) <= c.duration <= config.get("max_duration", float("inf"))
    )

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
            rank=0 if is_tarred else global_rank,
            world_size=1 if is_tarred else world_size,
        )
    else:
        # Non-bucketing, similar to NeMo's regular non-tarred manifests,
        # but we also use batch_duration instead of batch_size here.
        # Recommended for dev/test.
        sampler = DynamicCutSampler(
            cuts,
            max_duration=config.lhotse.batch_duration,
            shuffle=config.get("shuffle", False),
            drop_last=config.lhotse.get("drop_last", True),
            shuffle_buffer_size=config.lhotse.get("shuffle_buffer_size", 10000),
            rank=0 if is_tarred else global_rank,
            world_size=1 if is_tarred else world_size,
        )

    # 4. Dataset only maps CutSet -> batch of tensors.
    #    For non-shar data, I/O happens inside dataset __getitem__.
    #    For shar data, I/O happens in sampler iteration, so we put it together with the dataset
    #    into an iterable dataset based wrapper (see the next step).
    dataset = LhotseSpeechToTextBpeDataset(tokenizer=tokenizer, noise_cuts=config.lhotse.get("noise_cuts"))

    # 5. Creating dataloader (wrapper is explained in 4. and worker_init_fn in 1.).
    if is_tarred:
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


def read_as_cutset(config) -> Tuple[CutSet, bool]:
    """
    Reads NeMo configuration and creates a CutSet either from Lhotse or NeMo manifests.

    Returns a tuple of ``CutSet`` and a boolean indicating whether the data is tarred (True) or not (False).
    """
    # First, we'll figure out if we should read Lhotse manifest or NeMo manifest.
    use_nemo_manifest = all(config.lhotse.get(opt) is None for opt in ("cuts_path", "shar_path"))
    if use_nemo_manifest:
        assert (
            config.get("manifest_filepath") is not None
        ), "You must specify either: manifest_filepath, lhotse.cuts_path, or lhotse.shar_path"
        is_tarred = config.get("tarred_audio_filepaths") is not None
    else:
        is_tarred = config.lhotse.get("shar_path") is not None
    if use_nemo_manifest:
        # Read NeMo manifest -- use the right wrapper depending on tarred/non-tarred.
        if is_tarred:
            cuts = CutSet(
                LazyNeMoTarredIterator(
                    config["manifest_filepath"],
                    tar_paths=config["tarred_audio_filepaths"],
                    shuffle_shards=config.get("shuffle", False),
                )
            )
        else:
            cuts = CutSet(LazyNeMoIterator(config["manifest_filepath"], sampling_rate=config.get("sample_rate")))
    else:
        # Read Lhotse manifest (again handle both tarred(shar)/non-tarred).
        if is_tarred:
            # Lhotse Shar is the equivalent of NeMo's native "tarred" dataset.
            # The combination of shuffle_shards, and repeat causes this to
            # be an infinite manifest that is internally reshuffled on each epoch.
            # seed="trng" means we'll defer setting the seed until the iteration
            # is triggered, and we'll use system TRNG to get a completely random seed for each worker.
            # This results in every dataloading worker using full data but in a completely different order.
            # Note: there is also seed="randomized", but "trng" works around PyTorch-Lightning training loop
            # that apparently re-creates dataloader on each training "epoch", which results in identical sampling.
            if config.lhotse.get("cuts_path") is not None:
                warnings.warn("Note: lhotse.cuts_path will be ignored because lhotse.shar_path was provided.")
            cuts = CutSet.from_shar(in_dir=config.lhotse.shar_path, shuffle_shards=True, seed="trng").repeat()
        else:
            # Regular Lhotse manifest points to individual audio files (like native NeMo manifest).
            cuts = CutSet.from_file(config.lhotse.cuts_path)
    return cuts, is_tarred


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


class LazyNeMoIterator(lhotse.lazy.ImitatesDict):
    """
    ``LazyNeMoIterator`` reads a NeMo (non-tarred) JSON manifest and converts it on the fly to an ``Iterable[Cut]``.
    It's used to create a ``lhotse.CutSet``.

    Currently, it requires (and exclusively supports) the following keys in NeMo manifests:
    - "audio_filepath"
    - "duration"
    - "text"

    .. caution:: We assume every audio file has the same sampling rate, and it has to be explicitly provided
        in ``LazyNeMoIterator`` constructor.

    Example::

        >>> cuts = lhotse.CutSet(LazyNeMoIterator("nemo_manifests/train.json", sampling_rate=16000))
    """

    def __init__(self, path: str | Path, sampling_rate: int = 16000) -> None:
        self.source = lhotse.lazy.LazyJsonlIterator(path)
        self.sampling_rate = sampling_rate

    @property
    def path(self) -> str | Path:
        return self.source.path

    def __iter__(self):
        for data in self.source:
            recording = lhotse.Recording(
                id=Path(data["audio_filepath"]).name,
                sources=[lhotse.audio.AudioSource(type="file", channels=[0], source=data["audio_filepath"],)],
                sampling_rate=self.sampling_rate,
                duration=data["duration"],
                num_samples=lhotse.utils.compute_num_samples(data["duration"], self.sampling_rate),
            )
            cut = recording.to_cut()
            cut.supervisions.append(
                lhotse.SupervisionSegment(
                    id=cut.id, recording_id=cut.recording_id, start=0, duration=cut.duration, text=data["text"],
                )
            )
            yield cut

    def __len__(self) -> int:
        return len(self.source)

    def __add__(self, other) -> "lhotse.lazy.LazyIteratorChain":
        return lhotse.lazy.LazyIteratorChain(self, other)


class LazyNeMoTarredIterator(lhotse.lazy.ImitatesDict):
    """
    ``LazyNeMoTarredIterator`` reads a NeMo tarred JSON manifest and converts it on the fly to an ``Iterable[Cut]``.
    It's used to create a ``lhotse.CutSet``.



    Currently, it requires (and exclusively supports) the following keys in NeMo manifests:
    - "audio_filepath"
    - "duration"
    - "text"
    - "shard_id"

    Example of CutSet with inter-shard shuffling enabled::

        >>> cuts = lhotse.CutSet(LazyNeMoTarredIterator(
        ...     manifest_path="nemo_manifests/train.json",
        ...     tar_paths=["nemo_manifests/audio_0.tar", ...],
        ...     shuffle_shards=True,
        ... ))
    """

    def __init__(
        self, manifest_path: str | Path, tar_paths: str | Sequence[str | Path], shuffle_shards: bool = False,
    ) -> None:
        from cytoolz import groupby

        def strip_pipe(p):
            if isinstance(p, str) and p.startswith("pipe:"):
                return Path(p[5:])
            return p

        self.source = lhotse.lazy.LazyJsonlIterator(manifest_path)
        self.shard_id_to_manifest: Dict[int, str] = groupby("shard_id", self.source)
        tar_paths = expand_sharded_filepaths(tar_paths, shard_strategy="replicate", world_size=1, global_rank=0)
        self.shard_id_to_tar_path: Dict[int, Path] = {int(strip_pipe(p).stem.split("_")[1]): p for p in tar_paths}
        self.shuffle_shards = shuffle_shards
        self._validate()

    def _validate(self) -> None:
        shard_ids_tars = set(self.shard_id_to_tar_path)
        shard_ids_manifest = set(self.shard_id_to_manifest)
        assert shard_ids_tars == shard_ids_manifest, (
            f"Mismatch between shard IDs discovered from tar files ({len(shard_ids_tars)=}) and "
            f"JSON manifest ({len(shard_ids_manifest)=}): {shard_ids_tars - shard_ids_manifest=}"
        )

    @property
    def shard_ids(self) -> List[int]:
        return sorted(self.shard_id_to_manifest.keys())

    @property
    def path(self) -> str | Path:
        return self.source.path

    def __iter__(self):
        shard_ids = self.shard_ids

        if self.shuffle_shards:
            # Use TRNG for 100% randomness
            random.Random(secrets.randbelow(2 ** 32)).shuffle(shard_ids)

        for sid in shard_ids:
            shard_manifest = self.shard_id_to_manifest[sid]
            tar_path = self.shard_id_to_tar_path[sid]
            with tarfile.open(fileobj=lhotse.serialization.open_best(tar_path, mode="rb"), mode="r|*") as tar:
                for data, tar_info in zip(shard_manifest, tar):
                    raw_audio = tar.extractfile(tar_info).read()
                    recording = lhotse.Recording.from_bytes(raw_audio, recording_id=tar_info.path)
                    cut = recording.to_cut()
                    cut.supervisions.append(
                        lhotse.SupervisionSegment(
                            id=cut.id,
                            recording_id=cut.recording_id,
                            start=0,
                            duration=cut.duration,
                            text=data["text"],
                        )
                    )
                    yield cut

    def __len__(self) -> int:
        return len(self.source)

    def __add__(self, other) -> "lhotse.lazy.LazyIteratorChain":
        return lhotse.lazy.LazyIteratorChain(self, other)

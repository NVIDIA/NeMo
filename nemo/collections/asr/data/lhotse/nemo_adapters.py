import random
import secrets
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List

try:
    from lhotse.lazy import ImitatesDict
except ImportError:
    ImitatesDict = object

import soundfile

from nemo.collections.asr.data.audio_to_text import expand_sharded_filepaths


class LazyNeMoIterator(ImitatesDict):
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
        from lhotse.lazy import LazyJsonlIterator

        self.source = LazyJsonlIterator(path)
        self.sampling_rate = sampling_rate

    @property
    def path(self) -> str | Path:
        return self.source.path

    def __iter__(self):
        from lhotse import SupervisionSegment
        from lhotse.audio import AudioSource, Recording
        from lhotse.utils import compute_num_samples

        for data in self.source:
            recording = Recording(
                id=Path(data["audio_filepath"]).name,
                sources=[AudioSource(type="file", channels=[0], source=data["audio_filepath"],)],
                sampling_rate=self.sampling_rate,
                duration=data["duration"],
                num_samples=compute_num_samples(data["duration"], self.sampling_rate),
            )
            cut = recording.to_cut()
            cut.supervisions.append(
                SupervisionSegment(
                    id=cut.id, recording_id=cut.recording_id, start=0, duration=cut.duration, text=data["text"],
                )
            )
            yield cut

    def __len__(self) -> int:
        return len(self.source)

    def __add__(self, other):
        from lhotse.lazy import LazyIteratorChain

        return LazyIteratorChain(self, other)


class LazyNeMoTarredIterator(ImitatesDict):
    """
    ``LazyNeMoTarredIterator`` reads a NeMo tarred JSON manifest and converts it on the fly to an ``Iterable[Cut]``.
    It's used to create a ``lhotse.CutSet``.

    Currently, it requires (and exclusively supports) the following keys in NeMo manifests:
    - "audio_filepath"
    - "duration"
    - "text"
    - "shard_id"

    Args ``manifest_path`` and ``tar_paths`` can be either a path/string to a single file, or a string in NeMo format
    that indicates multiple paths (e.g. "[[data/bucket0/tarred_audio_paths.json],[data/bucket1/...]]").

    Example of CutSet with inter-shard shuffling enabled::

        >>> cuts = lhotse.CutSet(LazyNeMoTarredIterator(
        ...     manifest_path="nemo_manifests/train.json",
        ...     tar_paths=["nemo_manifests/audio_0.tar", ...],
        ...     shuffle_shards=True,
        ... ))
    """

    def __init__(self, manifest_path: str | Path, tar_paths: str | list, shuffle_shards: bool = False,) -> None:
        from cytoolz import groupby
        from lhotse.lazy import LazyJsonlIterator

        def strip_pipe(p):
            if isinstance(p, str):
                if p.startswith("pipe:"):
                    p = p[5:]
                return Path(p)
            return p

        self.source = LazyJsonlIterator(manifest_path)
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
        from lhotse.serialization import open_best
        from lhotse.audio import AudioSource, Recording
        from lhotse import SupervisionSegment

        shard_ids = self.shard_ids

        if self.shuffle_shards:
            # Use TRNG for 100% randomness
            random.Random(secrets.randbelow(2 ** 32)).shuffle(shard_ids)

        for sid in shard_ids:
            shard_manifest = self.shard_id_to_manifest[sid]
            tar_path = self.shard_id_to_tar_path[sid]
            with tarfile.open(fileobj=open_best(tar_path, mode="rb"), mode="r|*") as tar:
                for data, tar_info in zip(shard_manifest, tar):
                    raw_audio = tar.extractfile(tar_info).read()
                    # Note: Lhotse has a Recording.from_bytes() utility that we won't use here because
                    #       the profiling indicated significant overhead in torchaudio ffmpeg integration
                    #       that parses full audio instead of just reading the header for WAV files.
                    # recording = lhotse.Recording.from_bytes(raw_audio, recording_id=tar_info.path)
                    meta = soundfile.info(BytesIO(raw_audio))
                    recording = Recording(
                        id=tar_info.path,
                        sources=[AudioSource(type="memory", channels=list(range(meta.channels)), source=raw_audio)],
                        sampling_rate=int(meta.samplerate),
                        num_samples=meta.frames,
                        duration=meta.duration,
                    )
                    cut = recording.to_cut()
                    cut.supervisions.append(
                        SupervisionSegment(
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

    def __add__(self, other):
        from lhotse.lazy import LazyIteratorChain

        return LazyIteratorChain(self, other)

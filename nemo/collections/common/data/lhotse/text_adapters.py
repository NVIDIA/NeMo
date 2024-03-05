from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Iterator, TextIO

import numpy as np


@dataclass
class TextExample:

    # original string
    text: str

    # tokenized string, i.e. array of ints (agnostic to BPE, bytes, whatever else)
    tokens: np.ndarray | None = None

    custom: dict | None = None

    @property
    def duration(self) -> int:
        """
        Called "duration" for compatibility with lhotse in this prototype, but it's really "num_tokens";
        eventually might be called sth else, perhaps with Lhotse sampler API extended
        to support which attribute should be read for a given datatype.
        """
        # semi-HACK: lhotse workaround
        if self.tokens is not None:
            return len(self.tokens)
        return len(self.text)

    def resample(self, *args, **kwargs):
        # HACK: lhotse workaround
        return self


class LhotseTextZipAdapter:
    """
    ``LhotseTextZipAdapter`` is used to read a tuple of N text files
    (e.g., a pair of files with translations in different languages)
    and wrap them in a ``TextExample`` object to enable dataloading
    with Lhotse together with training examples in audio modality.
    """

    def __init__(self, paths: list[str] | str):
        self.paths = paths

    def __iter__(self) -> Iterator[TextExample]:
        with open_many(self.paths) as fs:
            for entries in zip(*fs):
                yield TextExample(entries)


@contextmanager
def open_many(paths: list[str]) -> Generator[list[TextIO], None, None]:
    fs = []
    try:
        for p in paths:
            fs.append(open(p))
        yield fs
    finally:
        for f in fs:
            f.close()

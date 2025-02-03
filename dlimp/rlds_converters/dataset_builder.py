"""Inspired by https://github.com/kpertsch/bridge_rlds_builder/blob/f0d16c5a8384c1476aa1c274a9aef3a5f76cbada/bridge_dataset/conversion_utils.py"""

import abc
import itertools
import multiprocessing as mp
from typing import Any, Callable, Dict, Iterable, Tuple, Union

import tensorflow_datasets as tfds
from absl import logging
from tensorflow_datasets.core import (
    dataset_builder,
    download,
    example_serializer,
    file_adapters,
    naming,
)
from tensorflow_datasets.core import split_builder as split_builder_lib
from tensorflow_datasets.core import splits as splits_lib
from tensorflow_datasets.core import writer as writer_lib
from tqdm import tqdm

Key = Union[str, int]
Example = Dict[str, Any]
ExampleInput = Any


class MultiThreadedSplitBuilder(split_builder_lib.SplitBuilder):
    """Multithreaded version of tfds.core.SplitBuilder. Removes Apache Beam support, only supporting Python generators."""

    def __init__(
        self,
        process_fn: Callable[[ExampleInput], Example],
        num_workers: int,
        chunksize: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._process_fn = process_fn
        self.num_workers = num_workers
        self.chunksize = chunksize

    def submit_split_generation(
        self,
        split_name: splits_lib.Split,
        generator: Iterable[Tuple[Key, ExampleInput]],
        filename_template: naming.ShardedFileTemplate,
        disable_shuffling: bool = False,
    ) -> splits_lib.SplitInfo:
        if self._max_examples_per_split is not None:
            logging.warning(
                "Splits capped at %s examples max.", self._max_examples_per_split
            )
            generator = itertools.islice(generator, self._max_examples_per_split)
            total_num_examples = self._max_examples_per_split
        else:
            # If dataset info has been pre-downloaded from the internet,
            # we can use the pre-computed number of example for the progression bar.
            split_info = self._split_dict.get(split_name)
            if split_info and split_info.num_examples:
                total_num_examples = split_info.num_examples
            else:
                total_num_examples = None

        serialized_info = self._features.get_serialized_info()
        writer = writer_lib.Writer(
            serializer=example_serializer.ExampleSerializer(serialized_info),
            filename_template=filename_template,
            hash_salt=split_name,
            disable_shuffling=disable_shuffling,
            file_format=self._file_format,
            shard_config=self._shard_config,
        )
        pbar = tqdm(
            total=total_num_examples,
            desc=f"Generating {split_name} examples...",
            unit=" examples",
            dynamic_ncols=True,
            miniters=1,
        )
        with mp.Pool(
            self.num_workers,
            initializer=MultiThreadedSplitBuilder._worker_init,
            initargs=(self._process_fn, self._features),
        ) as pool:
            logging.info(
                "Using %d workers with chunksize %d.", self.num_workers, self.chunksize
            )
            while True:
                curr = pbar.n
                iterator = itertools.islice(generator, self.chunksize)
                results = pool.map(MultiThreadedSplitBuilder._worker_fn, iterator)
                for key, example in results:
                    writer._shuffler.add(key, example)
                    writer._num_examples += 1
                    pbar.update(1)
                if pbar.n == curr:
                    break
        shard_lengths, total_size = writer.finalize()

        return splits_lib.SplitInfo(
            name=split_name,
            shard_lengths=shard_lengths,
            num_bytes=total_size,
            filename_template=filename_template,
        )

    @staticmethod
    def _worker_init(
        process_fn: Callable[[ExampleInput], Example],
        features: tfds.features.FeaturesDict,
    ):
        global __process_fn
        global __features
        global __serializer
        __process_fn = process_fn
        __features = features
        __serializer = example_serializer.ExampleSerializer(
            features.get_serialized_info()
        )

    @staticmethod
    def _worker_fn(example_input):
        global __process_fn
        global __features
        global __serializer
        key, example = __process_fn(example_input)
        return key, __serializer.serialize_example(__features.encode_example(example))


class MultiThreadedDatasetBuilder(tfds.core.GeneratorBasedBuilder):
    """Multithreaded version of tfds.core.GeneratorBasedBuilder."""

    # Defaults can be overridden by subclasses.
    NUM_WORKERS = 16  # number of parallel workers
    CHUNKSIZE = 1000  # number of examples to process in memory before writing to disk

    @classmethod
    @abc.abstractmethod
    def _process_example(cls, example_input: ExampleInput) -> Example:
        """Process a single example.

        This is the function that will be parallelized, so it should contain any heavy computation and I/O. It
        should return a feature dictionary compatible with `self.info.features` (see the FeatureConnector
        documenation) that is ready to be encoded and serialized.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _split_generators(
        self,
        dl_manager: download.DownloadManager,
    ) -> Dict[splits_lib.Split, Iterable[Tuple[Key, ExampleInput]]]:
        """Same as GeneratorBasedBuilder._split_generators, but returns generators of tuples (key,
        example_input) rather than (key, example). `example_input` will be passed to
        `_process_example` for further processing.
        """
        raise NotImplementedError()

    def _generate_examples(self, *args, **kwargs):
        """This is not actually called from TFDS code. I believe they left it in for legacy reasons. However,
        it must be overridden for TFDS to recognize the class as a valid dataset builder.
        """
        raise RuntimeError()

    def _download_and_prepare(
        self,
        dl_manager: download.DownloadManager,
        download_config: download.DownloadConfig,
    ) -> None:
        """Same as superclass `_download_and_prepare`, but removes Apache Beam stuff and uses
        MultiThreadedSplitBuilder instead of SplitBuilder.
        """
        split_builder = MultiThreadedSplitBuilder(
            process_fn=type(self)._process_example,
            num_workers=self.NUM_WORKERS,
            chunksize=self.CHUNKSIZE,
            split_dict=self.info.splits,
            features=self.info.features,
            dataset_size=self.info.dataset_size,
            max_examples_per_split=download_config.max_examples_per_split,
            beam_options=download_config.beam_options,
            beam_runner=download_config.beam_runner,
            file_format=self.info.file_format,
            shard_config=download_config.get_shard_config(),
        )

        split_generators = self._split_generators(dl_manager)
        dataset_builder._check_split_names(split_generators.keys())

        # Writer fail if the number of example yield is `0`, so we return here.
        if download_config.max_examples_per_split == 0:
            return

        # Start generating data for all splits
        path_suffix = file_adapters.ADAPTER_FOR_FORMAT[
            self.info.file_format
        ].FILE_SUFFIX

        split_infos = []
        for split_name, generator in split_generators.items():
            filename_template = naming.ShardedFileTemplate(
                split=split_name,
                dataset_name=self.name,
                data_dir=self.data_path,
                filetype_suffix=path_suffix,
            )
            split_info = split_builder.submit_split_generation(
                split_name=split_name,
                generator=generator,
                filename_template=filename_template,
                disable_shuffling=self.info.disable_shuffling,
            )
            split_infos.append(split_info)

        # Update the info object with the splits.
        split_dict = splits_lib.SplitDict(split_infos)
        self.info.set_splits(split_dict)

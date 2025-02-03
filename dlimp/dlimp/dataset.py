import inspect
import string
from functools import partial
from typing import Any, Callable, Dict, Sequence, Union

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core.dataset_builder import DatasetBuilder

from dlimp.utils import parallel_vmap


def _wrap(f, is_flattened):
    """Wraps a method to return a DLataset instead of a tf.data.Dataset."""

    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)
        if not isinstance(result, DLataset) and isinstance(result, tf.data.Dataset):
            # make the result a subclass of DLataset and the original class
            result.__class__ = type(
                "DLataset", (DLataset, type(result)), DLataset.__dict__.copy()
            )
            # propagate the is_flattened flag
            if is_flattened is None:
                result.is_flattened = f.__self__.is_flattened
            else:
                result.is_flattened = is_flattened
        return result

    return wrapper


class DLataset(tf.data.Dataset):
    """A DLimp Dataset. This is a thin wrapper around tf.data.Dataset that adds some utilities for working
    with datasets of trajectories.

    A DLataset starts out as dataset of trajectories, where each dataset element is a single trajectory. A
    dataset element is always a (possibly nested) dictionary from strings to tensors; however, a trajectory
    has the additional property that each tensor has the same leading dimension, which is the trajectory
    length. Each element of the trajectory is known as a frame.

    A DLataset is just a tf.data.Dataset, so you can always use standard methods like `.map` and `.filter`.
    However, a DLataset is also aware of the difference between trajectories and frames, so it provides some
    additional methods. To perform a transformation at the trajectory level (e.g., restructuring, relabeling,
    truncating), use `.traj_map`. To perform a transformation at the frame level (e.g., image decoding,
    resizing, augmentations) use `.frame_map`.

    Once there are no more trajectory-level transformation to perform, you can convert to DLataset to a
    dataset of frames using `.flatten`. You can still use `.frame_map` after flattening, but using `.traj_map`
    will raise an error.
    """

    def __getattribute__(self, name):
        # monkey-patches tf.data.Dataset methods to return DLatasets
        attr = super().__getattribute__(name)
        if inspect.ismethod(attr):
            return _wrap(attr, None)
        return attr

    def _apply_options(self):
        """Applies some default options for performance."""
        options = tf.data.Options()
        options.autotune.enabled = True
        options.deterministic = False
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.map_fusion = True
        options.experimental_optimization.map_and_filter_fusion = True
        options.experimental_optimization.inject_prefetch = False
        options.experimental_warm_start = True
        return self.with_options(options)

    def with_ram_budget(self, gb: int) -> "DLataset":
        """Sets the RAM budget for the dataset. The default is half of the available memory.

        Args:
            gb (int): The RAM budget in GB.
        """
        options = tf.data.Options()
        options.autotune.ram_budget = gb * 1024 * 1024 * 1024  # GB --> Bytes
        return self.with_options(options)

    @staticmethod
    def from_tfrecords(
        dir_or_paths: Union[str, Sequence[str]],
        shuffle: bool = True,
        num_parallel_reads: int = tf.data.AUTOTUNE,
    ) -> "DLataset":
        """Creates a DLataset from tfrecord files. The type spec of the dataset is inferred from the first file. The
        only constraint is that each example must be a trajectory where each entry is either a scalar, a tensor of shape
        (1, ...), or a tensor of shape (T, ...), where T is the length of the trajectory.

        Args:
            dir_or_paths (Union[str, Sequence[str]]): Either a directory containing .tfrecord files, or a list of paths
                to tfrecord files.
            shuffle (bool, optional): Whether to shuffle the tfrecord files. Defaults to True.
            num_parallel_reads (int, optional): The number of tfrecord files to read in parallel. Defaults to AUTOTUNE. This
                can use an excessive amount of memory if reading from cloud storage; decrease if necessary.
        """
        if isinstance(dir_or_paths, str):
            paths = tf.io.gfile.glob(tf.io.gfile.join(dir_or_paths, "*.tfrecord"))
        else:
            paths = dir_or_paths

        if len(paths) == 0:
            raise ValueError(f"No tfrecord files found in {dir_or_paths}")

        if shuffle:
            paths = tf.random.shuffle(paths)

        # extract the type spec from the first file
        type_spec = _get_type_spec(paths[0])

        # read the tfrecords (yields raw serialized examples)
        dataset = _wrap(tf.data.TFRecordDataset, False)(
            paths,
            num_parallel_reads=num_parallel_reads,
        )._apply_options()

        # decode the examples (yields trajectories)
        dataset = dataset.traj_map(partial(_decode_example, type_spec=type_spec))

        # broadcast traj metadata, as well as add some extra metadata (_len, _traj_index, _frame_index)
        dataset = dataset.enumerate().traj_map(_broadcast_metadata)

        return dataset

    @staticmethod
    def from_rlds(
        builder: DatasetBuilder,
        split: str = "train",
        shuffle: bool = True,
        num_parallel_reads: int = tf.data.AUTOTUNE,
    ) -> "DLataset":
        """Creates a DLataset from the RLDS format (which is a special case of the TFDS format).

        Args:
            builder (DatasetBuilder): The TFDS dataset builder to load the dataset from.
            data_dir (str): The directory to load the dataset from.
            split (str, optional): The split to load, specified in TFDS format. Defaults to "train".
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
            num_parallel_reads (int, optional): The number of tfrecord files to read in parallel. Defaults to AUTOTUNE. This
                can use an excessive amount of memory if reading from cloud storage; decrease if necessary.
        """
        dataset = _wrap(builder.as_dataset, False)(
            split=split,
            shuffle_files=shuffle,
            decoders={"steps": tfds.decode.SkipDecoding()},
            read_config=tfds.ReadConfig(
                skip_prefetch=True,
                num_parallel_calls_for_interleave_files=num_parallel_reads,
                interleave_cycle_length=num_parallel_reads,
            ),
        )._apply_options()

        dataset = dataset.enumerate().traj_map(_broadcast_metadata_rlds)

        return dataset

    def map(
        self,
        fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        num_parallel_calls=tf.data.AUTOTUNE,
        **kwargs,
    ) -> "DLataset":
        return super().map(fn, num_parallel_calls=num_parallel_calls, **kwargs)

    def traj_map(
        self,
        fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        num_parallel_calls=tf.data.AUTOTUNE,
        **kwargs,
    ) -> "DLataset":
        """Maps a function over the trajectories of the dataset. The function should take a single trajectory
        as input and return a single trajectory as output.
        """
        if self.is_flattened:
            raise ValueError("Cannot call traj_map on a flattened dataset.")
        return super().map(fn, num_parallel_calls=num_parallel_calls, **kwargs)

    def frame_map(
        self,
        fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        num_parallel_calls=tf.data.AUTOTUNE,
        **kwargs,
    ) -> "DLataset":
        """Maps a function over the frames of the dataset. The function should take a single frame as input
        and return a single frame as output.
        """
        if self.is_flattened:
            return super().map(fn, num_parallel_calls=num_parallel_calls, **kwargs)
        else:
            return super().map(
                parallel_vmap(fn, num_parallel_calls=num_parallel_calls),
                num_parallel_calls=num_parallel_calls,
                **kwargs,
            )

    def flatten(self, *, num_parallel_calls=tf.data.AUTOTUNE) -> "DLataset":
        """Flattens the dataset of trajectories into a dataset of frames."""
        if self.is_flattened:
            raise ValueError("Dataset is already flattened.")
        dataset = self.interleave(
            lambda traj: tf.data.Dataset.from_tensor_slices(traj),
            cycle_length=num_parallel_calls,
            num_parallel_calls=num_parallel_calls,
        )
        dataset.is_flattened = True
        return dataset

    def iterator(self, *, prefetch=tf.data.AUTOTUNE):
        if prefetch == 0:
            return self.as_numpy_iterator()
        return self.prefetch(prefetch).as_numpy_iterator()

    @staticmethod
    def choose_from_datasets(datasets, choice_dataset, stop_on_empty_dataset=True):
        if not isinstance(datasets[0], DLataset):
            raise ValueError("Please pass DLatasets to choose_from_datasets.")
        return _wrap(tf.data.Dataset.choose_from_datasets, datasets[0].is_flattened)(
            datasets, choice_dataset, stop_on_empty_dataset=stop_on_empty_dataset
        )

    @staticmethod
    def sample_from_datasets(
        datasets,
        weights=None,
        seed=None,
        stop_on_empty_dataset=False,
        rerandomize_each_iteration=None,
    ):
        if not isinstance(datasets[0], DLataset):
            raise ValueError("Please pass DLatasets to sample_from_datasets.")
        return _wrap(tf.data.Dataset.sample_from_datasets, datasets[0].is_flattened)(
            datasets,
            weights=weights,
            seed=seed,
            stop_on_empty_dataset=stop_on_empty_dataset,
            rerandomize_each_iteration=rerandomize_each_iteration,
        )

    @staticmethod
    def zip(*args, datasets=None, name=None):
        if datasets is not None:
            raise ValueError("Please do not pass `datasets=` to zip.")
        if not isinstance(args[0], DLataset):
            raise ValueError("Please pass DLatasets to zip.")
        return _wrap(tf.data.Dataset.zip, args[0].is_flattened)(*args, name=name)


def _decode_example(
    example_proto: tf.Tensor, type_spec: Dict[str, tf.TensorSpec]
) -> Dict[str, tf.Tensor]:
    features = {key: tf.io.FixedLenFeature([], tf.string) for key in type_spec.keys()}
    parsed_features = tf.io.parse_single_example(example_proto, features)
    parsed_tensors = {
        key: tf.io.parse_tensor(parsed_features[key], spec.dtype)
        if spec is not None
        else parsed_features[key]
        for key, spec in type_spec.items()
    }

    for key in parsed_tensors:
        if type_spec[key] is not None:
            parsed_tensors[key] = tf.ensure_shape(
                parsed_tensors[key], type_spec[key].shape
            )

    return parsed_tensors


def _get_type_spec(path: str) -> Dict[str, tf.TensorSpec]:
    """Get a type spec from a tfrecord file.

    Args:
        path (str): Path to a single tfrecord file.

    Returns:
        dict: A dictionary mapping feature names to tf.TensorSpecs.
    """
    data = next(iter(tf.data.TFRecordDataset(path))).numpy()
    example = tf.train.Example()
    example.ParseFromString(data)

    printable_chars = set(bytes(string.printable, "utf-8"))

    out = {}
    for key, value in example.features.feature.items():
        data = value.bytes_list.value[0]
        # stupid hack to deal with strings that are not encoded as tensors
        if all(char in printable_chars for char in data):
            out[key] = None
            continue
        tensor_proto = tf.make_tensor_proto([])
        tensor_proto.ParseFromString(data)
        dtype = tf.dtypes.as_dtype(tensor_proto.dtype)
        shape = [d.size for d in tensor_proto.tensor_shape.dim]
        if shape:
            shape[0] = None  # first dimension is trajectory length, which is variable
        out[key] = tf.TensorSpec(shape=shape, dtype=dtype)

    return out


def _broadcast_metadata(
    i: tf.Tensor, traj: Dict[str, tf.Tensor]
) -> Dict[str, tf.Tensor]:
    """
    Each element of a dlimp dataset is a trajectory. This means each entry must either have a leading dimension equal to
    the length of the trajectory, have a leading dimension of 1, or be a scalar. Entries with a leading dimension of 1
    and scalars are assumed to be trajectory-level metadata. This function broadcasts these entries to the length of the
    trajectory, as well as adds the extra metadata fields `_len`, `_traj_index`, and `_frame_index`.
    """
    # get the length of each dict entry
    traj_lens = {
        k: tf.shape(v)[0] if len(v.shape) > 0 else None for k, v in traj.items()
    }

    # take the maximum length as the canonical length (elements should either be the same length or length 1)
    traj_len = tf.reduce_max([l for l in traj_lens.values() if l is not None])

    for k in traj:
        # broadcast scalars to the length of the trajectory
        if traj_lens[k] is None:
            traj[k] = tf.repeat(traj[k], traj_len)
            traj_lens[k] = traj_len

        # broadcast length-1 elements to the length of the trajectory
        if traj_lens[k] == 1:
            traj[k] = tf.repeat(traj[k], traj_len, axis=0)
            traj_lens[k] = traj_len

    asserts = [
        # make sure all the lengths are the same
        tf.assert_equal(
            tf.size(tf.unique(tf.stack(list(traj_lens.values()))).y),
            1,
            message="All elements must have the same length.",
        ),
    ]

    assert "_len" not in traj
    assert "_traj_index" not in traj
    assert "_frame_index" not in traj
    traj["_len"] = tf.repeat(traj_len, traj_len)
    traj["_traj_index"] = tf.repeat(i, traj_len)
    traj["_frame_index"] = tf.range(traj_len)

    with tf.control_dependencies(asserts):
        return traj


def _broadcast_metadata_rlds(i: tf.Tensor, traj: Dict[str, Any]) -> Dict[str, Any]:
    """
    In the RLDS format, each trajectory has some top-level metadata that is explicitly separated out, and a "steps"
    entry. This function moves the "steps" entry to the top level, broadcasting any metadata to the length of the
    trajectory. This function also adds the extra metadata fields `_len`, `_traj_index`, and `_frame_index`.
    """
    steps = traj.pop("steps")

    traj_len = tf.shape(tf.nest.flatten(steps)[0])[0]

    # broadcast metadata to the length of the trajectory
    metadata = tf.nest.map_structure(lambda x: tf.repeat(x, traj_len), traj)

    # put steps back in
    assert "traj_metadata" not in steps
    traj = {**steps, "traj_metadata": metadata}

    assert "_len" not in traj
    assert "_traj_index" not in traj
    assert "_frame_index" not in traj
    traj["_len"] = tf.repeat(traj_len, traj_len)
    traj["_traj_index"] = tf.repeat(i, traj_len)
    traj["_frame_index"] = tf.range(traj_len)

    return traj

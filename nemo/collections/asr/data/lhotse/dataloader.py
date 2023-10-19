import torch.utils
from lhotse import CutSet
from lhotse.dataset import DynamicBucketingSampler, DynamicCutSampler, IterableDatasetWrapper, make_worker_init_fn

from nemo.collections.asr.data.lhotse.cutset import read_cutset_from_config


def get_lhotse_dataloader_from_config(config, global_rank: int, world_size: int, dataset: torch.utils.data.Dataset):
    """
    Setup a Lhotse training dataloder.

    Expects a typical NeMo dataset configuration format, with additional fields: "use_lhotse=True" and "lhotse: <dict>".
    Some fields in the original NeMo configuration are ignored (e.g. ``batch_size``).
    To learn about lhotse specific parameters, search this code for ``config.lhotse``.

    The ``dataset`` parameter should be an instance of a Lhotse-compatible PyTorch Dataset class.
    It only needs to define the following method ``__getitem__(self, cuts: CutSet) -> Dict[str, torch.Tensor]``.
    This dataset is not expected to refer to any actual data; it may be interpreted as a function
    mapping a Lhotse CutSet into a mini-batch of tensors.

    For example, see: :class:`nemo.collections.asr.data.audio_to_text_lhotse.LhotseSpeechToTextBpeDataset`,
    which is constructed from just a tokenizer and essentially loads and collates audio and tokenizes the transcript.
    """

    # 1. Load a manifest as a Lhotse CutSet.
    cuts, is_tarred = read_cutset_from_config(config)

    # Resample as a safeguard; it's a no-op when SR is already OK
    cuts = cuts.resample(config["sample_rate"])

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
            duration_bins=config.lhotse.get("duration_bins", None),
            num_cuts_for_bins_estimate=config.lhotse.get("num_cuts_for_bins_estimate", 10000),
            buffer_size=config.lhotse.get("buffer_size", 10000),
            shuffle_buffer_size=config.lhotse.get("shuffle_buffer_size", 10000),
            quadratic_duration=config.lhotse.get("quadratic_duration", None),
            seed=config.lhotse.get("seed", 0),
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
            seed=config.lhotse.get("seed", 0),
            rank=0 if is_tarred else global_rank,
            world_size=1 if is_tarred else world_size,
        )

    # 4. Creating dataloader.
    num_workers = config.get('num_workers', 0)
    if is_tarred:
        # Wrapper here is necessary when using NeMo tarred data or Lhotse Shar data,
        # because then I/O happens upon sampler iteration. Normally, the sampler resides
        # in the training loop process, but when we use iterable dataset, we can move it to
        # the dataloading worker process.
        # We use lhotse's own worker_init_fn which leverages information such as rank, world_size,
        # worker_id, etc. to set a different random seed for each (node, worker) combination.
        # This together with infinite datasets removes the need to split data across nodes/workers.
        dloader_kwargs = dict(
            dataset=IterableDatasetWrapper(dataset=dataset, sampler=sampler),
            worker_init_fn=make_worker_init_fn(rank=global_rank, world_size=world_size),
            persistent_workers=num_workers > 0,  # helps Lhotse Shar maintain shuffling state
        )
    else:
        # For non-tarred data, the sampler resides in the training loop process and
        # reads only light-weight JSON objects; it samples mini-batches and passes
        # the meta-data to Dataset, which performs the actual I/O inside its __getitem__ method.
        dloader_kwargs = dict(dataset=dataset, sampler=sampler)
    dloader = torch.utils.data.DataLoader(
        **dloader_kwargs, batch_size=None, num_workers=num_workers, pin_memory=config.get('pin_memory', False),
    )

    return dloader

#!/usr/bin/env python
import importlib
import math
import sys

import click
import pytorch_lightning as pl
import torch
from lhotse import compute_num_samples
from omegaconf import OmegaConf

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.models.aed_multitask_models import EncDecMultiTaskModel
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.utils import logging


class ProfilingBatchGenerator:
    """
    ProfilingBatchGenerator is used to generate artificial mini-batches for model training
    and tracking the progress of batch size optimization.

    The high-level usage API is the following::

        >>> gen = ProfilingBatchGenerator(schema)
        ... finished = False
        ... while not finished:
        ...     batch = gen(input_seq_len, output_seq_len)
        ...     try:
        ...         training_step(model, batch)
        ...         oom = False
        ...     except torch.cuda.OutOfMemoryError:
        ...         oom = True
        ...     finished = gen.advance(oom)
        ... solution = gen.max_batch_size  # The solution of the search problem.
        ... gen.reset()  # Can re-use for other sequence lengths now.


    In order to generate mini-batches compatible with a given model, the generator:

    * accepts a ``schema`` argument in its constructor, and

    * accepts input/output sequence lengths in each call to generate a mini-batch.

    ``schema`` has the following structure::

        >>> [{"type": NeuralType(...) | str, "seq_length": "input|output", "vocab_size": int}, {...}, ...]

    Each item in ``schema`` specifies a NeMo NeuralType which needs to have a defined ``elements_type``.
    The supported types are ``AudioSignal``, ``LengthsType`` and ``LabelsType``.
    If "type" is not a NeuralType, we interpret that as a placeholder tensor that's not relevant but expect by the model.

    In addition, ``"seq_length"`` key is used to determine whether we should apply input or output sequence length
    to a given tensor, and "vocab_size" is required for ``LabelsType`` so that we can generate proper label values.

    The search terminates once the difference between max working batch size and min OOM batch size
    divided by the latter is smaller than ``rel_gap_thresh`` that difference amounts to a single element.
    For example, a max working batch size is 96 and min OOM batch size is 100 indicates a gap of 0.04,
    which would terminate the search with threshold of 0.05.
    """

    def __init__(
        self,
        schema: list[dict] = None,
        start_batch_size: int = 1024,
        rel_gap_thresh: float = 0.05,
        device: str = "cuda",
    ):
        self.schema = schema
        self.start_batch_size = start_batch_size
        self.rel_gap_thresh = rel_gap_thresh
        self.device = device
        self.reset()

    def __call__(self, input_seq_len: int, output_seq_len: int):
        B = self._current
        batch = []
        for item in self.schema:
            nt = item["type"]
            if not isinstance(nt, NeuralType):  # placeholder
                tnsr = torch.tensor([])
            elif isinstance(nt.elements_type, AudioSignal):
                tnsr = torch.randn(B, input_seq_len, dtype=torch.float32, device=self.device)
            elif isinstance(nt.elements_type, LengthsType):
                seq_len = input_seq_len if item["seq_length"] == "input" else output_seq_len
                tnsr = torch.ones(B, dtype=torch.long, device=self.device) * seq_len
            elif isinstance(nt.elements_type, LabelsType):
                tnsr = torch.randint(0, item["vocab_size"], size=(B, output_seq_len), device=self.device)
            else:
                raise RuntimeError("Unexpected item in oomptimizer schema: {item}")
            batch.append(tnsr)
        return tuple(batch)

    @property
    def max_batch_size(self) -> int | None:
        """
        Return the solution of the batch size search problem.
        It will keep returning None until the search is done.
        """
        if (
            self._max_ok is not None
            and self._min_err is not None
            and (self.current_rel_gap <= self.rel_gap_thresh or self._min_err - self._max_ok <= 1)
        ):
            return self._max_ok
        return None

    @property
    def current_rel_gap(self) -> float | None:
        """
        Return the current gap between the largest batch that works and the smallest batch that triggers OOM.
        The gap is defined as the batch size difference divided by the larger element.
        E.g., if the best found batch size is 95 and the smallest that triggers OOM is 100, the gap is 0.05.
        """
        if self._min_err is None or self._max_ok is None:
            return None
        return (self._min_err - self._max_ok) / self._min_err

    def reset(self):
        """Reset the generator to prepare it for a new search."""
        self._current = self.start_batch_size
        self._max_ok = None  # max batch size that works
        self._min_err = None  # min batch size that doesn't work

    def advance(self, oom: bool) -> bool:
        """
        Adjusts the current batch size based on the outcome.
        Returns a bool indicating whether the calibration is complete.
        """
        if self.max_batch_size is not None:
            return True

        if oom:
            # Training step failed with OOM.
            # Update the minimum known batch size that causes an error.
            self._min_err = min(float("inf") if self._min_err is None else self._min_err, self._current)
            # Training step failed on OOM
            if self._max_ok is None:
                # We haven't found a batch size that works yet, keep going 2x down.
                self._current = round(self._current / 2)
            else:
                # Try the middle-point between the known extremes.
                self._current = round((self._max_ok + self._min_err) / 2)
        else:
            # Training step successful.
            # Update the maximum known batch size that works.
            self._max_ok = max(-1 if self._max_ok is None else self._max_ok, self._current)
            if self._min_err is None:
                # We haven't found a batch size that causes an error yet, keep going 2x higher
                self._current *= 2
            else:
                # Try the middle-point between the known extremes.
                self._current = round((self._max_ok + self._min_err) / 2)

        return False


class FloatList(click.Option):
    """Support passing bucket duration bins as [1.1,2.5,5.6,...]"""

    name = "list[float]"

    def type_cast_value(self, ctx, value):
        if isinstance(value, list) and all(isinstance(v, float) for v in value):
            return value
        try:
            import ast

            return ast.literal_eval(value)
        except ValueError:
            raise click.BadParameter(value)


@click.command(context_settings={'show_default': True})
@click.option(
    "-n",
    "--pretrained-name",
    type=str,
    default=None,
    help="Name of a pretrained model to use, e.g. 'nvidia/canary-1b'.",
)
@click.option(
    "-m",
    "--module-name",
    type=str,
    default=None,
    help="Full path to NeMo's module corresponding to CONFIG_PATH, e.g. 'nemo.collections.asr.models.EncDecMultiTaskModel'.",
)
@click.option(
    "-c", "--config-path", type=str, default=None, help="Path to the training configuration file for MODULE_NAME."
)
@click.option("-o", "--optimizer-name", type=str, default="adamw", help="Name of optimizer to use.")
@click.option(
    "-b",
    "--buckets",
    cls=FloatList,
    default=[5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
    help="List of upper-bound bucket bins (i.e. first bucket is [0.0 - item0), second bucket is [item0 - item1), etc.)",
)
@click.option(
    "-t",
    "--threshold",
    type=float,
    default=0.05,
    help="Search stopping criterion in range [0, 1], lower is more precise. Interpret as the uncerainty gap, i.e. (min_oom_batch_size - max_ok_batch_size) / min_oom_batch_size.",
)
@click.option("-s", "--start-batch-size", type=int, default=1024, help="Initial batch size to start the search from.")
@click.option(
    "-l",
    "--labels-per-second",
    type=int,
    default=15,  # conservative estimate towards longer transcripts
    help="How many labels/second should we simulate. More means longer output text sequences, and can increase memory consumption.",
)
@click.option(
    "-f",
    "--memory-fraction",
    type=float,
    default=0.95,
    help="Limits the use of CUDA memory for this process to MEMORY_FRACTION of the total device memory. "
    "By default we force 5% memory to be unused to account for non-training-loop related CUDA memory usage"
    "in actual training scripts.",
)
@click.option(
    "-d",
    "--device",
    default="cuda:0",
    help="Device string to be passed to torch.device; due to MEMORY_FRACTION option, "
    "it must specify the device index (e.g. cuda:0). "
    "You can also leave the default index and select a specific GPU using env var CUDA_VISIBLE_DEVICES=<idx>",
)
def oomptimizer(
    pretrained_name: str | None,
    module_name: str | None,
    config_path: str | None,
    optimizer_name: str,
    buckets: list[float],
    threshold: float,
    start_batch_size: int,
    labels_per_second: int,
    memory_fraction: float,
    device: str,
):
    """
    OOMptimizer finds the optimal batch sizes for training your model with bucketing dataloading.

    \b
    There are two main usage patterns: for using a pretrained model or an untrained model configuration.
    The latter is more flexible but requires the user to provide two separate arguments. Examples:
    * python oomptimizer.py --pretrained-name nvidia/canary-1b
    * python oomptimizer.py --module-name nemo.collections.asr.models.EncDecMultiTaskModel \
        --config-path examples/asr/conf/speech_multitask/fast-conformer_aed.yaml

    Dynamic bucketing is notoriously difficult to tune as you risk running into CUDA OOM many steps into the training.
    In order to simplify finding the optimal settings, OOMptimizer scans each bucket to find the maximum possible
    batch size that doesn't trigger a CUDA OOM.

    \b
    The suggested workflow is the following:
    1) Run scripts/speech_recognition/estimate_duration_bins.py to get the duration distribution of your data.
    2) Run OOMptimizer to find the optimal batch sizes for your specific model, optimizer, and GPU.
    3) Use these optimal settings in your actual training script and enjoy optimal GPU utilization OOM-free.
    """
    if all(opt is None for opt in (pretrained_name, module_name, config_path)):
        click.secho(
            "You need to provide either PRETRAINED_NAME or the pair of MODULE_NAME and CONFIG_PATH.", fg="yellow"
        )
        sys.exit(1)
    logging.setLevel(logging.CRITICAL)
    torch.cuda.set_per_process_memory_fraction(memory_fraction, device)

    trainer = pl.Trainer(barebones=True)
    trainer.log_every_n_steps = 1000000
    if pretrained_name is not None:
        assert (
            config_path is None and module_name is None
        ), "--pretrained-name cannot be used together with --module-name/--config-path"
        print(f"Intializing ASR model from pretrained checkpoint {pretrained_name}.")
        model = ASRModel.from_pretrained(pretrained_name, trainer=trainer).to(device)
    else:
        assert config_path is not None, "--module-name requires --config-path to be specified as well."
        assert module_name is not None, "--config-path requires --module-name to be specified as well."
        cfg = OmegaConf.load(config_path)
        namespace, name = module_name.rsplit('.', maxsplit=1)
        model_cls = getattr(importlib.import_module(namespace), name)
        model = model_cls(cfg=cfg.model, trainer=trainer).to(device)

    # TODO(pzelasko): ideally move into model @property e.g. "oomptimizer_schema" :D
    schema = [
        {"type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
        {"type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
        {
            "type": NeuralType(("B", "T"), LabelsType()),
            "seq_length": "output",
            "vocab_size": model.tokenizer.vocab_size,
        },
        {"type": NeuralType(("B",), LengthsType()), "seq_length": "output"},
    ]
    if isinstance(model, EncDecMultiTaskModel):
        schema.extend(
            [{"type": "dummy"}, {"type": "dummy"}]
        )  # multi-task has 2 extra tensors not needed for batch size tuning

    print("Setting up the optimizers.")
    optimizer, _ = model.setup_optimization({"name": optimizer_name, "lr": 1e-7, "weight_decay": 0.0})

    def get_max_seq_lens(buckets):
        # TODO(pzelasko): support text data inputs.
        return [
            (
                compute_num_samples(d, sampling_rate=16000),  # num_samples
                math.ceil(labels_per_second * d),  # num_labels; might need to go data-driven for optimal tuning
            )
            for d in buckets
        ]

    print("Starting profiling.")
    max_seq_lens = get_max_seq_lens(buckets)
    gen = ProfilingBatchGenerator(schema=schema, start_batch_size=start_batch_size, rel_gap_thresh=threshold)
    profile = {}

    for bucket, (seq_len_in, seq_len_out) in zip(buckets, max_seq_lens):
        print(f"The current sequence lengths are: input={seq_len_in} output={seq_len_out}.")
        gen.reset()
        batch_idx = 0

        def step():
            batch = gen(seq_len_in, seq_len_out)
            oom = False
            try:
                print(f"Current gap: {gen.current_rel_gap}. Attempting shapes: {[b.shape for b in batch]}", end=" ")
                optimizer.zero_grad()
                model.training_step(batch, batch_idx)
                optimizer.step()
            except torch.cuda.OutOfMemoryError:
                print(f"- OOM!")
                torch.cuda.memory.empty_cache()
                oom = True
            else:
                print(f"- OK!")
            return oom

        with torch.autocast(device, torch.bfloat16):
            oom = step()
            while not (finished := gen.advance(oom)):
                oom = step()

        print(f"Optimal setting for input={seq_len_in} output={seq_len_out} is max_batch_size={gen.max_batch_size}")
        profile[(bucket, seq_len_in, seq_len_out)] = gen.max_batch_size
        gen.start_batch_size = gen.max_batch_size

    print("The 1st stage profile is:")
    for (bucket, seq_len_in, seq_len_out), bs in profile.items():
        print(f"Bucket={bucket} (input={seq_len_in} output={seq_len_out}) => max_batch_size={bs}")

    print("Bucket merging stage...")
    final_profile = []
    for idx, ((bucket, seq_len_in, seq_len_out), bs) in enumerate(profile.items()):
        if idx == 0:
            final_profile.append([bucket, bs])
            continue
        if bs == final_profile[-1][1]:
            print(f"Merging bucket {idx} with bucket {idx-1} due to identical batch sizes.")
            final_profile[-1][0] = bucket
            continue
        final_profile.append([bucket, bs])

    print("The final profile is:")
    print("\tbucket_duration_bins=[", ",".join(str(seqlen) for seqlen, bs in final_profile), "]", sep="")
    print("\tbucket_batch_size=[", ",".join(str(bs) for seqlen, bs in final_profile), "]", sep="")


if __name__ == "__main__":
    oomptimizer()

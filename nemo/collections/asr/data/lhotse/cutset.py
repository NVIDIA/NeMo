import warnings
from pathlib import Path
from typing import Tuple

from lhotse import CutSet

from nemo.collections.asr.data.lhotse.nemo_adapters import LazyNeMoTarredIterator, LazyNeMoIterator


def read_cutset_from_config(config) -> Tuple[CutSet, bool]:
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
        cuts = read_nemo_manifest(config, is_tarred)
    else:
        # Read Lhotse manifest (again handle both tarred(shar)/non-tarred).
        cuts = read_lhotse_manifest(config, is_tarred)
    return cuts, is_tarred


def read_lhotse_manifest(config, is_tarred: bool) -> CutSet:
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
        if isinstance(config.lhotse.shar_path, (str, Path)):
            # Single dataset in Lhotse Shar format
            cuts = CutSet.from_shar(in_dir=config.lhotse.shar_path, shuffle_shards=True, seed="trng").repeat()
        else:
            # Multiple datasets in Lhotse Shar format: we will dynamically multiplex them
            # with probability approximately proportional to their size
            cutsets = []
            weights = []
            for lsp in config.lhotse.shar_path:
                cutsets.append(CutSet.from_shar(in_dir=lsp, shuffle_shards=True, seed="trng"))
                weights.append(len(cutsets[-1]))
            cuts = CutSet.mux(*cutsets, weights=weights)
    else:
        # Regular Lhotse manifest points to individual audio files (like native NeMo manifest).
        cuts = CutSet.from_file(config.lhotse.cuts_path)
    return cuts


def read_nemo_manifest(config, is_tarred: bool) -> CutSet:
    if is_tarred:
        if isinstance(config["manifest_filepath"], (str, Path)):
            # Single manifest / tar file
            cuts = CutSet(
                LazyNeMoTarredIterator(
                    config["manifest_filepath"],
                    tar_paths=config["tarred_audio_filepaths"],
                    shuffle_shards=config.get("shuffle", False),
                )
            )
        else:
            # Assume it's [[path1], [path2], ...] (same for tarred_audio_filepaths).
            # This is the format for multiple NeMo buckets.
            # Note: we set "weights" here to be proportional to the number of utterances in each data source.
            #       this ensures that we distribute the data from each source uniformly throughout each epoch.
            #       Setting equal weights would exhaust the shorter data sources closer towatds the beginning
            #       of an epoch (or over-sample it in the case of infinite CutSet iteration with .repeat()).
            cutsets = []
            weights = []
            for (mp,), (tp,) in zip(config["manifest_filepath"], config["tarred_audio_filepaths"]):
                cutsets.append(
                    CutSet(
                        LazyNeMoTarredIterator(
                            manifest_path=mp, tar_paths=tp, shuffle_shards=config.get("shuffle", False),
                        )
                    )
                )
                weights.append(len(cutsets[-1]))
            cuts = CutSet.mux(*cutsets, weights=weights)
    else:
        cuts = CutSet(LazyNeMoIterator(config["manifest_filepath"], sampling_rate=config.get("sample_rate")))
    return cuts

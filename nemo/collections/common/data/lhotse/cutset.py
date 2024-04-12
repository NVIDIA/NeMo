# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import warnings
from itertools import repeat
from pathlib import Path
from typing import Sequence, Tuple, List

from lhotse import CutSet

from nemo.collections.common.data.lhotse.nemo_adapters import LazyNeMoIterator, LazyNeMoTarredIterator


def read_cutset_from_config(config) -> Tuple[CutSet, bool]:
    """
    Reads NeMo configuration and creates a CutSet either from Lhotse or NeMo manifests.

    Returns a tuple of ``CutSet`` and a boolean indicating whether the data is tarred (True) or not (False).
    """
    # First, we'll figure out if we should read Lhotse manifest or NeMo manifest.
    use_nemo_manifest = all(config[opt] is None for opt in ("cuts_path", "shar_path"))
    if use_nemo_manifest:
        assert (
            config.manifest_filepath is not None
        ), "You must specify either: manifest_filepath, lhotse.cuts_path, or lhotse.shar_path"
        is_tarred = config.tarred_audio_filepaths is not None
    else:
        is_tarred = config.shar_path is not None
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
        # The parameter ``config.shard_seed`` is used to determine shard shuffling order. Options:
        # - "trng" means we'll defer setting the seed until the iteration
        #   is triggered, and we'll use system TRNG to get a completely random seed for each worker.
        #   This results in every dataloading worker using full data but in a completely different order.
        # - "randomized" means we'll defer setting the seed until the iteration
        #   is triggered, and we'll use config.seed to get a pseudo-random seed for each worker.
        #   This results in every dataloading worker using full data but in a completely different order.
        #   Unlike "trng", this is deterministic, and if you resume training, you should change the seed
        #   to observe different data examples than in the previous run.
        # - integer means we'll set a specific seed in every worker, and data would be duplicated across them.
        #   This is mostly useful for unit testing or debugging.
        shard_seed = config.shard_seed
        if config.cuts_path is not None:
            warnings.warn("Note: lhotse.cuts_path will be ignored because lhotse.shar_path was provided.")
        if isinstance(config.shar_path, (str, Path)):
            logging.info(f"Initializing Lhotse Shar CutSet (tarred) from a single data source: '{config.shar_path}'")
            cuts = CutSet.from_shar(in_dir=config.shar_path, shuffle_shards=True, seed=shard_seed).repeat()
        else:
            # Multiple datasets in Lhotse Shar format: we will dynamically multiplex them
            # with probability approximately proportional to their size
            logging.info(
                "Initializing Lhotse Shar CutSet (tarred) from multiple data sources with a weighted multiplexer. "
                "We found the following sources and weights: "
            )
            cutsets = []
            weights = []
            for item in config.shar_path:
                if isinstance(item, (str, Path)):
                    path = item
                    cs = CutSet.from_shar(in_dir=path, shuffle_shards=True, seed=shard_seed)
                    weight = len(cs)
                else:
                    assert isinstance(item, Sequence) and len(item) == 2 and isinstance(item[1], (int, float)), (
                        "Supported inputs types for config.shar_path are: "
                        "str | list[str] | list[tuple[str, number]] "
                        "where str is a path and number is a mixing weight (it may exceed 1.0). "
                        f"We got: '{item}'"
                    )
                    path, weight = item
                    cs = CutSet.from_shar(in_dir=path, shuffle_shards=True, seed=shard_seed)
                logging.info(f"- {path=} {weight=}")
                cutsets.append(cs.repeat())
                weights.append(weight)
            cuts = mux(*cutsets, weights=weights, max_open_streams=config.max_open_streams, seed=config.shard_seed)
    else:
        # Regular Lhotse manifest points to individual audio files (like native NeMo manifest).
        cuts = CutSet.from_file(config.cuts_path)
    return cuts


def get_random_questions(question_file):
    question_file_list = question_file.split(",") if isinstance(question_file, str) else question_file
    random_questions = []
    for filepath in question_file_list:
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line:
                    random_questions.append(line)
    return random_questions

import random

def sample_and_attach_question(cut, questions: List[str]):
    if hasattr(cut, 'question'):
        return cut
    q = random.sample(questions, 1)[0]
    cut.question = q
    return cut


def read_nemo_manifest(config, is_tarred: bool) -> CutSet:

    common_kwargs = {
        "text_field": config.get("text_field", "text"),
        "lang_field": config.get("lang_field", "lang"),
    }
    shuffle = config.get("shuffle", False)

    if is_tarred:
        if isinstance(config["manifest_filepath"], (str, Path)):
            logging.info(
                f"Initializing Lhotse CutSet from a single NeMo manifest (tarred): '{config['manifest_filepath']}'"
            )
            cuts = CutSet(
                LazyNeMoTarredIterator(
                    config["manifest_filepath"],
                    tar_paths=config["tarred_audio_filepaths"],
                    shuffle_shards=shuffle,
                    **common_kwargs,
                )
            )
        else:
            # Format option 1:
            #   Assume it's [[path1], [path2], ...] (same for tarred_audio_filepaths).
            #   This is the format for multiple NeMo buckets.
            #   Note: we set "weights" here to be proportional to the number of utterances in each data source.
            #         this ensures that we distribute the data from each source uniformly throughout each epoch.
            #         Setting equal weights would exhaust the shorter data sources closer the towards the beginning
            #         of an epoch (or over-sample it in the case of infinite CutSet iteration with .repeat()).
            # Format option 1:
            #   Assume it's [[path1, weight1], [path2, weight2], ...] (while tarred_audio_filepaths remain unchanged).
            #   Note: this option allows to manually set the weights for multiple datasets.
            logging.info(
                f"Initializing Lhotse CutSet from multiple tarred NeMo manifest sources with a weighted multiplexer. "
                f"We found the following sources and weights: "
            )
            cutsets = []
            weights = []
            question_file_set = (
                config["question_file_set"]
                if "question_file_set" in config
                else [None for i in range(len(config["manifest_filepath"]))]
            )
            for manifest_info, (tar_path,), question_file in zip(
                config["manifest_filepath"], config["tarred_audio_filepaths"], question_file_set
            ):
                if len(manifest_info) == 1:
                    (manifest_path,) = manifest_info
                    cs = CutSet(
                        LazyNeMoTarredIterator(
                            manifest_path=manifest_path, tar_paths=tar_path, shuffle_shards=shuffle, **common_kwargs
                        )
                    )
                    weight = len(cs)
                else:
                    assert (
                        isinstance(manifest_info, Sequence)
                        and len(manifest_info) == 2
                        and isinstance(manifest_info[1], (int, float))
                    ), (
                        "Supported inputs types for config.manifest_filepath are: "
                        "str | list[list[str]] | list[tuple[str, number]] "
                        "where str is a path and number is a mixing weight (it may exceed 1.0). "
                        f"We got: '{manifest_info}'"
                    )
                    manifest_path, weight = manifest_info
                    cs = CutSet(
                        LazyNeMoTarredIterator(
                            manifest_path=manifest_path, tar_paths=tar_path, shuffle_shards=shuffle, **common_kwargs
                        )
                    )
                if question_file is not None:
                    logging.info(f"Use random questions from {question_file} for {manifest_path}")
                    questions = get_random_questions(question_file)
                    cs = cs.map(partial(sample_and_attach_question, questions=questions))

                logging.info(f"- {manifest_path=} {weight=}")
                cutsets.append(cs)
                weights.append(weight)
            if (max_open_streams := config.get("max_open_streams", None)) is not None:
                cuts = CutSet.infinite_mux(*cutsets, weights=weights, max_open_streams=max_open_streams, seed="trng")
            else:
                cuts = CutSet.mux(*[cs.repeat() for cs in cutsets], weights=weights, seed="trng")
    else:
        if isinstance(config.manifest_filepath, (str, Path)):
            logging.info(
                f"Initializing Lhotse CutSet from a single NeMo manifest (non-tarred): '{config.manifest_filepath}'"
            )
            cuts = CutSet(LazyNeMoIterator(config.manifest_filepath, **common_kwargs))
            question_file = config.get("question_file_set", None)
            if question_file is not None:
                logging.info(f"Use random questions from {question_file} for {config.manifest_filepath}")
                assert len(question_file) == 1, "Only one question file is supported for non-tarred data."
                questions = get_random_questions(question_file[0])
                cuts = cuts.map(partial(sample_and_attach_question, questions=questions))
        else:
            logging.info(
                f"Initializing Lhotse CutSet from multiple NeMo manifests (non-tarred): '{config.manifest_filepath}'"
            )
            question_file_set = (
                config["question_file_set"]
                if "question_file_set" in config
                else [None for i in range(len(config["manifest_filepath"]))]
            )
            cutsets = []
            weights = []
            for manifest_info, question_file in zip(config["manifest_filepath"], question_file_set):

                if len(manifest_info) == 1:
                    cs = CutSet(LazyNeMoIterator(manifest_info, **common_kwargs))
                    weight = len(cs)
                else:
                    assert (
                        isinstance(manifest_info, Sequence)
                        and len(manifest_info) == 2
                        and isinstance(manifest_info[1], (int, float))
                    ), (
                        "Supported inputs types for config.manifest_filepath are: "
                        "str | list[list[str]] | list[tuple[str, number]] "
                        "where str is a path and number is a mixing weight (it may exceed 1.0). "
                        f"We got: '{manifest_info}'"
                    )
                    manifest_path, weight = manifest_info
                    cs = CutSet(LazyNeMoIterator(manifest_path, **common_kwargs))

                if question_file is not None:
                    logging.info(f"Use random questions from {question_file} for {manifest_info}")
                    questions = get_random_questions(question_file)
                    cs = cs.map(partial(sample_and_attach_question, questions=questions))
                logging.info(f"- {manifest_path=} {weight=}")
                weights.append(weight)
                cutsets.append(cs)
            if (max_open_streams := config.get("max_open_streams", None)) is not None:
                cuts = CutSet.infinite_mux(*cutsets, weights=weights, max_open_streams=max_open_streams, seed="trng")
            else:
                cuts = CutSet.mux(*[cs for cs in cutsets], weights=weights, seed="trng")
    return cuts

def mux(
    *cutsets: CutSet, weights: list[int | float], max_open_streams: int | None = None, seed: str | int = "trng"
) -> CutSet:
    """
    Helper function to call the right multiplexing method flavour in lhotse.
    The result is always an infinitely iterable ``CutSet``, but depending on whether ``max_open_streams`` is set,
    it will select a more appropriate multiplexing strategy.
    """
    if max_open_streams is not None:
        cuts = CutSet.infinite_mux(*cutsets, weights=weights, seed=seed, max_open_streams=max_open_streams)
    else:
        cuts = CutSet.mux(*[cs.repeat() for cs in cutsets], weights=weights, seed=seed)
    return cuts

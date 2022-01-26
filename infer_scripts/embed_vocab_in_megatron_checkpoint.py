#!/usr/bin/env python3
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
import json
import logging
import pathlib
import shutil
import sys

import torch

LOGGER = logging.getLogger("embed_vocab")


def _are_vocabs_equal(embed_vocab, vocab):
    vocab_items = sorted(vocab.items(), key=lambda t: t[0])
    embed_vocab_items = sorted(embed_vocab.items(), key=lambda t: t[0])
    return vocab_items == embed_vocab_items


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Embed vocabulary inside Megatron checkpoint")
    parser.add_argument("--checkpoint-path", help="Path to checkpoint dir", required=True)
    parser.add_argument("--vocab-path", help="Path to checkpoint dir", required=True)
    parser.add_argument(
        "--overwrite", action="store_true", help="Force overwrite of already existing vocab", default=False
    )
    args = parser.parse_args()

    log_level = logging.INFO
    log_format = "%(asctime)s %(levelname)s %(name)s %(message)s"
    logging.basicConfig(level=log_level, format=log_format)

    vocab_path = pathlib.Path(args.vocab_path).absolute()
    LOGGER.info(f"Loading vocab from {vocab_path}")
    with vocab_path.open("r") as vocab_file:
        vocab = json.load(vocab_file)

    checkpoint_path = pathlib.Path(args.checkpoint_path).absolute()
    model_optim_rng_paths = sorted(checkpoint_path.rglob("model_optim_rng.pt"))
    if not model_optim_rng_paths:
        LOGGER.error(f"Could not find model_optim_rng.pt in {checkpoint_path} - is it Megatron checkpoint?")
        sys.exit(1)

    model_optim_rng_path = model_optim_rng_paths[0]
    LOGGER.info(f"Loading checkpoint from {model_optim_rng_path}")
    checkpoint = torch.load(model_optim_rng_path.as_posix(), map_location=lambda storage, _: storage)
    if "vocab" in checkpoint:
        LOGGER.info("Vocabulary already in checkpoint file")
        embed_vocab = checkpoint["vocab"]
        vocabs_equal = _are_vocabs_equal(embed_vocab, vocab)
        LOGGER.info(
            f"Embedded vocab and vocab from json file "
            f"{'are equal' if vocabs_equal else 'differs. Use --overwrite to replace vocab'}"
        )

        if vocabs_equal or not args.overwrite:
            sys.exit(not vocabs_equal)

    model_optim_rng_path = model_optim_rng_paths[0]
    backup_path = model_optim_rng_path.with_suffix(".pt.backup")

    LOGGER.info(f"Making backup file {backup_path}")
    shutil.copy(model_optim_rng_path, backup_path)

    LOGGER.info(f"Loading checkpoint from {model_optim_rng_path}")
    checkpoint = torch.load(model_optim_rng_path.as_posix(), map_location=lambda storage, _: storage)
    checkpoint["vocab"] = vocab

    LOGGER.info(f"Saving updated checkpoint into {model_optim_rng_path}")
    torch.save(checkpoint, model_optim_rng_path)

    LOGGER.info(f"Loading vocab from {model_optim_rng_path}")
    updated_checkpoint = torch.load(model_optim_rng_path.as_posix(), map_location=lambda storage, _: storage)
    embed_vocab = updated_checkpoint["vocab"]

    vocabs_equal = _are_vocabs_equal(embed_vocab, vocab)
    LOGGER.info(f"Embedded vocab and vocab from json file {'are equal' if vocabs_equal else 'differs'}")


if __name__ == "__main__":
    main()

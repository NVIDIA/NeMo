# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=C0115,C0116,C0301

"""A CLI to run ImageTokenizer on plain images based on torch.jit.

Usage:
    python3 -m cosmos1.models.tokenizer.inference.image_cli \
        --image_pattern 'path/to/input/folder/*.jpg' \
        --output_dir ./reconstructions \
        --checkpoint_enc ./checkpoints/<model-name>/encoder.jit \
        --checkpoint_dec ./checkpoints/<model-name>/decoder.jit

    Optionally, you can run the model in pure PyTorch mode:
    python3 -m cosmos1.models.tokenizer.inference.image_cli \
        --image_pattern 'path/to/input/folder/*.jpg' \
        --mode torch \
        --tokenizer_type CI8x8 \
        --checkpoint_enc ./checkpoints/<model-name>/encoder.jit \
        --checkpoint_dec ./checkpoints/<model-name>/decoder.jit
"""

import os
import sys
from argparse import ArgumentParser, Namespace
from typing import Any

import numpy as np
from cosmos1.models.tokenizer.inference.image_lib import ImageTokenizer
from cosmos1.models.tokenizer.inference.utils import (
    get_filepaths,
    get_output_filepath,
    read_image,
    resize_image,
    write_image,
)
from cosmos1.models.tokenizer.networks import TokenizerConfigs
from loguru import logger as logging


def _parse_args() -> tuple[Namespace, dict[str, Any]]:
    parser = ArgumentParser(description="A CLI for running ImageTokenizer on plain images.")
    parser.add_argument(
        "--image_pattern",
        type=str,
        default="path/to/images/*.jpg",
        help="Glob pattern.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="JIT full Autoencoder model filepath.",
    )
    parser.add_argument(
        "--checkpoint_enc",
        type=str,
        default=None,
        help="JIT Encoder model filepath.",
    )
    parser.add_argument(
        "--checkpoint_dec",
        type=str,
        default=None,
        help="JIT Decoder model filepath.",
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        default=None,
        choices=[
            "CI8x8",
            "DI8x8",
            "CI16x16",
            "DI16x16",
            "CI8x8-LowRes",
            "CI16x16-LowRes",
            "DI8x8-LowRes",
            "DI16x16-LowRes",
        ],
        help="Specifies the tokenizer type.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["torch", "jit"],
        default="jit",
        help="Specify the backend: native 'torch' or 'jit' (default: 'jit')",
    )
    parser.add_argument(
        "--short_size",
        type=int,
        default=None,
        help="The size to resample inputs. None, by default.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Sets the precision. Default bfloat16.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for invoking the model.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory.")
    parser.add_argument(
        "--save_input",
        action="store_true",
        help="If on, the input image will be be outputed too.",
    )
    args = parser.parse_args()
    return args


logging.info("Initializes args ...")
args = _parse_args()
if args.mode == "torch" and args.tokenizer_type is None:
    logging.error("'torch' backend requires the tokenizer_type to be specified.")
    sys.exit(1)


def _run_eval() -> None:
    """Invokes the evaluation pipeline."""

    if args.checkpoint_enc is None and args.checkpoint_dec is None and args.checkpoint is None:
        logging.warning("Aborting. Both encoder or decoder JIT required. Or provide the full autoencoder JIT model.")
        return

    if args.mode == "torch":
        _type = args.tokenizer_type.replace("-", "_")
        _config = TokenizerConfigs[_type].value
    else:
        _config = None

    logging.info(
        f"Loading a torch.jit model `{os.path.dirname(args.checkpoint or args.checkpoint_enc or args.checkpoint_dec)}` ..."
    )
    autoencoder = ImageTokenizer(
        checkpoint=args.checkpoint,
        checkpoint_enc=args.checkpoint_enc,
        checkpoint_dec=args.checkpoint_dec,
        tokenizer_config=_config,
        device=args.device,
        dtype=args.dtype,
    )

    filepaths = get_filepaths(args.image_pattern)
    logging.info(f"Found {len(filepaths)} images from {args.image_pattern}.")

    for filepath in filepaths:
        logging.info(f"Reading image {filepath} ...")
        image = read_image(filepath)
        image = resize_image(image, short_size=args.short_size)
        batch_image = np.expand_dims(image, axis=0)

        logging.info("Invoking the autoencoder model in ... ")
        output_image = autoencoder(batch_image)[0]

        output_filepath = get_output_filepath(filepath, output_dir=args.output_dir)
        logging.info(f"Outputing {output_filepath} ...")
        write_image(output_filepath, output_image)

        if args.save_input:
            ext = os.path.splitext(output_filepath)[-1]
            input_filepath = output_filepath.replace(ext, "_input" + ext)
            write_image(input_filepath, image)


@logging.catch(reraise=True)
def main() -> None:
    _run_eval()


if __name__ == "__main__":
    main()

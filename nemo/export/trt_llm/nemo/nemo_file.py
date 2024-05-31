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


import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import tensorstore  # This is important even though not used. Otherwise zarr raises error.
import torch
import zarr
from transformers import AutoTokenizer, PreTrainedTokenizer

from nemo.export.tarutils import TarPath, ZarrPathStore
from nemo.export.trt_llm.nemo.nemo import UnpackedNemoCheckpointDir
from nemo.export.trt_llm.nemo.sentencepiece_tokenizer import SentencePieceTokenizer

LOGGER = logging.getLogger("NeMo")


def load_sharded_metadata_torch_dist(checkpoint_dir: Union[Path, TarPath], torch_tensor=True):
    fs_reader = TarFileSystemReader(checkpoint_dir)
    metadata = fs_reader.read_metadata()

    state_dict = {
        k: torch.empty(tp.size, dtype=tp.properties.dtype)
        for k, tp in metadata.state_dict_metadata.items()
        if isinstance(tp, TensorStorageMetadata)
    }
    load_state_dict(
        state_dict,
        storage_reader=fs_reader,
        no_dist=True,
    )

    if not torch_tensor:
        for k, v in state_dict.items():
            if v.dtype == torch.bfloat16:
                state_dict[k] = v.view(torch.int16).numpy().view(np_bfloat16)
            else:
                state_dict[k] = v.numpy()
    return state_dict


def load_sharded_metadata_zarr(checkpoint_dir: Union[Path, TarPath], torch_tensor=True):
    sharded_state_dict = {}
    for subdir in checkpoint_dir.iterdir():
        if not subdir.is_dir() or not (subdir / '.zarray').exists():
            continue
        key = subdir.name

        zstore = ZarrPathStore(subdir)
        arr = zarr.open(zstore, 'r')

        if torch_tensor:
            # sharded_state_dict[key] = torch.from_numpy(arr[:].astype("float32")).to(dtype=torch.bfloat16)
            if arr.dtype.name == "bfloat16":
                sharded_state_dict[key] = torch.from_numpy(arr[:].view(np.int16)).view(torch.bfloat16)
            else:
                sharded_state_dict[key] = torch.from_numpy(arr[:])
        else:
            sharded_state_dict[key] = arr[:]

    return sharded_state_dict


def load_sharded_metadata(checkpoint_dir: Union[Path, TarPath], torch_tensor=True):
    with (checkpoint_dir / 'metadata.json').open(mode='r') as f:
        config_dict = json.load(f)
    if config_dict['sharded_backend'] == 'zarr':
        return load_sharded_metadata_zarr(checkpoint_dir, torch_tensor)
    elif config_dict['sharded_backend'] == 'torch_dist':
        return load_sharded_metadata_torch_dist(checkpoint_dir, torch_tensor)
    else:
        raise NotImplementedError(f'Distributed checkpoint backend {config_dict["sharded_backend"]} not supported')


def update_tokenizer_paths(tokenizer_config: Dict, unpacked_checkpoints_dir):
    def _update_config_entry(key, file_pattern):
        old_path = tokenizer_config[key]
        if old_path is None:
            return
        old_path = Path(old_path)
        new_path = unpacked_checkpoints_dir.get_tokenizer_file_path("tokenizer", key, file_pattern)
        if new_path:
            LOGGER.debug(f"Update tokenizer {key} {old_path} -> {new_path}")
            tokenizer_config[key] = new_path
        elif not old_path.exists():
            LOGGER.warning(f"Tokenizer {key}'s path {old_path} does not exists: set it to None")
            tokenizer_config[key] = None

    _update_config_entry("model", "*.model")
    _update_config_entry("vocab_file", "*vocab*")
    _update_config_entry("merge_file", "*merge*.txt")

    return tokenizer_config


def copy_tokenizer_files(config, out_dir):
    basenames = {
        "model": "tokenizer",
        "vocab_file": "vocab",
        "merge_file": "merges",
    }

    for key in basenames.keys():
        if config[key] is None:
            continue

        path = config[key]

        if isinstance(path, str):
            path = Path(path)

        if not path.exists():
            LOGGER.debug(f"Tokenizer {key}: {path} file not found")
            continue

        dst_path = out_dir / f"{basenames[key]}{path.suffix}"
        LOGGER.debug(f"Copy tokenizer {key}: {path}->{dst_path}")

        # Copy 'path' to 'dst_path' without shutil.copy(...) because 'path' may be a TarPath
        with path.open('rb') as infile:
            with open(dst_path, 'wb') as outfile:
                outfile.write(infile.read())


def get_tokenzier(tokenizer_dir_or_path: Path) -> PreTrainedTokenizer:
    """Loads the tokenizer from the decoded NEMO weights dir."""
    if os.path.isdir(os.path.join(tokenizer_dir_or_path, "huggingface_tokenizer")):
        return AutoTokenizer.from_pretrained(os.path.join(tokenizer_dir_or_path, "huggingface_tokenizer"))

    model_path = tokenizer_dir_or_path / "tokenizer.model" if tokenizer_dir_or_path.is_dir() else tokenizer_dir_or_path
    tokenizer_config = {"library": "sentencepiece", "model": str(model_path)}
    return build_tokenizer(tokenizer_config)


def build_tokenizer(tokenizer):
    if isinstance(tokenizer, dict):
        tokenizer_config = tokenizer
        if tokenizer_config["library"] == "sentencepiece":
            return SentencePieceTokenizer(model_path=tokenizer_config["model"])
        elif "GPT2" in tokenizer_config["type"]:
            tokenizer = GPT2Tokenizer(tokenizer_config["vocab_file"], tokenizer_config["merge_file"])
        else:
            raise ValueError(f'Tokenizer type {tokenizer_config["library"]} not handled')

        if tokenizer.bos_token_id is None:
            tokenizer.add_special_tokens({"bos_token": "<s>"})
        if tokenizer.eos_token_id is None:
            tokenizer.add_special_tokens({"eos_token": "</s>"})
    else:
        try:
            # If NeMo tokenizer, monkey patch interface
            from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

            if isinstance(tokenizer, TokenizerSpec):

                def batch_encode_patch(self, ids):
                    if torch.is_tensor(ids):
                        ids = ids.cpu().numpy()
                    return self.ids_to_text(ids)

                tokenizer.bos_token_id = tokenizer.bos_id
                tokenizer.eos_token_id = tokenizer.eos_id
                tokenizer.encode = tokenizer.text_to_ids
                TokenizerSpec.batch_decode = batch_encode_patch
        except:
            raise TypeError(f'Unsupported tokenizer build input: {type(tokenizer)}')

    return tokenizer


def load_nemo_model(nemo_ckpt: Union[str, Path], nemo_export_dir: Union[str, Path]):

    if not os.path.exists(nemo_ckpt):
        LOGGER.error("%s does not exist", nemo_ckpt)
        sys.exit(1)

    if os.path.isdir(nemo_ckpt):
        nemo_dir = Path(nemo_ckpt)
    else:
        nemo_dir = TarPath(nemo_ckpt)

    try:
        unpacked_checkpoint_dir = UnpackedNemoCheckpointDir(nemo_dir, load_checkpoints_to_cpu=True)

        dist_ckpt_folder = nemo_dir / "model_weights"
        if dist_ckpt_folder.exists():
            model = load_sharded_metadata(dist_ckpt_folder)
            nemo_model_config = unpacked_checkpoint_dir.model_config

            if nemo_model_config["tokenizer"].get("library", None) == "huggingface":
                tokenizer = AutoTokenizer.from_pretrained(
                    nemo_model_config["tokenizer"]["type"],
                    use_fast=nemo_model_config["tokenizer"].get("use_fast", False),
                )
            else:
                tokenizer_config = update_tokenizer_paths(nemo_model_config["tokenizer"], unpacked_checkpoint_dir)
                copy_tokenizer_files(tokenizer_config, nemo_export_dir)

                tokenizer_config["model"] = os.path.join(nemo_export_dir, "tokenizer.model")
                tokenizer = build_tokenizer(tokenizer_config)
        else:
            raise Exception(
                "Not a supported nemo file format. " "Only distributed mcore nemo checkpoints are support."
            )
    finally:
        if isinstance(nemo_dir, TarPath):
            nemo_dir.tarobject.close()

    return model, nemo_model_config, tokenizer

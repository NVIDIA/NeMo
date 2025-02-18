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

import sys

import torch


def load_dcp(ckpt_dir):
    from pathlib import Path

    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader

    if not isinstance(ckpt_dir, Path):
        ckpt_dir = Path(ckpt_dir)
    fs_reader = FileSystemReader(ckpt_dir)
    metadata = fs_reader.read_metadata()

    state_dict = {
        k: torch.empty(tp.size, dtype=tp.properties.dtype)
        for k, tp in metadata.state_dict_metadata.items()
        if type(tp).__name__ == 'TensorStorageMetadata'
    }

    dcp.load(
        state_dict,
        storage_reader=fs_reader,
    )
    return state_dict

def compare_ckpts(a_item, b_item, current_key, mismatches):
    if isinstance(a_item, dict):
        if not isinstance(b_item, dict):
            mismatches.append(f"Mismatch at '{current_key}': Expected dict, got {type(b_item)}.")
            return
        # Compare keys in both dicts.
        # If you want to fail if 'b_item' has extra keys or missing keys,
        # you can add additional checks here.
        for k in a_item:
            sub_key = f"{current_key}.{k}" if current_key else k
            compare_ckpts(a_item[k], b_item[k], sub_key)
    elif isinstance(a_item, torch.Tensor):
        if not isinstance(b_item, torch.Tensor):
            mismatches.append(f"Mismatch at '{current_key}': Expected torch.Tensor, got {type(b_item)}.")
            return

        if a_item.dtype != b_item.dtype:
            mismatches.append(f"Mismatch at '{current_key}': Different dtypes ({a_item.dtype} vs {b_item.dtype}).")
        if a_item.device != b_item.device:
            mismatches.append(f"Mismatch at '{current_key}': Different devices ({a_item.device} vs {b_item.device}).")
        if a_item.shape != b_item.shape:
            mismatches.append(f"Mismatch at '{current_key}': Different shapes ({a_item.shape} vs {b_item.shape}).")
        # Use torch.equal for an element-wise equality check
        if not torch.equal(a_item, b_item):
            mismatches.append(f"Mismatch at '{current_key}': Different values:\n{a_item}\nvs\n{b_item}")
    else:
        # For simple Python objects, we can do a direct type and value check
        if type(a_item) != type(b_item):
            mismatches.append(
                f"Mismatch at '{current_key}': Different types ({type(a_item)} vs {type(b_item)})."
            )
        elif a_item != b_item:
            mismatches.append(
                f"Mismatch at '{current_key}': Different values ({a_item} vs {b_item})."
            )

def remove_module_from_key(x):
    # module.decoder.layers.mlp.router.weight -> decoder.layers.mlp.router.weight
    # optimizer.state.fp32_param.module.output.weight -> optimizer.state.fp32_param.output.weight
    assert isinstance(x, str)
    return '.'.join(filter(lambda x: x != 'module', x.split('.')))


def remove_module_from_dict_keys(d):
    assert isinstance(d, dict)
    return {remove_module_from_key(k): v for k, v in d.items()}


if __name__ == "__main__":
    load_n_rename = lambda x: remove_module_from_dict_keys(load_dcp(x))
    ckpt = load_n_rename(sys.argv[1])
    ckpt2 = load_n_rename(sys.argv[2])
    # compare_ckpts(ckpt, ckpt2)
    mismatches = []
    compare_ckpts(ckpt, ckpt2, '', mismatches)

    if len(mismatches) > 0:
        # Join all mismatch messages and raise as a single exception
        raise ValueError(
            "The following mismatches were found:\n" + "\n".join(mismatches)
        )
    else:
        print("All keys and tensors match!")

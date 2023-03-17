# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import os
from argparse import ArgumentParser
from typing import Dict, List

import torch
from pytorch_lightning import Trainer

from nemo.collections.nlp.parts.nlp_overrides import (
    NEMO_MEGATRON_MODEL_PARALLEL_APPSTATE_OVERRIDE,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
)
from nemo.utils import logging, model_utils
from nemo.utils.app_state import AppState

"""
Usage:
python megatron_change_num_partitions.py \
    --model_file=PATH_TO_SRC_FILE \
    --target_file=PATH_TO_TGT_FILE \
    --tensor_model_parallel_size=2 \
    --target_tensor_model_parallel_size=1 \
    --pipeline_model_parallel_size=1 \
    --target_pipeline_model_parallel_size=1

"""

DEBUG_PRINT = False


def merge_partition(model, partitions: Dict[int, List[List[torch.Tensor]]], write_path: str = None):
    # Extract the pp_rank and number of modules per tp rank in each pp rank
    pp_ranks = list(partitions.keys())
    pp_lens = []
    for pp_rank in pp_ranks:
        partition_pp = partitions[pp_rank]
        max_len = max([len(x) for x in partition_pp])  # Perform max as we need to iterate through all modules
        pp_lens.append(max_len)

    idx = 0
    pp_rank = 0
    # During merge - model is TP 1 PP 1 model with all parameters present in correct order.
    # Merge the parameters of the various PP X TP Y models into the TP 1 PP 1 model.
    for name, param in model.named_parameters():
        # Since the PP ranks each contain the list of all their TP rank parameters
        # We need to detect if we need to move to the next PP rank when we run out of tensors in current PP rank
        # Reset the index so that it indexes the new pp rank tensor list correctly
        if idx >= pp_lens[pp_rank]:
            pp_rank += 1
            idx = 0

        # Extract all TP ranks out of current PP rank
        partitions_pp = partitions[pp_rank]

        if DEBUG_PRINT:
            print("Model Param:", name, param.shape, "Partition Params:", [p[idx].shape for p in partitions_pp])

        # Logic from original TP rank change
        if param.shape == partitions_pp[0][idx].shape:
            concated = partitions_pp[0][idx].data
        elif param.shape[0] == partitions_pp[0][idx].shape[0]:
            concated = torch.cat([partitions_pp[i][idx].data for i in range(len(partitions_pp))], dim=-1)
        else:
            concated = torch.cat([partitions_pp[i][idx].data for i in range(len(partitions_pp))], dim=0)

        if concated.shape != param.shape:
            logging.info(
                f"Warning: Shape mismatch for parameter {name} required shape: {param.shape}, merged shape: {concated.shape}. Narrowing to match required size."
            )
            if concated.shape[1:] == param.shape[1:]:
                concated = torch.narrow(concated, 0, 0, param.shape[0])
            elif concated.shape[:-1] == param.shape[:-1]:
                concated = torch.narrow(concated, -1, 0, param.shape[-1])
            else:
                raise RuntimeError(
                    f"Can not handle parameter {name}, required shape: {param.shape}, merged shape: {concated.shape}."
                )
        param.data = concated
        idx += 1

    # Save the file iff the original file was PP 1 TP 1
    if write_path is not None:
        model.save_to(write_path)


def split_partition(
    model,
    partitions,
    pp_size: int,
    tp_size: int,
    pp_rank: int,
    offset: int,
    write_path: str = None,
    megatron_legacy: bool = False,
):
    if len(partitions) != 2:
        raise ValueError(
            "Can only split partitions of model with TP=1. For partitions of models with TP>1, merge first."
        )

    if tp_size < 1:
        raise ValueError("TP size must to be >= 1.")

    if pp_size < 1:
        raise ValueError("PP size must to be >= 1.")

    # Setup app state to mimic current PP and TP ranks with single merged module
    app_state = AppState()
    app_state.data_parallel_rank = 0
    app_state.pipeline_model_parallel_size = pp_size
    app_state.tensor_model_parallel_size = tp_size
    app_state.model_parallel_size = app_state.pipeline_model_parallel_size * app_state.tensor_model_parallel_size

    # Go in reverse for TP order, as PP 0 TP 0 will merge all preceding files
    app_state.pipeline_model_parallel_rank = pp_rank
    app_state.tensor_model_parallel_rank = tp_size - 1

    # Compute reverse offset of parameter index from global map
    num_params = sum([1 for _ in model.parameters()])  # Count number of parameters iteratively
    idx = offset - num_params + 1  # start index of current PP TP rank in global map

    assert (
        idx + num_params - 1 == offset
    ), f"idx = {idx}, num_params = {num_params}, sum = {idx + num_params}, offset = {offset}"

    # Special case for GPT models - whose last PP TP rank has a duplicate embedding tensor

    # Megatron GPT check for final PP rank duplicated embeddings
    if (
        pp_rank == (pp_size - 1) and hasattr(model, 'mode') and hasattr(model.model, 'word_embeddings')
    ):  # duplicate embedding copy (tied weights)
        duplicate_word_embedding_offset = 1
    else:
        duplicate_word_embedding_offset = 0
    idx += duplicate_word_embedding_offset  # add duplicate embedding offset to index

    print("Start Layer Idx:", idx, "Num Remaining layers:", num_params, "Offset:", offset)
    print("Duplicate_word_embedding_offset", duplicate_word_embedding_offset)
    print()

    splits = []

    # This is the PP X TP Y model with partial parameters present in correct order.
    # We need to extract the parameters from the global map in reverse order to fill in the
    # parameters of this model in forward order.
    for param_name, param in model.named_parameters():

        # Since we are moving forward, we may reach the end of the global map
        # but GPT has an additional word embedding as its last parameter
        # Therefore we check for this, and reset the index to the parameter of the PP 0 TP 0 rank
        # which holds the parameters of the embedding.
        if idx == (len(partitions[0])) and duplicate_word_embedding_offset > 0:
            print("Found duplicate embedding copy for GPT model, resetting index")
            idx = 0  # reset idx parameter to 0 if we have duplicate embedding copy

        if DEBUG_PRINT:
            print(
                "Model param:",
                param_name,
                param.shape,
                "Layer Idx:",
                idx,
                "Global params:",
                partitions[1][idx],
                partitions[0][idx].shape,
            )

        # Tensor Parallel Splitting
        if param.shape == partitions[0][idx].shape:
            split = [partitions[0][idx].data] * tp_size
        elif param.shape[0] == partitions[0][idx].shape[0]:
            split = torch.split(partitions[0][idx].data, param.shape[-1], dim=-1)
        else:
            # For T5-converted weights, the splitting needs to be strided such that q,k,v weights are bunched together on each tensor-parallel rank.
            if 'query_key_value.weight' in param_name and megatron_legacy:
                split_dim = partitions[0][idx].data.shape[0]
                if split_dim % (tp_size * 3) != 0:
                    raise ValueError(
                        f"Can not split Q,K,V parameter {param_name} with shape {param.shape} into tensor parallel size {tp_size}. Not divisible by {tp_size * 3}."
                    )
                tp_qkv_splits = torch.chunk(partitions[0][idx].data, tp_size * 3, dim=0)
                split = []
                for i in range(tp_size):
                    tp_qkv = torch.cat([tp_qkv_splits[item] for item in range(i, tp_size * 3, tp_size)])
                    split.append(tp_qkv)
            elif 'key_value.weight' in param_name and megatron_legacy:
                split_dim = partitions[0][idx].data.shape[0]
                if split_dim % (tp_size * 2) != 0:
                    raise ValueError(
                        f"Can not split K,V parameter {param_name} with shape {param.shape} into tensor parallel size {tp_size}. Not divisible by {tp_size * 2}."
                    )
                tp_qkv_splits = torch.chunk(partitions[0][idx].data, tp_size * 2, dim=0)
                split = []
                for i in range(tp_size):
                    tp_qkv = torch.cat([tp_qkv_splits[item] for item in range(i, tp_size * 2, tp_size)])
                    split.append(tp_qkv)
            # Regular split for Megatron and NeMo-Megatron models.
            else:
                split = torch.split(partitions[0][idx].data, param.shape[0], dim=0)
        splits.append(split)
        idx += 1

    # Compute the new offset for the next PP rank in reverse order
    # Add 1 to offset to account for last PP rank's duplicated Embedding
    offset_diff = offset - num_params
    if pp_rank == (pp_size - 1):
        offset_diff += 1
    new_offset = offset_diff

    # Save each of the TP ranks in reverse order
    # This is done so that the last PP rank will save the last TP rank only after all other PP TP ranks are saved
    # The final rank will then save a new NeMo file with all other ranks inside.
    for tp_rank in range(tp_size - 1, -1, -1):
        app_state.pipeline_model_parallel_rank = pp_rank
        app_state.tensor_model_parallel_rank = tp_rank

        idx = 0
        for name, param in model.named_parameters():
            split_val = splits[idx][tp_rank].clone()

            if param.shape != split_val.shape:
                logging.info(
                    f"Warning: Shape mismatch for parameter {name} required shape: {param.shape}, split shape: {split_val.shape}. Padding to match required size."
                )

                if split_val.shape[1:] == param.shape[1:]:
                    pad = [0, 0] * len(split_val.shape)
                    pad[-1] = param.shape[0] - split_val.shape[0]
                    split_val = torch.nn.functional.pad(split_val, pad, 'constant')
                elif split_val.shape[:-1] == param.shape[:-1]:
                    pad = [0, param.shape[-1] - split_val.shape[-1]]
                    split_val = torch.nn.functional.pad(split_val, pad, 'constant')
                else:
                    raise RuntimeError(
                        f"Can not handle parameter {name}, required shape: {param.shape}, split shape: {split_val.shape}."
                    )

            param.data = split_val
            idx += 1

        if write_path is not None:
            print(f"Writing pp rank {pp_rank} tp rank {tp_rank} to file {write_path}")
            model.save_to(write_path)

    return new_offset


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_file", type=str, required=True, help="Path to source .nemo file")
    parser.add_argument("--target_file", type=str, required=True, help="Path to write target .nemo file")
    parser.add_argument("--tensor_model_parallel_size", type=int, required=True, help="TP size of source model")
    parser.add_argument("--target_tensor_model_parallel_size", type=int, required=True, help="TP size of target model")
    parser.add_argument('--pipeline_model_parallel_size', type=int, required=True, help='PP size of source model')
    parser.add_argument(
        '--target_pipeline_model_parallel_size', type=int, required=True, help='PP size of target model'
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="nemo.collections.nlp.models.language_modeling.megatron_gpt_model.MegatronGPTModel",
        help="NeMo model class. This script should support all NeMo megatron models that use Tensor Parallel",
    )
    parser.add_argument("--precision", default=16, help="PyTorch Lightning Trainer precision flag")
    parser.add_argument('--num_gpu_per_node', default=8, help='Number of GPUs per node')
    parser.add_argument(
        "--megatron_legacy",
        action="store_true",
        help="Converter for legacy megatron modles that have different q,k,v weight splits",
    )
    parser.add_argument(
        "--tokenizer_model_path",
        type=str,
        required=False,
        default=None,
        help="Path to the tokenizer model path if your model uses a tokenizer model as an artifact. This is needed if your model uses a sentencepiece tokenizer.",
    )

    args = parser.parse_args()

    precision = args.precision
    num_gpu_per_node = args.num_gpu_per_node
    if args.precision in ["32", "16"]:
        precision = int(float(args.precision))

    if precision == "bf16":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            precision = "bf16"
        else:
            logging.warning("BF16 is not supported on this device. Using FP16 instead.")
            precision = 16

    if precision == 32:
        dtype = torch.float32
    elif precision == 16:
        dtype = torch.float16
    elif precision == "bf16":
        dtype = torch.bfloat16

    tp_size = args.tensor_model_parallel_size
    tgt_tp_size = args.target_tensor_model_parallel_size
    pp_size = args.pipeline_model_parallel_size
    tgt_pp_size = args.target_pipeline_model_parallel_size
    cls = model_utils.import_class_by_path(args.model_class)

    trainer = Trainer(devices=1, strategy=NLPDDPStrategy(), accelerator="cpu", precision=precision)
    app_state = AppState()
    app_state.data_parallel_rank = 0
    app_state.pipeline_model_parallel_size = pp_size
    app_state.tensor_model_parallel_size = tp_size
    app_state.model_parallel_size = app_state.pipeline_model_parallel_size * app_state.tensor_model_parallel_size

    world_size = pp_size * tp_size  # pseudo world size for simulating load of a specific rank on a single gpu

    app_state.tensor_model_parallel_rank = 0
    app_state.pipeline_model_parallel_rank = 0

    # If input model has TP > 1 or PP > 1
    # Reconstruct the model to have TP = 1 and PP = 1
    # Note that this is a forward loop that will process PP [0..N] TP [0..M] in sequential order.
    if tp_size > 1 or pp_size > 1:
        partitions = {}
        model = None
        for pp_rank in range(pp_size):
            app_state.pipeline_model_parallel_rank = pp_rank
            partitions[pp_rank] = []

            for tp_rank in range(tp_size):
                app_state.tensor_model_parallel_rank = tp_rank

                print("------------", pp_rank, tp_rank)

                # Override flag that forces Model to use AppState instead of Trainer
                # to determine the world size, global and local rank
                # Used for simulating load of a specific rank on a single gpu
                os.environ[NEMO_MEGATRON_MODEL_PARALLEL_APPSTATE_OVERRIDE] = "true"

                # Compute the global rank to load the correct subset of parameters
                global_rank = pp_rank * tp_size + tp_rank

                # Update AppState
                app_state.world_size = world_size
                app_state.global_rank = global_rank
                app_state.local_rank = global_rank % num_gpu_per_node
                app_state.pipeline_model_parallel_size = pp_size
                app_state.tensor_model_parallel_size = tp_size
                app_state.model_parallel_size = (
                    app_state.pipeline_model_parallel_size * app_state.tensor_model_parallel_size
                )

                model = cls.restore_from(
                    restore_path=args.model_file,
                    trainer=trainer,
                    map_location=torch.device("cpu"),
                    save_restore_connector=NLPSaveRestoreConnector(),
                )
                model.to(dtype=dtype)

                # Reset env flag
                os.environ.pop(NEMO_MEGATRON_MODEL_PARALLEL_APPSTATE_OVERRIDE, None)

                print(f"<<<<<<<< LOADED MODEL {pp_rank} {tp_rank} | GLOBAL RANK = {global_rank} >>>>>>>>>")
                params = [p for _, p in model.named_parameters()]
                partitions[pp_rank].append(params)

                # app_state is being updated incorrectly during restore
                app_state.data_parallel_rank = 0
                app_state.pipeline_model_parallel_rank = pp_rank
                app_state.tensor_model_parallel_rank = tp_rank
                app_state.pipeline_model_parallel_size = pp_size
                app_state.tensor_model_parallel_size = tp_size
                app_state.model_parallel_size = (
                    app_state.pipeline_model_parallel_size * app_state.tensor_model_parallel_size
                )

        # Build a unified model with PP 1 TP 1
        model.cfg.tensor_model_parallel_size = 1
        model.cfg.pipeline_model_parallel_size = 1
        app_state.tensor_model_parallel_rank = 0
        app_state.pipeline_model_parallel_size = 0
        app_state.model_parallel_size = 1

        trainer = Trainer(devices=1, strategy=NLPDDPStrategy(), accelerator="cpu", precision=precision)
        if args.tokenizer_model_path is not None:
            model.cfg.tokenizer.model = args.tokenizer_model_path

        model = cls(model.cfg, trainer).to('cpu')
        model._save_restore_connector = NLPSaveRestoreConnector()

        if tgt_tp_size > 1 or tgt_pp_size > 1:
            merge_partition(model, partitions)
        else:
            # Write out the PP 1 TP 1 model to disk
            merge_partition(model, partitions, args.target_file)

        # Empty cache memory of all parameters from all PP TP partitions
        partitions.clear()

    else:
        # If input model has TP = 1 and PP = 1
        app_state.model_parallel_size = 1
        model = cls.restore_from(restore_path=args.model_file, trainer=trainer, map_location=torch.device("cpu"))
        model.to(dtype=dtype)

    # If target model has TP > 1 or PP > 1
    if tgt_pp_size > 1 or tgt_tp_size > 1:

        # Preserve the TP 1 PP 1 model parameters and names
        global_params = []
        global_params.append([p for n, p in model.named_parameters()])  # params
        global_params.append([n for n, p in model.named_parameters()])  # names

        world_size = (
            tgt_pp_size * tgt_tp_size
        )  # pseudo world size for simulating load of a specific rank on a single gpu
        new_global_batch_size = model.cfg.micro_batch_size * world_size
        old_global_batch_size = model.cfg.global_batch_size

        global_offset = len(global_params[0]) - 1  # -1 cause this indexes the array, range [0, L-1]
        print("Global offset of layers: ", global_offset)

        for pp_rank in range(tgt_pp_size - 1, -1, -1):  # reverse order

            model.cfg.pipeline_model_parallel_size = tgt_pp_size
            model.cfg.tensor_model_parallel_size = tgt_tp_size
            model.cfg.global_batch_size = old_global_batch_size  # Used for restoration

            # Override flag that forces Model to use AppState instead of Trainer
            # to determine the world size, global and local rank
            # Used for simulating load of a specific rank on a single gpu
            os.environ[NEMO_MEGATRON_MODEL_PARALLEL_APPSTATE_OVERRIDE] = "true"

            # Compute the global rank
            global_rank = (
                pp_rank * tgt_tp_size + 0
            )  # tp_rank = 0 needed just for modules, all TP will be merged to this PP rank

            # Update AppState
            app_state.world_size = world_size
            app_state.global_rank = global_rank
            app_state.local_rank = global_rank % num_gpu_per_node
            app_state.pipeline_model_parallel_size = tgt_pp_size
            app_state.tensor_model_parallel_size = tgt_tp_size
            app_state.model_parallel_size = (
                app_state.pipeline_model_parallel_size * app_state.tensor_model_parallel_size
            )

            trainer = Trainer(devices=1, strategy=NLPDDPStrategy(), accelerator="cpu", precision=precision)
            if args.tokenizer_model_path is not None:
                model.cfg.tokenizer.model = args.tokenizer_model_path

            model = cls(model.cfg, trainer).to('cpu')
            model._save_restore_connector = NLPSaveRestoreConnector()
            model.to(dtype=dtype)

            # Update global batch size
            if old_global_batch_size % new_global_batch_size != 0 or old_global_batch_size < new_global_batch_size:
                logging.info(
                    f"Global batch size {old_global_batch_size} is not divisible by new global batch size {new_global_batch_size}."
                    f" The model config will be updated with new global batch size {new_global_batch_size}."
                )
                model.cfg.global_batch_size = new_global_batch_size

            print("Global rank: ", global_rank, "Local rank: ", app_state.local_rank, "World size: ", world_size)
            print("PP rank: ", pp_rank, "TP rank: ", 0)
            print()

            global_offset = split_partition(
                model,
                global_params,
                tgt_pp_size,
                tgt_tp_size,
                pp_rank,
                global_offset,
                args.target_file,
                args.megatron_legacy,
            )

            # Reset env flag
            os.environ.pop(NEMO_MEGATRON_MODEL_PARALLEL_APPSTATE_OVERRIDE, None)

        # Check if invalid global offset - after all PP splits, global offset should be -1
        if global_offset < -1:
            raise ValueError(
                f"Invalid global offset {global_offset} found for global rank {app_state.global_rank} "
                f"and local rank {app_state.local_rank}. Should be -1 if all parameters have been assigned. "
                f"Currently, seems some parameters were duplicated."
            )
        elif global_offset > -1:
            print()
            print("!" * 80)
            print("Error: Some parameters were not correctly added to model partitions.")
            print("Below is list of parameters skipped in reverse order: ")

            for param_id in range(global_offset, -1, -1):
                print("Param ID:", param_id, ":", global_params[1][param_id], global_params[0][param_id].shape)

            raise ValueError(
                f"Invalid global offset {global_offset} found for global rank {app_state.global_rank} "
                f"and local rank {app_state.local_rank}. Should be -1 if all parameters have been assigned. "
                f"Currently, seems some parameters were not assigned."
            )

    logging.info("Successfully finished changing partitions!")


if __name__ == '__main__':
    main()

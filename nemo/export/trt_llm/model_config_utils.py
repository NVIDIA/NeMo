# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Utils to load and process model_config."""

import copy
import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import List, Union

import numpy as np

from .model_config import LAYER_QKV, LINEAR_COLUMN, EmbeddingConfig, LinearConfig, ModelConfig


def _restore_model_config(model_config, weights):
    """Recursively restores the model_config from json and loads np.ndarray weights from weights."""
    if isinstance(model_config, dict):
        for k, v in model_config.items():
            if isinstance(v, str) and v.startswith("_np:"):
                model_config[k] = weights[v]
            else:
                _restore_model_config(v, weights)
    if isinstance(model_config, list):
        for i, v in enumerate(model_config):
            if isinstance(v, str) and v.startswith("_np:"):
                model_config[i] = weights[v]
            else:
                _restore_model_config(v, weights)


def load_model_configs(
    model_config_json: Union[str, Path], inference_tensor_parallel: int = 1
) -> List[ModelConfig]:
    """Loads the model_config saved from ammo export.

    Args:
        model_config_json: The exported json file from ammo describing the optimized model.
            Inside the same directory, each gpu rank will have its own npz file.
            The json file represents the general ModelConfig structure while the detailed
            weights for each rank are stored in the npz file.

    Returns:
        The list of `ModelConfig` loaded and constructed.
    """
    model_config_json = Path(model_config_json)
    assert model_config_json.exists()

    with open(model_config_json, "r") as f:
        model_config_template = json.load(f)

    tensor_parallel = model_config_template["tensor_parallel"]
    assert tensor_parallel > 0, f"Invalid tensor_parallel {tensor_parallel}"

    model_config_dir = model_config_json.parents[0]

    model_configs = []
    for i in range(tensor_parallel):
        decoder_type = model_config_template["layers"][0]["decoder_type"]
        weights_file = f"{decoder_type}_tp{tensor_parallel}_rank{i}.npz"
        weights = dict(np.load(model_config_dir / weights_file))
        model_config = copy.deepcopy(model_config_template)
        model_config["rank"] = i
        _restore_model_config(model_config, weights)
        model_configs.append(ModelConfig.from_dict(model_config))

    model_configs = _postprocess_model_configs(
        model_configs, inference_tensor_parallel=inference_tensor_parallel
    )

    return model_configs


def _same_array(arrays: List[np.ndarray]):
    return all(np.array_equal(arrays[0], array) for array in arrays[1:])


def _merge_model_configs_to_first(configs):
    """This method merges the tensor fields for linear config so the config can be used with fewer GPUs.

    The implementation is recursive.
    """
    merged_config = configs[0]

    if isinstance(merged_config, EmbeddingConfig):
        if merged_config.is_local:
            merged_config.weight = np.ascontiguousarray(
                np.concatenate([config.weight for config in configs], axis=0)
            )

    elif isinstance(merged_config, LinearConfig):
        # The scaling factors merge rule is summarized as below:

        # S: all ranks should have the same scaling factor.
        # M: Pick elementwise max among the ranks. Merged shape same as single rank.
        # C: Concat the scaling factors on dim 0. Merged shape == tensor_parallel * original shape.
        # RC: Reshape and concat. This is for QKV handling only. Merged shape == tensor_parallel * original shape.
        # NA: Not valid / present

        # ws: weight scaling factor
        # as: activation scaling factor
        # ps: prequant scaling factor

        # C: Colum Linear
        # R: Row Linear
        # Q: QKV layer

        # F: FP8
        # I: INT8 SQ

        # Merge Rules:
        #     ws  as  ps
        # FQ  M   M   NA
        # FC  M   M   NA
        # FR  M   M   NA
        # IQ  RC  M   S
        # IC  C   M   S
        # IR  M   M   C

        # Handling constants
        for field_name in ["activation_scaling_factor", "weights_scaling_factor"]:
            merged_field_value = getattr(merged_config, field_name)
            if merged_field_value is not None and merged_field_value.size == 1:
                # Scaling factor is a scalar.
                setattr(
                    merged_config,
                    field_name,
                    np.maximum.reduce([getattr(config, field_name) for config in configs]),
                )

        if merged_config.layer_type == LAYER_QKV:
            assert merged_config.linear_type == LINEAR_COLUMN
            out_dim = merged_config.weight.shape[0]
            new_out_dim = out_dim * len(configs)
            in_dim = merged_config.weight.shape[1]
            # For QKV weights, the QKV dim should be the out most dim.
            merged_config.weight = np.ascontiguousarray(
                np.concatenate(
                    [config.weight.reshape(3, out_dim * in_dim // 3) for config in configs], axis=1
                ).reshape(new_out_dim, in_dim)
            )
            for field_name in ["bias", "weights_scaling_factor"]:
                merged_field_value = getattr(merged_config, field_name)
                if merged_field_value is not None:
                    if merged_field_value.shape[0] == out_dim:
                        field_values = [getattr(config, field_name) for config in configs]
                        setattr(
                            merged_config,
                            field_name,
                            np.ascontiguousarray(
                                np.concatenate(
                                    [
                                        field_value.reshape(3, out_dim // 3)
                                        for field_value in field_values
                                    ],
                                    axis=1,
                                ).reshape(new_out_dim)
                            ),
                        )

            # No op for prequant_scaling_factor
            assert _same_array(
                [config.prequant_scaling_factor for config in configs]
            ), f"Failed to merge config {merged_config} with others"

        else:
            # For normal linear layers, we merge column linear on the dim 0 and row on the dim 1
            merge_axis = 0 if merged_config.linear_type == LINEAR_COLUMN else 1
            merged_config.weight = np.ascontiguousarray(
                np.concatenate([config.weight for config in configs], axis=merge_axis)
            )

            # Only cat the bias for column linear.
            if merged_config.linear_type == LINEAR_COLUMN and merged_config.bias is not None:
                merged_config.bias = np.ascontiguousarray(
                    np.concatenate([config.bias for config in configs], axis=0)
                )

            if merged_config.linear_type == LINEAR_COLUMN:
                if (
                    merged_config.weights_scaling_factor is not None
                    and merged_config.weights_scaling_factor.size != 1
                ):
                    # INT8 sq case
                    merged_config.weights_scaling_factor = np.ascontiguousarray(
                        np.concatenate(
                            [config.weights_scaling_factor for config in configs], axis=0
                        )
                    )
                if merged_config.prequant_scaling_factor is not None:
                    assert _same_array(
                        [config.prequant_scaling_factor for config in configs]
                    ), f"Failed to merge config {merged_config} with others"
            else:
                if merged_config.weights_scaling_factor is not None:
                    merged_config.weights_scaling_factor = np.maximum.reduce(
                        [config.weights_scaling_factor for config in configs]
                    )
                if merged_config.prequant_scaling_factor is not None:
                    merged_config.prequant_scaling_factor = np.ascontiguousarray(
                        np.concatenate(
                            [config.prequant_scaling_factor for config in configs], axis=0
                        )
                    )

    elif is_dataclass(merged_config):
        for field in fields(merged_config):
            _merge_model_configs_to_first([getattr(config, field.name) for config in configs])
    elif isinstance(merged_config, list):
        for i in range(len(merged_config)):
            _merge_model_configs_to_first([config[i] for config in configs])


def _merge_embedding(model_configs: List[ModelConfig]):
    """Merges and replicates the embedding weights to all configs."""
    for embedding_name in ["vocab_embedding", "positional_embedding"]:
        embedding_0 = getattr(model_configs[0], embedding_name)
        if embedding_0 and embedding_0.is_local:
            weights = [getattr(config, embedding_name).weight for config in model_configs]
            merged_weight = np.ascontiguousarray(np.concatenate(weights, axis=0))
            for config in model_configs:
                getattr(config, embedding_name).weight = merged_weight
                getattr(config, embedding_name).is_local = False


def _postprocess_model_configs(
    model_configs: List[ModelConfig], inference_tensor_parallel: int = 1
) -> List[ModelConfig]:
    """Postprocesses the model configs with trained tensor parallel to target inference tensor parallel."""
    if inference_tensor_parallel < len(model_configs):
        # Merge the model_configs to target inferencen tensor parallel.
        assert (
            len(model_configs) % inference_tensor_parallel == 0
        ), f"Cannot merge {len(model_configs)} configs to {inference_tensor_parallel}"

        num_configs_per_group = len(model_configs) // inference_tensor_parallel
        merged_model_configs = []
        for i in range(inference_tensor_parallel):
            model_config_slice = model_configs[
                i * num_configs_per_group : (i + 1) * num_configs_per_group
            ]
            _merge_model_configs_to_first(model_config_slice)
            model_config_slice[0].rank = i
            model_config_slice[0].tensor_parallel = inference_tensor_parallel
            merged_model_configs.append(model_config_slice[0])
    else:
        merged_model_configs = model_configs

    # So far we do not support parallel embedding layers yet.
    # We will merge the local embedding weights and replicate it to all ranks for now.
    _merge_embedding(merged_model_configs)

    return merged_model_configs

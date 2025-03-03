# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from typing import List, Optional, Tuple

from omegaconf import DictConfig


def separate_bucket_transcriptions(inference_config: str) -> tuple:
    """
    Separates manifests and audio file paths from different buckets.

    Args:
        inference_config (str): The configuration object for inference.

    Returns:
        tuple: A tuple containing:
            - manifests (list): A list of manifest file paths.
            - tarr_audio_files (list or None): A list of tarred audio file paths or None if
              the dataset is not tarred.
    """

    if hasattr(inference_config.predict_ds, "is_tarred") and inference_config.predict_ds.is_tarred:
        tarred_audio_filepaths = inference_config.predict_ds.tarred_audio_filepaths
        manifest_filepaths = inference_config.predict_ds.manifest_filepath
        if type(tarred_audio_filepaths) != str and len(tarred_audio_filepaths) > 1:
            manifests = []
            tarr_audio_files = []
            for manifest_filepath, tarred_audio_filepath in zip(manifest_filepaths, tarred_audio_filepaths):
                manifests.append(manifest_filepath[0])
                tarr_audio_files.append(tarred_audio_filepath[0])
            return manifests, tarr_audio_files
        else:
            return [manifest_filepaths], [tarred_audio_filepaths]
    else:
        if isinstance(inference_config.predict_ds.manifest_filepath, str):
            return [inference_config.predict_ds.manifest_filepath ], None
        else:
            return inference_config.predict_ds.manifest_filepath, None


def get_transcribed_names(manifest_filepaths: List[str], prefix: str, is_tarred: bool=False) -> List[List[str]]:
    """
    Generates a list of modified file paths by prepending 'transcribed_' to the filenames.
    The use case is for non AIStore datasets

    Args:
        manifest_filepaths (list of str): A list of file paths to be modified.

    Returns:
        list of list of str: A list where each element is a single-item list containing the updated file path.
    Example:
        >>> manifest_filepaths = [
        ...     "/path/to/manifest_1.json",
        ...     "/path/to/manifest_2.json"
        ... ]
        >>> get_transcribed_names(manifest_filepaths)
        [
            ["/path/to/prefix_transcribed_manifest_1.json"],
            ["/path/to/prefix_transcribed_manifest_2.json"]
        ]
    """
    # For manifest_filepath, modify the filenames by prepending 'prefix_transcribed_'
    transcribed_paths = []

    for file_path in manifest_filepaths:
        directory, filename = os.path.split(file_path)
        
        new_filename = (
            f"{prefix}_transcribed_{filename}" if is_tarred 
            else f"{prefix}_transcribed_manifest.json"
        )
        transcribed_paths.append([os.path.join(directory, new_filename)])

    return transcribed_paths


def update_training_sets(
    config: DictConfig,
    updated_manifest_filepaths: List[str],
    updated_tarred_audio_filepaths: Optional[List[str]] = None,
    prefix:str  = ""
) -> Tuple[str, str]:
    """
    Updates the training dataset configuration by adding pseudo-labeled datasets
    to the training paths based on the dataset type.

    Args:
        config (DictConfig): Training config file to be updated.
        updated_manifest_filepaths (List[str]): List of updated manifest file paths to be included.
        updated_tarred_audio_filepaths (Optional[List[str]]): List of updated tarred audio filepaths to be included.

    Returns:
        Tuple[str, str]: A tuple containing:
            - Updated manifest file paths as a string, formatted for Omegaconf.
            - Updated tarred audio file paths as a string, formatted for Omegaconf.
    """
    updated_manifest_filepaths = get_transcribed_names(updated_manifest_filepaths, prefix, is_tarred=config.model.train_ds.get("is_tarred", False))
    manifest_filepath = config.model.train_ds.manifest_filepath

    if updated_tarred_audio_filepaths:
        updated_tarred_audio_filepaths = [[path] for path in updated_tarred_audio_filepaths]

    # Updating the configuration based on dataset types
    if config.model.train_ds.get("is_tarred", False):
        tarred_audio_filepaths = config.model.train_ds.tarred_audio_filepaths
        if isinstance(tarred_audio_filepaths, str):
            updated_tarred_audio_filepaths.append([tarred_audio_filepaths])
            updated_manifest_filepaths.append([manifest_filepath])
        else:
            updated_tarred_audio_filepaths += tarred_audio_filepaths
            updated_manifest_filepaths += manifest_filepath
    else:
        if config.model.train_ds.get("use_lhotse", False):
            if isinstance(manifest_filepath, str):
                updated_manifest_filepaths.append([manifest_filepath])
            else:
                updated_manifest_filepaths += manifest_filepath
        else:
            updated_manifest_filepaths = [item for sublist in updated_manifest_filepaths for item in sublist]
            if isinstance(manifest_filepath, str):
                updated_manifest_filepaths.append(manifest_filepath)
            else:
                updated_manifest_filepaths += manifest_filepath

    # Returning strings formatted for Omegaconf
    return (
        str(updated_manifest_filepaths).replace(", ", ","),
        str(updated_tarred_audio_filepaths).replace(", ", ",") if updated_tarred_audio_filepaths else None,
    )

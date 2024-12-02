import glob
import json
import os
from typing import List, Optional, Tuple, Union

from omegaconf import OmegaConf


def separate_multiple_transcriptions(inference_config: str) -> Tuple[List[str], Optional[List[str]]]:
    """
    Separates and returns the manifest and tarred audio file paths from the configuration.
    This function makes it easier to run transcribe_speech_parallel for each bucket separately

    Args:
        inference_config (str): Path to the inference configuration file.
    Returns:
        Tuple[List[str], Optional[List[str]]]: A tuple containing:
            - A list of manifest file paths.
            - An optional list of tarred audio file paths, or None if not applicable.
    """

    config = OmegaConf.load(inference_config)

    if hasattr(config.predict_ds, "is_tarred") and config.predict_ds.is_tarred:
        tarred_audio_filepaths = config.predict_ds.tarred_audio_filepaths
        manifest_filepaths = config.predict_ds.manifest_filepath
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

        return [config.predict_ds.manifest_filepath], None


def create_transcribed_shard_manifests(
    prediction_filepaths: List[str],
) -> List[str]:
    """
    Processes prediction files and generates transcribed shard manifests.

    This function reads prediction JSON files grouped by `shard_id` from
    specified directories, organizes the entries by shard, and writes the
    results to new JSON manifest files.

    Args:
        prediction_filepaths (List[str]): A list of filepaths to directories
            containing prediction JSON files (named like `predictions_[0-9]*.json`).

    Returns:
        List[str]: A list of filepaths to the created manifest files.
    """
    all_manifest_filepaths = []
    for prediction_filepath in prediction_filepaths:
        max_shard_id = 0
        shard_data = {}
        for full_path in glob.glob(os.path.join(prediction_filepath, "predictions_[0-9]*.json")):
            # Collect data based on their shard id
            with open(full_path, 'r') as f:
                for line in f.readlines():
                    data_entry = json.loads(line)
                    shard_id = data_entry.get("shard_id")
                    max_shard_id = max(max_shard_id, shard_id)
                    shard_data.setdefault(shard_id, []).append(data_entry)

        # Write each shard's data to a new JSON file in the output directory
        for shard_id, entries in shard_data.items():
            output_filename = os.path.join(prediction_filepath, f"transcribed_manifest_{shard_id}.json")
            with open(output_filename, 'w') as f:
                for data_entry in entries:
                    if data_entry['audio_filepath'].endswith(".wav"):
                        json.dump(data_entry, f, ensure_ascii=False)
                        f.write("\n")
        shard_manifest_filepath = os.path.join(
            prediction_filepath, f"transcribed_manifest__OP_0..{max_shard_id}_CL_.json"
        )

        all_manifest_filepaths.append([shard_manifest_filepath])
    return all_manifest_filepaths


def create_transcribed_manifests(
    prediction_filepaths: List[str],
) -> List[str]:
    """
    Renames prediction files to 'transcribed_manifest.json' for each directory
    and returns a list of the new file paths.
    Args:
        prediction_filepaths (List[str]): A list of file paths to directories
            containing the 'predictions_all.json' file.
    Returns:
        List[str]: A list of file paths to the renamed 'transcribed_manifest.json' files.
    """
    all_manifest_filepaths = []

    for prediction_filepath in prediction_filepaths:
        prediction_name = os.path.join(prediction_filepath, "predictions_all.json")
        transcripted_name = os.path.join(prediction_filepath, "transcribed_manifest.json")

        os.rename(prediction_name, transcripted_name)
        all_manifest_filepaths.append(transcripted_name)

    return all_manifest_filepaths


def write_sampled_shard_transcriptions(manifest_filepaths: List[str]) -> List[List[str]]:
    """
    Updates transcriptions by merging predicted shard data and transcribed manifest data.

    This function processes prediction and transcribed manifest files, merges them
    by matching the shard_id and audio file paths. For each shard, the corresponding
    data entries are written to a new file.

    Args:
        manifest_filepaths (List[str]): A list of file paths to directories containing
            prediction and transcribed manifest files.

    Returns:
        List[List[str]]: A list of lists containing the file paths to the generated
            transcribed shard manifest files.
    """
    all_manifest_filepaths = []

    # Process each prediction directory
    for prediction_filepath in manifest_filepaths:
        predicted_shard_data = {}

        # Collect entries from prediction files based on shard id
        for prediction_path in glob.glob(os.path.join(prediction_filepath, "predictions_[0-9]*.json")):
            with open(prediction_path, 'r') as f:
                for line in f:
                    data_entry = json.loads(line)
                    shard_id = data_entry.get("shard_id")
                    audio_filepath = data_entry['audio_filepath']
                    predicted_shard_data.setdefault(shard_id, {})[audio_filepath] = data_entry

        # Collect entries from transcribed manifest files
        all_data_entries = []
        max_shard_id = 0

        for full_path in glob.glob(os.path.join(prediction_filepath, "transcribed_manifest_[0-9]*.json")):

            with open(full_path, 'r') as f:
                for line in f:
                    data_entry = json.loads(line)
                    shard_id = data_entry.get("shard_id")
                    max_shard_id = max(max_shard_id, shard_id)
                    all_data_entries.append(data_entry)

            # Write the merged data to a new manifest file keeping new transcriptions
            output_filename = os.path.join(prediction_filepath, f"transcribed_manifest_{shard_id}.json")
            with open(output_filename, 'w') as f:
                for data_entry in all_data_entries:
                    audio_filepath = data_entry['audio_filepath']
                    # Escape duplicated audio files that end with *dup
                    if audio_filepath.endswith(".wav"):
                        if shard_id in predicted_shard_data and audio_filepath in predicted_shard_data[shard_id]:
                            predicted_data_entry = predicted_shard_data[shard_id][audio_filepath]
                            json.dump(predicted_data_entry, f, ensure_ascii=False)
                        else:
                            json.dump(data_entry, f, ensure_ascii=False)
                        f.write("\n")

        shard_manifest_filepath = os.path.join(
            prediction_filepath, f"transcribed_manifest__OP_0..{max_shard_id}_CL_.json"
        )
        all_manifest_filepaths.append([shard_manifest_filepath])

    return all_manifest_filepaths


def write_sampled_transcriptions(manifest_filepaths: List[str]) -> List[str]:
    """
    Updates transcriptions by merging predicted data with transcribed manifest data.


    This function processes prediction files and transcribed manifest files, merging
    them by matching audio file paths. The merged data is then written to a new file.

    Args:
        manifest_filepaths (List[str]): A list of file paths to directories containing
            prediction and transcribed manifest files.
    Returns:
        List[str]: A list of file paths to the generated transcribed manifest files.
    """
    all_manifest_filepaths = []

    # Process each prediction directory
    for prediction_filepath in manifest_filepaths:
        predicted_data = {}

        # Collect entries from prediction files
        prediction_path = os.path.join(prediction_filepath, "predictions_all.json")
        with open(prediction_path, 'r') as f:
            for line in f:
                data_entry = json.loads(line)
                path = data_entry['audio_filepath']
                predicted_data[path] = data_entry

        # Collect entries from transcribed manifest file
        transcribed_manifest_path = os.path.join(prediction_filepath, "transcribed_manifest.json")
        all_data_entries = []
        with open(transcribed_manifest_path, 'r') as f:
            for line in f:
                data_entry = json.loads(line)
                all_data_entries.append(data_entry)

        # Merge predicted data with transcribed data and write to a new file
        output_filename = os.path.join(prediction_filepath, "transcribed_manifest.json")
        with open(output_filename, 'w') as f:
            for data_entry in all_data_entries:
                audio_filepath = data_entry['audio_filepath']
                if audio_filepath in predicted_data:
                    predicted_data_entry = predicted_data[audio_filepath]
                    json.dump(predicted_data_entry, f, ensure_ascii=False)
                else:
                    json.dump(data_entry, f, ensure_ascii=False)
                f.write("\n")

        all_manifest_filepaths.append(output_filename)

    return all_manifest_filepaths


def update_training_sets(
    merged_config: OmegaConf, final_cache_manifests: list, tarred_audio_filepaths: Union[list, str]
) -> OmegaConf:
    """
    Adds pseudo-labeled sets to the training datasets based on dataset type and
    handles tarred audio files differently. The function updates the 'manifest_filepath'
    and 'tarred_audio_filepaths' fields in the training dataset configuration.
    Args:
        merged_config: The configuration object containing the model and dataset settings.
        final_cache_manifests: A list of paths to the manifest files for the pseudo-labeled data.
        tarred_audio_filepaths: A string or list of tarred audio file paths to be added to the training set.

    Returns:
        merged_config: The updated configuration object with the new training datasets.
    """

    if merged_config.model.train_ds.get("is_tarred", False):
        if isinstance(tarred_audio_filepaths, str):
            if isinstance(merged_config.model.train_ds['tarred_audio_filepaths'], str):
                merged_config.model.train_ds['tarred_audio_filepaths'] = [
                    [merged_config.model.train_ds['tarred_audio_filepaths']],
                    [tarred_audio_filepaths],
                ]
            else:
                merged_config.model.train_ds.tarred_audio_filepaths.append(tarred_audio_filepaths)
        else:
            if isinstance(merged_config.model.train_ds.tarred_audio_filepaths, str):
                merged_config.model.train_ds.tarred_audio_filepaths = [
                    [merged_config.model.train_ds.tarred_audio_filepaths]
                ]
            merged_config.model.train_ds.tarred_audio_filepaths += tarred_audio_filepaths

        if isinstance(merged_config.model.train_ds.manifest_filepath, str):
            merged_config.model.train_ds.manifest_filepath = [merged_config.model.train_ds.manifest_filepath]

        merged_config.model.train_ds.manifest_filepath += final_cache_manifests

    else:
        if isinstance(merged_config.model.train_ds.manifest_filepath, str):
            merged_config.model.train_ds.manifest_filepath = [merged_config.model.train_ds.manifest_filepath]

        if merged_config.model.train_ds.get("use_lhotse", False):
            merged_config.model.train_ds.manifest_filepath = [merged_config.model.train_ds.manifest_filepath]
            merged_config.model.train_ds.manifest_filepath.append(final_cache_manifests)
        else:
            merged_config.model.train_ds.manifest_filepath += final_cache_manifests

        print(f" changed {merged_config.model.train_ds.manifest_filepath}")

    return merged_config


def count_files_for_pseudo_labeling(manifest_filepath: str, is_tarred: bool) -> int:
    """
    Counts the number of files for pseudo-labeling.

    Args:
        manifest_filepath (str): The path to the manifest file(s).
        is_tarred (bool): Flag to determine whether to count files for multiple shard manifests.

    Returns:
        int: The total number of audio files given for pseudo labeling.
    """
    if is_tarred:
        dir_path, filename = os.path.split(manifest_filepath)
        prefix = filename.split('_', 1)[0]
        number_of_files = 0
        for full_path in glob.glob(os.path.join(dir_path, f"{prefix}_[0-9]*.json")):
            with open(full_path, 'r') as f:
                number_of_files += len(f.readlines())
    else:
        with open(manifest_filepath, 'r') as f:
            number_of_files = len(f.readlines())

    return number_of_files

import argparse
import io
import json
import tarfile
from pathlib import Path

import webdataset as wds
from PIL import Image
from tqdm import tqdm


def filter_cambrian737k_dataset(
    data_dir: Path,
    metadata_json_path: Path,
    filtered_json_path: Path,
):
    """
    Filter the Cambrian737k dataset by checking if the image exists.

    Args:
        data_dir: Path to the data directory
        metadata_json_path: Path to the metadata JSON file
        filtered_json_path: Path to the filtered JSON file
    """

    with metadata_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # A snapshot of Cambrian dataset contains tar files.
    # Iterate over all files and folders in the image_root directory.
    for tar_path in data_dir.glob('*.tar'):
        print(f'Extracting {tar_path}...')
        # --- Check for existence of the destination folder ---
        # Derive the expected directory name from the tar filename (e.g., "images.tar" -> "images")
        expected_dir_name = tar_path.stem
        expected_dir_path = data_dir / expected_dir_name

        # Skip the tar file if the directory already exists.
        if expected_dir_path.exists():
            print(f"Directory '{expected_dir_name}' already exists. Skipping extraction.")
            continue

        try:
            with tarfile.open(tar_path, 'r') as tar:
                # Extract all files into the image_root directory.
                tar.extractall(path=data_dir)
            print(f'Successfully extracted {tar_path}.')
        except Exception as e:
            print(f'Error extracting {tar_path}: {e}')

    result = []

    for item in tqdm(data):
        image_path = item.get("image")
        if image_path is not None:
            full_path = data_dir / image_path
            if full_path.exists():
                result.append({
                    "image": image_path,
                    "conversations": item.get("conversations")
                })
            else:
                print(f"Image {image_path} does not exist.")
    print(f"{len(result)} conversations will be saced")

    with filtered_json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Filtering done and saved to {filtered_json_path}.")


def convert_cambrian737k_dataset_to_webdataset(
    data_dir: Path,
    metadata_json_path: Path,
    output_dir: Path,
):
    """
    Convert the Cambrian737k dataset to the webdataset format.

    Args:
        data_dir: Path to the data directory
        metadata_json_path: Path to the metadata JSON file
        output_dir: Path to the output directory
    """

    # Load data
    with metadata_json_path.open('r') as f:
        data = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    with wds.ShardWriter(
        str(output_dir / 'pretrain-%d.tar'), maxcount=10000
    ) as shard_writer:
        for entry in tqdm(data):
            img_path = data_dir / entry["image"]
            if img_path.exists():
                # Image to jpec bytes.
                with Image.open(img_path) as image:
                    buf = io.BytesIO()
                    image.convert("RGB").save(buf, format="JPEG")
                    image_data = buf.getvalue()
                sample = {
                    "__key__": str(entry['id']),
                    "jpg": image_data,
                    "json": json.dumps(entry['conversations']).encode("utf-8"),
                }
                shard_writer.write(sample)

    print("Dataset successfully converted to the webdataset format.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare the Cambrian737k dataset in the webdataset format.")
    parser.add_argument(
        "--data-dir", type=Path,
        default='/datasets/Cambrian737k',
        help="Path to dataset directory.")
    parser.add_argument(
        "--output-dir", type=Path,
        default='/datasets/wds',
        help="Path to the output directory")
    args = parser.parse_args()

    original_metadata_path = args.data_dir / 'Cambrian737k.json'
    filtered_metadata_path = args.data_dir / 'Cambrian737k_filtered.json'

    # Filter the dataset by checking if the image exists.
    filter_cambrian737k_dataset(
        args.data_dir,
        original_metadata_path,
        filtered_metadata_path)

    convert_cambrian737k_dataset_to_webdataset(
        args.data_dir,
        filtered_metadata_path,
        args.output_dir)

import argparse
import io
import json
import tarfile
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

import webdataset as wds
from PIL import Image
from tqdm import tqdm


def process_single_image(entry, data_dir):
    """
    Process a single image entry and return the sample data.

    Args:
        entry: Dictionary containing image path and conversations
        data_dir: Path to the data directory

    Returns:
        Dictionary with processed sample data or None if image doesn't exist
    """
    img_path = data_dir / entry["image"]
    if not img_path.exists():
        return None

    try:
        # Image to JPEG bytes
        with Image.open(img_path) as image:
            with io.BytesIO() as buf:
                image.convert("RGB").save(buf, format="JPEG")
                # getvalue() returns a copy of the buffer content
                image_data = buf.getvalue()

        sample = {
            "__key__": entry["image"],
            "jpg": image_data,
            "json": json.dumps(entry['conversations']).encode("utf-8"),
        }
        return sample
    except Exception as e:
        print(f"Error processing image {entry['image']}: {e}")
        return None


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
        expected_dir_name = tar_path.stem
        expected_dir_path = data_dir / expected_dir_name

        if expected_dir_path.exists():
            print(f"Directory '{expected_dir_name}' already exists. Skipping extraction.")
            continue

        try:
            with tarfile.open(tar_path, 'r') as tar:
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
    num_workers: int | None = None,
):
    """
    Convert the Cambrian737k dataset to the webdataset format using multiprocessing.

    Args:
        data_dir: Path to the data directory
        metadata_json_path: Path to the metadata JSON file
        output_dir: Path to the output directory
        num_workers: Number of worker processes (default: CPU count // 2)
    """
    if num_workers is None:
        # Use half of CPU cores for I/O bound tasks like image processing
        num_workers = max(1, cpu_count() // 2)

    # Load data
    with metadata_json_path.open('r') as f:
        data = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a partial function with data_dir fixed
    process_func = partial(process_single_image, data_dir=data_dir)

    print(f"Processing {len(data)} images using {num_workers} workers...")

    with wds.ShardWriter(
        str(output_dir / 'Cambrian737k-%05d.tar'), maxcount=10000
    ) as shard_writer:
        # Use multiprocessing to process images in parallel
        with Pool(processes=num_workers) as pool:
            # Process images in chunks to avoid memory issues
            chunk_size = max(1, len(data) // (num_workers * 4))

            # Use imap for better memory efficiency and progress tracking
            results = pool.imap(process_func, data, chunksize=chunk_size)

            # Write results to shard writer with progress bar
            for sample in tqdm(results, total=len(data), desc="Processing images"):
                if sample is not None:
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
    parser.add_argument(
        "-f", "--force-filtering", action="store_true",
        help="Force to re-run data filtering in the metadata json file.")
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Number of worker processes for parallel processing (default: CPU count // 2)")
    args = parser.parse_args()

    original_metadata_path = args.data_dir / 'Cambrian737k.json'
    filtered_metadata_path = args.data_dir / 'Cambrian737k_filtered.json'

    # Filter the dataset by checking if the image exists.
    if args.force_filtering or not filtered_metadata_path.exists():
        filter_cambrian737k_dataset(
            args.data_dir,
            original_metadata_path,
            filtered_metadata_path)
    else:
        print(f"Filtered metadata already exists at {filtered_metadata_path}. Skipping filtering.")

    convert_cambrian737k_dataset_to_webdataset(
        args.data_dir,
        filtered_metadata_path,
        args.output_dir,
        args.num_workers)

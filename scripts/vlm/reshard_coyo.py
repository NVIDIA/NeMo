import os
import tarfile
from itertools import islice
from tqdm import tqdm
import glob
import argparse
from multiprocessing import Pool, cpu_count


def chunk(iterable, size):
    """Yield successive chunks of a given size from an iterable."""
    it = iter(iterable)
    return iter(lambda: tuple(islice(it, size)), ())


def merge_tar_chunk(args):
    """Function to merge a chunk of tar files into a single tar file."""
    tar_chunk, output_tar_path, log_errors = args

    # Skip processing if the output tar file already exists
    if os.path.exists(output_tar_path):
        return

    os.makedirs(os.path.dirname(output_tar_path), exist_ok=True)

    # Temporary file for atomic write
    temp_output_path = output_tar_path + ".tmp"

    try:
        with tarfile.open(temp_output_path, "w") as output_tar:
            for input_tar_path in tar_chunk:
                try:
                    with tarfile.open(input_tar_path, "r") as input_tar:
                        for member in input_tar.getmembers():
                            file_data = input_tar.extractfile(member)
                            if file_data:  # Ensure the member is a file
                                output_tar.addfile(member, file_data)
                except Exception as e:
                    if log_errors:
                        print(f"Error processing {input_tar_path}: {e}")

        # Rename temporary file to final file after successful write
        os.rename(temp_output_path, output_tar_path)
    except Exception as e:
        if log_errors:
            print(f"Error creating {output_tar_path}: {e}")
        # Clean up the temporary file if an error occurs
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)


def merge_tar_files(input_dir, output_dir, tars_to_merge, log_errors, tars_per_folder):
    # Use glob to match all tar files in the input directory
    input_pattern = os.path.join(input_dir, "**", "*.tar")
    input_tars = sorted(glob.glob(input_pattern))

    # Check if there are enough input files
    num_input_tars = len(input_tars)
    if num_input_tars == 0:
        print(f"No tar files found in {input_dir}.")
        return

    # Calculate the number of output tar files
    num_output_tars = (num_input_tars + tars_to_merge - 1) // tars_to_merge
    print(f"Found {num_input_tars} input tar files. Creating {num_output_tars} merged tar files...")

    # Prepare arguments for multiprocessing
    tasks = []
    for idx, tar_chunk in enumerate(chunk(input_tars, tars_to_merge)):
        # Determine the folder for the current tar
        folder_index = idx // tars_per_folder
        folder_path = os.path.join(output_dir, f"part_{folder_index:05d}")
        output_tar_path = os.path.join(folder_path, f"merged-{idx:05d}.tar")

        # Only add the task if the file doesn't already exist
        if not os.path.exists(output_tar_path):
            tasks.append((tar_chunk, output_tar_path, log_errors))

    if not tasks:
        print("All output files already exist. Nothing to do.")
        return

    # Use multiprocessing to process chunks in parallel
    num_workers = min(cpu_count(), len(tasks), 80)  # Limit to available CPUs or number of tasks
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap(merge_tar_chunk, tasks), total=len(tasks), desc="Merging"))

    print("Merging complete!")


if __name__ == "__main__":
    # Setup argparse
    parser = argparse.ArgumentParser(
        description="Merge tar files into fewer larger tar files with resume and atomic writes.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input tar files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save merged tar files.")
    parser.add_argument("--tars_to_merge", type=int, required=True,
                        help="Number of input tar files to merge into each output tar file.")
    parser.add_argument("--log_errors", action="store_true", help="Log errors during processing.")
    parser.add_argument("--tars_per_folder", type=int, default=1000,
                        help="Number of merged tar files per output folder.")

    args = parser.parse_args()

    # Call the merging function with parsed arguments
    merge_tar_files(args.input_dir, args.output_dir, args.tars_to_merge, args.log_errors, args.tars_per_folder)

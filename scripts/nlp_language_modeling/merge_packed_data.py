import argparse
import os

import numpy as np
from tqdm import tqdm


def load_packed_arrays(prefix):
    # Load using memory mapping for constant memory usage
    input_ids = np.load(f"{prefix}.input_ids.npy", mmap_mode='r')
    loss_mask = np.load(f"{prefix}.loss_mask.npy", mmap_mode='r')
    seq_start_id = np.load(f"{prefix}.seq_start_id.npy", mmap_mode='r')
    return input_ids, loss_mask, seq_start_id


def merge_packed_arrays(prefixes, output_prefix):
    total_samples = 0
    max_input_len = 0
    max_seq_starts = 0

    arrays_info = []
    for prefix in prefixes:
        print("Loading prefix:", prefix)
        input_ids, loss_mask, seq_start_id = load_packed_arrays(prefix)
        arrays_info.append((prefix, input_ids, loss_mask, seq_start_id))
        total_samples += input_ids.shape[0]
        max_input_len = max(max_input_len, input_ids.shape[1])
        max_seq_starts = max(max_seq_starts, seq_start_id.shape[1])

    print("Initializing merged arrays with total samples:", total_samples)
    # Allocate merged arrays as memory-mapped files
    merged_input_ids = np.lib.format.open_memmap(
        f"{output_prefix}.input_ids.npy", mode='w+', dtype=np.int32, shape=(total_samples, max_input_len)
    )
    merged_loss_mask = np.lib.format.open_memmap(
        f"{output_prefix}.loss_mask.npy", mode='w+', dtype=np.bool_, shape=(total_samples, max_input_len)
    )
    merged_seq_start_id = np.lib.format.open_memmap(
        f"{output_prefix}.seq_start_id.npy", mode='w+', dtype=np.int32, shape=(total_samples, max_seq_starts)
    )
    # Initialize with default values
    merged_input_ids[:] = -1
    merged_loss_mask[:] = True
    merged_seq_start_id[:] = -1

    sample_idx = 0
    for prefix, input_ids, loss_mask, seq_start_id in arrays_info:
        n_samples = input_ids.shape[0]
        print("Re-saving prefix:", prefix)
        for i in tqdm(range(n_samples)):
            curr_input = input_ids[i]
            curr_loss = loss_mask[i]
            curr_seq_start = seq_start_id[i]
            merged_input_ids[sample_idx, : len(curr_input)] = curr_input
            merged_loss_mask[sample_idx, : len(curr_loss)] = curr_loss
            merged_seq_start_id[sample_idx, : len(curr_seq_start)] = curr_seq_start
            sample_idx += 1

    print(f"Merged arrays saved with prefix '{output_prefix}'.")


def main():
    parser = argparse.ArgumentParser(description="Merge multiple packed npy files using constant memory.")
    parser.add_argument(
        "--input_prefixes",
        nargs="+",
        required=True,
        help="List of file prefixes to merge (each prefix should have .input_ids.npy, .loss_mask.npy, .seq_start_id.npy)",
    )
    parser.add_argument("--output_prefix", required=True, help="Output file prefix for the merged arrays")
    args = parser.parse_args()

    merge_packed_arrays(args.input_prefixes, args.output_prefix)


if __name__ == "__main__":
    main()

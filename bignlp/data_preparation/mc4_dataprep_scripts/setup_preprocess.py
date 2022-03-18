import os
import glob
import shutil
import math
import argparse

from prepare import ALL_LANGS

"""
Example usage:
 python setup_preprocess.py \
    --c4-path=<path/to/c4> \
    --soft-link-path=<path/to/save/softlinks> \
    --languages='all' \
    --node-array-size=20 \
    --workers-per-node=8 \
    --worker-mapping-file=<path/to/preprocess_mapping_file>
"""


def split_languages(c4_path, languages, max_split_size, soft_link_path, cleaned_en=False):
    if languages == 'all':
        langs = ALL_LANGS
    else:
        langs = languages.split(',')

    if soft_link_path is None:
        soft_link_path = os.path.join(c4_path, "multilingual_soft_links")
    os.makedirs(soft_link_path, exist_ok=True)

    lang_splits_info = []
    for lang in langs:
        assert lang in ALL_LANGS, f"Language `{lang}` cannot be recognized."
        if lang == 'en' and cleaned_en:
            file_list = sorted(glob.glob(os.path.join(c4_path, f"en/c4-train.*.json.gz")))
            print(" ****** Using cleaned english data.")
        else:
            file_list = sorted(glob.glob(os.path.join(c4_path, f"multilingual/c4-{lang}.tfrecord-*.json.gz")))
        file0 = file_list[0]
        file_size = os.path.getsize(file0) * 1.0 / 1024 ** 3  # convert bytes to GB
        num_files = len(file_list)
        total_size = file_size * num_files
        num_splits = max(2 ** (math.ceil(math.log2(total_size / max_split_size))), 1)
        assert num_files % num_splits == 0, f"Language `{lang}` cannot be properly splitted."
        for ind in range(num_splits):
            lang_split = os.path.join(soft_link_path, "{:s}_{:03d}-{:03d}".format(lang, ind, num_splits))
            os.makedirs(lang_split, exist_ok=True)
            chunk_size = len(file_list) // num_splits  # number of files in each split
            file_chunk = file_list[ind * chunk_size: (ind + 1) * chunk_size]
            for src in file_chunk:
                dst = os.path.join(lang_split, os.path.basename(src))
                if os.path.exists(dst):
                    os.remove(dst)
                os.symlink(src, dst)
            lang_splits_info.append((lang_split, file_size, chunk_size, total_size / num_splits))
    print(f" ****** Prepare workers mapping to preprocess following language splits...")
    for i, (lang_split, _, _, split_size) in enumerate(lang_splits_info):
        print("{:>4d} {:>7.2f}GB  {:s}".format(i + 1, split_size, lang_split))

    return lang_splits_info


def distribute_lang_splits(lang_splits_info, avail_nodes, workers_per_node, max_split_size, worker_mapping_file):
    avail_workers = avail_nodes * workers_per_node
    distributed_splits = [[] for _ in range(avail_workers)]
    distributed_size = [0] * avail_workers
    lang_splits_info.sort(key=lambda x: -x[-1])
    for i, (lang_split, _, _, split_size) in enumerate(lang_splits_info):
        min_ind = distributed_size.index(min(distributed_size))
        distributed_splits[min_ind].append(lang_split)
        distributed_size[min_ind] += split_size

    zipped_lists = zip(distributed_size, distributed_splits)
    sorted_pairs = sorted(zipped_lists)

    tuples = zip(*sorted_pairs)
    distributed_size, distributed_splits = [list(tuple) for tuple in tuples]

    output = "\n".join([",".join(distributed_splits[i]) for i in range(avail_workers)])
    with open(worker_mapping_file, 'w') as file:
        file.write(output)
    print(f" ****** Workers mapping saved to {worker_mapping_file} ...")
    for i in range(avail_workers):
        print("{:>4d} {:>7.2f}GB  {:s}".format(i + 1, distributed_size[i],
                                               ",".join([os.path.basename(split) for split in distributed_splits[i]])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup (m)C4 preprocessing")
    parser.add_argument("--c4-path", help="Path to (m)C4 dataset repo folder", required=True)
    parser.add_argument("--soft-link-path", help="Path to languages soft links for preprocessing")
    parser.add_argument("--languages", help="Specify the language list e.g. `en,es,zh,de,...` or "
                                            "use `all` to preprocess all languages. All specified "
                                            "languages have to be downloaded first", required=True)
    parser.add_argument("--node-array-size", help="Size of node array in download step", required=True,
                        type=int)
    parser.add_argument("--workers-per-node", default=8, help="Number of workers per node in preprocessing step",
                        type=int)
    parser.add_argument("--max-split-size", default=70, help="The language files are distributed in to smaller shards "
                                                             "for preprocessing, the size of each shard is less than "
                                                             "max-split-size. (unit in GB)", type=int)
    parser.add_argument("--worker-mapping-file", help="Where to save worker mapping file", required=True)
    parser.add_argument("--cleaned-en", action="store_true", help="Whether to use cleaned C4 en dataset instead."
                                                                  "of uncleaned mC4 en")
    args = parser.parse_args()

    print(f" ****** Removing git lfs cache files in {args.c4_path} ...")

    # Remove git lfs cached files
    if os.path.exists(os.path.join(args.c4_path, ".git", "lfs")):
        shutil.rmtree(os.path.join(args.c4_path, ".git", "lfs"))
    lang_splits_info = split_languages(args.c4_path, args.languages, args.max_split_size, args.soft_link_path,
                                       args.cleaned_en)
    distribute_lang_splits(lang_splits_info, args.node_array_size, args.workers_per_node, args.max_split_size,
                           args.worker_mapping_file)

import os
import sys
import subprocess
import argparse

"""
Example usage:
 python prepare.py \
    --data-path=<path/to/data/folder> \
    --git-lfs-path=<path/to/git/lfs/folder> \
    --languages='all' \
    --node-array-size=20 \
    --worker-mapping-file=<path/to/download_mapping_file>
"""

ALL_LANGS = [
    "af",
    "am",
    "ar",
    "az",
    "be",
    "bg",
    "bn",
    "ca",
    "co",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "eo",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fr",
    "fy",
    "ga",
    "gd",
    "gl",
    "gu",
    "ha",
    "hi",
    "ht",
    "hu",
    "hy",
    "id",
    "ig",
    "is",
    "it",
    "iw",
    "ja",
    "jv",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "ku",
    "ky",
    "la",
    "lb",
    "lo",
    "lt",
    "lv",
    "mg",
    "mi",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "mt",
    "my",
    "ne",
    "nl",
    "no",
    "ny",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "sd",
    "si",
    "sk",
    "sl",
    "sm",
    "sn",
    "so",
    "sq",
    "sr",
    "st",
    "su",
    "sv",
    "sw",
    "ta",
    "te",
    "tg",
    "th",
    "tr",
    "uk",
    "ur",
    "uz",
    "vi",
    "xh",
    "yi",
    "yo",
    "zh",
    "zu",
    "ceb",
    "fil",
    "haw",
    "hmn",
]

LANG_SPLIT = {
    "en_cleaned": [("en_cleaned", "en/c4-train.*-of-01024.json.gz")],
    "en": [
        ("en0", "multilingual/c4-en.tfrecord-0[01]*.json.gz"),
        ("en1", "multilingual/c4-en.tfrecord-0[23]*.json.gz"),
        ("en2", "multilingual/c4-en.tfrecord-0[45]*.json.gz"),
        ("en3", "multilingual/c4-en.tfrecord-0[67]*.json.gz"),
        ("en4", "multilingual/c4-en.tfrecord-0[89]*.json.gz"),
        ("en5", "multilingual/c4-en.tfrecord-1*.json.gz"),
    ],
    "ru": [
        ("ru0", "multilingual/c4-ru.tfrecord-0[01]*.json.gz"),
        ("ru1", "multilingual/c4-ru.tfrecord-0[234]*.json.gz"),
    ],
}


def setup_git_lfs(git_lfs_path):
    print(f" ****** Setting up git lfs under {git_lfs_path} ...")
    if not os.path.exists(os.path.join(git_lfs_path, "install.sh")):
        os.makedirs(git_lfs_path, exist_ok=True)
        os.system(
            f"cd {git_lfs_path} && "
            f"wget https://github.com/git-lfs/git-lfs/releases/download/v3.0.2/git-lfs-linux-amd64-v3.0.2.tar.gz && "
            f"tar -xvf git-lfs-linux-amd64-v3.0.2.tar.gz"
        )
    os.system(f"cd {git_lfs_path} && ./install.sh")


def prepare_c4_repo(data_path):
    c4_path = os.path.join(data_path, "c4")
    print(f" ****** Preparing (m)C4 dataset repo under {c4_path} ...")
    os.system(
        f"cd {data_path} && "
        f"GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4"
    )
    os.system(f"cd {c4_path} && git lfs install")


def distribute_languages(data_path, languages, avail_nodes, worker_mapping_file, cleaned_en=False):
    if languages == "all":
        langs = ALL_LANGS
    else:
        langs = languages.split(",")

    c4_path = os.path.join(data_path, "c4")
    lang_info = []
    for lang in langs:
        assert lang in ALL_LANGS, f"Language `{lang}` cannot be recognized."
        if lang == "en" and cleaned_en:
            lang = "en_cleaned"
            pattern = f"en/c4-train.00000-of-*.json.gz"
            print(" ****** Using cleaned english data.")
        else:
            pattern = f"multilingual/c4-{lang}.tfrecord-00000-*.json.gz"
        stdout = subprocess.check_output(
            f"cd {c4_path} && git lfs ls-files -s " f"-I '{pattern}'", shell=True
        )
        stdout = stdout.decode("utf-8").split()
        file_name = stdout[2]
        file_size = int(stdout[-2].strip("("))
        num_files = int(file_name.split("-")[-1].split(".")[0])
        if lang in LANG_SPLIT:
            for split, pattern in LANG_SPLIT[lang]:
                num_files = subprocess.check_output(
                    f"cd {c4_path} && " f"git lfs ls-files -I '{pattern}' | wc -l", shell=True
                )
                num_files = int(num_files.decode("utf-8"))
                total_size = file_size * num_files
                lang_info.append((split, file_size, num_files, total_size))
        else:
            total_size = file_size * num_files
            lang_info.append((lang, file_size, num_files, total_size))
    print(f" ****** Prepare workers mapping to download following languages...")
    for i, (lang, _, _, total_size) in enumerate(lang_info):
        print("{:>4d} {:>8.1f}GB  {:s}".format(i + 1, total_size / 1024, lang))

    distributed_langs = [[] for _ in range(avail_nodes)]
    distributed_size = [0] * avail_nodes
    lang_info.sort(key=lambda x: -x[-1])
    for lang, _, _, total_size in lang_info:
        min_ind = distributed_size.index(min(distributed_size))
        distributed_langs[min_ind].append(lang)
        distributed_size[min_ind] += total_size

    output = "\n".join([",".join(distributed_langs[i]) for i in range(avail_nodes)])
    with open(worker_mapping_file, "w") as file:
        file.write(output)
    print(f" ****** Workers mapping saved to {worker_mapping_file} ...")
    for i in range(avail_nodes):
        print(
            "{:>4d} {:>8.1f}GB  {:s}".format(
                i + 1, distributed_size[i] / 1024, ",".join(distributed_langs[i])
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup (m)C4 download")
    parser.add_argument("--data-path", help="Path to data storage folder", required=True)
    parser.add_argument("--git-lfs-path", help="Path to git lfs", required=True)
    parser.add_argument(
        "--languages",
        help="Specify the language list e.g. `en,es,zh,de,...` or "
        "use `all` to download all languages",
        required=True,
    )
    parser.add_argument(
        "--node-array-size", help="Size of node array in download step", required=True, type=int
    )
    parser.add_argument(
        "--worker-mapping-file", help="Where to save worker mapping file", required=True
    )
    parser.add_argument(
        "--cleaned-en",
        action="store_true",
        help="Whether to use cleaned C4 en dataset instead." "of uncleaned mC4 en",
    )
    args = parser.parse_args()
    avail_nodes = args.node_array_size

    setup_git_lfs(args.git_lfs_path)
    prepare_c4_repo(args.data_path)
    distribute_languages(
        args.data_path, args.languages, avail_nodes, args.worker_mapping_file, args.cleaned_en
    )

import argparse
import multiprocessing
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from nemo_text_processing.text_normalization.normalize import Normalizer
from tqdm.auto import tqdm

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.append(f"{BASE_DIR}")

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest


def get_normalizer():
    with tempfile.TemporaryDirectory() as data_dir:
        # data_dir = BASE_DIR / "data" / "normalizer"
        # data_dir.mkdir(parents=True, exist_ok=False)

        normalizer = Normalizer(
            lang="en",
            input_case="cased",
            whitelist=None,
            overwrite_cache=True,
            cache_dir=None,  # str(data_dir / "tts_cache_dir"),
        )
    return normalizer


def normalize(text):
    text_normalizer_call_kwargs = {"verbose": False, "punct_pre_process": True, "punct_post_process": True}
    return normalizer.normalize(text, **text_normalizer_call_kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="original manifest")
    parser.add_argument("--dst", type=str, help="path to save manifest")
    parser.add_argument("--src-key", type=str, default="text")
    parser.add_argument("--dst-key", type=str, default="normalized_text")
    args = parser.parse_args()

    records: List[Dict[str, Any]] = read_manifest(args.src)

    # there is a problem with picking normalizer object
    # you can avoid global var by passing normalizer to each call of normalize(...)
    # but it will be ~1.5x slower, than current approach with global variable
    global normalizer
    normalizer = get_normalizer()

    text_key = args.src_key
    text_normalized_key = args.dst_key
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        texts_normalized = list(tqdm(p.imap(normalize, [record[text_key] for record in records]), total=len(records)))

    for record, text_normalized in zip(records, texts_normalized):
        record[text_normalized_key] = text_normalized

    write_manifest(args.dst, records)


if __name__ == "__main__":
    main()

import json
import multiprocessing
import shutil
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from nemo.utils import logging


def create_audio_file(filepath, sample_rate, duration):
    data = np.random.uniform(-1, 1, size=(sample_rate * duration,))
    # Write out audio as 24bit PCM WAV
    sf.write(filepath, data, sample_rate, subtype='PCM_24')


def process(cfg):
    i = cfg["idx"]
    wav_dir = Path(cfg["wav_dir"])
    sample_rate = cfg["sample_rate"]
    total_duration = cfg["total_duration"]
    sample_duration = cfg["sample_duration"]
    audio_filepath = wav_dir / Path(f"audio_{i}.wav")
    create_audio_file(str(audio_filepath), sample_rate, total_duration)
    item = {
        "audio_filepath": str(audio_filepath.absolute()),
        "label": "0 1",
        "offset": 0.0,
        "duration": sample_duration,
    }
    # time.sleep(0.1)
    return item


def generate_dataset(root_dir, num_samples, sample_duration, total_duration=900, sample_rate=32000):
    root_dir = Path(root_dir)

    if root_dir.is_dir():
        logging.info("Found existing output dir, removing...")
        shutil.rmtree(str(root_dir), ignore_errors=True)

    logging.info("Creating output dir...")
    root_dir.mkdir(parents=True)
    wav_dir = root_dir / Path("wavs")
    wav_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = root_dir / Path("synth_manifest.json")

    candidates = []
    cfg = {
        "wav_dir": wav_dir,
        "sample_rate": sample_rate,
        "sample_duration": sample_duration,
        "total_duration": total_duration,
    }

    logging.info("Start generating data...")
    for i in range(num_samples):
        cfg_i = deepcopy(cfg)
        cfg_i["idx"] = i
        candidates.append(cfg_i)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as p:
        manifest_data = list(tqdm(p.imap(process, candidates), total=len(candidates)))

    total_hours = sample_duration * len(manifest_data) / 3600
    logging.info(f"Generated audio dataset of {total_hours:.2f} hours.")
    with manifest_path.open('w') as fout:
        for item in manifest_data:
            fout.write(f"{json.dumps(item)}\n")

    return str(manifest_path)


def generate_simple_dataset(
    root_dir,
    num_samples,
    sample_duration,
    manifest_name="synth_manifest.json",
    total_duration=900,
    sample_rate=32000,
    num_audios=10,
):
    root_dir = Path(root_dir)

    if root_dir.is_dir():
        logging.info("Found existing output dir, removing...")
        shutil.rmtree(str(root_dir), ignore_errors=True)

    logging.info("Creating output dir...")
    root_dir.mkdir(parents=True)
    wav_dir = root_dir / Path("wavs")
    wav_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = root_dir / Path(manifest_name)

    for i in range(num_audios):
        audio_filepath = Path(wav_dir) / Path(f"audio_{i}.wav")
        create_audio_file(str(audio_filepath), sample_rate, total_duration)

    manifest_data = []

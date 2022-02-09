# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import fnmatch
import functools
import multiprocessing
import pytorch_lightning as pl
import torch
import subprocess
import tarfile
import urllib.request
from pathlib import Path
import contextlib
import glob
import json
import os

from dataclasses import dataclass
from typing import Optional

import wget
from joblib import Parallel, delayed
from omegaconf import OmegaConf


from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.metrics.rnnt_wer import RNNTDecodingConfig
from nemo.utils import logging, model_utils

from tqdm import tqdm

from nemo_text_processing.text_normalization.normalize_with_audio import NormalizerWithAudio

parser = argparse.ArgumentParser(description='Download LibriTTS and create manifests')
parser.add_argument("--data-root", required=True, type=Path)
parser.add_argument("--data-sets", default="dev_clean", type=str)

parser.add_argument("--num-workers", default=4, type=int)

parser.add_argument("--normalization-source", default="dataset", type=str, choices=[None, "dataset", "nemo"])
parser.add_argument("--num-workers-for-normalizer", default=12, type=int)

parser.add_argument("--save-google-normalization-separately", action="store_true", default=False)

parser.add_argument("--pretrained-model", default="stt_en_citrinet_1024", type=str)

parser.add_argument('--whitelist-path', type=str, default=None)
parser.add_argument("--overwrite-cache-dir", action="store_true", default=False)

parser.add_argument('--without-download', action='store_true', default=False)
parser.add_argument('--without-extract', action='store_true', default=False)


args = parser.parse_args()

URLS = {
    'TRAIN_CLEAN_100': "https://www.openslr.org/resources/60/train-clean-100.tar.gz",
    'TRAIN_CLEAN_360': "https://www.openslr.org/resources/60/train-clean-360.tar.gz",
    'TRAIN_OTHER_500': "https://www.openslr.org/resources/60/train-other-500.tar.gz",
    'DEV_CLEAN': "https://www.openslr.org/resources/60/dev-clean.tar.gz",
    'DEV_OTHER': "https://www.openslr.org/resources/60/dev-other.tar.gz",
    'TEST_CLEAN': "https://www.openslr.org/resources/60/test-clean.tar.gz",
    'TEST_OTHER': "https://www.openslr.org/resources/60/test-other.tar.gz",
}


#########################
# Start of copy-paste from examples/asr/transcribe_speech.py

@dataclass
class TranscriptionConfig:
    # Required configs
    model_path: Optional[str] = None  # Path to a .nemo file
    pretrained_name: Optional[str] = None  # Name of a pretrained model
    audio_dir: Optional[str] = None  # Path to a directory which contains audio files
    dataset_manifest: Optional[str] = None  # Path to dataset's JSON manifest

    # General configs
    output_filename: Optional[str] = None
    batch_size: int = 32
    num_workers: int = min(batch_size, os.cpu_count() - 1)

    # Set `cuda` to int to define CUDA device. If 'None', will look for CUDA
    # device anyway, and do inference on CPU only if CUDA device is not found.
    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    amp: bool = False
    audio_type: str = "wav"

    # Recompute model transcription, even if the output folder exists with scores.
    overwrite_transcripts: bool = True

    # Decoding strategy for RNNT models
    rnnt_decoding: RNNTDecodingConfig = RNNTDecodingConfig()


def transcribe_manifest(cfg: TranscriptionConfig) -> TranscriptionConfig:
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    if cfg.model_path is None and cfg.pretrained_name is None:
        raise ValueError("Both cfg.model_path and cfg.pretrained_name cannot be None!")
    if cfg.audio_dir is None and cfg.dataset_manifest is None:
        raise ValueError("Both cfg.audio_dir and cfg.dataset_manifest cannot be None!")

    # setup GPU
    if cfg.cuda is None:
        if torch.cuda.is_available():
            cfg.cuda = 0  # use 0th CUDA device
        else:
            cfg.cuda = -1  # use CPU

    device = torch.device(f'cuda:{cfg.cuda}' if cfg.cuda >= 0 else 'cpu')

    # setup model
    if cfg.model_path is not None:
        # restore model from .nemo file path
        model_cfg = ASRModel.restore_from(restore_path=cfg.model_path, return_config=True)
        classpath = model_cfg.target  # original class path
        imported_class = model_utils.import_class_by_path(classpath)  # type: ASRModel
        logging.info(f"Restoring model : {imported_class.__name__}")
        asr_model = imported_class.restore_from(restore_path=cfg.model_path, map_location=device)  # type: ASRModel
        model_name = os.path.splitext(os.path.basename(cfg.model_path))[0]
    else:
        # restore model by name
        asr_model = ASRModel.from_pretrained(model_name=cfg.pretrained_name, map_location=device)  # type: ASRModel
        model_name = cfg.pretrained_name

    trainer = pl.Trainer(gpus=[cfg.cuda] if cfg.cuda >= 0 else 0)
    asr_model.set_trainer(trainer)
    asr_model = asr_model.eval()

    # Setup decoding strategy
    if hasattr(asr_model, 'change_decoding_strategy'):
        asr_model.change_decoding_strategy(cfg.rnnt_decoding)

    # get audio filenames
    if cfg.audio_dir is not None:
        filepaths = list(glob.glob(os.path.join(cfg.audio_dir, f"*.{cfg.audio_type}")))
    else:
        # get filenames from manifest
        filepaths = []
        with open(cfg.dataset_manifest, 'r') as f:
            for line in f:
                item = json.loads(line)
                filepaths.append(item['audio_filepath'])
    logging.info(f"\nTranscribing {len(filepaths)} files...\n")

    # setup AMP (optional)
    if cfg.amp and torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
        logging.info("AMP enabled!\n")
        autocast = torch.cuda.amp.autocast
    else:

        @contextlib.contextmanager
        def autocast():
            yield

    # Compute output filename
    if cfg.output_filename is None:
        # create default output filename
        if cfg.audio_dir is not None:
            cfg.output_filename = os.path.dirname(os.path.join(cfg.audio_dir, '.')) + '.json'
        else:
            cfg.output_filename = cfg.dataset_manifest.replace('.json', f'_{model_name}.json')

    # if transcripts should not be overwritten, and already exists, skip re-transcription step and return
    if not cfg.overwrite_transcripts and os.path.exists(cfg.output_filename):
        logging.info(
            f"Previous transcripts found at {cfg.output_filename}, and flag `overwrite_transcripts`"
            f"is {cfg.overwrite_transcripts}. Returning without re-transcribing text."
        )

        return cfg

    # transcribe audio
    with autocast():
        with torch.no_grad():
            transcriptions = asr_model.transcribe(filepaths, batch_size=cfg.batch_size)
    logging.info(f"Finished transcribing {len(filepaths)} files !")

    logging.info(f"Writing transcriptions into file: {cfg.output_filename}")

    # if transcriptions form a tuple (from RNNT), extract just "best" hypothesis
    if type(transcriptions) == tuple and len(transcriptions) == 2:
        transcriptions = transcriptions[0]

    # write audio transcriptions
    with open(cfg.output_filename, 'w', encoding='utf-8') as f:
        if cfg.audio_dir is not None:
            for idx, text in enumerate(transcriptions):
                item = {'audio_filepath': filepaths[idx], 'pred_text': text}
                f.write(json.dumps(item) + "\n")
        else:
            with open(cfg.dataset_manifest, 'r') as fr:
                for idx, line in enumerate(fr):
                    item = json.loads(line)
                    item['pred_text'] = transcriptions[idx]
                    f.write(json.dumps(item) + "\n")

    logging.info("Finished writing predictions !")
    return cfg

# End of copy-paste from examples/asr/transcribe_speech.py
##########################


def _normalize_line(normalizer: NormalizerWithAudio, line: str):
    line = json.loads(line)

    normalized_texts = normalizer.normalize(
        text=line["text"],
        n_tagged=100,
        punct_post_process=True,
    )

    normalized_text, _ = normalizer.select_best_match(
        normalized_texts=normalized_texts,
        input_text=line["text"],
        pred_text=line["pred_text"],
        remove_punct=True,
    )
    line["normalized_text"] = normalized_text
    return line


def normalize_manifest(normalizer, manifest_file, num_workers):
    manifest_out = manifest_file.replace('.json', '_normalized.json')

    print(f'Normalizing of {manifest_file}...')
    with open(manifest_file, 'r') as f:
        lines = f.readlines()

    # normalized_lines = []
    # for line in tqdm(lines):
    #     normalized_lines.append(_normalize_line(normalizer, line))
    normalized_lines = Parallel(n_jobs=num_workers)(delayed(_normalize_line)(normalizer, line) for line in tqdm(lines))

    with open(manifest_out, 'w') as f_out:
        for line in normalized_lines:
            f_out.write(json.dumps(line, ensure_ascii=False) + '\n')

    print(f'Normalized version saved at {manifest_out}')


def __maybe_download_file(source_url, destination_path):
    if not destination_path.exists():
        tmp_file_path = destination_path.with_suffix('.tmp')
        urllib.request.urlretrieve(source_url, filename=str(tmp_file_path))
        tmp_file_path.rename(destination_path)


def __extract_file(filepath, data_dir):
    try:
        tar = tarfile.open(filepath)
        tar.extractall(data_dir)
        tar.close()
    except Exception:
        print(f"Error while extracting {filepath}. Already extracted?")


def __process_transcript(file_path: str, normalization_source="dataset"):
    entries = []

    with open(file_path, encoding="utf-8") as fin:
        original_text = fin.readlines()[0].strip()

    norm_text = None
    if normalization_source == "dataset":
        with open(file_path.replace("original.txt", "normalized.txt"), encoding="utf-8") as fin:
            norm_text = fin.readlines()[0].strip()

    wav_file = file_path.replace(".original.txt", ".wav")
    assert os.path.exists(wav_file), f"{wav_file} not found!"

    duration = subprocess.check_output(f"soxi -D {wav_file}", shell=True)

    entity = {'audio_filepath': os.path.abspath(wav_file), 'duration': float(duration), 'text': original_text}
    if norm_text is not None:
        entity['normalized_text'] = norm_text

    entries.append(entity)

    return entries


def __process_data(
        data_folder, manifest_file, num_workers, num_workers_for_normalizer,
        normalization_source="dataset", normalizer=None, pretrained_model=None,
        save_google_normalization_separately=False):
    files = []
    entries = []

    for root, dirnames, filenames in os.walk(data_folder):
        for filename in fnmatch.filter(filenames, '*.original.txt'):
            files.append(os.path.join(root, filename))

    with multiprocessing.Pool(num_workers) as p:
        processing_func = functools.partial(
            __process_transcript,
            normalization_source=normalization_source
        )
        results = p.imap(processing_func, files)
        for result in tqdm(results, total=len(files)):
            entries.extend(result)

    with open(manifest_file, 'w') as fout:
        for m in entries:
            fout.write(json.dumps(m) + '\n')

    if save_google_normalization_separately:
        google_manifest_file = Path(manifest_file).parent / f"{Path(manifest_file).stem}_google.json"
        entries = []
        for p in files:
            with open(p.replace("original.txt", "normalized.txt"), encoding="utf-8") as fin:
                norm_text = fin.readlines()[0].strip()

            entity = {
                'audio_filepath': os.path.abspath(p.replace(".original.txt", ".wav")),
                'normalized_text': norm_text
            }
            entries.append(entity)

        with open(google_manifest_file, 'w') as fout:
            for m in entries:
                fout.write(json.dumps(m) + '\n')

    if normalization_source == "nemo":
        # TODO(oktai15): flags
        output_filename = manifest_file.replace('.json', f'_{pretrained_model}.json')
        cfg = TranscriptionConfig(
            pretrained_name=pretrained_model,
            dataset_manifest=manifest_file,
            output_filename=output_filename,
            num_workers=num_workers,
            cuda=0,
            amp=True,
            overwrite_transcripts=False
        )
        transcribe_manifest(cfg)
        normalize_manifest(normalizer, output_filename, num_workers=num_workers_for_normalizer)

# python scripts/dataset_processing/tts/libritts/get_data.py --data-root=/data_4tb/datasets2 --data-set=test_clean \
# --num-workers=4 --normalization-source nemo --whitelist-path ./nemo_text_processing/text_normalization/en/data/whitelist_lj_speech_libri_tts.tsv \
# --save-google-normalization-separately
def main():
    data_root = args.data_root
    data_sets = args.data_sets
    num_workers = args.num_workers
    num_workers_for_normalizer = args.num_workers if args.num_workers_for_normalizer is None else args.num_workers_for_normalizer

    data_root = data_root / "LibriTTS"
    data_root.mkdir(exist_ok=True, parents=True)

    normalizer = None
    if args.normalization_source == "nemo":
        whitelist_path = args.whitelist_path
        if whitelist_path is None:
            # TODO(oktai15): change the branch after merging to main
            wget.download(
                "https://raw.githubusercontent.com/NVIDIA/NeMo/upd_tts_libritts_get_data/nemo_text_processing/text_normalization/en/data/whitelist_lj_speech_libri_tts.tsv",
                out=str(data_root),
            )
            whitelist_path = data_root / "whitelist_lj_speech_libri_tts.tsv"

        normalizer = NormalizerWithAudio(
            lang="en",
            input_case="cased",
            whitelist=whitelist_path,
            overwrite_cache=args.overwrite_cache_dir,
            cache_dir=data_root / "cache_dir",
        )

    if data_sets == "ALL":
        data_sets = "dev_clean,dev_other,train_clean_100,train_clean_360,train_other_500,test_clean,test_other"
    if data_sets == "mini":
        data_sets = "dev_clean,train_clean_100"
    for data_set in data_sets.split(','):
        filepath = data_root / f"{data_set}.tar.gz"

        if not args.without_download:
            __maybe_download_file(URLS[data_set.upper()], filepath)

        if not args.without_extract:
            # We need to use data_root.parent, because tarred file contains LibriTTS directory
            __extract_file(str(filepath), str(data_root.parent))

        __process_data(
            data_folder=str(data_root / data_set.replace("_", "-")),
            manifest_file=str(data_root / f"{data_set}.json"),
            num_workers=num_workers,
            num_workers_for_normalizer=num_workers_for_normalizer,
            normalization_source=args.normalization_source,
            normalizer=normalizer,
            pretrained_model=args.pretrained_model,
            save_google_normalization_separately=args.save_google_normalization_separately
        )


if __name__ == "__main__":
    main()

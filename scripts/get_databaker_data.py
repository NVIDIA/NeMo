#!/usr/bin/env python

# =============================================================================
# Copyright 2019 Pursuit Engineering Team APAC at NVIDIA. All Rights Reserved.
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
# =============================================================================

import argparse
import glob
import os
import urllib.request

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pypinyin import lazy_pinyin, Style
import librosa
import json
import random
from tqdm import tqdm
from nemo import logging


URLS = {
    'DATABAKER_CSMSC':
    "https://weixinxcxdb.oss-cn-beijing.aliyuncs.com/gwYinPinKu/BZNSYP.rar"
}


def __maybe_download_file(destination, source):
    """
    Downloads source to destination if it doesn't exist.
    If exists, skips download
    Args:
        destination: local filepath
        source: url of resource
    Returns:
    """
    source = URLS[source]
    if not os.path.exists(destination):
        logging.info("{0} does not exist. Downloading ...".format(destination))
        urllib.request.urlretrieve(source, filename=destination + ".tmp")
        os.rename(destination + ".tmp", destination)
        logging.info("Downloaded {0}.".format(destination))
    else:
        logging.info("Destination {0} exists. Skipping.".format(destination))
    return destination


def __extract_rar(rar_path, dest_dir):
    """
    Extracts rar compressed file using unrar.
    If unrar not installed, remind user to install unrar.
    If already decompressed, skips extracting.
    Args:
        rar_path: path of the rar file
        dest_dir: extraction destination dir
    Returns:
    """
    if not os.path.exists(dest_dir):
        if os.system("which unrar > /dev/null") != 0:
            message = "Please install unrar and run the script again.\n" \
                    "On Ubuntu/Debian, run: sudo apt-get install unrar -y"
            logging.info(message)
            exit(1)
        os.makedirs(dest_dir)
        logging.info("Extracting... This might take a few minutes.", flush=True)
        status = os.system(
            "unrar x {0} {1} > /dev/null".format(rar_path, dest_dir))
        if status != 0:
            logging.info("Extraction failed.")
            exit(1)
    else:
        logging.info("Skipping extracting. Data already there {0}.".format(data_dir))


def __convert_waves(wavedir, converted_wavedir, wavename, sr):
    """
    Converts a wav file to target sample rate.
    """
    wavepath = os.path.join(wavedir, wavename)
    converted_wavepath = os.path.join(converted_wavedir, wavename)
    y, sr = librosa.load(wavepath, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)
    librosa.output.write_wav(converted_wavepath, y, sr)
    return wavename, round(duration, 2)


def __is_chinese(_char):
    """
    Checks if a char is a Chinese character.
    """
    if not '\u4e00' <= _char <= '\u9fa5':
        return False
    return True


def __replace_symbols(transcript):
    """
    Replaces Chinese symbols with English symbols.
    """
    transcript = transcript.replace("。", ".")
    transcript = transcript.replace("，", ",")
    transcript = transcript.replace("！", "!")
    transcript = transcript.replace("？", "?")
    return transcript


def __convert_transcript(raw_transcript):
    """
    Converts a Chinese transcript to a Chinese pinyin sequence.
    """
    waveid, raw_trans = raw_transcript.split("\t")[:2]
    wavename = waveid + ".wav"
    symbols = ",.!?"
    # For simplicity, we only retain the Chinese chars and symbols
    trans = ''.join(
        [_char for _char in __replace_symbols(raw_trans) if __is_chinese(
            _char) or _char in symbols])
    pinyin_trans = []
    for pinyin in lazy_pinyin(trans, style=Style.TONE3):
        if pinyin not in symbols and not pinyin[-1].isdigit():
            pinyin_trans.append(pinyin + "0")
        else:
            pinyin_trans.append(pinyin)
    return wavename, " ".join(pinyin_trans)


def __prepare_databaker_csmsc(data_root, train_size, sr=22050):
    """
    Prepare Databaker Chinese Standard Mandarin Speech Copus(10000 Sentences).
    Convert the sample rate of wav files to 22050.
    Generate train manifest json and eval manifest json.
    """
    dataset_name = "DATABAKER_CSMSC"
    copyright_statement = "Chinese Standard Mandarin Speech Copus and its " \
        "download link is provided by Databaker (Beijing) Technology Co.," \
        "Ltd. Supports Non-Commercial use only. \nFor more info about this" \
        " dataset, visit: https://www.data-baker.com/open_source.html"
    logging.info(copyright_statement)
    rar_path = os.path.join(data_root, dataset_name+'.rar')
    dataset_dir = os.path.join(data_root, dataset_name)
    __maybe_download_file(rar_path, dataset_name)
    __extract_rar(rar_path, dataset_dir)
    wavedir = os.path.join(dataset_dir, "Wave")
    wavepaths = glob.glob(os.path.join(wavedir, "*.wav"))
    logging.info("Found {} wav files, converting them to {} HZ sample rate...".format(
        len(wavepaths), sr), flush=True)
    converted_wavedir = os.path.join(dataset_dir, str(sr))
    if not os.path.exists(converted_wavedir):
        os.mkdir(converted_wavedir)
    executor = ProcessPoolExecutor(max_workers=80)
    durations = []
    duration_dict = {}
    for wavepath in wavepaths:
        wavename = os.path.basename(wavepath)
        durations.append(executor.submit(
            partial(
                __convert_waves, wavedir, converted_wavedir, wavename, sr)))
    for duration in tqdm(durations):
        wavename, dur = duration.result()
        duration_dict[wavename] = dur
    del durations
    logging.info("Phoneticizing transcripts...", flush=True)
    transcriptfile = os.path.join(
        dataset_dir, "ProsodyLabeling", "000001-010000.txt")
    with open(transcriptfile, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    raw_transcripts = all_lines[::2]
    pinyin_transcripts = []
    pinyin_transcripts_dist = {}
    for raw_transcript in raw_transcripts:
        pinyin_transcripts.append(executor.submit(
            partial(__convert_transcript, raw_transcript)))
    for pinyin_transcript in tqdm(pinyin_transcripts):
        wavename, pinyin_trans = pinyin_transcript.result()
        pinyin_transcripts_dist[wavename] = pinyin_trans
    del raw_transcripts
    del pinyin_transcripts
    wavenames = list(duration_dict.keys())
    random.shuffle(wavenames)
    train_num = int(len(wavenames) * train_size)
    train_wavenames = wavenames[:train_num]
    eval_wavenames = wavenames[train_num:]
    train_lines = []
    eval_lines = []
    logging.info("Generating Manifest...", flush=True)
    for wavename in tqdm(train_wavenames):
        tmp_dict = {}
        tmp_dict["audio_filepath"] = os.path.join(converted_wavedir, wavename)
        tmp_dict["duration"] = duration_dict[wavename]
        tmp_dict["text"] = pinyin_transcripts_dist[wavename]
        train_lines.append(json.dumps(tmp_dict))
    for wavename in tqdm(eval_wavenames):
        tmp_dict = {}
        tmp_dict["audio_filepath"] = os.path.join(converted_wavedir, wavename)
        tmp_dict["duration"] = duration_dict[wavename]
        tmp_dict["text"] = pinyin_transcripts_dist[wavename]
        eval_lines.append(json.dumps(tmp_dict))
    del duration_dict
    del pinyin_transcripts_dist
    tr_mani = "databaker_csmsc_train.json"
    ev_mani = "databaker_csmsc_eval.json"
    if len(train_lines) > 0:
        with open(os.path.join(data_root, tr_mani), "w", encoding="utf-8") \
                as f:
            for line in train_lines:
                f.write("%s\n" % line)
    if len(eval_lines) > 0:
        with open(os.path.join(data_root, ev_mani), "w", encoding="utf-8") \
                as f:
            for line in eval_lines:
                f.write("%s\n" % line)
    logging.info("Complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Databaker Mandarin TTS Dataset")
    parser.add_argument("--dataset_name", default='databaker_csmsc', type=str)
    parser.add_argument("--data_root", required=True, type=str)
    parser.add_argument("--train_size", type=float, default=0.9)
    args = parser.parse_args()

    if args.train_size > 1 or args.train_size <= 0:
        logging.info("train_size should > 0 and <= 1")

    if not os.path.exists(args.data_root):
        os.makedirs(args.data_root)

    if args.dataset_name == 'databaker_csmsc':
        __prepare_databaker_csmsc(args.data_root, args.train_size)
    else:
        logging.info("Unsupported dataset.")


if __name__ == "__main__":
    main()

#
# USAGE: python get_timit_data.py --data_root=<where timit is>
#        --data_new_root=<where to put new data .json>

import argparse
import fnmatch
import json
import os
import subprocess
import sys
import tarfile
import urllib.request

from sox import Transformer
from tqdm import tqdm

#
# This script proposes to create the *json manifest necessary to use the TIMIT
# dataset. Please note that this is the PHONEME classification task of TIMIT.
#

#
# TIMIT IS NOT FREE, please first dowload the dataset
# https://catalog.ldc.upenn.edu/LDC93S1
#

#
# DEFINE STANDARD PHONEMES MAPPING (to 39)
# Based on: https://github.com/espnet/espnet/tree/master/egs/timit/asr1/conf
# Paper: Speaker-independent phone recognition using hidden Markov models
#

PHN_DICT = {
    "aa": "aa",
    "ae": "ae",
    "ah": "ah",
    "ao": "aa",
    "aw": "aw",
    "ax": "ah",
    "ax-h": "ah",
    "axr": "er",
    "ay": "ay",
    "b": "b",
    "bcl": "sil",
    "ch": "ch",
    "d": "d",
    "dcl": "sil",
    "dh": "dh",
    "dx": "dx",
    "eh": "eh",
    "el": "l",
    "em": "m",
    "en": "n",
    "eng": "ng",
    "epi": "sil",
    "er": "er",
    "ey": "ey",
    "f": "f",
    "g": "g",
    "gcl": "sil",
    "h#": "sil",
    "hh": "hh",
    "hv": "hh",
    "ih": "ih",
    "ix": "ih",
    "iy": "iy",
    "jh": "jh",
    "k": "k",
    "kcl": "sil",
    "l": "l",
    "m": "m",
    "n": "n",
    "ng": "ng",
    "nx": "n",
    "ow": "ow",
    "oy": "oy",
    "p": "p",
    "pau": "sil",
    "pcl": "sil",
    "q": "",
    "r": "r",
    "s": "s",
    "sh": "sh",
    "t": "t",
    "tcl": "sil",
    "th": "th",
    "uh": "uh",
    "uw": "uw",
    "ux": "uw",
    "v": "v",
    "w": "w",
    "y": "y",
    "z": "z",
    "zh": "sh",
    "h#": "",
}

#
# DEFINE STANDARD SPEAKERS LISTS
# Based on: https://github.com/espnet/espnet/tree/master/egs/timit/asr1/conf
#

DEV_LIST = {
    'faks0',
    'fdac1',
    'fjem0',
    'mgwt0',
    'mjar0',
    'mmdb1',
    'mmdm2',
    'mpdf0',
    'fcmh0',
    'fkms0',
    'mbdg0',
    'mbwm0',
    'mcsh0',
    'fadg0',
    'fdms0',
    'fedw0',
    'mgjf0',
    'mglb0',
    'mrtk0',
    'mtaa0',
    'mtdt0',
    'mthc0',
    'mwjg0',
    'fnmr0',
    'frew0',
    'fsem0',
    'mbns0',
    'mmjr0',
    'mdls0',
    'mdlf0',
    'mdvc0',
    'mers0',
    'fmah0',
    'fdrw0',
    'mrcs0',
    'mrjm4',
    'fcal1',
    'mmwh0',
    'fjsj0',
    'majc0',
    'mjsw0',
    'mreb0',
    'fgjd0',
    'fjmg0',
    'mroa0',
    'mteb0',
    'mjfc0',
    'mrjr0',
    'fmml0',
    'mrws1',
}

TEST_LIST = {
    'mdab0',
    'mwbt0',
    'felc0',
    'mtas1',
    'mwew0',
    'fpas0',
    'mjmp0',
    'mlnt0',
    'fpkt0',
    'mlll0',
    'mtls0',
    'fjlm0',
    'mbpm0',
    'mklt0',
    'fnlp0',
    'mcmj0',
    'mjdh0',
    'fmgd0',
    'mgrt0',
    'mnjm0',
    'fdhc0',
    'mjln0',
    'mpam0',
    'fmld0',
}

parser = argparse.ArgumentParser(description='TIMIT data processing')
parser.add_argument("--data_root", required=True, default=None, type=str)
parser.add_argument("--data_new_root", required=True, default=None, type=str)
args = parser.parse_args()


def __process_data(data_folder: str, dst_folder: str):
    """
    Build manifests's json
    Args:
        data_folder: timit root
        dst_folder: where json manifests will be stored

    Returns:

    """

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    files_train = []
    files_dev = []
    files_test = []

    for data_set in ['TRAIN', 'TEST']:
        for r, d, f in os.walk(os.path.join(data_folder, data_set)):
            spk = r.split('/')[-1].lower()
            for filename in fnmatch.filter(f, '*.PHN'):
                if data_set == 'TRAIN':
                    if 'SA' not in filename and 'SA' not in filename:
                        files_train.append(os.path.join(r, filename))
                else:
                    if spk in DEV_LIST and 'SA' not in filename:
                        files_dev.append(os.path.join(r, filename))
                    elif spk in TEST_LIST and 'SA' not in filename:
                        files_test.append(os.path.join(r, filename))

    print("Training samples:" + str(len(files_train)))
    print("Validation samples:" + str(len(files_dev)))
    print("Test samples:" + str(len(files_test)))

    for data_set in ['train', 'dev', 'test']:

        print("Processing: " + data_set)
        entries = []
        if data_set == 'train':
            files = files_train
        elif data_set == 'dev':
            files = files_dev
        else:
            files = files_test

        for transcripts_file in tqdm(files):
            with open(transcripts_file, encoding="utf-8") as fin:

                phn_transcript = ""
                for line in fin:

                    # MAPPING TO THE 39 PHN SET
                    phn = line.split(" ")[2].split("\n")[0]
                    mapped = PHN_DICT[phn]
                    phn_transcript += mapped + " "

                wav_file = transcripts_file.split(".")[0] + ".WAV"
                duration = subprocess.check_output("soxi -D {0}".format(wav_file), shell=True)
                entry = dict()
                entry['audio_filepath'] = os.path.abspath(wav_file)
                entry['duration'] = float(duration)
                entry['text'] = phn_transcript
                entries.append(entry)

        with open(os.path.join(dst_folder, data_set + ".json"), 'w') as fout:
            for m in entries:
                fout.write(json.dumps(m) + '\n')


def main():

    data_root = args.data_root
    data_new_root = args.data_new_root

    __process_data(data_root, data_new_root)
    print('Done!')


if __name__ == "__main__":
    main()

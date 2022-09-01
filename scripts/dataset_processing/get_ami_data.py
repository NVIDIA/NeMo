# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
#
# Download the AMI test dataset used to evaluate Speaker Diarization
# More information here: https://groups.inf.ed.ac.uk/ami/corpus/
# USAGE: python get_ami_data.py --data_root=<where to put data> --manifest_filepath AMItest_input_manifest.json
import argparse
import os

from nemo.collections.asr.parts.utils.manifest_utils import create_manifest

test_set_ids = [
    "EN2002a",
    "EN2002b",
    "EN2002c",
    "EN2002d",
    "ES2004a",
    "ES2004b",
    "ES2004c",
    "ES2004d",
    "ES2014a",
    "ES2014b",
    "ES2014c",
    "ES2014d",
    "IS1009a",
    "IS1009b",
    "IS1009c",
    "IS1009d",
    "TS3003a",
    "TS3003b",
    "TS3003c",
    "TS3003d",
    "TS3007a",
    "TS3007b",
    "TS3007c",
    "TS3007d",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the AMI Test Corpus Dataset for Speaker Diarization")
    parser.add_argument("--manifest_filepath", help="path to output manifest file", type=str, required=True)
    parser.add_argument("--data_root", help="path to output manifest file", type=str, default="ami_dataset")
    args = parser.parse_args()

    data_path = os.path.abspath(args.data_root)
    os.makedirs(data_path, exist_ok=True)
    audio_path = os.path.join(data_path, "audio")
    os.makedirs(audio_path, exist_ok=True)
    rttm_path = os.path.join(data_path, "split_rttms")

    for id in test_set_ids:
        os.system(
            f"wget -P {audio_path} https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus/{id}/audio/{id}.Mix-Headset.wav"
        )

    os.system(
        f"wget -P {data_path} https://raw.githubusercontent.com/tango4j/diarization_annotation/main/AMI_corpus/test/split_rttms.tar.gz"
    )
    os.system(f"tar -xzvf {data_path}/split_rttms.tar.gz -C {data_path}")

    audio_files_path = 'audio_files.txt'
    rttm_files_path = 'rttm_files.txt'
    with open(audio_files_path, 'w') as f:
        f.write('\n'.join(os.path.join(audio_path, p) for p in os.listdir(audio_path)))

    with open(rttm_files_path, 'w') as f:
        f.write('\n'.join(os.path.join(rttm_path, p) for p in os.listdir(rttm_path)))

    create_manifest(
        audio_files_path, args.manifest_filepath, rttm_path=rttm_files_path,
    )

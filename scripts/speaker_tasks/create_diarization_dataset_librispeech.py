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

import argparse
import json
import os
import random
import wave

from nemo.collections.asr.parts.preprocessing.segment import AudioSegment

random.seed(42)

"""
This script creates a synthetic diarization dataset using the LibriSpeech dataset.
"""

#read manifest file, from NeMo/examples/nlp/token_classification/punctuate_capitalize_infer.py
#TODO add support for multiple input manifest files
def read_manifest(manifest):
    manifest_data = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            manifest_data.append(data)
    return manifest_data

#randomly select 2 speaker ids from loaded dict
#TODO make parameterizable
#TODO enforce exclusivity
def get_speaker_ids(list):
    file1 = list[random.randint(0, len(list)-1)]
    file2 = list[random.randint(0, len(list)-1)]

    fn1 = file1['audio_filepath'].split('/')[-1]
    fn2 = file2['audio_filepath'].split('/')[-1]

    speaker_id1 = fn1.split('-')[0]
    speaker_id2 = fn2.split('-')[0]

    return [speaker_id1,speaker_id2]

#get a list of the samples for the two specified speakers
#TODO enforce exclusion within one clip (avoid repetition)?
def get_speaker_samples(file_list, speaker_ids):
    speaker_lists = {'sp1': [], 'sp2': []}
    for file in file_list:
        fn = file['audio_filepath'].split('/')[-1]
        spid = fn.split('-')[0]
        if spid == speaker_ids[0]:
            speaker_lists['sp1'].append(file)
        elif spid == speaker_ids[1]:
            speaker_lists['sp2'].append(file)
    return speaker_lists

#load a sample for the selected speaker id
def load_speaker_sample(speaker_lists, speaker_turn):
    if (speaker_trun == 0):
        speaker_id = 'sp1'
    elif (speaker_trun == 1):
        speaker_id = 'sp2'
    file_id = random.randint(0,len(speaker_lists[speaker_id]))
    file = speaker_lists[speaker_id][file_id]
    return file

#add new entry to dict (to write to output manifest file)
def create_new_entry(new_file, start, speaker_id):
    end = start + new_file['duration']
    return [start,end,speaker_id]

def main(
    input_manifest_filepath, output_manifest_filepath, output_wav_filepath
):
    if os.path.exists(output_manifest_filepath):
        os.remove(output_manifest_filepath)
    if os.path.exists(output_wav_filepath):
        os.remove(output_wav_filepath)

    #load librispeech manifest file
    input_file = read_manifest(input_manifest_filepath)

    #get speaker ids for a given diarization session
    speaker_ids = get_speaker_ids(input_file) #randomly select 2 speaker ids

    #build list of samples for speaker1 / speaker2
    #TODO replace with more efficient sampling method
    speaker_lists = get_speaker_samples(speaker_ids)

    session_length = 60*10 #initially assuming 10 minute meeting length
    running_length = 0

    #assume alternating between speakers 1 & 2
    speaker_turn = 0

    #TODO assuming no silence padding, no overlap
    current_dict = []

    outfile = 'librispeech_diarization_0001'
    with wave.open(output_wav_filepath+outfile+'.wav', 'wb') as wav_out:

        while (running_length < session_length):
            #randomly sample from each speaker (TODO enforce exclusion)
            file = load_speaker_sample(speaker_lists, speaker_turn)
            filepath = file['audio_filepath']

            #load wav file
            #new_wav = AudioSegment.from_file(filepath, target_sr=16000)

            with wave.open(new_file, 'rb') as wav_in:
                if not wav_out.getnframes():
                    wav_out.setparams(wav_in.getparams())
                wav_out.writeframes(wav_in.readframes(wav_in.getnframes()))

            #build dict as you go
            #TODO fixed size dict before loop?
            new_entry = add_new_entry(file, running_length, speaker_ids[speaker_turn])
            current_dict.append(new_entry)

            #update speaker turn
            speaker_turn = (speaker_turn + 1) % 2

    wav_out.close()
    #write manifest file
    labels_to_rttmfile(current_dict, outfile, output_rttm_filepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_manifest_filepath", help="path to input manifest file", type=str, required=True)
    parser.add_argument("--output_rttm_filepath", help="path to output manifest file", type=str, required=True)
    parser.add_argument("--output_wav_filepath", help="path to output wav files", type=str, required=True)
    args = parser.parse_args()

    main(
        args.input_manifest_filepath,
        args.output_manifest_filepath,
        args.output_wav_filepath
    )

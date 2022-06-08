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


import os
import random
import json
import numpy as np
import librosa
import soundfile as sf

from nemo.collections.asr.parts.utils.speaker_utils import labels_to_rttmfile

#from scripts/speaker_tasks/filelist_to_manifest.py - move function?
def read_manifest(manifest):
    data = []
    with open(manifest, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return data

class LibriSpeechGenerator(object):
    """
    Librispeech Diarization Session Generator.

    Args:
        manifest_path (str): Manifest file with paths to librispeech audio files
        sr (int): sampling rate of the audio files
        num_speakers (int): number of unique speakers per diarization session
        session_length (int): length of each diarization session (seconds)
        output_dir (str): output directory
        output_filename (str): output filename for the wav and rttm files
        rng: Random number generator
    """
    def __init__(
        self,
        manifest_path=None,
        sr=16000,
        num_speakers=2,
        session_length=60,
        output_dir='output',
        output_filename='librispeech_diarization',
        rng=None,
    ):
        self._manifest_path = manifest_path
        self._sr = sr
        self._num_speakers = num_speakers
        self._session_length = session_length
        self._output_dir = output_dir
        self._output_filename = output_filename
        self._rng = random.Random() if rng is None else rng

        self._manifest = read_manifest(manifest_path)

    #Get/Set Methods
    def set_session_length(self, new_sl):
        self._session_length = new_sl

    def set_output_filename(self, new_fn):
        self._output_filename = new_fn
    #TODO add more get/set methods here as needed

    #TODO add method to load all parameters from a config file (yaml)

    #randomly select 2 speaker ids from loaded dict
    def get_speaker_ids(self):
        file1 = self._manifest[random.randint(0, len(list)-1)]
        file2 = self._manifest[random.randint(0, len(list)-1)]

        fn1 = file1['audio_filepath'].split('/')[-1]
        fn2 = file2['audio_filepath'].split('/')[-1]

        speaker_id1 = fn1.split('-')[0]
        speaker_id2 = fn2.split('-')[0]

        return [speaker_id1,speaker_id2]

    #get a list of the samples for the two specified speakers
    def get_speaker_samples(self, speaker_ids):
        speaker_lists = {'sp1': [], 'sp2': []}
        for file in self._manifest:
            fn = file['audio_filepath'].split('/')[-1]
            spid = fn.split('-')[0]
            if spid == speaker_ids[0]:
                speaker_lists['sp1'].append(file)
            elif spid == speaker_ids[1]:
                speaker_lists['sp2'].append(file)
        return speaker_lists

    #load a sample for the selected speaker id
    def load_speaker_sample(self, speaker_lists, speaker_turn):
        if (speaker_turn == 0):
            speaker_id = 'sp1'
        elif (speaker_turn == 1):
            speaker_id = 'sp2'
        file_id = random.randint(0,len(speaker_lists[speaker_id])-1)
        file = speaker_lists[speaker_id][file_id]
        return file

    #add new entry to dict (to write to output manifest file)
    def create_new_rttm_entry(self, new_file, start, speaker_id):
        end = start + new_file['duration']
        return str(start) + ' ' + str(end) + ' ' + str(speaker_id)

    def generate_session(self):
        speaker_ids = get_speaker_ids() #randomly select 2 speaker ids
        speaker_lists = get_speaker_samples(speaker_ids) #get list of samples per speaker

        speaker_turn = 0 #assume alternating between speakers 1 & 2
        running_length = 0

        wavpath = os.path.join(self._output_dir, self._output_filename + '.wav')
        array = np.zeros(self._session_length*self._sr)
        manifest_list = []

        while (running_length < self._session_length):
            file = load_speaker_sample(speaker_lists, speaker_turn)
            filepath = file['audio_filepath']
            audio_file, sr = librosa.load(filepath, sr=self._sr)

            duration = file['duration']
            if (running_length + duration) > self._session_length:
                duration = self._session_length - running_length

            start = int(running_length*self._sr)
            length = int(duration*self._sr)
            array[start:start+length] = audio_file[:length]

            new_entry = create_new_rttm_entry(file, running_length, speaker_ids[speaker_turn])
            manifest_list.append(new_entry)

            speaker_turn = (speaker_turn + 1) % 2
            running_length += duration

        sf.write(wavpath, array, sampling_rate)
        labels_to_rttmfile(manifest_list, self._output_filename, self._output_dir)

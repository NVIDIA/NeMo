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
from typing import Dict, List, Optional

import torch

from nemo.collections.asr.parts import collections
from nemo.core.classes import Dataset
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.utils import logging

__all__ = ['AudioToSpeechLabelDataSet']


class AudioToSpeechLabelDataSet(Dataset):
    """Data Layer for general speech classification.
    Module which reads speech recognition with target label. It accepts comma-separated
    JSON manifest files describing the correspondence between wav audio files
    and their target labels. JSON files should be of the following format::
        {"audio_filepath": path_to_wav_0, "duration": time_in_sec_0, "label": \
target_label_0, "offset": offset_in_sec_0}
        ...
        {"audio_filepath": path_to_wav_n, "duration": time_in_sec_n, "label": \
target_label_n, "offset": offset_in_sec_n}
    Args:
        manifest_filepath (str): Dataset parameter.
            Path to JSON containing data.
        labels (list): Dataset parameter.
            List of target classes that can be output by the speaker recognition model.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        trim_silence (bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        load_audio (bool): Dataset parameter.
            Controls whether the dataloader loads the audio signal and
            transcript or just the transcript.
            Defaults to True.
        time_length (float): max seconds to consider in a batch # Pass this only for speaker recognition and VAD task 
        shift_length (float): amount of shift of window for generating the frame for VAD task. in a batch # Pass this only for VAD task during inference.
        
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
        """
        return {
            'audio_signal': NeuralType(
                ('B', 'T'),
                AudioSignal(freq=self._sample_rate)
                if self is not None and hasattr(self, '_sample_rate')
                else AudioSignal(),
            ),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'label': NeuralType(tuple('B'), LabelsType()),
            'label_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(
        self,
        *,
        manifest_filepath: str,
        labels: List[str],
        featurizer,
        min_duration: Optional[float] = 0.1,
        max_duration: Optional[float] = None,
        trim: bool = False,
        load_audio: bool = True,
        time_length: Optional[float] = 8,
        shift_length: Optional[float] = 1,
    ):
        super().__init__()
        self.collection = collections.ASRSpeechLabel(
            manifests_files=manifest_filepath.split(','), min_duration=min_duration, max_duration=max_duration,
        )

        self.featurizer = featurizer
        self.trim = trim
        self.load_audio = load_audio
        self.time_length = time_length 
        self.shift_length = shift_length

        logging.info("Time length considered for collate func is {}".format(time_length))
        logging.info("Shift length considered for collate func is {}".format(time_length))

        self.labels = labels if labels else self.collection.uniq_labels
        self.num_classes = len(self.labels)

        self.label2id, self.id2label = {}, {}
        for label_id, label in enumerate(self.labels):
            self.label2id[label] = label_id
            self.id2label[label_id] = label

        for idx in range(len(self.labels[:5])):
            logging.debug(" label id {} and its mapped label {}".format(idx, self.id2label[idx]))

    def fixed_seq_collate_fn(self, batch):
        """collate batch of audio sig, audio len, tokens, tokens len
        Args:
            batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
                LongTensor):  A tuple of tuples of signal, signal lengths,
                encoded tokens, and encoded tokens length.  This collate func
                assumes the signals are 1d torch tensors (i.e. mono audio).
        """
        fixed_length = self.featurizer.sample_rate * self.time_length
        _, audio_lengths, _, tokens_lengths = zip(*batch)

        has_audio = audio_lengths[0] is not None
        fixed_length = int(min(fixed_length, max(audio_lengths)))

        audio_signal, tokens, new_audio_lengths = [], [], []
        for sig, sig_len, tokens_i, _ in batch:
            if has_audio:
                sig_len = sig_len.item()
                chunck_len = sig_len - fixed_length

                if chunck_len < 0:
                    repeat = fixed_length // sig_len
                    rem = fixed_length % sig_len
                    sub = sig[-rem:] if rem > 0 else torch.tensor([])
                    rep_sig = torch.cat(repeat * [sig])
                    signal = torch.cat((rep_sig, sub))
                    new_audio_lengths.append(torch.tensor(fixed_length))
                else:
                    start_idx = torch.randint(0, chunck_len, (1,)) if chunck_len else torch.tensor(0)
                    end_idx = start_idx + fixed_length
                    signal = sig[start_idx:end_idx]
                    new_audio_lengths.append(torch.tensor(fixed_length))

                audio_signal.append(signal)
            tokens.append(tokens_i)

        if has_audio:
            audio_signal = torch.stack(audio_signal)
            audio_lengths = torch.stack(new_audio_lengths)
        else:
            audio_signal, audio_lengths = None, None
        tokens = torch.stack(tokens)
        tokens_lengths = torch.stack(tokens_lengths)

        return audio_signal, audio_lengths, tokens, tokens_lengths

    def sliced_seq_collate_fn(self, batch):
        """collate batch of audio sig, audio len, tokens, tokens len
        Args:
            batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
                LongTensor):  A tuple of tuples of signal, signal lengths,
                encoded tokens, and encoded tokens length.  This collate func
                assumes the signals are 1d torch tensors (i.e. mono audio).
        """
        slice_length = self.featurizer.sample_rate * self.time_length
        _, audio_lengths, _, tokens_lengths = zip(*batch)
        slice_length = min(slice_length, max(audio_lengths))
        shift = 1 * self.featurizer.sample_rate
        has_audio = audio_lengths[0] is not None

        audio_signal, num_slices, tokens, audio_lengths = [], [], [], []
        for sig, sig_len, tokens_i, _ in batch:
            if has_audio:
                sig_len = sig_len.item()
                slices = sig_len // slice_length
                if slices <= 0:
                    repeat = slice_length // sig_len
                    rem = slice_length % sig_len
                    sub = sig[-rem:] if rem > 0 else torch.tensor([])
                    rep_sig = torch.cat(repeat * [sig])
                    signal = torch.cat((rep_sig, sub))
                    audio_signal.append(signal)
                    num_slices.append(1)  # single embedding
                    tokens.extend([tokens_i] * 1)
                    audio_lengths.extend([slice_length] * 1)
                else:
                    slices = (sig_len - slice_length) // shift + 1
                    for slice_id in range(slices):
                        start_idx = slice_id * shift
                        end_idx = start_idx + slice_length
                        signal = sig[start_idx:end_idx]
                        audio_signal.append(signal)

                    num_slices.append(slices)
                    tokens.extend([tokens_i] * slices)
                    audio_lengths.extend([slice_length] * slices)

        if has_audio:
            audio_signal = torch.stack(audio_signal)
            audio_lengths = torch.tensor(audio_lengths)
        else:
            audio_signal, audio_lengths = None, None
        tokens = torch.stack(tokens)
        tokens_lengths = torch.tensor(num_slices)  # each embedding length

        return audio_signal, audio_lengths, tokens, tokens_lengths

    def vad_frame_seq_collate_fn(self, batch):
        """collate batch of audio sig, audio len, tokens, tokens len
        Args:
            batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
                LongTensor):  A tuple of tuples of signal, signal lengths,
                encoded tokens, and encoded tokens length.  This collate func
                assumes the signals are 1d torch tensors (i.e. mono audio).
                batch size equals to 1.      
        """
        slice_length = int(self.featurizer.sample_rate * self.time_length)
        _, audio_lengths, _, tokens_lengths = zip(*batch)
        slice_length = min(slice_length, max(audio_lengths))
        shift = int(self.featurizer.sample_rate * self.shift_length)
        has_audio = audio_lengths[0] is not None

        audio_signal, num_slices, tokens, audio_lengths = [], [], [], []
      
        append_len = int(slice_length / 2) - 1
        for sig, sig_len, tokens_i, _ in batch:
            start = torch.zeros(append_len) 
            end = torch.zeros(append_len)
            sig = torch.cat((start, sig, end))
            sig_len +=  append_len * 2

            if has_audio:
                slices = (sig_len - slice_length) // shift + 1
                for slice_id in range(slices):
                    start_idx = slice_id * shift
                    end_idx = start_idx + slice_length
                    signal = sig[start_idx:end_idx]
                    audio_signal.append(signal)

                num_slices.append(slices)
                tokens.extend([tokens_i] * slices)
                audio_lengths.extend([slice_length] * slices)
                
        if has_audio:
            audio_signal = torch.stack(audio_signal)
            audio_lengths = torch.tensor(audio_lengths)
        else:
            audio_signal, audio_lengths = None, None
            
        tokens = torch.stack(tokens)
        tokens_lengths = torch.tensor(num_slices)  
        return audio_signal, audio_lengths, tokens, tokens_lengths
    
    def __len__(self):
        return len(self.collection)

    def __getitem__(self, index):
        sample = self.collection[index]
        if self.load_audio:
            offset = sample.offset

            if offset is None:
                offset = 0

            features = self.featurizer.process(
                sample.audio_file, offset=offset, duration=sample.duration, trim=self.trim
            )
            f, fl = features, torch.tensor(features.shape[0]).long()
        else:
            f, fl = None, None

        t = self.label2id[sample.label]
        tl = 1  # For compatibility with collate_fn used later

        return f, fl, torch.tensor(t).long(), torch.tensor(tl).long()

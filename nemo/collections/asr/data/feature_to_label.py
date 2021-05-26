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

from nemo.collections.common.parts.preprocessing import collections
from nemo.core.classes import Dataset
from nemo.core.neural_types import AcousticEncodedRepresentation, LabelsType, LengthsType, NeuralType
from nemo.utils import logging


def _feature_collate_fn(batch):
    """collate batch of feat sig, feat len, tokens, tokens len
    Args:
        batch (FloatTensor, LongTensor, LongTensor, LongTensor):  A tuple of tuples of feature, feature lengths,
               encoded tokens, and encoded tokens length. 
    """
    _, feat_lengths, _, tokens_lengths = zip(*batch)
    feat_signal, tokens = [], []
    for sig, sig_len, tokens_i, tokens_i_len in batch:
        feat_signal.append(sig)
        tokens.append(tokens_i)

    feat_signal = torch.stack(feat_signal)
    feat_lengths = torch.stack(feat_lengths)

    tokens = torch.stack(tokens)
    tokens_lengths = torch.stack(tokens_lengths)

    return feat_signal, feat_lengths, tokens, tokens_lengths


class _FeatureSeqSpeakerLabelDataset(Dataset):
    """
    Dataset that loads tensors via a json file containing paths to feature files, sequences of labels. 
    Each new line is a different sample. Example below:
    and their target labels. JSON files should be of the following format:
        {"feature_filepath": "/path/to/feature_0.p", "seq_label": speakerA speakerB SpeakerA ....} \
        ...
        {"feature_filepath": "/path/to/feature_n.p", "seq_label": target_seq_label_n} 
    target_seq_label_n is the string of sequence of speaker label, separated by space.

    Args:
        manifest_filepath (str): Dataset parameter. Path to JSON containing data.
        labels (Optional[list]): Dataset parameter. List of unique labels collected from all samples.
        feature_loader : Dataset parameter. Feature loader to load (external) feature.       
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
        """
        # TODO output type for external features
        output_types = {
            'external_feat': NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            'feat_length': NeuralType(tuple('B'), LengthsType()),
        }

        if self.is_speaker_emb:
            output_types.update(
                {
                    'embs': NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
                    'embs_length': NeuralType(tuple('B'), LengthsType()),
                    'label': NeuralType(('B', 'T'), LabelsType()),
                    'label_length': NeuralType(tuple('B'), LengthsType()),
                }
            )
        else:
            output_types.update(
                {'label': NeuralType(('B', 'T'), LabelsType()), 'label_length': NeuralType(tuple('B'), LengthsType()),}
            )

        return output_types

    def __init__(
        self, *, manifest_filepath: str, labels: List[str], feature_loader, is_speaker_emb: bool = False,
    ):
        super().__init__()
        self.collection = collections.ASRFeatureSequenceLabel(manifests_files=manifest_filepath.split(','),)

        self.feature_loader = feature_loader
        self.labels = labels if labels else self.collection.uniq_labels
        self.is_speaker_emb = is_speaker_emb

        self.label2id, self.id2label = {}, {}
        for label_id, label in enumerate(self.labels):
            self.label2id[label] = label_id
            self.id2label[label_id] = label

        for idx in range(len(self.labels[:5])):
            logging.debug(" label id {} and its mapped label {}".format(idx, self.id2label[idx]))

    def __len__(self):
        return len(self.collection)

    def __getitem__(self, index):
        sample = self.collection[index]

        features = self.feature_loader.process(sample.feature_file)
        f, fl = features, torch.tensor(features.shape[0]).long()

        t = torch.tensor(sample.seq_label).float()
        tl = torch.tensor(len(sample.seq_label)).long()

        return f, fl, t, tl


class FeatureToSeqSpeakerLabelDataset(_FeatureSeqSpeakerLabelDataset):
    """
    Dataset that loads tensors via a json file containing paths to feature
    files and sequence of speakers. Each new line is a
    different sample. Example below:
    {"feature_filepath": "/path/to/feature_0.p", "seq_label": speakerA speakerB SpeakerA ....} \
    ...
    {"feature_filepath": "/path/to/feature_n.p", "seq_label": target_seq_label_n} 
    target_seq_label_n is the string of sequence of speaker label, separated by space.

    Args:
        manifest_filepath (str): Path to manifest json as described above. Canbe comma-separated paths.
        labels (Optional[list]): String containing all the possible labels to map to
            if None then automatically picks from ASRFeatureSequenceLabel collection.
        feature_loader, Feature load to loader (external) feature.
    
    """

    def _collate_fn(self, batch):
        return _feature_collate_fn(batch)

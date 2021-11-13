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

"""
This script serves two goals:
    (1) Demonstrate how to use NeMo Models outside of PytorchLightning
    (2) Shows example of batch speaker recognition inference    
Usage:
python speaker_reco_infer.py --spkr_model='/path/to/.nemo/file' --train_manifest=/path/to/train/manifest/file'
--test_manifest=/path/to/train/manifest/file' --batch_size=32
train_manifest file is used to map the labels from which model was trained so it is mandatory to 
pass the train manifest file

for finetuning tips see: https://colab.research.google.com/github/NVIDIA/NeMo/blob/main/tutorials/speaker_tasks/Speaker_Recognition_Verification.ipynb
"""


import json
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
from tqdm import tqdm

from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.utils import logging

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


can_gpu = torch.cuda.is_available()


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--spkr_model", type=str, default="ecapa_tdnn", required=True, help="Pass your trained .nemo model",
    )
    parser.add_argument(
        "--train_manifest", type=str, required=True, help="path to train manifest file to match labels"
    )
    parser.add_argument(
        "--test_manifest", type=str, required=True, help="path to test manifest file to perform inference"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    if args.spkr_model.endswith('.nemo'):
        logging.info(f"Using local speaker model from {args.spkr_model}")
        speaker_model = EncDecSpeakerLabelModel.restore_from(restore_path=args.spkr_model)
    else:
        logging.error(f"Please pass a trained .nemo file")
        sys.exit()

    labels = []
    with open(args.train_manifest, 'rb') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            item = json.loads(line)
            labels.append(item['label'])

    labels_map = sorted(set(labels))
    label2id, id2label = {}, {}
    for label_id, label in enumerate(labels_map):
        label2id[label] = label_id
        id2label[label_id] = label

    speaker_model.setup_test_data(
        test_data_layer_params={
            'sample_rate': 16000,
            'manifest_filepath': args.test_manifest,
            'labels': labels_map,
            'batch_size': args.batch_size,
            'trim_silence': False,
            'shuffle': False,
        }
    )
    if can_gpu:
        speaker_model = speaker_model.cuda()
    speaker_model.eval()

    speaker_model.test_dataloader()
    all_labels = []
    all_logits = []
    for test_batch in tqdm(speaker_model.test_dataloader()):
        if can_gpu:
            test_batch = [x.cuda() for x in test_batch]
        with autocast():
            audio_signal, audio_signal_len, labels, _ = test_batch
            logits, _ = speaker_model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)

            all_logits.extend(logits.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_logits, true_labels = np.asarray(all_logits), np.asarray(all_labels)
    infer_labels = all_logits.argmax(axis=1)

    out_manifest = os.path.basename(args.test_manifest).split('.')[0] + '_infer.json'
    out_manifest = os.path.join(os.path.dirname(args.test_manifest), out_manifest)
    with open(args.test_manifest, 'rb') as f1, open(out_manifest, 'w') as f2:
        lines = f1.readlines()
        for idx, line in enumerate(lines):
            line = line.strip()
            item = json.loads(line)
            item['infer'] = id2label[infer_labels[idx]]
            json.dump(item, f2)
            f2.write('\n')

    logging.info("Inference labels have been written to {} manifest file".format(out_manifest))


if __name__ == '__main__':
    main()

# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
This script uses multi-processing to add normalized text into manifest files.

$ python <nemo_root_path>/scripts/dataset_processing/tts/add_pretrained_speaker_embedding.py \
    --src=<data_root_path>/manifest.json \
    --dst=<data_root_path>/manifest.json \
    --feature_dir=<data_root_path>/supplementary_dir \
    --model=titanet_large
"""
import os
import argparse
import multiprocessing
import sys
import tempfile
import torch
from pathlib import Path
from typing import Any, Dict, List

import nemo.collections.asr as nemo_asr
from tqdm import tqdm

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
from nemo.collections.tts.parts.utils.tts_dataset_utils import get_base_dir

def get_pretrained_speaker_verification_model(model):
    model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model)
    model.eval().cuda()
    return model

def save_embedding(index):
    record, emb = records[index], embs[index]
    rel_audio_path = Path(record["audio_filepath"]).relative_to(base_data_dir).with_suffix("")
    rel_audio_path_as_text_id = str(rel_audio_path).replace("/", "_")
    save_path = os.path.join(embedding_path, f'{rel_audio_path_as_text_id}.pt')
    if not os.path.exists(save_path):
        torch.save(emb, save_path)  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", type=str, help="original manifest")
    parser.add_argument("--feature-dir", type=str, help="path to save feature data")
    parser.add_argument("--model", type=str, default="titanet_large")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    
    global records, embs, embedding_path, base_data_dir
    
    embedding_path = os.path.join(args.feature_dir, 'reference_speaker_embedding')
    os.makedirs(embedding_path, exist_ok=True)
    
    records = read_manifest(args.manifest_path)
    base_data_dir = get_base_dir([record["audio_filepath"] for record in records])
    for r in records: r['label'] = r['speaker']
    
    model = get_pretrained_speaker_verification_model(args.model)
    with tempfile.NamedTemporaryFile() as temp:
        write_manifest(temp.name, records)
        embs, *_ = model.batch_inference(manifest_filepath=temp.name,
                                         batch_size=args.batch_size, 
                                         device='cuda')
        embs = torch.from_numpy(embs)
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        _ = list(tqdm(p.imap(save_embedding, range(len(records))), total=len(records)))
    

if __name__ == "__main__":
    main()

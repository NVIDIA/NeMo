# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import fiddle as fdl
import lhotse
import lightning.pytorch as pl
import numpy as np
import torch
import transformers
from datasets import Audio, load_dataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.asr.data.audio_to_text_lhotse import LhotseSpeechToTextBpeDataset
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.speechlm.models.hf_auto_model_for_speech_seq2seq import HFAutoModelForSpeechSeq2Seq


def get_dataset(processor):
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    input_features = []
    for d in ds:
        audio_sample = d["audio"]
        features = processor(audio_sample["array"], sampling_rate=audio_sample["sampling_rate"], return_tensors="pt", text=d["text"])
        input_features.append(features)
        
    return input_features

class AudioDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx]

def prepare_dataset(batch):
    audio_array = [array["array"] for array in batch["audio"]]
    batch = processor(audio_array, text=batch["text"])
    return batch


@dataclass
class DataCollatorCTCWithPadding:
    processor: transformers.AutoProcessor
    padding: Union[bool, str] = "longest"

    def __call__(self, features) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        #print("collate called")

        #print(features)

        input_features = [{"input_features": feature["input_features"][0]} for feature in features]
        label_features = [{"labels": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, padding=self.padding, return_tensors="pt")

        labels_batch = self.processor.feature_extractor.pad(labels=label_features, padding=self.padding, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["labels"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='openai/whisper-large-v3')
    parser.add_argument('--strategy', type=str, default='auto', choices=['auto', 'ddp', 'fsdp'])
    parser.add_argument('--devices', default=1)
    parser.add_argument('--accelerator', default='gpu', choices=['gpu'])
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--model-save-path', type=str, default=None)
    args = parser.parse_args()


    model = HFAutoModelForSpeechSeq2Seq(model_name=args.model)
    processor = model.processor
    tokenizer = model.tokenizer

    train_dataset = get_dataset(processor)
    train_dataset = AudioDataset(train_dataset)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1, collate_fn=DataCollatorCTCWithPadding(processor))


    llm.api.finetune(
        model=model,
        data=train_dataloader,
        trainer=nl.Trainer(
            devices=args.devices,
            max_steps=args.max_steps,
            accelerator=args.accelerator,
            strategy=args.strategy,
            log_every_n_steps=1,
            limit_val_batches=0.0,
            num_sanity_val_steps=0,
            accumulate_grad_batches=10,
            gradient_clip_val=0.5,
            use_distributed_sampler=False,
            callbacks=[],
            logger=None,
        ),
        optim=fdl.build(llm.adam.pytorch_adam_with_flat_lr(lr=1e-5)),
        log=None,
    )



    if args.model_save_path is not None:
        model.save_pretrained(args.model_save_path)
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

import fiddle as fdl
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from datasets import Audio, load_dataset
import transformers
import torch

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.asr.models.hf_auto_model_for_speech_seq2seq import HFAutoModelForSpeechSeq2Seq
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


def get_dataset(dataset_id):
    ds = load_dataset(dataset_id)
    ds = ds.train_test_split(test_size=0.2)
    return ds.remove_columns(["english_transcription", "intent_class", "lang_id"])


def uppercase(dataset):
    return {"transcription": dataset["transcription"].upper()}


def preprocess(dataset, processor):
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
    return dataset.map(uppercase)


def prep_dataset(batch):
    audio = batch["audio"]
    print(audio["sampling_rate"])
    batch = processor(audio["array"], sampling_rate=audio["sampling_rate"], text=batch["transcription"])
    batch["input_length"] = len(batch["input_values"][0])
    return batch


def prepare_dataset(dataset):
    return dataset.map(prep_dataset, remove_columns=dataset.column_names["train"], num_proc=4)


@dataclass
class DataCollatorCTCWithPadding:
    processor: transformers.AutoProcessor
    padding: Union[bool, str] = "longest"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"][0]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")

        labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

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

    #train_dataset = get_dataset("distil-whisper/librispeech_long")
    train_dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    #train_dataset = prepare_dataset(train_dataset)
    print(train_dataset)

    #data_collator = DataCollatorCTCWithPadding(processor=processor, padding="longest")
    #train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16, collate_fn=data_collator)

    '''
    llm.api.finetune(
        model=model,
        data=data,
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
    '''


    if args.model_save_path is not None:
        model.save_pretrained(args.model_save_path)
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
import torch
from datasets import Audio, DatasetDict, load_dataset
from omegaconf import OmegaConf

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.speechlm.models.hf_auto_model_for_speech_seq2seq import HFAutoModelForSpeechSeq2Seq


def get_dataset(feature_extractor):
    common_voice = DatasetDict()

    common_voice["train"] = load_dataset(
        "mozilla-foundation/common_voice_11_0",
        "hi",
        split="train+validation",
    )
    common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="test")
    common_voice = common_voice.remove_columns(
        ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"]
    )

    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
    common_voice = common_voice.map(
        prepare_dataset,
        fn_kwargs={"feature_extractor": feature_extractor},
        remove_columns=common_voice.column_names["train"],
        num_proc=4,
    )
    return common_voice


def prepare_dataset(batch, feature_extractor):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

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

    hf_model = HFAutoModelForSpeechSeq2Seq(model_name=args.model)
    hf_model.configure_model(train=False)
    processor = hf_model.processor
    tokenizer = hf_model.tokenizer

    ds = get_dataset(processor.feature_extractor)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=hf_model.model.config.decoder_start_token_id,
    )

    from torch.utils.data.dataloader import DataLoader

    dl = DataLoader(ds["train"], shuffle=True, batch_size=1, collate_fn=data_collator)

    d = next(iter(dl))
    print(d["input_features"].shape)
    print(d["labels"].shape)

    hf_model.model.generation_config.language = "hindi"
    hf_model.model.generation_config.task = "transcribe"

    hf_model.model.generation_config.forced_decoder_ids = None

    llm.api.finetune(
        model=hf_model,
        data=dl,
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
        hf_model.save_pretrained(args.model_save_path)

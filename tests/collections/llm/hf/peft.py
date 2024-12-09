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
from lightning.pytorch.loggers import WandbLogger
from nemo import lightning as nl
from nemo.collections import llm

def make_dummy_dataset(tokenizer, seq_len, batch_size, n=100):
    data = {'text': "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nThe sky is usually clear above the desert and the sunshine duration is extremely high everywhere in the Sahara. Most of the desert enjoys more than 3,600 h of bright sunshine annually or over 82% of the time and a wide area in the eastern part experiences in excess of 4,000 h of bright sunshine a year or over 91% of the time, and the highest values are very close to the theoretical maximum value. A value of 4,300 h or 98% of the time would be recorded in Upper Egypt (Aswan, Luxor) and in the Nubian Desert (Wadi Halfa). The annual average direct solar irradiation is around 2,800 kWh/(m2 year) in the Great Desert. The Sahara has a huge potential for solar energy production. The constantly high position of the sun, the extremely low relative humidity, the lack of vegetation and rainfall make the Great Desert the hottest continuously large area worldwide and certainly the hottest place on Earth during summertime in some spots. The average high temperature exceeds 38 °C (100.4 °F) - 40 °C (104 °F) during the hottest month nearly everywhere in the desert except at very high mountainous areas. The highest officially recorded average high temperature was 47 °C (116.6 °F) in a remote desert town in the Algerian Desert called Bou Bernous with an elevation of 378 meters above sea level. It's the world's highest recorded average high temperature and only Death Valley, California rivals it. Other hot spots in Algeria such as Adrar, Timimoun, In Salah, Ouallene, Aoulef, Reggane with an elevation between 200 and 400 meters above sea level get slightly lower summer average highs around 46 °C (114.8 °F) during the hottest months of the year. Salah, well known in Algeria for its extreme heat, has an average high temperature of 43.8 °C (110.8 °F), 46.4 °C (115.5 °F), 45.5 (113.9 °F). Furthermore, 41.9 °C (107.4 °F) in June, July, August and September. In fact, there are even hotter spots in the Sahara, but they are located in extremely remote areas, especially in the Azalai, lying in northern Mali. The major part of the desert experiences around 3 – 5 months when the average high strictly exceeds 40 °C (104 °F). The southern central part of the desert experiences up to 6 – 7 months when the average high temperature strictly exceeds 40 °C (104 °F) which shows the constancy and the length of the really hot season in the Sahara. Some examples of this are Bilma, Niger and Faya-Largeau, Chad. The annual average daily temperature exceeds 20 °C (68 °F) everywhere and can approach 30 °C (86 °F) in the hottest regions year-round. However, most of the desert has a value in excess of 25 °C (77 °F). The sand and ground temperatures are even more extreme. During daytime, the sand temperature is extremely high as it can easily reach 80 °C (176 °F) or more. A sand temperature of 83.5 °C (182.3 °F) has been recorded in Port Sudan. Ground temperatures of 72 °C (161.6 °F) have been recorded in the Adrar of Mauritania and a value of 75 °C (167 °F) has been measured in Borkou, northern Chad. Due to lack of cloud cover and very low humidity, the desert usually features high diurnal temperature variations between days and nights. However, it's a myth that the nights are cold after extremely hot days in the Sahara. The average diurnal temperature range is typically between 13 °C (55.4 °F) and 20 °C (68 °F). The lowest values are found along the coastal regions due to high humidity and are often even lower than 10 °C (50 °F), while the highest values are found in inland desert areas where the humidity is the lowest, mainly in the southern Sahara. Still, it's true that winter nights can be cold as it can drop to the freezing point and even below, especially in high-elevation areas.\n\n### Input:\nWhat percent of time is the sun generally over most of the   desert?\n\n### Response:\n82% of the timeds = endoftext|>"}

    def fmt(example):
        nvtx.push_range("tokenize", domain="model")
        ans = tokenizer(example['text'])
        tokens = ans['input_ids'][:seq_len]
        nvtx.pop_range(domain="model")
        return {'tokens': tokens, 'labels': tokens[1:] + [tokens[-1]], }

    class CachedFormatter:
        def __init__(self, fn):
            self.cache = None
            self.fn = fn

        def __call__(self, *args, **kwargs):
            if self.cache is None:
                self.cache = self.fn(*args, **kwargs)
            return self.cache

    from datasets import load_dataset

    dataset = Dataset.from_dict({"text": [data['text'] for _ in range(n)]})
    dataset = dataset.map(CachedFormatter(fmt), batched = False, batch_size = batch_size)
    return dataset


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--strategy', type=str, default='auto', choices=['auto', 'ddp', 'fsdp'])
    parser.add_argument('--devices', default=1)
    parser.add_argument('--accelerator', default='gpu', choices=['gpu'])
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--wandb-project', type=str, default=None)
    args = parser.parse_args()

    wandb = None
    if args.wandb_project is not None:
        model = '_'.join(args.model.split('/')[-2:])
        wandb = WandbLogger(
            project=args.wandb_project,
            name=f'{model}_dev{args.devices}_strat_{args.strategy}',
        )
    grad_clip = 0.5
    if args.strategy == 'fsdp':
        # See: https://github.com/Lightning-AI/pytorch-lightning/blob/8ad3e29816a63d8ce5c00ac104b14729a4176f4f/src/lightning/pytorch/plugins/precision/fsdp.py#L81
        grad_clip = None
    use_dist_samp = False
    tokenizer = llm.HFAutoModelForCausalLM.configure_tokenizer(args.model)

    llm.api.finetune(
        model=llm.HFAutoModelForCausalLM(args.model),
        data=llm.HFDatasetDataModule(
            mk_hf_dataset(tokenizer.tokenizer), pad_token_id=tokenizer.tokenizer.eos_token_id
        ),
        trainer=nl.Trainer(
            devices=args.devices,
            max_steps=args.max_steps,
            accelerator=args.accelerator,
            strategy=args.strategy,
            log_every_n_steps=1,
            limit_val_batches=0.0,
            num_sanity_val_steps=0,
            accumulate_grad_batches=10,
            gradient_clip_val=grad_clip,
            use_distributed_sampler=use_dist_samp,
            logger=wandb,
        ),
        optim=fdl.build(llm.adam.pytorch_adam_with_flat_lr(lr=1e-5)),
        log=None,
        peft=llm.peft.LoRA(
            target_modules=['*_proj'],
            dim=32,
        ),
    )

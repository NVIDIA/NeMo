# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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


from argparse import ArgumentParser

import torch
from pytorch_lightning.trainer.trainer import Trainer
from torch.utils.data import DataLoader
from omegaconf.omegaconf import OmegaConf, open_dict

"""
Usage:
    a. If you need to run model on a few prompts from the file:
        python megatron_gpt_eval.py \
            --model_file=PATH_TO_MODEL \
            --path_to_file=PATH_TO_FILE \
            --tokens_to_generate=32 \
            --prompt .

    b. If you need to run model on a prompt from the CLI:
        python megatron_gpt_eval.py \
            --model_file=PATH_TO_MODEL \
            --tokens_to_generate=32 \
            --prompt=YOUR_PROMPT
"""

from nemo.collections.nlp.data.language_modeling.megatron.gpt_request_dataset import GPTRequestDataset
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.megatron_utils import compute_model_parallel_rank
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin
from nemo.utils import logging
from nemo.utils.app_state import AppState

assert torch.cuda.is_available()

# +
precision = 16
tensor_model_parallel_size = 1
model_file = "prompt_tuned_megatron_gpt3.nemo"

# cast precision to int if 32 or 16
if precision in ["32", "16"]:
    precision = int(float(precision))

# trainer required for restoring model parallel models
trainer = Trainer(plugins=NLPDDPPlugin(), gpus=tensor_model_parallel_size, precision=precision)

app_state = AppState()

if tensor_model_parallel_size is not None and tensor_model_parallel_size > 1:
    app_state.model_parallel_size = tensor_model_parallel_size
    app_state.model_parallel_rank = compute_model_parallel_rank(trainer.local_rank, app_state.model_parallel_size)

cfg = OmegaConf.load("conf/megatron_gpt_config.yaml")

# Set current model params
cfg.model.encoder_seq_length = 2048

# Set prompt tuning params
cfg.model.optim.lr = 2e-4
cfg.model.optim.sched.min_lr = 2e-6
cfg.model.use_soft_prompts = True
cfg.model.prompt_length = 10
cfg.model.data.train_ds = 'prompt_tuning_ner_train.json'
cfg.model.data.valid_ds = 'prompt_tuning_ner_val.json'
cfg.model.data.test_ds = 'prompt_tuning_ner_test.json'
cfg.model.data.batch_size = 32
cfg.model.data.data_prefix = None
cfg.model.existing_prompt_tags = ["NER-Yes-No", "NER-Complete"]
cfg.model.optim.sched.warmup_steps = 50
cfg.model.optim.sched.constant_steps = 100
cfg.trainer.max_steps = 200
    
model = MegatronGPTModel.restore_from(model_file, cfg.model, trainer=trainer)

model.freeze()

# +
request = [
            {
                "prompt_tag": "NER-Yes-No",
                "prompt": 'are there entities in: "Fine needle aspiration cytology of primary thyroid lymphoma: a report of ten cases" answer: "',
                "tokens_to_generate": 4,
                "stop_after_sentence": False,
            },
            {
                "prompt_tag": "NER-Yes-No",
                "prompt": 'are there entities in: "All the patients received combination chemotherapy (CHOP regime) with local radiotherapy.  Five patients are alive and are free of disease till date, whereas, two patients died of the disease." answer: "',
                "tokens_to_generate": 4,
                "stop_after_sentence": False,
            },
            {
                "prompt_tag": "NER-Complete",
                "prompt": 'find entities: "All the patients received combination chemotherapy (CHOP regime) with local radiotherapy.  Five patients are alive and are free of disease till date, whereas, two patients died of the disease." answer: "',
                "tokens_to_generate": 64,
                "stop_after_sentence": True,
            },
            {
                "prompt_tag": "NER-Yes-No",
                "prompt": 'are there entities in: "Over the last two decades there has been rapid progress in synthetic organic chemistry associated with the search for new organic compound derivatives with desirable properties.  Such compounds are widely used in the pharmaceutical industry.  Among the several FDA approved pharmaceutical drugs, the pyrazole core is found in rimonabant (1), and celecoxib (2) (Figure\u00a0 1) [1]." answer: "',
                "tokens_to_generate": 4,
                "stop_after_sentence": True,
            },
            {
                "prompt_tag": "NER-Complete",
                "prompt": 'find entities: "Over the last two decades there has been rapid progress in synthetic organic chemistry associated with the search for new organic compound derivatives with desirable properties.  Such compounds are widely used in the pharmaceutical industry.  Among the several FDA approved pharmaceutical drugs, the pyrazole core is found in rimonabant (1), and celecoxib (2) (Figure\u00a0 1) [1]." answer: "',
                "tokens_to_generate": 64,
                "stop_after_sentence": True,
            },
            {
                "prompt_tag": "NER-Yes-No",
                "prompt": 'are there entities in: "Downregulation of survivin expression and concomitant induction of apoptosis by celecoxib and its non-cyclooxygenase-2-inhibitory analog, dimethyl-celecoxib (DMC), in tumor cells in vitro and in vivo"',
                "tokens_to_generate": 4,
                "stop_after_sentence": True,
            },
            {
                "prompt_tag": "NER-Complete",
                "prompt": 'find entities: "Downregulation of survivin expression and concomitant induction of apoptosis by celecoxib and its non-cyclooxygenase-2-inhibitory analog, dimethyl-celecoxib (DMC), in tumor cells in vitro and in vivo" answer: "',
                "tokens_to_generate": 64,
                "stop_after_sentence": True,
            },
            {
                "prompt_tag": "NER-Yes-No",
                "prompt": 'are there entities in: "Each progeny DEN-4 sequence was compared with the parent DEN-4 DNA sequence (GenBank accession number: AF375822).  The results in Table 2 revealed that there were 18 nucleotide mutations resulting in 13 amino acid changes in DEN-4 propagated in Vero cells using serum-containing medium, and 11 nucleotide mutations resulting in 6 amino acid changes in DEN-4 propagated in Vero cells grown in serum-free medium." answer: "',
                "tokens_to_generate": 4,
                "stop_after_sentence": True,
            },
            {
                "prompt_tag": "NER-Complete",
                "prompt": 'find entities: "Each progeny DEN-4 sequence was compared with the parent DEN-4 DNA sequence (GenBank accession number: AF375822).  The results in Table 2 revealed that there were 18 nucleotide mutations resulting in 13 amino acid changes in DEN-4 propagated in Vero cells using serum-containing medium, and 11 nucleotide mutations resulting in 6 amino acid changes in DEN-4 propagated in Vero cells grown in serum-free medium." answer: "',
                "tokens_to_generate": 64,
                "stop_after_sentence": True,
            }
        ]

dataset = GPTRequestDataset(request, model.tokenizer)
request_dl = DataLoader(dataset)
response = trainer.predict(model, request_dl)
# -

for res in response[0]:
    print(res['completion']['text'])
    print('\n')

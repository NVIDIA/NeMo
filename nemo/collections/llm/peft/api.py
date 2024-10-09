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

from nemo.collections.llm.peft.lora import LoRA
from nemo.collections.llm.utils import factory
from nemo.lightning.pytorch.callbacks.peft import PEFT


import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Optional, Union

import nemo_run as run
import pytorch_lightning as pl
from typing_extensions import Annotated

from nemo.lightning import AutoResume, NeMoLogger, OptimizerModule, Trainer, io
from nemo.lightning.pytorch.callbacks import PEFT, ModelTransform
from nemo.utils import logging
from nemo.lightning.io import load_context, ModelConnector
from nemo.collections.llm.api import _set_with_io
from nemo.collections import llm




@factory
def gpt_lora() -> PEFT:
    return LoRA()

def merge_lora(
    model: pl.LightningModule,
    trainer: Trainer,
    log: Annotated[Optional[NeMoLogger], run.Config[NeMoLogger]] = None, 
    resume: Annotated[Optional[AutoResume], run.Config[AutoResume]] = None,
):
    _log = log or NeMoLogger()
    #logger will setup paths in trainer
    app_state = _log.setup(
        trainer,
        resume_if_exists=getattr(resume, "resume_if_exists", False),
        task_config=None,
    )
    #resume will import hf LLM to default path if it doesn't already exists.
    #if exists -> ok; 
    # if doesnt exist -> will download to a new(maybe default) dir; 
    #                 if new dir == old dir -> ok, otherwise throw error                         
    resume.setup(trainer, model)
    lora = load_context(resume.get_context_path(), "model.model_transform")
    if lora:
        _set_with_io(model, "model_transform", lora)
        trainer.callbacks.append(lora)
    import pdb; pdb.set_trace()
    #need to init the lora transform from checkpoint dir

    predict_dataloader = llm.SquadDataModule(seq_length=2048, micro_batch_size=2, global_batch_size=8, num_workers=0)
    trainer.predict(model, dataloaders=predict_dataloader)
   

__all__ = ["gpt_lora",
           "merge_lora"]

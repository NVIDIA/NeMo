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
from typing import Any, Callable, Optional, Union, Iterator, Dict

import nemo_run as run
import pytorch_lightning as pl
from typing_extensions import Annotated

from nemo.lightning import AutoResume, NeMoLogger, OptimizerModule, Trainer, io
from nemo.lightning.pytorch.callbacks import PEFT, ModelTransform
from nemo.utils import logging
from nemo.lightning.io import load_context, ModelConnector
from nemo.collections.llm.api import _set_with_io
from nemo.collections import llm
from pytorch_lightning.loops import _PredictionLoop 
from nemo.utils.get_rank import is_global_rank_zero
import torch
from pytorch_lightning.trainer.states import TrainerFn





@factory
def gpt_lora() -> PEFT:
    return LoRA()

def merge_lora(
    model: pl.LightningModule,
    trainer: Trainer,
    output_path: str,
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
    else:
        raise("Cannot find LoRA config")

    predict_dataloader = llm.SquadDataModule(seq_length=2048, micro_batch_size=2, global_batch_size=8, num_workers=0)
    class LoRAMergeLoop(_PredictionLoop):  #PredictionLoop is internal now X_X
        def __init__(self, trainer, inference_mode: bool = True):
            super().__init__(trainer, inference_mode)

        def on_run_start(self):
            print("Start merging lora")
            self._on_predict_start() # enter PEFT load ckpt for the second time, load adapter state dict

        def _predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int, dataloader_iter: Optional[Iterator]):
            import pdb; pdb.set_trace()

            if trainer.state.fn == TrainerFn.PREDICTING:            
                base_state_dict = {k:v for k,v in trainer.model.state_dict().items() if 'adapter' not in k and 'extra_state' not in k }
                lora_sharded_dict = {k:v.data.data for k, v in trainer.model.sharded_state_dict().items() if 'adapter' in k and 'extra_state' not in k}
                merged_weights = self._merge_lora_weights(base_model_state_dict = base_state_dict, 
                                        lora_state_dict = lora_sharded_dict, 
                                        num_layers = trainer.model._modules['0'].config.num_layers, 
                                        tp_size = trainer.strategy.tensor_model_parallel_size,
                                        rank =torch.distributed.get_rank())
            import pdb; pdb.set_trace()
            #trainer.model.load_state_dict(merged_weights) cannot load, keys dont match after model.walk. TODO: Directly dump state dict without model.load
            #We cannot reuse peft.py save_checkpoint because it saves adapter weights only
            trainer.strategy.checkpoint_io.save_checkpoint(trainer.model.sharded_state_dict(), output_path)
            if is_global_rank_zero():
                trainer.model.io_dump(output_path)
            print("Dump state here")
            
        def on_run_end(self):
            print("Some checks here")

        def _merge_lora_weights(self, base_model_state_dict: Dict[str, Any],
                                lora_state_dict: Dict[str, Any],
                                num_layers: int,
                                tp_size: int,
                                rank: int):
            mcore_layer_to_lora = {}
            """
            'self_attention.linear_qkv.adapter.linear_in.weight' 
            'self_attention.linear_qkv.adapter.linear_out.weight', 
            'self_attention.linear_proj.adapter.linear_in.weight'
            'self_attention.linear_proj.adapter.linear_out.weight',
            'mlp.linear_fc1.adapter.linear_in.weight',
            'mlp.linear_fc1.adapter.linear_out.weight', 
            'mlp.linear_fc2.adapter.linear_in.weight',
            'mlp.linear_fc2.adapter.linear_out.weight', 
            """

            mcore_layer_to_lora["attention_qkv"] = {
                "base_model_layer": "self_attention.linear_qkv.weight",
                "lora_in": "self_attention.linear_qkv.adapter.linear_in.weight",
                "lora_out": "self_attention.linear_qkv.adapter.linear_out.weight",
            }
            mcore_layer_to_lora["attention_dense"] = {
                "base_model_layer": "self_attention.linear_proj.weight",
                "lora_in": "self_attention.linear_proj.adapter.linear_in.weight",
                "lora_out": "self_attention.linear_proj.adapter.linear_out.weight",
            }
            mcore_layer_to_lora["mlp_fc1"] = {
                "base_model_layer": "mlp.linear_fc1.weight",
                "lora_in": "mlp.linear_fc1.adapter.linear_in.weight",
                "lora_out": "mlp.linear_fc1.adapter.linear_out.weight",
            }
            mcore_layer_to_lora["mlp_fc2"] = {
                "base_model_layer": "mlp.linear_fc2.weight",
                "lora_in": "mlp.linear_fc2.adapter.linear_in.weight",
                "lora_out": "mlp.linear_fc2.adapter.linear_out.weight",
            }

            for nl in range(num_layers):
                for key in mcore_layer_to_lora.keys():
                    ##TODO: prefix should be model or module or 0.module?
                    key_base = f'0.module.decoder.layers.{nl}.{mcore_layer_to_lora[key]["base_model_layer"]}'
                    key_lora_in = f'module.decoder.layers.{nl}.{mcore_layer_to_lora[key]["lora_in"]}'
                    key_lora_out = f'module.decoder.layers.{nl}.{mcore_layer_to_lora[key]["lora_out"]}'
                    if key_lora_in in lora_state_dict and key_lora_out in lora_state_dict:
                        if tp_size > 1:
                            gathered_lora_in = [torch.zeros_like(lora_state_dict[key_lora_in]) for _ in range(tp_size)]
                            gathered_lora_out = [torch.zeros_like(lora_state_dict[key_lora_out]) for _ in range(tp_size)]
                            torch.distributed.all_gather(gathered_lora_in, lora_state_dict[key_lora_in])
                            torch.distributed.all_gather(gathered_lora_out, lora_state_dict[key_lora_out])

                            if is_global_rank_zero():
                                print(f"RANK{torch.distributed.get_rank()} has {key_lora_in} shape {lora_state_dict[key_lora_in].shape}") #gathered lorain{gathered_lora_in}")
                                print(f"RANK{torch.distributed.get_rank()} has {key_lora_out} shape {lora_state_dict[key_lora_out].shape}") #gathered loraout {gathered_lora_out}")
                            ## TODO: Who decides what dim they split?
                            tp_dim_lora_in = 1 if key in ["attention_dense", 'mlp_fc2'] else 0
                            wt_lora_in = torch.cat(gathered_lora_in, dim=tp_dim_lora_in).float()
                            wt_lora_out = torch.cat(gathered_lora_out, dim=0).float()
                            wt_lora = wt_lora_out @ wt_lora_in
                            tp_dim_base = 0 if key in ["attention_qkv", "mlp_fc1"] else 1
                            wt_lora_current_rank = torch.chunk(wt_lora, tp_size, dim=tp_dim_base)[rank]
                        else: #when tp==1
                            if key == 'attention_qkv' and nl==31:
                                import pdb; pdb.set_trace()
                            wt_lora_in = lora_state_dict[key_lora_in]
                            wt_lora_out = lora_state_dict[key_lora_out]
                            wt_lora = wt_lora_out @ wt_lora_in
                            wt_lora_current_rank = wt_lora

                        wt_base = base_model_state_dict[key_base]
                        logging.info(f"Full {key_base} wt_lora_in {wt_lora_in.shape}, wt_lora_out {wt_lora_out.shape}, wt_lora {wt_lora.shape}, wt_base {wt_base.shape}")

                        
                        base_model_state_dict[key_base] = (wt_base.float() + wt_lora_current_rank.to(wt_base.device)).type_as(wt_base)
                        logging.info(f'merging for weight {key_base}')

            return base_model_state_dict
        
    #import pdb; pdb.set_trace()
    trainer.predict_loop = LoRAMergeLoop(trainer)
    trainer.predict(model, dataloaders=predict_dataloader)
   

__all__ = ["gpt_lora",
           "merge_lora"]

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

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import nemo_run as run
import pytorch_lightning as pl
import torch
from megatron.core import dist_checkpointing
from pytorch_lightning.loops import _PredictionLoop
from pytorch_lightning.trainer.states import TrainerFn
from typing_extensions import Annotated

from nemo.collections import llm
from nemo.collections.llm.api import _set_with_io
from nemo.collections.llm.peft.lora import LoRA
from nemo.collections.llm.utils import factory
from nemo.lightning import AutoResume, NeMoLogger, Trainer
from nemo.lightning.ckpt_utils import ckpt_to_context_subdir, ckpt_to_weights_subdir
from nemo.lightning.io import load_context
from nemo.lightning.io.pl import TrainerContext
from nemo.lightning.pytorch.callbacks import PEFT
from nemo.lightning.pytorch.callbacks.peft import PEFT
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero


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
    # logger will setup paths in trainer
    _log.setup(
        trainer,
        resume_if_exists=getattr(resume, "resume_if_exists", False),
        task_config=None,
    )
    resume.setup(trainer, model)
    lora = load_context(resume.get_context_path(), "model.model_transform")
    if lora:
        _set_with_io(model, "model_transform", lora)
        trainer.callbacks.append(lora)
    else:
        raise Exception("Cannot find LoRA config")

    predict_dataloader = llm.SquadDataModule(seq_length=2048, micro_batch_size=2, global_batch_size=8, num_workers=0)

    class LoRAMergeLoop(_PredictionLoop):  # PredictionLoop is internal now X_X
        def __init__(self, trainer, inference_mode: bool = True):
            super().__init__(trainer, inference_mode)

        def run(self):
            self._on_predict_start()  # trigger ModelTransform on_predict_start hook to enter PEFT load ckpt for the second time, loading adapter state_dict
            if trainer.state.fn == TrainerFn.PREDICTING:  # no need ?
                # base_state_dict = {k:v for k,v in trainer.model.state_dict().items() if 'adapter' not in k and 'extra_state' not in k }
                base_sharded_dict = {k: v for k, v in trainer.model.sharded_state_dict().items() if 'adapter' not in k}
                lora_sharded_dict = {
                    k: v.data.data
                    for k, v in trainer.model.sharded_state_dict().items()
                    if 'adapter' in k and 'extra_state' not in k
                }
                merged_weights = self._merge_lora_weights(
                    base_model_state_dict=base_sharded_dict,
                    lora_state_dict=lora_sharded_dict,
                    num_layers=trainer.model.config.num_layers,
                    tp_size=trainer.strategy.tensor_model_parallel_size,
                    rank=torch.distributed.get_rank(),
                )
            weight_path = ckpt_to_weights_subdir(output_path)
            Path(weight_path).mkdir(parents=True, exist_ok=True)
            dist_checkpointing.save(merged_weights, str(ckpt_to_weights_subdir(weight_path)))
            if is_global_rank_zero():
                # trainer.model.io_dump(output_path)
                if hasattr(trainer.model, "__io__") and hasattr(trainer.model.tokenizer, '__io__'):
                    trainer.model.__io__.tokenizer = trainer.model.tokenizer.__io__
                TrainerContext.from_trainer(trainer).io_dump(ckpt_to_context_subdir(output_path), yaml_attrs="model")
            logging.info(f"Merged checkpoint saved to {output_path}")

        def _merge_lora_weights(
            self,
            base_model_state_dict: Dict[str, Any],
            lora_state_dict: Dict[str, Any],
            num_layers: int,
            tp_size: int,
            rank: int,
        ):
            mcore_layer_keys = [
                "self_attention.linear_qkv.weight",
                "self_attention.linear_proj.weight",
                "mlp.linear_fc1.weight",
                "mlp.linear_fc2.weight",
            ]
            print("###### TP_SIZE", tp_size)
            for nl in range(num_layers):
                for key in mcore_layer_keys:
                    key_base = f'module.decoder.layers.{nl}.{key}'
                    key_lora_in = (
                        f'module.decoder.layers.{nl}.{key.rsplit(".weight", 1)[0] + ".adapter.linear_in.weight"}'
                    )
                    key_lora_out = (
                        f'module.decoder.layers.{nl}.{key.rsplit(".weight", 1)[0] + ".adapter.linear_out.weight"}'
                    )
                    if key_lora_in in lora_state_dict and key_lora_out in lora_state_dict:
                        if tp_size > 1:
                            gathered_lora_in = [torch.zeros_like(lora_state_dict[key_lora_in]) for _ in range(tp_size)]
                            gathered_lora_out = [
                                torch.zeros_like(lora_state_dict[key_lora_out]) for _ in range(tp_size)
                            ]
                            torch.distributed.all_gather(gathered_lora_in, lora_state_dict[key_lora_in])
                            torch.distributed.all_gather(gathered_lora_out, lora_state_dict[key_lora_out])

                            if is_global_rank_zero():
                                print(
                                    f"RANK{torch.distributed.get_rank()} has {key_lora_in} shape {lora_state_dict[key_lora_in].shape}"
                                )  # gathered lorain{gathered_lora_in}")
                                print(
                                    f"RANK{torch.distributed.get_rank()} has {key_lora_out} shape {lora_state_dict[key_lora_out].shape}"
                                )  # gathered loraout {gathered_lora_out}")
                            tp_dim_lora_in = (
                                1 if key in ["self_attention.linear_proj.weight", "mlp.linear_fc2.weight"] else 0
                            )
                            wt_lora_in = torch.cat(gathered_lora_in, dim=tp_dim_lora_in).float()
                            wt_lora_out = torch.cat(gathered_lora_out, dim=0).float()
                            wt_lora = wt_lora_out @ wt_lora_in
                            tp_dim_base = (
                                0 if key in ["self_attention.linear_qkv.weight", "mlp.linear_fc1.weight"] else 1
                            )
                            wt_lora_current_rank = torch.chunk(wt_lora, tp_size, dim=tp_dim_base)[rank]
                        else:  # when tp==1
                            wt_lora_in = lora_state_dict[key_lora_in]
                            wt_lora_out = lora_state_dict[key_lora_out]
                            wt_lora = wt_lora_out @ wt_lora_in
                            wt_lora_current_rank = wt_lora

                        wt_base = base_model_state_dict[key_base].data.data
                        logging.info(
                            f"Full {key_base} wt_lora_in {wt_lora_in.shape}, wt_lora_out {wt_lora_out.shape}, wt_lora {wt_lora.shape}, wt_base {wt_base.shape}"
                        )

                        base_model_state_dict[key_base].data.data = (
                            wt_base.float() + wt_lora_current_rank.to(wt_base.device)
                        ).type_as(wt_base)
                        logging.info(f'merging for weight {key_base}')

            return base_model_state_dict  # reference, no need to return. return for clarity??

    trainer.predict_loop = LoRAMergeLoop(trainer)
    trainer.predict(model, dataloaders=predict_dataloader)  # How to get rid of this dummy data loader??


__all__ = ["gpt_lora", "merge_lora"]

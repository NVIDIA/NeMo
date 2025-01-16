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

from typing import Any, Dict, Optional

from lightning.pytorch.callbacks import Callback

from nemo.collections.common.parts.perf_metrics_utils import LLM_VOCAB_SIZE_MAP
from nemo.utils import logging
from collections import deque
import nemo_run as run
from nemo.utils.exp_manager import TimingCallback
from nemo.collections.llm.gpt.model import GPTConfig
from nemo.lightning import Trainer
from lightning.pytorch import LightningDataModule

__all__ = ["FLOPsMeasurementCallback"]


class FLOPsMeasurementCallback(Callback):
    """
    Calculate FLOPs per second after last train step for a given job run.

    Args:
        model_config (Dict[str, Any]): params for running the experiment/job.
            For NeMo1.0, expects a nested dictionary with parent keys
                1. run- for assessing model name (Eg. 'gpt3', 'llama2', etc.) from sub-key 'name'.
                    'name' usually has value like- train_gpt3_5b_*, which is matched to model name 'gpt3'.
                2. trainer- for accessing 'num_nodes' and 'devices'
                3. model- Hyperparams for the model. Specifically- global batch size, sequence length,
                    hidden size,  ffn hidden size, num_layers, num_attention_heads, num_query_groups,
                    moe_router_topk. (list might increase with new models as required)
        trainer (nemo.lightning.Trainer): Required with NeMo 2.0 to access 'num_nodes' and 'devices'
        data_module (LightningDataModule): Required with NeMo 2.0 to access 'global_batch_size'
        model_name (Optional[str]): If present, will override 'name' under 'run' in model_config.
            Defaults to None.
    """

    _higher_is_better: bool = True

    log_per_step: bool = False
    log_avg_on_train_end: bool = True
    
    # maintain a FIFO queue of fixed length <step_time_buffer_size>; used to calculate average tflops_per_sec_per_gpu
    # over last <step_time_buffer_size> steps of job/experiment.
    # added 1 to 'step_time_buffer' for skipping warmup step 0 in case job/experiment has max_steps <= buffer_size
    step_time_buffer_size: int = 20
    step_time_buffer = deque(maxlen=step_time_buffer_size+1)

    def __init__(
        self,
        model_config: Dict[str, Any] | run.Config[GPTConfig] | GPTConfig,
        trainer: Optional[run.Config[Trainer] | Trainer] = None, # Required with NeMo 2.0
        data_module: Optional[run.Config[LightningDataModule] | LightningDataModule] = None, # Required with NeMo 2.0
        model_name: Optional[str] = None,
    ):
        if isinstance(model_config, dict): # NeMo 1.0
            self.cfg = model_config

            self.run_cfg = self.cfg.get('run', {})
            self.train_cfg = self.cfg.get('trainer', {})
            self.model_cfg = self.cfg.get('model', {})

            # use config params only when NOT provided explicitly
            self.model = self.run_cfg.get('name', "") if model_name is None else model_name

            self.num_nodes = self.train_cfg.get('num_nodes', None)
            self.num_gpus_per_node = self.train_cfg.get('devices', None)

            self.gbs = self.model_cfg.get('global_batch_size', None)
            self.enc_seq_len = self.model_cfg.get('encoder_seq_length', None)
            self.hs = self.model_cfg.get('hidden_size', None)
            self.layers = self.model_cfg.get('num_layers', None)
            self.ffn_hs = self.model_cfg.get('ffn_hidden_size', None)
            self.attention_heads = self.model_cfg.get('num_attention_heads', None)
            self.moe_router_topk = self.model_cfg.get('moe_router_topk', None)

            # this handles both- 1. key is present, value is None; 2. key is absent
            self.query_groups = self.model_cfg.get('num_query_groups', None)
            if self.query_groups is None:
                self.query_groups = self.attention_heads

            self.model = self.model.lower() if self.model is not None else self.model
        else:
            assert trainer is not None, (f"'trainer' arg (nemo.lightning.Trainer) not passed to 
                                         {self.__class__.__name__}. Required to calculate TFLOPs per sec")
            self.num_nodes = trainer.num_nodes
            self.num_gpus_per_node = trainer.devices
            
            assert data_module is not None, (f"'data' arg (lightning.pytorch.LightningDataModule) not passed to
                                             {self.__class__.__name__}. Required to calculate TFLOPs per sec")
            self.gbs = data_module.global_batch_size
            
            self.enc_seq_len = model_config.seq_length
            self.hs = model_config.hidden_size
            self.layers = model_config.num_layers
            self.ffn_hs = model_config.ffn_hidden_size
            self.attention_heads = model_config.num_attention_heads
            self.moe_router_topk = model_config.moe_router_topk

            # this handles both- 1. key is present, value is None; 2. key is absent
            self.query_groups = model_config.num_query_groups
            if self.query_groups is None:
                self.query_groups = self.attention_heads

    def setup(self, trainer, pl_module, stage):
        """
        PyTorch Lightning callback hook to calculate model FLOPs before fit/test
        """
        timing_callback_enabled = False
        callbacks = [] if trainer.callbacks is None else trainer.callbacks
        for callback in callbacks:
            if isinstance(callback, TimingCallback):
                timing_callback_enabled = True
            elif isinstance(callback, run.Config) and callback.__fn_or_cls__ == TimingCallback:
                timing_callback_enabled = True
            
        assert timing_callback_enabled, (
            f"TimingCallback is disabled. Enable it to calculate TFLOPs per sec or disable {self.__class__.__name__}"
            )

        try:
            _, flops_per_gpu = self.eval_model_flops()
            self.tflops_per_gpu = flops_per_gpu / 1e12
        except Exception as exc:
            logging.warning(f"Failed to calculate model flops. Skipping computing TFLOPs per sec per gpu per step.")
            self.tflops_per_gpu = -1

    def on_train_end(self, trainer, pl_module):
        """
        PyTorch Lightning callback hook to calculate TFLOPs per sec per GPU after training
        """
        if self.log_avg_on_train_end and len(self.step_time_buffer) > 1:
            self.step_time_buffer.popleft() # skip warmup step 0 in case list contains value for step 0
            avg_tflops_per_sec_per_gpu = self.tflops_per_gpu / sum(self.step_time_buffer) / len(self.step_time_buffer)

            logging.info(f"TFLOPs per sec per GPU={avg_tflops_per_sec_per_gpu:.2f}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        metric_log_name = "train_step_timing" + " in s"
        if hasattr(pl_module, metric_log_name) and self.tflops_per_gpu != -1:
            step_time = getattr(pl_module, metric_log_name)
            self.step_time_buffer.append(step_time)
            
            if self.log_per_step:
                pl_module.log(
                    f"tflops_per_sec_per_gpu",
                    round(self.tflops_per_gpu / step_time, 2),
                    on_step=True,
                    on_epoch=False,
                    batch_size=1,
                    prog_bar=True,
                )

    def eval_model_flops(self):
        """
        Calculate model FLOPs for a given model
        """

        model_flops_map = {
            "gpt3": self._gpt3,
            "llama2": self._llama2,
            "llama3": self._llama3,
            "nemotron": self._nemotron,
            "mixtral": self._mixtral,
            "bert": self._bert,
        }

        if self.model is not None:
            model_matches = [model for model in model_flops_map if model in self.model]
            self.model = model_matches[0] if len(model_matches) > 0 else self.model
        if self.model not in model_flops_map:
            logging.info(f"FLOPs measurement supported for {list(model_flops_map.keys())}")
            raise KeyError(f"Failed to extract valid model name from or missing FLOPs calculations for {self.model}")

        total_flops = model_flops_map[self.model]()
        flops_per_gpu = total_flops / (self.num_nodes * self.num_gpus_per_node)

        return total_flops, flops_per_gpu

    def _gpt3(self):
        """Model FLOPs for GPT3 family"""

        vocab_size = LLM_VOCAB_SIZE_MAP["gpt3"]

        return (
            24 * self.gbs * self.enc_seq_len * self.hs * self.hs
            + 4 * self.gbs * self.enc_seq_len * self.enc_seq_len * self.hs
        ) * (3 * self.layers) + (6 * self.gbs * self.enc_seq_len * self.hs * vocab_size)

    def _llama2(self):
        """Model FLOPs for llama2 family"""
        vocab_size = LLM_VOCAB_SIZE_MAP["llama2"]

        return (
            self.gbs
            * self.enc_seq_len
            * self.layers
            * self.hs
            * self.hs
            * (
                12
                + (12 * self.query_groups / self.attention_heads)
                + (18 * self.ffn_hs / self.hs)
                + (12 * self.enc_seq_len / self.hs)
                + (6 * vocab_size / (self.layers * self.hs))
            )
        )

    def _llama3(self):
        """Model FLOPs for llama3 family"""
        vocab_size = LLM_VOCAB_SIZE_MAP["llama3"]

        return (
            self.gbs
            * self.enc_seq_len
            * self.layers
            * self.hs
            * self.hs
            * (
                12
                + (12 * self.query_groups / self.attention_heads)
                + (18 * self.ffn_hs / self.hs)
                + (12 * self.enc_seq_len / self.hs)
                + (6 * vocab_size / (self.layers * self.hs))
            )
        )

    def _nemotron(self):
        """Model FLOPs for nemotron family"""
        vocab_size = LLM_VOCAB_SIZE_MAP["nemotron"]

        return (
            self.gbs
            * self.enc_seq_len
            * self.layers
            * self.hs
            * self.hs
            * (
                12
                + (12 * self.query_groups / self.attention_heads)
                + (12 * self.ffn_hs / self.hs)
                + (12 * self.enc_seq_len / self.hs)
                + (6 * vocab_size / (self.layers * self.hs))
            )
        )

    def _mixtral(self):
        """Model FLOPs for mixtral family"""
        vocab_size = LLM_VOCAB_SIZE_MAP["mixtral"]

        return (
            self.gbs
            * self.enc_seq_len
            * self.layers
            * self.hs
            * self.hs
            * (
                12
                + (12 * self.query_groups / self.attention_heads)
                + (18 * self.moe_router_topk * self.ffn_hs / self.hs)
                + (12 * self.enc_seq_len / self.hs)
                + (6 * vocab_size / (self.layers * self.hs))
            )
        )

    def _bert(self):
        """Model FLOPs for BERT family"""
        vocab_size = LLM_VOCAB_SIZE_MAP["bert"]

        return (
            72
            * self.gbs
            * self.layers
            * self.enc_seq_len
            * self.hs
            * self.hs
            * (1 + (self.enc_seq_len / (6 * self.hs)) + (vocab_size / (12 * self.hs * self.layers)))
        )

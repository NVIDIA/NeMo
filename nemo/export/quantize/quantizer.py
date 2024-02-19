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

import copy
import os
import tarfile
from typing import Optional

import ammo.torch.quantization as atq
import torch.distributed as dist
from ammo.torch.export import export_model_config
from ammo.torch.utils import print_rank_0
from megatron.core import parallel_state
from omegaconf import OmegaConf
from omegaconf.omegaconf import DictConfig, open_dict
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.collections.nlp.parts.utils_funcs import torch_dtype_from_precision
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero

from .utils_wip import copy_artifacts, temporary_directory  # TODO: Find a good place for these utils

QUANT_CFG_CHOICES = {
    "int8": atq.INT8_DEFAULT_CFG,
    "int8_sq": atq.INT8_SMOOTHQUANT_CFG,
    "fp8": atq.FP8_DEFAULT_CFG,
    "int4_awq": atq.INT4_AWQ_CFG,
    "w4a8_awq": atq.W4A8_AWQ_BETA_CFG,
}


class Quantizer:

    """
    Post-training quantization of Nemo checkpoints.

    PTQ converts selected model layers to low-precision format (e.g., INT4, FP8) for efficient serving.
    The process consist of several steps:

        1. Loading a Nemo model from disk using appropriate parallelism strategy
        2. Calibrating the model to obtain appropriate algorithm-specific scaling factors
        3. Producing .qnemo tarball with model config (JSON), quantized weights (safetensors)
           and tokenizer config (yaml).

    The .qnemo file produced is intended consumed by TensorRT-LLM toolbox for inference.
    This can be achieved using Nemo inference containers.

    Currently supported and tested model family is Llama2. Model type needs to be specified in
    the quantization command with decoder_type parameter on exporting (see below). Quantizing other
    model families is experimental and might not be fully supported.

    Available quantization methods are listed in QUANT_CFG_CHOICES dictionary on top of this file.
    Please consult AMMO docummentation for details. You can also ispect different choices in
    examples/nlp/language_modeling/conf/megatron_llama_quantization.yaml for quantization algorithms and
    calibration data as well as recommended settings.
    """

    def __init__(self, quantization_config: DictConfig, inference_config: DictConfig, trainer_config: DictConfig):
        self.quantization_config = quantization_config
        self.inference_config = inference_config
        self.trainer_config = trainer_config
        assert self.quantization_config.algorithm in QUANT_CFG_CHOICES
        atq_config = QUANT_CFG_CHOICES[quantization_config.algorithm]
        if quantization_config.algorithm != "fp8":
            # disable quantization for the last output layer
            atq_config = copy.deepcopy(atq_config)
            atq_config["quant_cfg"]["*.output_layer.*"] = {"enable": False}
        self.atq_config = atq_config

    def _load_model(self, model_file, tensor_model_parallel_size: Optional[int] = None):
        trainer = Trainer(strategy=NLPDDPStrategy(), **self.trainer_config)
        connector = NLPSaveRestoreConnector()

        if os.path.isdir(model_file):
            connector.model_extracted_dir = model_file

        model_cfg = self._restore_and_modify_config(model_file, trainer, connector, tensor_model_parallel_size)

        model = MegatronGPTModel.restore_from(
            restore_path=model_file, trainer=trainer, override_config_path=model_cfg, save_restore_connector=connector,
        )
        model.freeze()

        try:
            model.model.module.language_model.encoder.activations_checkpoint_method = None
        except AttributeError:
            pass
        print_rank_0(model)
        self._check_ddp_initialized(model)
        return model

    def _check_ddp_initialized(self, model):
        if parallel_state.is_unitialized():

            def dummy():
                return

            if model.trainer.strategy.launcher is not None:
                model.trainer.strategy.launcher.launch(dummy, trainer=model.trainer)
            model.trainer.strategy.setup_environment()

    def _restore_and_modify_config(
        self,
        model_file: str,
        trainer: Trainer,
        connector: NLPSaveRestoreConnector,
        tensor_model_parallel_size: Optional[int] = None,
    ):
        model_cfg = MegatronGPTModel.restore_from(
            restore_path=model_file, trainer=trainer, save_restore_connector=connector, return_config=True,
        )
        with open_dict(model_cfg):
            model_cfg.activations_checkpoint_method = None
            model_cfg.activations_checkpoint_granularity = None
            if tensor_model_parallel_size is not None:
                model_cfg.tensor_model_parallel_size = tensor_model_parallel_size
            model_cfg.name = "ammo"  # Model needs to be loaded in "ammo" layer spec

        return model_cfg

    def quantize(self, model_file: str, dataloader, tensor_model_parallel_size: Optional[int] = None):
        model = self._load_model(model_file, tensor_model_parallel_size)
        model.set_inference_config(OmegaConf.to_container(self.inference_config))

        def forward_loop():
            for i, batch in enumerate(dataloader):
                print_rank_0(f"Calibrating batch {i}")
                model.predict_step(batch, i)

        atq.quantize(model, self.atq_config, forward_loop)
        return model

    def export(self, model, output_file: str, decoder_type: str, dtype: str, inference_tensor_parallel: int):
        supported_dtype = [16, "16", "bf16"]  # FIXME: Move that to top
        assert dtype in supported_dtype, f"{dtype} not supported. Supported dtypes are {supported_dtype}"
        torch_dtype = torch_dtype_from_precision(dtype)

        with temporary_directory() as tmp_dir:
            export_model_config(
                model=model,
                decoder_type=decoder_type,
                dtype=torch_dtype,
                export_dir=tmp_dir,
                inference_tensor_parallel=inference_tensor_parallel,
            )
            dist.barrier()  # Wait until all ranks complete export_model_config step
            if is_global_rank_zero():
                logging.info(f"Exporting quantized weights, tokenizer config, and model artifacts to {output_file}...")
                with tarfile.open(output_file, "w:gz") as tar:
                    config = copy_artifacts(model, tmp_dir)
                    OmegaConf.save(config.tokenizer, os.path.join(tmp_dir, "tokenizer_config.yaml"))
                    tar.add(tmp_dir, arcname="./")

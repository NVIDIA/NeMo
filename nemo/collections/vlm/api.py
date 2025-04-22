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

# This file reuses the API endpoints defined in nemo.collections.llm.api
# but registers them under the 'vlm' namespace for the NeMo CLI.

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import lightning.pytorch as pl
import nemo_run as run
import torch
from typing_extensions import Annotated

import nemo.lightning as nl
from nemo.collections.llm import api as llm_api
from nemo.collections.llm.evaluation.api import EvaluationConfig, EvaluationTarget
from nemo.collections.llm.modelopt import ExportConfig, PruningConfig, QuantizationConfig
from nemo.lightning import AutoResume, NeMoLogger, OptimizerModule, Trainer
from nemo.lightning.pytorch.callbacks import PEFT, ModelTransform

if TYPE_CHECKING:
    from megatron.core.inference.common_inference_params import CommonInferenceParams
    from megatron.core.inference.inference_request import InferenceRequest

TokenizerType = Any
AnyPath = Union[Path, str]


# --- Reuse LLM API functions under the 'vlm' namespace ---

_llm_train = llm_api.train
_llm_pretrain = llm_api.pretrain
_llm_finetune = llm_api.finetune
_llm_validate = llm_api.validate
_llm_prune = llm_api.prune
_llm_distill = llm_api.distill
_llm_ptq = llm_api.ptq
_llm_deploy = llm_api.deploy
_llm_evaluate = llm_api.evaluate
_llm_import_ckpt = llm_api.import_ckpt
_llm_export_ckpt = llm_api.export_ckpt
_llm_generate = llm_api.generate


@run.cli.entrypoint(namespace="vlm")
def train(
    model: pl.LightningModule,
    data: pl.LightningDataModule,
    trainer: Trainer,
    log: Annotated[Optional[NeMoLogger], run.Config[NeMoLogger]] = None,
    resume: Annotated[Optional[AutoResume], run.Config[AutoResume]] = None,
    optim: Optional[OptimizerModule] = None,
    tokenizer: Optional[TokenizerType] = None,
    model_transform: Optional[Union[PEFT, ModelTransform, Callable]] = None,
) -> Path:
    """
    Trains a VLM model using the specified data and trainer.
    (Reuses llm.train functionality)
    """
    return _llm_train(
        model=model,
        data=data,
        trainer=trainer,
        log=log,
        resume=resume,
        optim=optim,
        tokenizer=tokenizer,
        model_transform=model_transform,
    )


@run.cli.entrypoint(namespace="vlm")
def pretrain(
    model: pl.LightningModule,
    data: pl.LightningDataModule,
    trainer: Trainer,
    log: Annotated[Optional[NeMoLogger], run.Config[NeMoLogger]] = None,
    resume: Annotated[Optional[AutoResume], run.Config[AutoResume]] = None,
    optim: Optional[OptimizerModule] = None,
) -> Path:
    """
    Pretrains a VLM model using the specified data and trainer.
    (Reuses llm.pretrain functionality)
    """
    return _llm_pretrain(model=model, data=data, trainer=trainer, log=log, resume=resume, optim=optim)


@run.cli.entrypoint(namespace="vlm")
def finetune(
    model: pl.LightningModule,
    data: pl.LightningDataModule,
    trainer: Trainer,
    log: Annotated[Optional[NeMoLogger], run.Config[NeMoLogger]] = None,
    resume: Annotated[Optional[AutoResume], run.Config[AutoResume]] = None,
    optim: Optional[OptimizerModule] = None,
    peft: Optional[Union[PEFT, ModelTransform, Callable]] = None,
) -> Path:
    """
    Finetunes a VLM model using the specified data and trainer.
    (Reuses llm.finetune functionality)
    """
    return _llm_finetune(model=model, data=data, trainer=trainer, log=log, resume=resume, optim=optim, peft=peft)


@run.cli.entrypoint(namespace="vlm")
def validate(
    model: pl.LightningModule,
    data: pl.LightningDataModule,
    trainer: Trainer,
    log: Annotated[Optional[NeMoLogger], run.Config[NeMoLogger]] = None,
    resume: Annotated[Optional[AutoResume], run.Config[AutoResume]] = None,
    optim: Optional[OptimizerModule] = None,
    tokenizer: Optional[TokenizerType] = None,
    model_transform: Optional[Union[PEFT, ModelTransform, Callable]] = None,
) -> Path:
    """
    Validates a VLM model using the specified data and trainer.
    (Reuses llm.validate functionality)
    """
    return _llm_validate(
        model=model,
        data=data,
        trainer=trainer,
        log=log,
        resume=resume,
        optim=optim,
        tokenizer=tokenizer,
        model_transform=model_transform,
    )


@run.cli.entrypoint(name="prune", namespace="vlm")
def prune(
    nemo_checkpoint: str,
    save_path: str,
    pruning_config: PruningConfig,
    devices: int = 1,
    num_nodes: int = 1,
    tp_size: int = 1,
    pp_size: int = 1,
    num_train_samples: int = 1024,
    data: pl.LightningDataModule | None = None,
    tokenizer_path: str | None = None,
    legacy_ckpt: bool = False,
) -> str:
    """
    Prunes a VLM model.
    (Reuses llm.prune functionality)
    """
    return _llm_prune(
        nemo_checkpoint=nemo_checkpoint,
        save_path=save_path,
        pruning_config=pruning_config,
        devices=devices,
        num_nodes=num_nodes,
        tp_size=tp_size,
        pp_size=pp_size,
        num_train_samples=num_train_samples,
        data=data,
        tokenizer_path=tokenizer_path,
        legacy_ckpt=legacy_ckpt,
    )


@run.cli.entrypoint(name="distill", namespace="vlm")
def distill(
    student_model_path: AnyPath,
    teacher_model_path: AnyPath,
    data: pl.LightningDataModule,
    trainer: Trainer,
    log: Annotated[Optional[NeMoLogger], run.Config[NeMoLogger]] = None,
    resume: Annotated[Optional[AutoResume], run.Config[AutoResume]] = None,
    optim: Optional[OptimizerModule] = None,
    tokenizer: Optional[TokenizerType] = None,
    model_transform: Optional[Union[PEFT, ModelTransform, Callable]] = None,
) -> Path:
    """
    Distills a teacher VLM model into a student VLM model.
    (Reuses llm.distill functionality)
    """
    return _llm_distill(
        student_model_path=student_model_path,
        teacher_model_path=teacher_model_path,
        data=data,
        trainer=trainer,
        log=log,
        resume=resume,
        optim=optim,
        tokenizer=tokenizer,
        model_transform=model_transform,
    )


@run.cli.entrypoint(name="ptq", namespace="vlm")
def ptq(
    model_path: str,
    export_config: ExportConfig,
    calibration_tp: int = 1,
    calibration_pp: int = 1,
    num_layers_in_first_pipeline_stage: int | None = None,
    num_layers_in_last_pipeline_stage: int | None = None,
    devices: int | None = None,
    num_nodes: int | None = None,
    quantization_config: Annotated[Optional[QuantizationConfig], run.Config[QuantizationConfig]] = None,
    forward_loop: Callable | None = None,
    tokenizer_path: str | None = None,
    legacy_ckpt: bool = False,
    trust_remote_code: bool = False,
) -> Path:
    """
    Applies Post-Training Quantization (PTQ) to a VLM model.
    (Reuses llm.ptq functionality)
    """
    return _llm_ptq(
        model_path=model_path,
        export_config=export_config,
        calibration_tp=calibration_tp,
        calibration_pp=calibration_pp,
        num_layers_in_first_pipeline_stage=num_layers_in_first_pipeline_stage,
        num_layers_in_last_pipeline_stage=num_layers_in_last_pipeline_stage,
        devices=devices,
        num_nodes=num_nodes,
        quantization_config=quantization_config,
        forward_loop=forward_loop,
        tokenizer_path=tokenizer_path,
        legacy_ckpt=legacy_ckpt,
        trust_remote_code=trust_remote_code,
    )


@run.cli.entrypoint(namespace="vlm")
def deploy(
    nemo_checkpoint: AnyPath = None,
    backend: str = "in-framework",
    model_type: str = "llama",
    triton_model_name: str = "triton_model",
    triton_model_version: Optional[int] = 1,
    triton_http_port: int = 8000,
    triton_grpc_port: int = 8001,
    triton_http_address: str = "0.0.0.0",
    triton_model_repository: AnyPath = None,
    start_fastapi_server: bool = True,
    fastapi_http_address: str = "0.0.0.0",
    fastapi_port: int = 8080,
    num_gpus: int = 1,
    num_nodes: int = 1,
    tensor_parallelism_size: int = 1,
    pipeline_parallelism_size: int = 1,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    dtype: str = "bfloat16",
    max_input_len: int = 4096,
    max_output_len: int = 256,
    max_batch_size: int = 8,
    output_context_logits: bool = True,
    output_generation_logits: bool = True,
):
    """
    Deploys a VLM model on a PyTriton server.
    (Reuses llm.deploy functionality)
    """
    # Note: deploy itself doesn't return anything significant for the CLI flow
    _llm_deploy(
        nemo_checkpoint=nemo_checkpoint,
        backend=backend,
        model_type=model_type,
        triton_model_name=triton_model_name,
        triton_model_version=triton_model_version,
        triton_http_port=triton_http_port,
        triton_grpc_port=triton_grpc_port,
        triton_http_address=triton_http_address,
        triton_model_repository=triton_model_repository,
        start_fastapi_server=start_fastapi_server,
        fastapi_http_address=fastapi_http_address,
        fastapi_port=fastapi_port,
        num_gpus=num_gpus,
        num_nodes=num_nodes,
        tensor_parallelism_size=tensor_parallelism_size,
        pipeline_parallelism_size=pipeline_parallelism_size,
        context_parallel_size=context_parallel_size,
        expert_model_parallel_size=expert_model_parallel_size,
        dtype=dtype,
        max_input_len=max_input_len,
        max_output_len=max_output_len,
        max_batch_size=max_batch_size,
        output_context_logits=output_context_logits,
        output_generation_logits=output_generation_logits,
    )


@run.cli.entrypoint(namespace="vlm")
def evaluate(
    target_cfg: EvaluationTarget,
    eval_cfg: EvaluationConfig = EvaluationConfig(type="gsm8k"),
) -> dict:
    """
    Evaluates a deployed VLM model.
    (Reuses llm.evaluate functionality)
    """
    return _llm_evaluate(target_cfg=target_cfg, eval_cfg=eval_cfg)


@run.cli.entrypoint(name="import", namespace="vlm")
def import_ckpt(
    model: pl.LightningModule,
    source: str,
    output_path: Optional[AnyPath] = None,
    overwrite: bool = False,
    **kwargs,
) -> Path:
    """
    Imports a checkpoint into a VLM model.
    (Reuses llm.import_ckpt functionality)
    """
    return _llm_import_ckpt(model=model, source=source, output_path=output_path, overwrite=overwrite, **kwargs)


@run.cli.entrypoint(name="export", namespace="vlm")
def export_ckpt(
    path: AnyPath,
    target: str,
    output_path: Optional[AnyPath] = None,
    overwrite: bool = False,
    load_connector: Callable[[Path, str], nl.io.ModelConnector] = llm_api.load_connector_from_trainer_ckpt,
    **kwargs,
) -> Path:
    """
    Exports a checkpoint from a VLM model.
    (Reuses llm.export_ckpt functionality)
    """
    return _llm_export_ckpt(
        path=path,
        target=target,
        output_path=output_path,
        overwrite=overwrite,
        load_connector=load_connector,
        **kwargs,
    )


@run.cli.entrypoint(name="generate", namespace="vlm")
def generate(
    path: AnyPath,
    trainer: nl.Trainer,
    prompts: Optional[list[str]] = None,
    encoder_prompts: Optional[list[str]] = None,
    input_dataset: Optional[Union[pl.LightningDataModule, str]] = None,
    params_dtype: torch.dtype = torch.bfloat16,
    add_BOS: bool = False,
    max_batch_size: int = 4,
    random_seed: Optional[int] = None,
    inference_batch_times_seqlen_threshold: int = 1000,
    inference_params: Optional["CommonInferenceParams"] = None,
    text_only: bool = False,
    output_path: Optional[AnyPath] = None,
) -> list[Union["InferenceRequest", str]]:
    """
    Generates text using a VLM model.
    (Reuses llm.generate functionality)
    """
    return _llm_generate(
        path=path,
        trainer=trainer,
        prompts=prompts,
        encoder_prompts=encoder_prompts,
        input_dataset=input_dataset,
        params_dtype=params_dtype,
        add_BOS=add_BOS,
        max_batch_size=max_batch_size,
        random_seed=random_seed,
        inference_batch_times_seqlen_threshold=inference_batch_times_seqlen_threshold,
        inference_params=inference_params,
        text_only=text_only,
        output_path=output_path,
    )

from megatron.core.inference.common_inference_params import CommonInferenceParams
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed
from nemo import lightning as nl
import nemo_run as run
from nemo.collections import llm
import torch
import pytorch_lightning as pl
from pathlib import Path
from megatron.core.optimizer import OptimizerConfig
from nemo.collections.llm import Llama2Config7B
from typing import List, Optional
from nemo.lightning.io.mixin import IOMixin
from run_sft import trainer, local_executor_torchrun
import os

input_data="/workspace/data/verilog/test.jsonl"
base_llama_path = "/root/.cache/nemo/models/Llama-2-7b-hf"
sft_ckpt_path=str(next((d for d in Path("/workspace/sft_log/checkpoints").iterdir() if d.is_dir() and d.name.endswith("-last")), None))

os.makedirs("/workspace/inference", exist_ok=True)
output_path_base="/workspace/inference/base_llama_prediction.jsonl"
output_path_sft="/workspace/inference/sft_prediction.jsonl"

# Configure inference to predict on base model checkpoint
def configure_inference_base():
    return run.Partial(
        llm.generate,
        path=str(base_llama_path),
        trainer=trainer(),
        input_dataset=input_data,
        inference_params=CommonInferenceParams(num_tokens_to_generate=50, top_k=1),
        output_path=output_path_base,
    )

# Configure inference to predict on trained DAPT checkpoint
def configure_inference_sft():
    return run.Partial(
        llm.generate,
        path=str(sft_ckpt_path),
        trainer=trainer(),
        input_dataset=input_data,
        inference_params=CommonInferenceParams(num_tokens_to_generate=50, top_k=1),
        output_path=output_path_sft,
    )

if __name__ == '__main__':
    print("running inference on base model")
    run.run(configure_inference_base(), executor=local_executor_torchrun())
    print("running inference on supervise fine tuned model")
    run.run(configure_inference_sft(), executor=local_executor_torchrun())
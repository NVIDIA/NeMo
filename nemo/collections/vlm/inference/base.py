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

from typing import Optional

import pytorch_lightning as pl
import torch
import torch.distributed
from megatron.core.inference.common_inference_params import CommonInferenceParams
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import InferenceWrapperConfig

import nemo.lightning as nl
from nemo.collections import vlm
from nemo.collections.vlm.inference.vlm_engine import VLMEngine
from nemo.collections.vlm.inference.vlm_inference_controller import VLMTextGenerationController
from nemo.collections.vlm.inference.vlm_inference_wrapper import VLMInferenceWrapper


def _setup_trainer_and_restore_model(path: str, trainer: nl.Trainer, model: pl.LightningModule):
    fabric = trainer.to_fabric()
    model = fabric.load_model(path, model)
    return model


def setup_model_and_tokenizer(
    path: str,
    trainer: Optional[nl.Trainer] = None,
    params_dtype: torch.dtype = torch.bfloat16,
    inference_batch_times_seqlen_threshold: int = 1000,
):
    # model: io.TrainerContext = io.load_context(path=path, subpath="model")
    # trainer = trainer or io.load_context(path=path, subpath="trainer")
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = processor.tokenizer
    config = vlm.MLlamaConfig11BInstruct()
    model = vlm.MLlamaModel(config, tokenizer=tokenizer)
    _setup_trainer_and_restore_model(path=path, trainer=trainer, model=model)

    mcore_model = model.module.cuda()
    mcore_model = mcore_model.to(params_dtype)
    inference_wrapped_model = VLMInferenceWrapper(
        mcore_model,
        InferenceWrapperConfig(
            hidden_size=config.language_model_config.hidden_size,
            params_dtype=params_dtype,
            inference_batch_times_seqlen_threshold=inference_batch_times_seqlen_threshold,
            padded_vocab_size=tokenizer.vocab_size,
        ),
    )

    return inference_wrapped_model, processor


def generate(
    model: VLMInferenceWrapper,
    processor,
    prompts: list[str],
    images,
    max_batch_size: int = 4,
    random_seed: Optional[int] = None,
    inference_params: Optional[CommonInferenceParams] = None,
) -> dict:
    text_generation_controller = VLMTextGenerationController(inference_wrapped_model=model, processor=processor)
    mcore_engine = VLMEngine(
        text_generation_controller=text_generation_controller, max_batch_size=max_batch_size, random_seed=random_seed
    )

    common_inference_params = inference_params or CommonInferenceParams(num_tokens_to_generate=50)

    results = mcore_engine.generate(
        prompts=prompts,
        images=images,
        common_inference_params=common_inference_params,
    )

    return results

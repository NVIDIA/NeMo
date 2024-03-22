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

import asyncio
import json
import os
import re
import threading
import torch
from omegaconf import OmegaConf, open_dict
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from pytorch_lightning.trainer.trainer import Trainer
from torch.utils.data import DataLoader, Dataset

from nemo.collections.multimodal.models.neva.neva_model import MegatronNevaModel
from nemo.collections.multimodal.parts.utils import create_neva_model_and_processor
from nemo.collections.nlp.modules.common.text_generation_utils import generate
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.collections.nlp.parts.peft_config import PEFT_CONFIG_MAP
from nemo.core.config import hydra_runner
from nemo.utils.app_state import AppState
from nemo.utils.model_utils import inject_model_parallel_rank

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

try:
    import ammo.torch.quantization as atq

    HAVE_AMMO = True

except (ImportError, ModuleNotFoundError):

    HAVE_AMMO = False

"""
This is the script to run GPT text generation.

Usage:
    Assume the model has TP=1, PP=1 in the following use cases.
    a. run greedy inference from a nemo file:
        python neva_evaluation.py \
            neva_model_file=PATH_TO_MODEL \
            inference.greedy=True \
            inference.add_BOS=True \
            trainer.devices=1 \
            trainer.num_nodes=1 \
            tensor_model_parallel_size=-1 \
            pipeline_model_parallel_size=-1 \
            prompts=[prompt1,prompt2]

    b. run greedy inference from a PTL checkpoint file:
        python neva_evaluation.py \
            checkpoint_dir=PATH_TO_CHECKPOINT_FILE \
            checkpoint_name=CHECKPOINT_FILE_NAME \
            hparams_file=HPARAMS_FILE \
            inference.greedy=True \
            inference.add_BOS=True \
            trainer.devices=1 \
            trainer.num_nodes=1 \
            tensor_model_parallel_size=-1 \
            pipeline_model_parallel_size=-1 \
            prompts=[prompt1,prompt2]

    c. run top_p inference from a nemo file:
        python neva_evaluation.py \
            neva_model_file=PATH_TO_MODEL \
            inference.greedy=False \
            inference.top_k=0 \
            inference.top_p=0.9 \
            inference.repetition_penalty=1.2 \
            inference.add_BOS=True \
            trainer.devices=1 \
            trainer.num_nodes=1 \
            tensor_model_parallel_size=-1 \
            pipeline_model_parallel_size=-1 \
            prompts=[prompt1,prompt2]

    d. If you don't need to generate tokens and need model to compute logprobs:
         python neva_evaluation.py \
            neva_model_file=PATH_TO_MODEL \
            inference.compute_logprob=True \
            trainer.devices=1 \
            trainer.num_nodes=1 \
            tensor_model_parallel_size=-1 \
            pipeline_model_parallel_size=-1 \
            prompts=[text to get logprob]

    e. Launch the inference server
         python neva_evaluation.py \
            neva_model_file=PATH_TO_MODEL \
            trainer.devices=1 \
            trainer.num_nodes=1 \
            tensor_model_parallel_size=-1 \
            pipeline_model_parallel_size=-1 \
            server=True
        
        To send a request to the server, here is one example code:
        ```python
        import json
        import requests

        batch_size = 8
        port_num = 5555
        headers = {"Content-Type": "application/json"}


        def request_data(data):
            resp = requests.put('http://localhost:{}/generate'.format(port_num),
                                data=json.dumps(data),
                                headers=headers)
            sentences = resp.json()['sentences']
            return sentences


        data = {
            "sentences": [""] * batch_size,
            "images" : [] * batch_size,
            "tokens_to_generate": 300,
            "temperature": 1.0,
            "add_BOS": True,
            "top_k": 0,
            "top_p": 0.9,
            "greedy": False,
            "all_probs": False,
            "repetition_penalty": 1.2,
            "min_tokens_to_generate": 2,
        }

        sentences = request_data(data)
        ```
"""

if not torch.cuda.is_available():
    raise EnvironmentError("GPU is needed for the inference")


class RequestDataSet(Dataset):
    def __init__(self, sentences):
        super().__init__()
        self.sentences = sentences

    def __len__(self,):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


@hydra_runner(config_path="conf", config_name="neva_inference")
def main(cfg) -> None:
    model, image_processor = create_neva_model_and_processor(cfg)

    length_params: LengthParam = {
        "max_length": cfg.inference.tokens_to_generate,
        "min_length": cfg.inference.min_tokens_to_generate,
    }

    sampling_params: SamplingParam = {
        "use_greedy": cfg.inference.greedy,
        "temperature": cfg.inference.temperature,
        "top_k": cfg.inference.top_k,
        "top_p": cfg.inference.top_p,
        "repetition_penalty": cfg.inference.repetition_penalty,
        "add_BOS": cfg.inference.add_BOS,
        "all_probs": cfg.inference.all_probs,
        "compute_logprob": cfg.inference.compute_logprob,
        "end_strings": cfg.inference.end_strings,
    }

    with open(cfg.prompt_file, 'r') as f:
        lines = f.readlines()

    final_prompts = []
    for line in lines:
        prompt_dict = json.loads(line)
        assert 'prompt' in prompt_dict or 'text' in prompt_dict
        if 'prompt' not in prompt_dict:
            prompt_dict['prompt'] = prompt_dict['text']
        if cfg.inference.insert_image_token == 'left':
            prompt_dict['prompt'] = '<image>' + prompt_dict['prompt']
        elif cfg.inference.insert_image_token == 'right':
            prompt_dict['prompt'] = prompt_dict['prompt'] + '<image>'
        if 'image' in prompt_dict:
            prompt_dict['image_path'] = prompt_dict['image']
            prompt_dict['image'] = image_processor(os.path.join(cfg.inference.images_base_path, prompt_dict['image']))
        final_prompts.append(prompt_dict)

    responses = model.generate(
        input_prompts=final_prompts, length_params=length_params, sampling_params=sampling_params, inference_config=cfg
    )

    # =================== Start Quantization ====================
    #  see https://gitlab-master.nvidia.com/omniml/ammo/-/tree/main/examples/nemo/neva for details
    if HAVE_AMMO and cfg.quantization.enable == True:
        print(f"Using quantization algorithm: {cfg.quantization.algorithm}")
        if cfg.quantization.algorithm == "int8_sq":
            atq_config = atq.INT8_SMOOTHQUANT_CFG
        elif cfg.quantization.algorithm == "fp8":
            atq_config = atq.FP8_DEFAULT_CFG
        elif cfg.quantization.algorithm == "awq":
            atq_config = atq.INT4_AWQ_CFG
        else:
            raise ValueError(f"Unsupported quantization algorithm: {cfg.quantization.algorithm}")

        def forward_loop():
            model.generate(
                input_prompts=final_prompts,
                length_params=length_params,
                sampling_params=sampling_params,
                inference_config=cfg,
            )

        atq.quantize(model, atq_config, forward_loop)

        responses = model.generate(
            input_prompts=final_prompts,
            length_params=length_params,
            sampling_params=sampling_params,
            inference_config=cfg,
        )
    # ============== Quantization End =========================

    results = []
    for response, prompt in zip(responses, final_prompts):
        prompt['full_text'] = response["clean_text"]
        prompt['text'] = response["clean_response"]
        prompt['model_id'] = cfg.neva_model_file
        if 'image_path' in prompt:
            prompt['image'] = prompt.pop('image_path')
        if 'answer_id' not in prompt:
            prompt['answer_id'] = 0
        if 'metadata' not in prompt:
            prompt['metadata'] = {}
        results.append(prompt)

    with open(cfg.output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    """ 
    # Second method of running text generation, call trainer.predict
    ds = RequestDataSet(final_prompts)
    request_dl = DataLoader(dataset=ds, batch_size=1)
    config = OmegaConf.to_container(cfg.inference)
    model.set_inference_config(config)
    response = trainer.predict(model, request_dl)

    print("***************************")
    print(response)
    print("***************************")
    """


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter

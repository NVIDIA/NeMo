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

import argparse
from pathlib import Path

from nemo import lightning as nl
from nemo.collections import llm


STRING_TO_MODEL_CLASS = {
    "baichuan2": llm.Baichuan2Model,
    "chatgml2": llm.ChatGLMModel,
    "gemma2": llm.Gemma2Model,
    "gemma": llm.GemmaModel,
    "llama2": llm.LlamaModel,
    "llama31": llm.LlamaModel,
    "mistral": llm.MistralModel,
    "mixtral": llm.MixtralModel,
    "qwen2": llm.Qwen2Model,
    "starcoder": llm.StarcoderModel,
    "starcoder2": llm.Starcoder2Model,
    "starcoder2-7b": llm.Starcoder2Model,
}

STRING_TO_SMALLEST_CONFIG = {
    "baichuan2": llm.Baichuan2Config7B,
    "chatgml2": llm.ChatGLM2Config6B,
    "gemma2": llm.Gemma2Config2B,
    "gemma": llm.GemmaConfig2B,
    "llama2": llm.Llama2Config7B,
    "llama31": llm.Llama31Config8B,
    "mistral": llm.MistralConfig7B,
    "mixtral": llm.MixtralConfig8x3B,
    "qwen2": llm.Qwen2Config500M,
    "starcoder": llm.StarcoderConfig15B,
    "starcoder2": llm.Starcoder2Config3B,
    "starcoder2-7b": llm.Starcoder2Config7B,
}


def get_args():
    parser = argparse.ArgumentParser(description='Test Llama2 7B model model conversion from HF')
    parser.add_argument('--hf_model', type=str, help="Original HF model")
    parser.add_argument('--output_path', type=str, help="NeMo 2.0 export path")
    parser.add_argument("-mt",
                        "--model_type",
                        type=str,
                        help="NeMo 2.0 model type",
                        choices=list(STRING_TO_MODEL_CLASS.keys()),
                        default="llama2")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    model_type = args.model_type
    model_class = STRING_TO_MODEL_CLASS[model_type]
    model_config = STRING_TO_SMALLEST_CONFIG[model_type]

    model = model_class(model_config)
    nemo2_path = llm.import_ckpt(model, "hf://" + args.hf_model, output_path=Path(args.output_path))

    trainer = nl.Trainer(
        devices=1,
        strategy=nl.MegatronStrategy(tensor_model_parallel_size=1),
        plugins=nl.MegatronMixedPrecision(precision='fp16'),
    )
    fabric = trainer.to_fabric()
    trainer.strategy.setup_environment()
    fabric.load_model(nemo2_path)

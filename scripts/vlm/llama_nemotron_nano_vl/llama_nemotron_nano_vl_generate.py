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

"""
Example:
  python scripts/vlm/llama_nemotron_nano_vl_generate.py \
    --local_model_path=/path/to/nemo/model
"""

import argparse
import os

import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor
import nemo.lightning as nl
from nemo.collections.vlm import LlamaNemotronNanoVLConfig8B
from nemo.collections.vlm.neva.model.llama_nemotron_vl import LlamaNemotronVLModel
from nemo.utils import logging


def load_image(image_url: str) -> Image.Image:
    # pylint: disable=C0115,C0116
    if os.path.exists(image_url):
        return Image.open(image_url)
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw)
        return image
    except requests.exceptions.RequestException as e:
        print(f"Error loading image from {image_url}: {e}")
        return None


def main(args) -> None:
    # pylint: disable=C0115,C0116
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        ckpt_include_optimizer=False,
    )
    trainer = nl.Trainer(
        devices=1,
        max_steps=1000,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        val_check_interval=1000,
        limit_val_batches=50,
    )

    # # Tokenize the input texts
    # processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    #

    from nemo.collections.common.tokenizers import AutoTokenizer

    hf_tokenizer = AutoTokenizer("meta-llama/Llama-3.1-8B-Instruct")
    image_token_index = 128256
    hf_tokenizer.tokenizer.chat_template = """{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- set date_string = \"26 Jul 2024\" %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = none %}\n{%- endif %}\n\n{%- if system_message is not none %}{#- System message + builtin tools #}\n{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n{%- if builtin_tools is defined or tools is not none %}\n    {{- \"Environment: ipython\\n\" }}\n{%- endif %}\n{%- if builtin_tools is defined %}\n    {{- \"Tools: \" + builtin_tools | reject('equalto', 'code_interpreter') | join(\", \") + \"\\n\\n\"}}\n{%- endif %}{%- if tools is not none and not tools_in_user_message %}\n    {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- \"<|eot_id|>\" }}\n\n{%-endif %}{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n{%- endif %}\n    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n    {{- first_user_message + \"<|eot_id|>\"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- \"<|python_tag|>\" + tool_call.name + \".call(\" }}\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\n                {{- arg_name + '=\"' + arg_val + '\"' }}\n                {%- if not loop.last %}\n                    {{- \", \" }}\n                {%- endif %}\n                {%- endfor %}\n            {{- \")\" }}\n        {%- else  %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n            {{- '\"parameters\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- \"}\" }}\n        {%- endif %}\n        {%- if builtin_tools is defined %}\n            {#- This means we're in ipython mode #}\n            {{- \"<|eom_id|>\" }}\n        {%- else %}\n            {{- \"<|eot_id|>\" }}\n        {%- endif %}\n    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- \"<|eot_id|>\" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n"""
    config = LlamaNemotronNanoVLConfig8B()

    new_special_tokens = {
        "additional_special_tokens": [
            "<image>",
            "<img>",
            "</img>",
            "<quad>",
            "</quad>",
            "<ref>",
            "</ref>",
            "<box>",
            "</box>",
        ]
    }
    hf_tokenizer.tokenizer.add_special_tokens(new_special_tokens)
    fabric = trainer.to_fabric()

    # Decide whether to import or load the model based on the input arguments
    if args.load_from_mlm:
        # EOS path
        model = fabric.import_model(f"pyt://{args.load_from_mlm}", LlamaNemotronVLModel)
    else:
        model = LlamaNemotronVLModel(config, tokenizer=hf_tokenizer)
        model = fabric.load_model(args.local_model_path, model)

    model = model.module.cuda()
    model.eval()

    raw_image = load_image(args.image_url)
    conversation = [
        {"role": "system", "content": "Answer the questions."},
        {
            "role": "user",
            "content": f"<img><image></img>\n{args.prompt}",
        },
    ]

    prompt = hf_tokenizer.tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    input_ids = hf_tokenizer.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)['input_ids'].cuda()

    processor = AutoImageProcessor.from_pretrained("nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1", trust_remote_code=True)
    imgs = processor.preprocess(raw_image)['pixel_values']
    images = imgs.cuda()
    num_image_tiles = torch.tensor([len(imgs)], dtype=torch.int).cuda()

    generated_ids = input_ids.clone()
    position_ids = (
        torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
    )
    # Greedy generation loop
    for _ in range(100):
        with torch.no_grad():
            output = model(
                images=images,
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
                num_image_tiles=num_image_tiles,
                image_token_index=image_token_index,
            )

            next_token_ids = torch.argmax(output[:, -1], dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)

            input_ids = generated_ids
            position_ids = (
                torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
                .unsqueeze(0)
                .expand_as(input_ids)
            )

            # If the generated token is the end of sequence token, stop generating
            if next_token_ids.item() == hf_tokenizer.tokenizer.eos_token_id:
                break

    # generated_ids[generated_ids == -200] = 0
    generated_texts = hf_tokenizer.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
    logging.info("======== GENERATED TEXT OUTPUT ========")
    logging.info(f"{generated_texts}")
    logging.info("=======================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA Multimodal Inference")
    parser.add_argument(
        "--load_from_mlm",
        type=str,
        default=None,
        help="Flag to indicate whether to load the model from megatron checkpoint.",
    )
    parser.add_argument(
        "--local_model_path",
        type=str,
        default=None,
        help="Local path to the model if not loading from Hugging Face.",
    )
    parser.add_argument(
        "--image_url",
        type=str,
        default="https://github.com/NVIDIA/NeMo/releases/download/v2.3.0/example-image-for-vlm-generation-task.png",
        help="URL of the image to use for inference.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What MOE Switch XXL training what is speed of H100 over A100 and H100 with NV Link over A100?",
        help="Custom prompt to use for inference.",
    )
    args = parser.parse_args()

    main(args)

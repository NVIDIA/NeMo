# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import argparse, base64, io, os
import asyncio, json, time, logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from fastapi import FastAPI

import gradio as gr
import PIL.Image
from omegaconf import OmegaConf

from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.collections.multimodal.parts.utils import create_neva_model_and_processor

CFG_STRING = """
trainer:
  devices: 1
  num_nodes: 1
  accelerator: gpu
  logger: False # logger provided by exp_manager
  precision: bf16 # 16, 32, or bf16

inference:
  greedy: True # Whether or not to use sampling ; use greedy decoding otherwise
  top_k: 0  # The number of highest probability vocabulary tokens to keep for top-k-filtering.
  top_p: 0.9 # If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
  temperature: 0.2 # sampling temperature
  add_BOS: True # add the bos token at the begining of the prompt
  tokens_to_generate: 64 # The minimum length of the sequence to be generated.
  all_probs: False  # whether return the log prob for all the tokens in vocab
  repetition_penalty: 1.2  # The parameter for repetition penalty. 1.0 means no penalty.
  min_tokens_to_generate: 0  # The minimum length of the sequence to be generated.
  compute_logprob: False  # a flag used to compute logprob of all the input text, a very special case of running inference, default False
  end_strings: ["<extra_id_1>","<extra_id_7>",]  # generation will stop when one of these tokens is generated
  images_base_path: /pwd/images
  insert_image_token: null # `left` or `right` or `null`

cluster_type: BCP
tensor_model_parallel_size: 1
pipeline_model_parallel_size: 1
pipeline_model_parallel_split_rank: 0 # used for encoder and decoder model (0 for others)

neva_model_file: /pwd/nemo_experiments/nemo_llava.nemo #neva_22b_tp8_finetuned_v1.nemo neva_8b_tp4_finetuned_v1.nemo
base_model_file: null
checkpoint_dir: null #/pwd/nemo_multimodal/nemo_experiments/nemo_llava_finetune/checkpoints # checkpoint file dir. This is used to load the PTL checkpoint generated during the Kosmos training
checkpoint_name: null #megatron_clip--val_loss=0.41-step=13499-consumed_samples=431904.0.ckpt # PTL checkpoint file name, only used for PTL checkpoint loading
hparams_file: null #/pwd/nemo_multimodal/nemo_experiments/nemo_llava_finetune/version_0/hparams.yaml # model configuration file, only used for PTL checkpoint loading
"""

def predict_impl(input_prompts):

    cfg.inference.tokens_to_generate = 1024

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
        "random_seed": cfg.inference.get('random_seed'),
    }

    # Generate model responses
    responses = model.generate(
        input_prompts=input_prompts,  # Adjust based on your model's requirements
        length_params=length_params,  # Define these parameters as in your original code
        sampling_params=sampling_params,  # Define these parameters as in your original code
        inference_config=cfg,
    )

    return responses



# Function to handle predictions (it also dumps images/prompts)
def predict(prompt, image=None):
    if image is not None:
        with open("prompts.jsonl", "r") as file:
            lines = file.readlines()
            current_line_count = len(lines)

        current_line_count += 1
        # Convert image to RGB and save as .jpg file
        image = image.convert("RGB")
        image_path = f"image-{current_line_count}.jpg"
        image.save(image_path)
        #image.close()

    # Append the prompt to prompts.jsonl file (to pass to rank > 0)
    import json
    with open("prompts.jsonl", "a") as file:
        json_line = json.dumps({"prompt": prompt})
        file.write(json_line + "\n")

    
    input_data = {"prompt": prompt}
    if image is not None:
        input_data["image"] = image_processor(image)
        print("Got image size", image.size, "processed into:", input_data["image"])

    # Call "main" predict (rank = 0)
    responses = predict_impl([input_data])
    return responses[0]["clean_response"].replace("<extra_id_1>", "").strip()


# Function to monitor prompts.jsonl for new lines and call predict in the parallel threads
def monitor_prompts():
    last_line_count = 0
    file_path = "prompts.jsonl"
    
    while True:
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                lines = file.readlines()
                current_line_count = len(lines)
                
                if current_line_count > last_line_count:
                    # New line detected
                    new_lines = lines[last_line_count:]
                    for line in new_lines:
                        data = json.loads(line)
                        prompt = data.get("prompt")                        
                        # Call predict_impl with the new prompt
                        image = None
                        if "<image>" in prompt:
                            image = PIL.Image.open(f"image-{current_line_count}.jpg")

                        predict(prompt, image)
                    
                    last_line_count = current_line_count
        time.sleep(1)



def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def get_vlm_response_neva(history, image, max_tokens = 512):
    if image is not None:
        # PIL image to jpeg
        image = PIL.Image.fromarray(image)
        image = image.convert("RGB")
        print("Passing image size", image.size)

        prompt = "<image>\n"
    else:
        prompt = ""

    for i, turn in enumerate(history):
        if i: prompt += "<extra_id_1>User\n"    # NeMo adds the first one... 
        prompt += turn[0].strip() + "\n"
        if turn[1]:
          prompt += "<extra_id_1>Assistant\n" + turn[1].strip() + "\n"

    prompt = prompt.strip()

    yield predict(prompt, image)           


def bot(history, image):
    history[-1][1] = ""
    # 64 tokens
    print(history)
    for delta in get_vlm_response_neva(history, image):
        print("Delta:", delta)
        history[-1][1] += delta
        yield history


# Minimalistic OpenAI Compatible API implementation
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "neva-gpt-model"
    messages: List[Message]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.2
    stream: Optional[bool] = False

async def process_queue():
    while True:
        batch = []
        while len(batch) < 4:
            try:
                request_entry = await asyncio.wait_for(request_queue.get(), timeout=0.05)
                batch.append(request_entry)
            except asyncio.TimeoutError:
                break
        
        if batch:
            # Text-only for now
            def get_prompt(messages):
                # prompt template gets applied in NeMo already
                # so we only add user content and filter out system
                prompt, first = "", True
                for message in messages:
                    if message.role.lower() == "system": continue

                    if not first: prompt += f"<extra_id_1>{message.role.capitalize()}\n"
                    prompt += message.content.strip() + "\n"
                    first = False

                return prompt.strip()

            input_prompts = [{
                                "prompt": get_prompt(entry["request"].messages),
                                "image" : None                            
                            } for entry in batch]
            responses = predict_impl(input_prompts)
            
            for entry, response in zip(batch, responses):
                response_content = response["clean_response"].replace("<extra_id_1>", "").strip()
                entry["response"].set_result({
                    "id": str(time.time()),
                    "object": "chat.completion",
                    "created": time.time(),
                    "model": entry["request"].model,
                    "choices": [{"message": Message(role="assistant", content=response_content)}],
                })
                logging.info(f"Request {entry['id']} response prepared")






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="distckpt.nemo")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--api", action="store_true")
    args = parser.parse_args()

    # Create Model
    cfg = OmegaConf.create(CFG_STRING)
    cfg.neva_model_file = args.model_path
    cfg.base_model_file = args.model_base
    cfg.tensor_model_parallel_size = args.tp
    cfg.pipeline_model_parallel_size = args.pp
    cfg.trainer.devices = args.tp * args.pp
    model, image_processor,_ = create_neva_model_and_processor(cfg)
    print(predict("<image>\nPlease describe this image in detail.", PIL.Image.open("640px-AgamaSinaita.jpg")))


    with gr.Blocks(analytics_enabled = False) as demo:
        with gr.Row(equal_height=True):
            with gr.Column(scale=4):
                gallery = gr.Image(
                    label="Upload an image",
                )

        with gr.Column(scale=8):
            with gr.Row():
                chatbot = gr.Chatbot(
                    [],
                    elem_id="chatbot",
                    bubble_full_width=False,
                    height = 1024
                )

            with gr.Row():
                with gr.Column(scale=5):
                    txt = gr.Textbox(
                        show_label=False,
                        placeholder="Enter text and press enter",
                        container=False,
                    )

                with gr.Column(scale=1):
                    submit = gr.Button("Submit", variant="primary")

        with gr.Row():
            gr.Examples([
                        ["Hello, please describe this image in detail."],
                        ["Please, ignore the image and tell something about pinguins."],
                    ],
                    txt,
                )

        txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
            bot, [chatbot, gallery], [chatbot], api_name="bot_response", #postprocess = False
        )
        submit_msg = submit.click(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
            bot, [chatbot, gallery], [chatbot], api_name="bot_response", #postprocess = False
        )
        #txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False).then(
        #    bot, chatbot, chatbot
        #)


        if args.api:
            app = FastAPI(title="OpenAI-compatible API")
            app = gr.mount_gradio_app(app, demo, path="/gradio")
            request_queue = asyncio.Queue()


            @app.post("/chat/completions")
            async def chat_completions(request: ChatCompletionRequest):
                request_id = str(time.time())
                request_entry = {"id": request_id, "request": request, "response": asyncio.Future()}
                await request_queue.put(request_entry)
                logging.info(f"Request {request_id} added to the queue")
                response = await request_entry["response"]
                logging.info(f"Request {request_id} processed")
                return response
                
            @app.on_event("startup")
            async def startup_event():
                asyncio.create_task(process_queue())

            import uvicorn
            uvicorn.run(app, host="0.0.0.0", port=8000)
            logging.basicConfig(level=logging.INFO)


        else:
            demo.queue()
            demo.launch(share=True, server_name = "0.0.0.0")

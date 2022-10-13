# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import json

import gradio as gr
import requests

port_num = 5555
headers = {"Content-Type": "application/json"}


def request_data(data):
    resp = requests.put('http://localhost:{}/generate'.format(port_num), data=json.dumps(data), headers=headers)
    sentences = resp.json()['sentences']
    return sentences


def get_generation(prompt, greedy, add_BOS, token_to_gen, min_tokens, temp, top_p, top_k, repetition):
    data = {
        "sentences": [prompt],
        "tokens_to_generate": int(token_to_gen),
        "temperature": temp,
        "add_BOS": add_BOS,
        "top_k": top_k,
        "top_p": top_p,
        "greedy": greedy,
        "all_probs": False,
        "repetition_penalty": repetition,
        "min_tokens_to_generate": int(min_tokens),
    }
    sentences = request_data(data)
    return sentences[0]


def get_demo(share, username, password):
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=2, width=200):
                greedy_flag = gr.Checkbox(label="Greedy")
                add_BOS = gr.Checkbox(label="Add BOS token", value=False)
                token_to_gen = gr.Number(label='Number of Tokens to generate', value=300, type=int)
                min_token_to_gen = gr.Number(label='Min number of Tokens to generate', value=1, type=int)
                temperature = gr.Slider(minimum=0.0, maximum=10.0, value=1.0, label='Temperature', step=0.1)
                top_p = gr.Slider(minimum=0.0, maximum=1.0, step=0.02, value=0.9, label='Top P')
                top_k = gr.Slider(minimum=0, maximum=10000, step=2, value=0, label='Top K')
                repetition_penality = gr.Slider(
                    minimum=0.0, maximum=5.0, step=0.02, value=1.2, label='Repetition penalty'
                )
            with gr.Column(scale=1, min_width=800):
                input_prompt = gr.Textbox(
                    label="Input",
                    value="Ariel was playing basketball. 1 of her shots went in the hoop. 2 of her shots did not go in the hoop. How many shots were there in total?",
                    lines=5,
                )
                output_box = gr.Textbox(value="", label="Output")
                btn = gr.Button(value="Submit")
                btn.click(
                    get_generation,
                    inputs=[
                        input_prompt,
                        greedy_flag,
                        add_BOS,
                        token_to_gen,
                        min_token_to_gen,
                        temperature,
                        top_p,
                        top_k,
                        repetition_penality,
                    ],
                    outputs=[output_box],
                )
    demo.launch(share=share, server_port=13570, server_name='0.0.0.0', auth=(username, password))

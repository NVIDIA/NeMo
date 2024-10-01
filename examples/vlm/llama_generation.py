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

import requests
import torch
from PIL import Image
import argparse

from nemo.collections import vlm

from nemo import lightning as nl


model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

def main(args) -> None:
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        ckpt_load_optimizer=False,
        ckpt_save_optimizer=False,
    )
    trainer = nl.Trainer(
        devices=args.tp_size,
        max_steps=1000,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        val_check_interval=1000,
        limit_val_batches=50,
    )

    fabric = trainer.to_fabric()

    # Decide whether to import or load the model based on the input arguments
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = processor.tokenizer

    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "If I had to write a haiku for this one, it would be: "}
        ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    batch = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    )

    model = vlm.MLlamaModel(
        vlm.MLlamaModelConfig(
            language_model_config=vlm.CrossAttentionTextModelConfig8B(parallel_output=False),
            vision_model_config=vlm.CrossAttentionVisionModelConfig(num_layers=32, hidden_size=1280,
                                                                    num_attention_heads=16, vision_chunk_size=560,
                                                                    vision_max_num_chunks=4, ),
            # vlm.CrossAttentionVisionModelConfig(num_layers=32, hidden_size=1280, num_attention_heads=16, vision_chunk_size=448, vision_max_num_chunks=4,),
        ),
        tokenizer=tokenizer)
    # import pdb; pdb.set_trace()
    # local_model_path = "/root/.cache/nemo/models/evian3-11b-vision-final_vv1_zarr/"
    local_model_path = "/root/.cache/nemo/models/meta-llama/Llama-3.2-11B-Vision-Instruct_zarr/"
    # local_model_path = "/lustre/fsw/coreai_dlalgo_llm/nemo_home/models/evian3-11b-vision-early_vv1_vision_only/"
    model = fabric.load_model(local_model_path, model)
    model = model.module.cuda()
    model.eval()
    model = model.to(torch.bfloat16)

    # input = torch.load("/lustre/fsw/coreai_dlalgo_genai/yuya/evian3/evian3_input.pt")
    # batch = torch.load("/lustre/fsw/coreai_dlalgo_llm/chcui/tmp/mllama_energon_batch.pt")

    input_ids = batch["input_ids"].cuda(non_blocking=True)
    position_ids = (
        torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
    )

    min_prompt_len = position_ids.shape[-1]

    input_ids = input_ids[:, :min_prompt_len]
    generated_ids = input_ids.clone()

    for cur_pos in range(min_prompt_len, min_prompt_len+1):
        with torch.no_grad():
            position_ids = torch.arange(
                0, cur_pos, dtype=torch.long, device="cuda"
            ).reshape(1, -1)

            output = model(
                batch_images=batch["pixel_values"].cuda(non_blocking=True),
                batch_masks=[[[5, 512]]],
                aspect_ratio_ids=batch["aspect_ratio_ids"].cuda(non_blocking=True),
                tokens=generated_ids,
                position_ids=position_ids,
            )

            next_token_ids = torch.argmax(output[:, -1], dim=-1, keepdim=True)
            # Broadcast the tensor from rank 0 to all other ranks
            # torch.distributed.broadcast(next_token_ids, src=0)
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
            if (next_token_ids == tokenizer.eos_token_id).all():
                break

    generated_ids = generated_ids.tolist()
    # print(generated_ids)
    print(tokenizer.decode(generated_ids[0][min_prompt_len:]))
    # print(tokenizer.decode(generated_ids[1][min_prompt_len:]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--devices", type=int, required=False, default=1)
    parser.add_argument("--tp_size", type=int, required=False, default=1)
    parser.add_argument("--pp_size", type=int, required=False, default=1)
    parser.add_argument("--encoder_pp_size", type=int, required=False, default=0)


    args = parser.parse_args()
    main(args)

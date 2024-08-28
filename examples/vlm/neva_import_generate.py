import requests
import torch
from PIL import Image
from transformers import AutoProcessor

from nemo import lightning as nl
from nemo.collections.vlm.neva.model.llava import LlavaModel


def main() -> None:

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

    # Tokenize the input texts
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    # Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
    # Each value in "content" has to be a list of dicts with types ("text", "image")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What are these?"},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    hf_tokenizer = processor.tokenizer

    image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
    raw_image = Image.open(requests.get(image_file, stream=True).raw)
    inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)
    input_ids = inputs['input_ids'].cuda()
    input_ids[input_ids == 32000] = -200
    media = inputs['pixel_values'].cuda()
    media = media.reshape(media.size(0), 1, 1, 3, 336, 336)
    position_ids = (
        torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
    )

    fabric = trainer.to_fabric()
    model = fabric.import_model("hf://llava-hf/llava-1.5-7b-hf", LlavaModel)
    model = model.module.cuda()
    model.eval()
    generated_ids = input_ids.clone()

    # Greedy generation loop
    for _ in range(20):
        with torch.no_grad():
            output = model(
                media=media,
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
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
            if next_token_ids.item() == hf_tokenizer.eos_token_id:
                break
    generated_ids[generated_ids == -200] = 0
    generated_texts = hf_tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
    print(generated_texts)


main()

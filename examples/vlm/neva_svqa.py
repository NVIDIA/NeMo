"""
python /opt/NeMo/examples/vlm/neva_svqa.py \
  --local_model_path '/lustre/fsw/coreai_dlalgo_genai/yuya/debug/neva_finetune_mlp2x_gelu/neva_finetune_pyt_mlp--reduced_train_loss=0.6185-epoch=2-last' \
    --question_file /lustre/fsw/coreai_dlalgo_genai/datasets/eval/scienceqa/llava_test_CQM-A.json \
    --image_folder_path /lustre/fsw/coreai_dlalgo_genai/datasets/eval/scienceqa/images/test \
    --answers_file /opt/test.json \
    --max_length 10

PYTHONPATH=${PYTHONPATH}:/lustre/fsw/coreai_dlalgo_genai/yuya/LLaVA \
python /lustre/fsw/coreai_dlalgo_genai/yuya/LLaVA/llava/eval/eval_science_qa.py \
    --base-dir /lustre/fsw/coreai_dlalgo_genai/datasets/eval/scienceqa \
    --result-file /lustre/fsw/coreai_dlalgo_genai/datasets/eval/scienceqa/answers/${NAME}.jsonl \
    --output-file /lustre/fsw/coreai_dlalgo_genai/datasets/eval/scienceqa/answers/${NAME}_output.jsonl \
    --output-result /lustre/fsw/coreai_dlalgo_genai/datasets/eval/scienceqa/answers/${NAME}_result.json
"""


import argparse
import json
import math
import os

import requests
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

from nemo import lightning as nl
from nemo.collections.vlm import LlavaModel, Llava1_5Config7B
from nemo.utils.get_rank import is_global_rank_zero
from nemo.collections.multimodal.data.neva.neva_dataset import process_image


def generate(model, input_ids, media, position_ids, tokenizer, max_length=20, attention_mask=None):
    """
    Performs greedy generation on the model using provided inputs.

    Args:
        model: The model to use for generation.
        input_ids: The input IDs for text generation.
        media: The processed image inputs (or None if no image is used).
        position_ids: The position IDs for the input tokens.
        tokenizer: Tokenizer used to handle EOS token checks.
        max_length: Maximum length of the generated sequence.
        attention_mask: Mask used for attention during generation (optional).
    """
    generated_ids = input_ids.clone()
    predicted_ids = torch.tensor([], dtype=input_ids.dtype, device=generated_ids.device)
    for _ in range(max_length):
        with torch.no_grad():
            output = model(
                media=media,
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )

            next_token_ids = torch.argmax(output[:, -1], dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
            predicted_ids = torch.cat([predicted_ids, next_token_ids], dim=-1)
            input_ids = generated_ids

            # Update position ids for the new input length
            position_ids = (
                torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
                .unsqueeze(0)
                .expand_as(input_ids)
            )

            # Stop if the end-of-sequence token is generated
            if next_token_ids.item() == tokenizer.eos_token_id:
                break

    return generated_ids, predicted_ids


def main(args) -> None:
    # Trainer setup
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

    # Load processor and tokenizer
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    hf_tokenizer = processor.tokenizer

    # Load model
    fabric = trainer.to_fabric()
    if args.load_from_hf:
        model = fabric.import_model("hf://llava-hf/llava-1.5-7b-hf", LlavaModel)
    else:
        model = LlavaModel(Llava1_5Config7B(), tokenizer=hf_tokenizer)
        model = fabric.load_model(args.local_model_path, model)

    model = model.module.cuda()
    model.eval()

    predict_answers(args, model, processor)


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i: i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def predict_answers(args, model, processor):
    """
    Function to predict answers based on the input questions and the multimodal model.

    Args:
        args: Arguments for processing.
        model: The model used for predictions.
        processor: Processor to handle image and text inputs.
    """
    tokenizer = processor.tokenizer
    image_processor = processor.image_processor

    # Load questions from the question file
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))

    # Set up answers file and ensure directory exists
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    # Only the global rank 0 will write results (for multi-GPU setups)
    if is_global_rank_zero():
        ans_file = open(answers_file, "w")

    # Iterate over the questions
    for i, line in enumerate(tqdm(questions, disable=(not is_global_rank_zero()))):
        idx = line["id"]
        question = line['conversations'][0]['value'].replace('<image>', '').strip()
        cur_prompt = question
        # If the question includes an image, process it
        if 'image' in line:
            cur_prompt = '<image>\n' + cur_prompt
            image_path = os.path.join(args.image_folder_path, line['image'])
            image = Image.open(image_path).convert('RGB')
            image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
        else:
            image = None

        cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."
        # Tokenize the input prompt and prepare input tensors
        prompt = processor.apply_chat_template([
            {
                "role": "system",
                "content": [{"type": "text", "text": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": cur_prompt}]
            }
        ], add_generation_prompt=True)

        # Prepare the input tensors for the model
        inputs = processor(prompt, image, return_tensors='pt').to(0, torch.float16)
        input_ids = inputs['input_ids'][:, 1:].cuda()  # strip bos token
        input_ids[input_ids == 32000] = -200  # Handle specific token in the input
        media = inputs['pixel_values'].cuda().reshape(inputs['pixel_values'].size(0), 3, 336, 336) if image else None
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device).unsqueeze(
            0).expand_as(input_ids)

        attention_mask = inputs.get('attention_mask', None).cuda() if 'attention_mask' in inputs else None

        # Generate the response using the model
        generated_ids, predicted_ids = generate(model, input_ids, media, position_ids, tokenizer, attention_mask=attention_mask, max_length=args.max_length)

        # Post-process and decode the generated response
        generated_texts = tokenizer.batch_decode(predicted_ids, skip_special_tokens=False)
        outputs = generated_texts[0]

        # If global rank zero, write the response to the answers file
        if is_global_rank_zero():
            import shortuuid
            ans_id = shortuuid.uuid()
            ans_file.write(
                json.dumps(
                    {
                        "question_id": idx,
                        "prompt": cur_prompt,
                        "text": outputs,
                        "answer_id": ans_id,
                        "model_id": args.local_model_path,
                        "metadata": {},
                    }
                )
                + "\n"
            )
            ans_file.flush()

    # Close the file if opened
    if is_global_rank_zero():
        ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA Multimodal Inference")
    parser.add_argument(
        "--load_from_hf",
        action="store_true",
        help="Flag to indicate whether to load the model from Hugging Face hub.",
    )
    parser.add_argument(
        "--local_model_path",
        type=str,
        default="default",
        help="Local path to the model if not loading from Hugging Face.",
    )
    parser.add_argument(
        "--image_folder_path",
        type=str,
        default="./images",
        help="Path to the folder containing images for inference.",
    )
    parser.add_argument(
        "--question_file",
        type=str,
        required=True,
        help="Path to the JSON file containing questions.",
    )
    parser.add_argument(
        "--answers_file",
        type=str,
        required=True,
        help="Path to save the answers in JSON format.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=20,
        help="Maximum length for the generated responses.",
    )
    args = parser.parse_args()

    main(args)
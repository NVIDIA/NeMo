"""Gemma3 language model generate"""

import torch
from transformers import AutoTokenizer

from nemo import lightning as nl
from nemo.collections.llm.gpt.model.gemma3 import Gemma3Model

HF_MODEL_NAME = "google/gemma-3-1b-it"


def main():
    """Entrypoint"""

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        sequence_parallel=False,
        setup_optimizers=False,
        store_optimizer_states=False,
    )

    trainer = nl.Trainer(
        accelerator="gpu",
        devices=1,
        num_nodes=1,
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        enable_checkpointing=False,
    )
    fabric = trainer.to_fabric()
    model = fabric.import_model(f"hf://{HF_MODEL_NAME}", Gemma3Model)
    model = model.module.cuda()
    model.eval()

    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Who are you?"},
                ],
            },
        ],
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # print(model)

    with torch.no_grad():
        input_ids = inputs['input_ids'].clone().to("cuda")
        print(input_ids)
        generated_ids = input_ids
        for _ in range(10):
            seq_len = input_ids[0].shape[0]
            position_ids = torch.arange(seq_len, dtype=torch.int64).to("cuda")
            output = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
            )
            next_token_ids = torch.argmax(output[:, -1], dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
            input_ids = generated_ids

        outputs = tokenizer.batch_decode(generated_ids)
        # ['<bos><start_of_turn>user\nWho are you?<end_of_turn>\n<start_of_turn>model\nHi there! I’m Gemma, a large']
        print(outputs)


if __name__ == "__main__":
    main()

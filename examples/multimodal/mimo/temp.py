from nemo.collections.multimodal.mimo.data.mock import MockDataModule
import torch
from nemo.collections.multimodal.mimo.model.base import BaseMimoConfig
from nemo.collections.multimodal.mimo.model.base import BaseMimoModel
from nemo import lightning as nl
from transformers import AutoProcessor
from nemo.collections.llm import import_ckpt

if __name__ == '__main__':
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=2,
        ckpt_include_optimizer=False,
    )
    processor = AutoProcessor.from_pretrained("hf://llava-hf/llava-v1.6-vicuna-7b-hf")
    tokenizer = processor.tokenizer
    trainer = nl.Trainer(
        devices=2,
        max_steps=1000,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="32"),
        val_check_interval=1000,
        limit_val_batches=50,
    )
    fabric = trainer.to_fabric()
    data_module = MockDataModule(vocab_size = tokenizer.vocab_size )
    data_module.setup()
    dataloader = data_module.test_dataloader()
    # base_config = BaseMimoConfig(vocab_size = tokenizer.vocab_size)
    # model = BaseMimoModel(config=base_config, tokenizer=tokenizer)
    # fabric.launch()
    # model.configure_model()
    
    # model = fabric.setup_module(model)
    # import csv
    # state_dict_nemo_module = model.state_dict()

    # # Prepare data: Save only tensor keys with their shapes
    # state_dict_data = [
    #     (key, tuple(tensor.shape))
    #     for key, tensor in state_dict_nemo_module.items()
    #     if isinstance(tensor, torch.Tensor)
    # ]

    # # Save to CSV
    # csv_file_path = "nemo_model_state_dict_keys_shapes.csv"  # Adjust path if needed
    # with open(csv_file_path, mode="w", newline="") as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["Key", "Shape"])  # Header
    #     writer.writerows(state_dict_data)

    # print(f"State dictionary tensor keys and shapes saved to: {csv_file_path}")
    # breakpoint()
    
    # model = model.module
    # device = model.device
    # model.eval()
    # batch = next(iter(dataloader))

    # batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

    # forward_args = {
    #     "images": batch["images"],
    #     "input_ids": batch["tokens"],
    #     "position_ids": batch["position_ids"],
    #     "attention_mask": batch.get("attention_mask", None),
    #     "loss_mask": batch.get("loss_mask", None),
    #     # "labels": batch.get("labels", None),
    #      "labels": None,
    # }
    # fw_out = model(**forward_args)
    # import_ckpt(model=BaseMimoModel(BaseMimoConfig()),
    #         source="hf://llava-hf/llava-v1.6-vicuna-7b-hf",
    #         )
    model = fabric.import_model("hf://llava-hf/llava-v1.6-vicuna-7b-hf", BaseMimoModel)
    model = model.module
    model.eval()
    device = model.device
    batch = next(iter(dataloader))
    
    forward_args = {
        "images": batch["images"],
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch.get("attention_mask", None),
        "loss_mask": batch.get("loss_mask", None),
        # "labels": batch.get("labels", None),
         "labels": None,
    }
    fw_out = model(**forward_args)
    breakpoint()
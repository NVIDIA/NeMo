from nemo import lightning as nl
from nemo.collections import vlm

if __name__ == "__main__":
    gbs = 8
    mbs = 1
    seq_length = 256

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

    data = vlm.MockDataModule(
        seq_length=seq_length,
        global_batch_size=gbs,
        micro_batch_size=mbs,
        tokenizer=None,
        image_processor=None,
        num_workers=0,
    )

    from nemo.collections.vlm.neva.model.llava import Llava1_5Config7B, LlavaModel
    model = LlavaModel(Llava1_5Config7B())
    ckpt_path = model.import_ckpt("hf://llava-hf/llava-1.5-7b-hf")

    # from nemo.collections.llm import LlamaModel, Llama2Config7B
    # model = LlamaModel(Llama2Config7B())
    # ckpt_path = model.import_ckpt("hf://lmsys/vicuna-7b-v1.5")
    print(ckpt_path)
    # trainer.fit(model, data, ckpt_path=ckpt_path)

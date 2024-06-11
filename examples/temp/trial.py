from pathlib import Path
from pytorch_lightning import Trainer
from nemo import lightning as nl
from nemo.collections import llm


if __name__ == '__main__':

    strategy = nl.MegatronStrategy(tensor_model_parallel_size=1)
    trainer = Trainer(  # nl.Trainer(
        devices=1,
        max_steps=100,
        accelerator='gpu',
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision('bf16-mixed')
    )

    data = llm.MockDataModule(global_batch_size=4)

    config = llm.GPTConfigV2(num_layers=4, hidden_size=256, ffn_hidden_size=256, num_attention_heads=4, seq_length=data.seq_length)
    config.optim.lr = 5e-5
    config.tokenizer_filepath = None  # using default GPT2 tokenizer or put filepath to spe tokenizer.model

    model = llm.GPTModelV2(config, trainer=trainer)

    trainer.fit(model, data)
    
    model.save_to("./mini_gpt")

    import os
    os.environ['HF_TOKEN'] = ''
    model.push_to_hf_hub("smajumdar/abc5", pack_nemo_file=False)

import torch

from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning.pytorch.callbacks import ModelTransform
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from pathlib import Path
if __name__ == "__main__":
    trainer = nl.Trainer(
        devices=1,
        max_steps=200,
        accelerator="gpu",
        strategy=nl.MegatronStrategy(
            tensor_model_parallel_size=1,
            ckpt_include_optimizer=False,
        ),        
        # plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        # use_distributed_sampler=False,
        limit_val_batches=5,
        # val_check_interval=200,
        num_sanity_val_steps=0,
    )

    optim = nl.MegatronOptimizerModule(
        config=OptimizerConfig(
            optimizer="adam",
            lr=5e-6,
            use_distributed_optimizer=True,
            bf16=False,
            params_dtype=torch.float32,
        ),
        # lr_scheduler=nl.lr_scheduler.CosineAnnealingScheduler(),
    )

    # tokenizer = get_nmt_tokenizer(model_name='/ckpts/nemotron/nemotron3-8b-nemo/tokenizer.model', tokenizer_model='/ckpts/nemotron/nemotron3-8b-nemo/tokenizer.model')
    # model = llm.NemotronModel(llm.Nemotron3Config8B(), optim=optim)
    # ckpt_path = model.import_ckpt("hf:///ckpts/nemotron/nemotron3-8b-hf-new/")
    tokenizer = get_nmt_tokenizer(library='megatron', model_name='BertWordPieceLowerCase')
    model = llm.BertModel(llm.GoogleBERTBaseConfig(), optim=optim, tokenizer=tokenizer)

    data = llm.BERTPreTrainingDataModule(
        paths=[1., '/aot/datasets/the_pile/bert/my-bert_text_sentence'],
        seq_length=512,
        micro_batch_size=4, 
        global_batch_size=4,
        tokenizer=model.tokenizer, 
        num_workers=0)

    trainer.fit(model, data)

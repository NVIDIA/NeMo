import torch
import argparse
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from pytorch_lightning.loggers import WandbLogger
from nemo.collections.llm.api import _setup
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

def get_args():
    parser = argparse.ArgumentParser(description='Train a small GPT model using NeMo 2.0')
    parser.add_argument('--devices', type=int, help="Number of devices to use for training")
    parser.add_argument('--max-steps', type=int, help="Number of steps to train for")
    parser.add_argument('--experiment-dir', type=str, help="directory to write results and checkpoints to")
    parser.add_argument('--model-path', type=str, help="Path to model checkpoint")
    parser.add_argument('--tokenizer-model-path', type=str, default=None, help="Path to tokenizer model, defaults to None")
    return parser.parse_args()

if __name__ == "__main__":

    args = get_args()
    
    # Checkpoint callback setup
    checkpoint_callback = nl.ModelCheckpoint(
        save_best_model=True,
        save_last=False,
        monitor="reduced_train_loss",
        save_top_k=1,
        every_n_train_steps=10,
        enable_nemo_ckpt_io=False,
        dirpath=args.experiment_dir,
    )

    trainer = nl.Trainer(
        devices=args.devices,
        max_steps=args.max_steps,
        accelerator="gpu",
        strategy=nl.MegatronStrategy(ckpt_include_optimizer=False,
                                     tensor_model_parallel_size=1,),        
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed",
                                          params_dtype=torch.bfloat16,
                                          ),
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        limit_val_batches=5,
        val_check_interval=10,
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
    )

    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=1e-5,
        min_lr=1e-5,
        use_distributed_optimizer=False,
        bf16=True,
    )

    optim = MegatronOptimizerModule(config=opt_config)
    model_config = llm.BaseMambaConfig130m()
    model_config.tokenizer_model_path = args.tokenizer_model_path

    tokenizer = get_nmt_tokenizer(
        library=model_config.tokenizer_library,
        model_name=model_config.tokenizer_name,
        tokenizer_model=model_config.tokenizer_model_path,
        use_fast=True,
    )

    model = llm.GPTModel(
                        model_config,
                        optim=optim, 
                        tokenizer=tokenizer
                    )
    
    ckpt_path = model.import_ckpt(
                                  path="pytorch://"+args.model_path, 
                                  model_config=model_config,
                                )
    
    data = llm.SquadDataModule(
                               seq_length=512, 
                               micro_batch_size=2, 
                               global_batch_size=4,
                               tokenizer=model.tokenizer, 
                               num_workers=0, 
                               pad_to_max_length=True
                            )
    
    app_state = _setup(
        model=model,
        data=data,
        trainer=trainer,
        log=None,
        resume=None,
        optim=optim,
        tokenizer=model.tokenizer,
        model_transform=None,
    )

    trainer.fit(model, data, ckpt_path=ckpt_path)

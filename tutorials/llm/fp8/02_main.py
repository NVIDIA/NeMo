import os                                                                                                                                                                                      
import torch                                                                                                                                                                                
import fiddle as fdl
from typing import List, Optional

from nemo import lightning as nl                                                                                                                                                               
from nemo.collections import llm       
from nemo.collections.llm import import_ckpt
from nemo.lightning.io.mixin import IOMixin
from lightning.pytorch.loggers import TensorBoardLogger,WandbLogger     
from nemo.lightning.pytorch.callbacks import ModelCheckpoint                                                                                                                                   
from nemo.collections.llm.gpt.data.fine_tuning import FineTuningDataModule    
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer                                                                                                              
from nemo.collections.llm.gpt.model.llama import Llama31Config8B, LlamaModel
from nemo.collections.llm.recipes.optim.adam import distributed_fused_adam_with_cosine_annealing                                                                        

sequence_length=8192
tensor_parallel_size=1
pipeline_parallel_size=1
virtual_pipeline_parallel_size=0
context_parallel_size=1
sequence_parallel=False    

hf_tokenizer_path='/workspace/nemo/models/Llama-3.1-8B-Instruct/'                                                                                                                                 

micro_batch_size=4
global_batch_size=256
load_optimizer=False

experiment_name="tensorwise-fp8-sft-llama-3.1-nemo2-mcore"
wandb_project_name='nemo2-sft-tutorial'

learning_rate= 5e-6
warmup_steps=50
min_lr=5e-7

recipe = "tensorwise_fp8"

nodes=1
gpu_devices=8
max_steps=2000
log_every_n_steps=5
val_check_interval=100
limit_val_batches=8  

tokenizer = get_nmt_tokenizer(library='huggingface', model_name=hf_tokenizer_path)
config = Llama31Config8B()
model = LlamaModel(config=config, tokenizer=tokenizer)

experiment_log_dir='/workspace/nemo/sft/nemo2/Llama-3.1-8B-Instruct-logs/results_llama_nemo2{}'.format(experiment_name)
resume_dir='/workspace/nemo/sft/nemo2/Llama-3.1-8B-Instruct-logs/results_llama_nemo2{}'.format(experiment_name)
resume_path='/workspace/nemo/models/Llama-3.1-8B-Instruct-Nemo/' 

if __name__ == "__main__":
    
    train_dl = llm.SquadDataModule(seq_length=sequence_length, tokenizer=hf_tokenizer_path, micro_batch_size=micro_batch_size, global_batch_size=global_batch_size)

    if recipe == "bf16":
        plugins = nl.MegatronMixedPrecision(
        precision="bf16-mixed",)

    if recipe == "delayed":
        plugins = nl.MegatronMixedPrecision(
        precision="bf16-mixed",
        fp8="hybrid",
        fp8_recipe="delayed",
        fp8_margin=0,
        fp8_amax_history_len=1024,
        fp8_amax_compute_algo="max",
        fp8_param_gather=True)
        
    if recipe == "mxfp8":
        plugins = nl.MegatronMixedPrecision(
        precision="bf16-mixed",
        fp8="hybrid",
        fp8_recipe="mxfp8",
        fp8_param_gather=True)
    
    if recipe == "tensorwise_fp8":
        plugins = nl.MegatronMixedPrecision(
        precision="bf16-mixed",
        fp8="hybrid",
        fp8_recipe="tensorwise",
        first_last_layers_bf16=True,
        num_layers_at_start_in_bf16=1,
        num_layers_at_end_in_bf16=1,
        fp8_param_gather=True)
    
    if recipe == "blockwise_fp8":
        plugins = nl.MegatronMixedPrecision(
        precision="bf16-mixed",
        fp8="hybrid",
        fp8_recipe="blockwise",
        fp8_param_gather=True)

    wandb = WandbLogger(
        project=wandb_project_name,
        name=experiment_name)

    optim_config = distributed_fused_adam_with_cosine_annealing(
        max_lr=learning_rate,
        min_lr=min_lr,
        warmup_steps=warmup_steps,
        adam_beta2=0.98
    )

    optim = fdl.build(optim_config)

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=tensor_parallel_size,
        pipeline_model_parallel_size=pipeline_parallel_size,
        pipeline_dtype=torch.bfloat16,
        sequence_parallel=sequence_parallel,
        ckpt_load_optimizer=load_optimizer,
        ckpt_load_strictness="log_all")

    trainer = nl.Trainer(
        num_nodes=nodes,
        devices=gpu_devices,
        max_steps=max_steps,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=val_check_interval,
        limit_val_batches=limit_val_batches,
        accelerator="gpu",
        strategy=strategy,
        plugins=plugins,
        logger=wandb
    )

    resume = nl.AutoResume(
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
        resume_from_directory=resume_dir,
        restore_config=nl.RestoreConfig(path=resume_path) if resume_path else None,)
    
    checkpoint_callback = ModelCheckpoint(
        monitor="reduced_train_loss",
        save_top_k=2,
        every_n_train_steps=val_check_interval,)

    logger = nl.NeMoLogger(
        name=experiment_name,
        log_dir=experiment_log_dir,
        ckpt=checkpoint_callback,
        tensorboard=TensorBoardLogger(os.path.join(experiment_log_dir, experiment_name)),
        update_logger_directory=False,
        wandb=wandb
    )

    llm.finetune(model=model, data=train_dl, trainer=trainer, optim=optim, log=logger, resume=resume)
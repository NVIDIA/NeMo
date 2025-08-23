"""
Recipe factory for creating training recipes with LoRA configuration.
"""

import nemo_run as run
from config import TrainingConfig
from data_modules import CustomHFDataModule

from nemo.collections import llm
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer


def create_lora_recipe(config: TrainingConfig) -> run.Config:
    """
    Create a LoRA fine-tuning recipe based on the provided configuration.

    Args:
        config: Training configuration object

    Returns:
        Configured LoRA fine-tuning recipe
    """
    # Create base recipe
    recipe = llm.hf_auto_model_for_causal_lm.finetune_recipe(
        model_name=config.model.name,
        dir=config.paths.checkpoint_dir,
        name=f"{config.trainer.checkpoint_filename}_recipe",
        num_nodes=config.compute.nodes,
        num_gpus_per_node=config.compute.gpus_per_node,
        peft_scheme='lora',
    )

    # Configure LoRA parameters
    recipe.peft = run.Config(
        llm.peft.LoRA,
        target_modules=config.lora.target_modules,
        dim=config.lora.dim,
        dropout=config.lora.dropout,
        lora_A_init_method=config.lora.lora_A_init_method,
        lora_B_init_method=config.lora.lora_B_init_method,
    )

    # Configure tokenizer
    tokenizer = llm.HFAutoModelForCausalLM.configure_tokenizer(config.model.name)

    # Configure data module
    recipe.data = run.Config(
        CustomHFDataModule,
        path_or_dataset=config.data.dataset_name,
        seq_length=config.data.seq_length,
        micro_batch_size=config.data.micro_batch_size,
        split=config.data.split,
        pad_token_id=tokenizer.tokenizer.eos_token_id,
        tokenizer=run.Config(AutoTokenizer, pretrained_model_name=config.model.name),
    )

    # Configure checkpoint settings
    recipe.log.ckpt.filename = config.trainer.checkpoint_filename + "_v" + str(config.trainer.version)
    recipe.log.ckpt.save_optim_on_train_end = True

    # Configure trainer parameters
    recipe.trainer.max_steps = config.trainer.max_steps
    recipe.trainer.num_sanity_val_steps = config.trainer.num_sanity_val_steps
    recipe.trainer.val_check_interval = config.trainer.val_check_interval
    recipe.trainer.log_every_n_steps = config.trainer.log_every_n_steps
    recipe.trainer.limit_val_batches = 1

    # Configure optimizer
    recipe.optim.optimizer_fn.lr = config.optimizer.lr
    recipe.optim.optimizer_fn.weight_decay = config.optimizer.weight_decay
    recipe.optim.optimizer_fn.betas = config.optimizer.betas

    # Configure learning rate scheduler
    recipe.optim.lr_scheduler.warmup_steps = config.optimizer.warmup_steps
    recipe.optim.lr_scheduler.constant_steps = config.optimizer.constant_steps
    recipe.optim.lr_scheduler.min_lr = config.optimizer.min_lr

    return recipe


def create_full_finetune_recipe(config: TrainingConfig) -> run.Config:
    """
    Create a full fine-tuning recipe (without LoRA) based on the provided configuration.

    Args:
        config: Training configuration object

    Returns:
        Configured full fine-tuning recipe
    """
    # Create base recipe
    recipe = llm.hf_auto_model_for_causal_lm.finetune_recipe(
        model_name=config.model.name,
        dir=config.paths.checkpoint_dir,
        name=f"{config.trainer.checkpoint_filename}_full_recipe",
        num_nodes=config.compute.nodes,
        num_gpus_per_node=config.compute.gpus_per_node,
    )

    # Configure tokenizer
    tokenizer = llm.HFAutoModelForCausalLM.configure_tokenizer(config.model.name)

    # Configure data module
    recipe.data = run.Config(
        CustomHFDataModule,
        path_or_dataset=config.data.dataset_name,
        seq_length=config.data.seq_length,
        micro_batch_size=config.data.micro_batch_size,
        split=config.data.split,
        pad_token_id=tokenizer.tokenizer.eos_token_id,
        tokenizer=run.Config(AutoTokenizer, pretrained_model_name=config.model.name),
    )

    # Configure checkpoint settings
    recipe.log.ckpt.filename = config.trainer.checkpoint_filename

    # Configure trainer parameters
    recipe.trainer.max_steps = config.trainer.max_steps
    recipe.trainer.num_sanity_val_steps = config.trainer.num_sanity_val_steps
    recipe.trainer.val_check_interval = config.trainer.val_check_interval
    recipe.trainer.log_every_n_steps = config.trainer.log_every_n_steps

    # Configure optimizer with potentially different settings for full fine-tuning
    recipe.optim.optimizer_fn.lr = config.optimizer.lr * 0.1  # Lower LR for full fine-tuning
    recipe.optim.optimizer_fn.weight_decay = config.optimizer.weight_decay
    recipe.optim.optimizer_fn.betas = config.optimizer.betas

    # Configure learning rate scheduler
    recipe.optim.lr_scheduler.warmup_steps = config.optimizer.warmup_steps
    recipe.optim.lr_scheduler.constant_steps = config.optimizer.constant_steps
    recipe.optim.lr_scheduler.min_lr = config.optimizer.min_lr

    return recipe


def create_recipe(config: TrainingConfig, recipe_type: str = "lora") -> run.Config:
    """
    Create a training recipe based on the specified type.

    Args:
        config: Training configuration object
        recipe_type: Type of recipe ("lora" or "full")

    Returns:
        Configured training recipe

    Raises:
        ValueError: If recipe_type is not supported
    """
    if recipe_type.lower() == "lora":
        return create_lora_recipe(config)
    elif recipe_type.lower() == "full":
        return create_full_finetune_recipe(config)
    else:
        raise ValueError(f"Unsupported recipe type: {recipe_type}. Choose 'lora' or 'full'.")

#!/usr/bin/env python3
"""
finetune_llama.py

Fine-tunes the LLaMA-3.2 1B embedding model on the AllNLI triplet-format dataset using NeMo.

Input dataset:
- A JSON file (`allnli_triplet_train.json`) containing entries like:
  {
    "query":   "<anchor sentence>",
    "pos_doc": "<positive sentence>",
    "neg_doc": "<negative sentence>"
  }

Input model:
- Pre-trained LLaMA 3.2 1B model in NeMo format (`Llama-3.2-1B.nemo`)

Output:
- Fine-tuned `.nemo` model saved to current directory
"""

import os
import nemo_run as run
from nemo.collections import llm

# Dataset path (produced by download_allnli_triplet.py)
TRAIN_DATA_PATH = "allnli_triplet.json"

# Pretrained LLaMA model checkpoint (converted by import_llama1b.py)
PRETRAINED_NEMO_MODEL = "Llama-3.2-1B.nemo"

# Set NeMo working directory
os.environ["NEMO_HOME"] = os.getcwd()


def get_custom_dataloader(
    data_path,
    dataset_identifier='allnli_llama_triplet',
    seq_length=512,
    micro_batch_size=16,
    global_batch_size=64,
    tokenizer=None,
    num_workers=8
):
    """
    Creates a CustomRetrievalDataModule for triplet training with LLaMA.
    """
    return run.Config(
        llm.CustomRetrievalDataModule,
        data_root=data_path,
        dataset_identifier=dataset_identifier,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        tokenizer=tokenizer,
        num_workers=num_workers,
        query_key="query",
        pos_doc_key="pos_doc",
        neg_doc_key="neg_doc",
    )


def train_llama_on_allnli(json_file_path):
    """
    Fine-tunes the LLaMA-3.2 1B embedding model on the AllNLI triplet dataset.
    """
    pretrained_model_path = os.path.abspath(PRETRAINED_NEMO_MODEL)

    # Create data loader config
    datamodule = get_custom_dataloader(
        data_path=json_file_path,
        dataset_identifier="allnli_llama_triplet"
    )

    # Load fine-tuning recipe
    recipe = llm.recipes.llama_embedding_1b.finetune_recipe(
        name="allnli_llama_finetune",
        resume_path=pretrained_model_path,
        num_nodes=1,
        num_gpus_per_node=1,
    )

    # Customize recipe settings
    recipe.optim.config.lr = 5e-6
    recipe.optim.lr_scheduler.min_lr = 5e-7
    recipe.trainer.max_steps = 100
    recipe.trainer.val_check_interval = 10
    recipe.trainer.limit_val_batches = 5
    recipe.data = datamodule

    # Run training
    run.run(recipe, executor=run.LocalExecutor())


if __name__ == "__main__":
    train_llama_on_allnli(TRAIN_DATA_PATH)


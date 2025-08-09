#!/usr/bin/env python3
"""
finetune_e5.py

Fine-tunes the E5-Large-V2 embedding model on a triplet-format dataset using the NeMo framework.

Expected input:
- A JSON file (`allnli_triplet.json`) containing records with:
  {
    "query":   "<anchor sentence>",
    "pos_doc": "<positive sentence>",
    "neg_doc": "<negative sentence>"
  }

Expected model:
- Pre-trained E5 model in NeMo format (`e5-large-v2.nemo`)

Output:
- Fine-tuned model checkpoint stored in NeMo format
"""

import os
import nemo_run as run
from nemo.collections import llm

# Path to your dataset (downloaded via download_allnli_triplet.py)
TRAIN_DATA_PATH = "allnli_triplet.json"

# Pretrained E5 checkpoint converted via import_e5_large.py
PRETRAINED_NEMO_MODEL = "e5-large-v2.nemo"

# NeMo working directory
os.environ["NEMO_HOME"] = os.getcwd()


def get_custom_dataloader(
    data_path,
    dataset_identifier='allnli_e5_triplet',
    seq_length=512,
    micro_batch_size=16,
    global_batch_size=64,
    tokenizer=None,
    num_workers=8
):
    """
    Creates a CustomRetrievalDataModule for triplet training with E5.
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


def train_e5_on_allnli(json_file_path):
    """
    Main function to fine-tune the E5 model on the AllNLI triplet dataset.
    """
    pretrained_model_path = os.path.abspath(PRETRAINED_NEMO_MODEL)

    # Create data module config
    datamodule = get_custom_dataloader(
        data_path=json_file_path,
        dataset_identifier="allnli_e5_triplet"
    )

    # Load fine-tuning recipe
    recipe = llm.recipes.e5_340m.finetune_recipe(
        name="allnli_e5_large_finetune",
        resume_path=pretrained_model_path,
        num_nodes=1,
        num_gpus_per_node=1,
    )

    # Customize recipe parameters
    recipe.model.config.global_in_batch_negatives = True
    recipe.optim.config.lr = 5e-6
    recipe.optim.lr_scheduler.min_lr = 5e-7
    recipe.trainer.max_steps = 100
    recipe.trainer.val_check_interval = 10
    recipe.trainer.limit_val_batches = 5
    recipe.data = datamodule

    # Launch training
    run.run(recipe, executor=run.LocalExecutor())


if __name__ == "__main__":
    train_e5_on_allnli(TRAIN_DATA_PATH)


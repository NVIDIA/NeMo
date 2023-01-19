import json
import os
from typing import Any, AnyStr, List

import numpy as np
import pytorch_lightning as ptl
import torch
import torch.nn.functional as F
from apex.transformer import parallel_state
from omegaconf import OmegaConf
from omegaconf.omegaconf import open_dict
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer.trainer import Trainer
from scripts.upscale_scripts.models import EmbeddingLinearCombination
from torch import nn
from torch.utils.data import DataLoader, Dataset

from nemo.collections.nlp.data.language_modeling.megatron.gpt_prompt_learning_dataset import GPTPromptLearningDataset
from nemo.collections.nlp.models.language_modeling.megatron_gpt_prompt_learning_model import (
    MegatronGPTPromptLearningModel,
)
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.core.config import hydra_runner


def load_prompt_learning_model(virtual_prompt_model_file, trainer_cfg, base_model_path):
    trainer = Trainer(strategy=NLPDDPStrategy(), **trainer_cfg)
    prompt_learning_cfg = MegatronGPTPromptLearningModel.restore_from(
        virtual_prompt_model_file, trainer=trainer, return_config=True,
    )

    with open_dict(prompt_learning_cfg):
        prompt_learning_cfg.save_nemo_on_validation_end = False
        prompt_learning_cfg.micro_batch_size = 1
        prompt_learning_cfg.global_batch_size = 1
        prompt_learning_cfg.language_model_path = base_model_path

    model = MegatronGPTPromptLearningModel.restore_from(
        restore_path=virtual_prompt_model_file, trainer=trainer, override_config_path=prompt_learning_cfg,
    )
    return model


def get_word_type_embeddings(model):
    word_embeddings = model.frozen_model.model.language_model.embedding.word_embeddings.weight.data
    pos_embeddings = model.frozen_model.model.language_model.embedding.position_embeddings.weight.data
    return word_embeddings, pos_embeddings


def get_dataset(model, data_paths: List[AnyStr]):
    dataset = GPTPromptLearningDataset(
        data=data_paths,
        tokenizer=model.tokenizer,
        virtual_prompt_source=model.virtual_prompt_source,
        task_templates=model.task_templates,
        pseudo_tokens=model.pseudo_tokens,
        pad_token_id=model.pad_token_id,
        max_seq_length=model.frozen_model.cfg.encoder_seq_length,
        min_seq_length=1,
        add_bos=model.cfg.data.get('add_bos', False),
        add_eos=model.cfg.data.get('add_eos', True),
        for_train=True,
        tokens_to_generate=None,
        cache_data_path=None,
        load_cache=None,
    )
    return dataset


class LinearCombinationDataset(Dataset):
    def __init__(self, prompt_embeddings, word_embeddings, num_repeats) -> None:
        super().__init__()
        self.examples = [(word_embeddings, prompt_embeddings) for _ in range(num_repeats)]

    def __len__(self,):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


@hydra_runner(config_path="./", config_name="linear_combination")
def main(cfg) -> None:

    # trainer required for restoring model parallel models
    sm_model = load_prompt_learning_model(cfg.small_prompt_learning_model, cfg.nemo_trainer, cfg.small_model_path)
    sm_word_embeddings = sm_model.frozen_model.model.language_model.embedding.word_embeddings.weight.data
    sm_pos_embeddings = sm_model.frozen_model.model.language_model.embedding.position_embeddings.weight.data
    sm_prompt_learning_embs = sm_model.prompt_table.prompt_table[cfg.taskname].prompt_embeddings.weight.data

    lg_model = load_prompt_learning_model(cfg.large_prompt_learning_model, cfg.nemo_trainer, cfg.large_model_path)
    lg_word_embeddings = lg_model.frozen_model.model.language_model.embedding.word_embeddings.weight.data
    lg_pos_embeddings = lg_model.frozen_model.model.language_model.embedding.position_embeddings.weight.data
    lg_prompt_learning_embs = lg_model.prompt_table.prompt_table[cfg.taskname].prompt_embeddings.weight.data

    sm_prompt_tokens = sm_prompt_learning_embs + sm_pos_embeddings[:sm_prompt_learning_embs.shape[0], :]
    lg_prompt_tokens = lg_prompt_learning_embs + lg_pos_embeddings[:lg_prompt_learning_embs.shape[0], :]

    y_prompts = sm_prompt_tokens
    x_prompts = torch.cat((sm_word_embeddings, sm_pos_embeddings))

    val_y_prompts = lg_prompt_tokens
    val_x_prompts = torch.cat((lg_word_embeddings, lg_pos_embeddings))
    train = LinearCombinationDataset(y_prompts, x_prompts, 100)
    val = LinearCombinationDataset(val_y_prompts, val_x_prompts, 1)
    train_dataloader = DataLoader(
        train, batch_size=1, shuffle=False
    )
    val_dataloader = DataLoader(
        val, batch_size=1, shuffle=False
    )  
    linear_combiner = EmbeddingLinearCombination(x_prompts.shape[0], x_prompts.shape[1], y_prompts.shape[0], cfg.linear_combiner)
    wblogger = WandbLogger(**cfg.linear_combiner.wandb)
    # saves top-K checkpoints based on "val_loss" metric
    with open_dict(cfg.linear_combiner.trainer):
        cfg.linear_combiner.trainer.val_check_interval=1

    trainer = ptl.Trainer(
        **cfg.linear_combiner.trainer, logger=wblogger,
    )
    trainer.fit(model=linear_combiner, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

if __name__ == '__main__':
    main()

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.utils.data as pt_data
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from nemo.collections.common.losses import SmoothedCrossEntropyLoss
from nemo.collections.common.metrics import Perplexity, GlobalAverageLossMetric
from nemo.collections.common.parts import transformer_weights_init
from nemo.collections.nlp.data import TarredOneSideTranslationDataset, TranslationOneSideDataset
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.collections.nlp.modules.common.transformer import TransformerEmbedding, TransformerEncoder
from nemo.core.classes.common import typecheck
from nemo.core.classes.modelPT import ModelPT
from nemo.utils import logging

__all__ = ["TransformerLMModel"]


class TransformerLMModel(ModelPT):
    """
    Left-to-right Transformer language model.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_gpus

        # shared params for dataset and data loaders
        self.dataset_cfg = cfg.dataset

        vocab_file = cfg.language_model.get("vocab_file", None)

        if vocab_file is not None:
            vocab_file = self.register_artifact("language_model.vocab_file", vocab_file)

        tokenizer_model = cfg.language_model.get("tokenizer_model", None)

        if tokenizer_model is not None:
            tokenizer_model = self.register_artifact("language_model.tokenizer_model", tokenizer_model)

        if cfg.language_model.special_tokens:
            special_tokens = OmegaConf.to_container(cfg.language_model.special_tokens, resolve=True)
        else:
            special_tokens = None

        self.tokenizer = get_tokenizer(
            tokenizer_name=cfg.language_model.tokenizer,
            vocab_file=vocab_file,
            special_tokens=special_tokens,
            tokenizer_model=tokenizer_model,
        )

        # make vocabulary size divisible by 8 for fast fp16 training
        vocab_size = 8 * math.ceil(self.tokenizer.vocab_size / 8)

        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        self.embedding_layer = TransformerEmbedding(
            vocab_size=vocab_size,
            hidden_size=cfg.language_model.hidden_size,
            max_sequence_length=cfg.language_model.max_seq_length,
            embedding_dropout=cfg.language_model.get("embedding_dropout", 0.0),
            learn_positional_encodings=False,
        )
        self.encoder = TransformerEncoder(
            num_layers=cfg.language_model.num_layers,
            hidden_size=cfg.language_model.hidden_size,
            mask_future=True,
            num_attention_heads=cfg.language_model.num_attn_heads,
            inner_size=cfg.language_model.inner_size,
            ffn_dropout=cfg.language_model.get("ffn_dropout", 0.0),
            hidden_act=cfg.language_model.get("inner_activation", "relu"),
            attn_score_dropout=cfg.language_model.get("attn_score_dropout", 0.0),
            attn_layer_dropout=cfg.language_model.get("attn_layer_dropout", 0.0),
        )
        self.log_softmax = TokenClassifier(
            hidden_size=cfg.language_model.hidden_size, num_classes=vocab_size, log_softmax=True,
        )

        std_init_range = 1 / math.sqrt(cfg.language_model.hidden_size)
        self.apply(lambda module: transformer_weights_init(module, std_init_range))

        # tie weights of embedding and softmax matrices
        self.log_softmax.mlp.layer0.weight = self.embedding_layer.token_embedding.weight

        self.loss_fn = SmoothedCrossEntropyLoss(pad_id=self.tokenizer.pad_id)
        self.eval_loss = GlobalAverageLossMetric(dist_sync_on_step=False, take_avg_loss=True)
        self.eval_ppl = Perplexity()

    @typecheck()
    def forward(self, input_ids, attention_mask):
        """
        No special modification required for Lightning, define it as you normally would
        in the `nn.Module` in vanilla PyTorch.
        """
        token_embeddings = self.embedding_layer(input_ids)
        hidden_states = self.encoder(token_embeddings, attention_mask)
        log_probs = self.log_softmax(hidden_states=hidden_states)

        return log_probs

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        for i in range(len(batch)):
            if batch[i].ndim == 3:
                # Dataset returns already batched data and the first dimension of size 1
                # added by DataLoader is excess.
                batch[i] = batch[i].squeeze(dim=0)
        ids, mask = batch
        input_ids, labels = ids[:, :-1], ids[:, 1:]
        input_mask = mask[:, :-1]
        log_probs = self(input_ids=input_ids, attention_mask=input_mask)

        train_loss = self.loss_fn(log_probs=log_probs, labels=labels)

        tensorboard_logs = {
            "train_loss": train_loss,
            "lr": self._optimizer.param_groups[0]["lr"],
        }
        return {"loss": train_loss, "log": tensorboard_logs}

    def eval_step(self, batch, batch_idx, mode):
        for i in range(len(batch)):
            if batch[i].ndim == 3:
                # Dataset returns already batched data and the first dimension of size 1
                # added by DataLoader is excess.
                batch[i] = batch[i].squeeze(dim=0)
        ids, mask = batch
        input_ids, labels = ids[:, :-1], ids[:, 1:]
        input_mask, output_mask = mask[:, :-1], mask[:, 1:]
        log_probs = self(input_ids=input_ids, attention_mask=input_mask)

        eval_loss = self.loss_fn(log_probs=log_probs, labels=labels)
        self.eval_loss(loss=eval_loss, num_measurements=log_probs.shape[0] * log_probs.shape[1])
        self.eval_ppl(log_probs=log_probs, labels=labels, mask=output_mask)
        return {}

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'test')
    
    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        return self.eval_step(batch, batch_idx, 'val')

    def eval_epoch_end(self, outputs, mode):
        eval_loss = self.eval_loss.compute()
        eval_ppl = self.eval_ppl.compute()
        ans = {f"{mode}_loss": eval_loss, f"{mode}_ppl": eval_ppl}
        ans['log'] = dict(ans)
        logging.info(f"\n\n\n\n Validation PPL: {np.round(eval_ppl.item(), 2)}")
        return ans

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        self.log_dict(self.eval_epoch_end(outputs, 'val'), sync_dist=True)
        self.eval_loss.reset()
        self.eval_ppl.reset()

    def test_epoch_end(self, outputs):
        return self.eval_epoch_end(outputs, 'test')

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config)

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config)

    def _setup_dataloader_from_config(self, cfg: DictConfig, predict_last_k=0):

        if cfg.get("use_tarred_dataset", False):
            if cfg.get("metadata_file") is None:
                raise FileNotFoundError("Trying to use tarred data set but could not find metadata path in config.")
            else:
                metadata_file = cfg.get('metadata_file')
                with open(metadata_file) as metadata_reader:
                    metadata = json.load(metadata_reader)
                if cfg.get('tar_files') is None:
                    tar_files = metadata.get('tar_files')
                    if tar_files is not None:
                        logging.info(f'Loading from tarred dataset {tar_files}')
                    else:
                        raise FileNotFoundError("Could not find tarred dataset in config or metadata.")
                else:
                    tar_files = cfg.get('tar_files')
                    if metadata.get('tar_files') is not None:
                        raise ValueError(
                            'Tar files specified in config and in metadata file. Tar files should only be specified once.'
                        )
            dataset = TarredOneSideTranslationDataset(
                text_tar_filepaths=tar_files,
                metadata_path=metadata_file,
                tokenizer=self.tokenizer,
                shuffle_n=cfg.get("tar_shuffle_n", 100),
                shard_strategy=cfg.get("shard_strategy", "scatter"),
                global_rank=self.global_rank,
                world_size=self.world_size,
            )
            return torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=1,
                num_workers=cfg.get("num_workers", 2),
                pin_memory=cfg.get("pin_memory", False),
                drop_last=cfg.get("drop_last", False),
            )
        else:
            dataset = TranslationOneSideDataset(
                tokenizer=self.tokenizer,
                dataset=cfg.file_name,
                tokens_in_batch=cfg.tokens_in_batch,
                clean=cfg.get("clean", False),
                max_seq_length=cfg.get("max_seq_length", 512),
                min_seq_length=cfg.get("min_seq_length", 1),
                cache_ids=cfg.get("cache_ids", False),
            )
        if cfg.shuffle:
            sampler = pt_data.RandomSampler(dataset)
        else:
            sampler = pt_data.SequentialSampler(dataset)
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=cfg.get("num_workers", 2),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=cfg.get("drop_last", False),
        )

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass

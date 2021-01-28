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

import math
from typing import Dict, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from nemo.collections.common.losses import SmoothedCrossEntropyLoss
from nemo.collections.common.metrics import Perplexity
from nemo.collections.common.parts import transformer_weights_init
from nemo.collections.nlp.data import L2RLanguageModelingDataset, TarredL2RLanguageModelingDataset
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
        self.global_rank = 0
        self.world_size = 1
        if trainer is not None:
            self.global_rank = (trainer.node_rank * trainer.num_gpus) + trainer.local_rank
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

        if hasattr(self.tokenizer, 'pad_token'):
            pad_id = self.tokenizer.pad_id
        else:
            raise ValueError(
                "The tokenizer must support a special `pad_token`. Provide it using" "the `special_tokens` dictionary."
            )

        self.training_loss = SmoothedCrossEntropyLoss(pad_id=pad_id)
        self.validation_loss = SmoothedCrossEntropyLoss(
            pad_id=pad_id, predict_last_k=self.dataset_cfg.get("predict_last_k", 0),
        )

        self.training_perplexity = Perplexity(dist_sync_on_step=True)
        self.validation_perplexity = Perplexity(compute_on_step=False)

        # Optimizer setup needs to happen after all model weights are ready
        self.setup_optimization()

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
        input_ids, input_mask, labels = batch
        log_probs = self(input_ids=input_ids, attention_mask=input_mask)

        train_loss = self.training_loss(log_probs=log_probs, labels=labels)
        training_perplexity = self.training_perplexity(logits=log_probs)

        tensorboard_logs = {
            "train_loss": train_loss,
            "lr": self._optimizer.param_groups[0]["lr"],
            "train_ppl": training_perplexity,
        }
        return {"loss": train_loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        input_ids, input_mask, labels = batch
        log_probs = self(input_ids=input_ids, attention_mask=input_mask)

        val_loss = self.validation_loss(log_probs=log_probs, labels=labels)
        self.validation_perplexity(logits=log_probs)

        tensorboard_logs = {"val_loss": val_loss}

        return {"val_loss": val_loss, "log": tensorboard_logs}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """

        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        validation_perplexity = self.validation_perplexity.compute()
        tensorboard_logs = {"val_loss": avg_loss, "val_ppl": validation_perplexity}
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        """
        Lightning calls this inside the test loop with the data from the test dataloader
        passed in as `batch`.
        """
        output_dict = self.validation_step(batch, batch_idx)
        result = {"test_loss": output_dict['val_loss'], "log": {}}
        for k, v in output_dict['log'].items():
            new_k = k.replace("val", "test")
            result['log'][new_k] = v

        return result

    def test_epoch_end(self, outputs):
        """
        Called at the end of test step to aggregate outputs.
        :param outputs: list of individual outputs of each test step.
        """

        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        validation_perplexity = self.validation_perplexity.compute()
        tensorboard_logs = {"test_loss": avg_loss, "test_ppl": validation_perplexity}
        return {"test_loss": avg_loss, "log": tensorboard_logs}

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if 'is_tarred' in train_data_config and train_data_config['is_tarred']:
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if isinstance(self._trainer.limit_train_batches, float):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches
                    * math.ceil((len(self._train_dl.dataset) / self.world_size) / train_data_config['batch_size'])
                )

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        self._validation_dl = self._setup_dataloader_from_config(
            cfg=val_data_config, predict_last_k=self.dataset_cfg.get("predict_last_k", 0),
        )

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        self._test_dl = self._setup_dataloader_from_config(
            cfg=test_data_config, predict_last_k=self.dataset_cfg.get("predict_last_k", 0),
        )

    def _setup_dataloader_from_config(self, cfg: DictConfig, predict_last_k=0):
        if cfg.get('use_cache', False):
            logging.info("Constructing tokenized dataset cache...")

        shuffle = cfg.shuffle

        if cfg.get('is_tarred', False):
            if ('tarred_text_filepaths' in cfg and cfg['tarred_text_filepaths'] is None) or (
                'file_name' in cfg and cfg['file_name'] is None
            ):
                logging.warning(
                    "Could not load dataset as `file_name` was None or "
                    f"`tarred_text_filepaths` is None. Provided config : {cfg}"
                )
                return None

            shuffle_n = cfg.get('shuffle_n', 4 * cfg['batch_size']) if shuffle else 0
            dataset = TarredL2RLanguageModelingDataset(
                text_tar_filepaths=cfg['tarred_text_filepaths'],
                metadata_path=cfg['file_name'],
                tokenizer=self.tokenizer,
                max_seq_length=self.dataset_cfg.max_seq_length,
                batch_step=predict_last_k,
                shuffle_n=shuffle_n,
                shard_strategy=cfg.get("tarred_shard_strategy", "scatter"),
                global_rank=self.global_rank,
                world_size=self.world_size,
            )

            shuffle = False
        else:
            if "file_name" in cfg and cfg.file_name is None:
                logging.warning(f"Could not load dataset as `file_name` was None. Provided config : {cfg}")
                return None

            dataset = L2RLanguageModelingDataset(
                tokenizer=self.tokenizer,
                dataset=cfg.file_name,
                max_seq_length=self.dataset_cfg.max_seq_length,
                batch_step=predict_last_k,
                use_cache=cfg.get('use_cache', False),
            )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.dataset_cfg.get("num_workers", 2),
            pin_memory=self.dataset_cfg.get("pin_memory", False),
            drop_last=self.dataset_cfg.get("drop_last", False),
        )

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass

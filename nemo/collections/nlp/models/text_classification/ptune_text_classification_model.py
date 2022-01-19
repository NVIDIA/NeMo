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


import os
from typing import Dict, List, Optional

import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.common.losses import CrossEntropyLoss
from nemo.collections.nlp.data.text_classification.ptune_text_classification_dataset import BankPTextClassificationDataset, token_wrapper 
from nemo.collections.nlp.metrics.classification_report import ClassificationReport
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common import SequenceClassifier
from nemo.collections.nlp.modules.common.lm_utils import get_lm_model
from nemo.collections.nlp.parts.utils_funcs import tensor2list
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.neural_types import NeuralType
from nemo.utils import logging
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.prompt_encoder import PromptEncoder
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.nlp.modules.common.megatron.megatron_init import (
    initialize_model_parallel_for_nemo,
)
from torch.nn.utils.rnn import pad_sequence
from nemo.utils import logging

__all__ = ['PTuneTextClassificationModel']


class PTuneTextClassificationModel(NLPModel, Exportable):

    # @property
    # def input_types(self) -> Optional[Dict[str, NeuralType]]:
    #     return self.bert_model.input_types

    # @property
    # def output_types(self) -> Optional[Dict[str, NeuralType]]:
    #     return self.classifier.output_types

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """Initializes the BERTTextClassifier model."""
        super().__init__(cfg=cfg, trainer=trainer)

        initialize_model_parallel_for_nemo(
            world_size=trainer.world_size,
            global_rank=trainer.global_rank,
            local_rank=trainer.local_rank,
            tensor_model_parallel_size=cfg.get('tensor_model_parallel_size', 1),
            seed=cfg.get('seed', 1234),
        )

        # shared params for dataset and data loaders
        self.dataset_cfg = cfg.dataset
        # tokenizer needs to get initialized before the super.__init__()
        # as dataloaders and datasets need it to process the data
        self.tokenizer = get_nmt_tokenizer(
            library=cfg.tokenizer.library,
            model_name=cfg.tokenizer.type,
            tokenizer_model=self.register_artifact("tokenizer.model", cfg.tokenizer.model),
            vocab_file=self.register_artifact("tokenizer.vocab_file", cfg.tokenizer.vocab_file),
            merges_file=self.register_artifact("tokenizer.merges_file", cfg.tokenizer.merge_file),
        )

        self.class_weights = None

        self.model = MegatronGPTModel.restore_from(self.register_artifact('language_model.nemo_file', cfg.language_model.get('nemo_file', None)),
                                                   trainer=trainer).half()

        for param in self.model.parameters():
            param.requires_grad = cfg.use_lm_finetune

        hidden_size = self.model.cfg.hidden_size



        # register the file containing the labels into the artifacts to get stored in the '.nemo' file later
        self.classes = cfg.dataset.classes

        self.embeddings = self.model.model.language_model.embedding.word_embeddings

        # set allowed vocab set
        self.vocab = self.tokenizer.tokenizer.get_vocab()

        self.allowed_vocab_ids = set(self.vocab[token_wrapper(k)] for k in cfg.dataset.classes)

        # map from id to label
        self.allowed_vocab = {}
        label_ids = {}
        for i, k in enumerate(cfg.dataset.classes):
            self.allowed_vocab[self.vocab[token_wrapper(k)]] = i
            label_ids[k] = i

        # setup to track metrics
        self.classification_report = ClassificationReport(
            num_classes=len(self.classes), label_ids=label_ids, mode='micro', dist_sync_on_step=True
        )


        self.template = cfg.prompt_encoder.template

        self.prompt_encoder = PromptEncoder(
            template=cfg.prompt_encoder.template,
            hidden_size=hidden_size,
            lstm_dropout=cfg.prompt_encoder.dropout
        )

        # load prompt encoder
        self.hidden_size = hidden_size
        self.tokenizer.add_special_tokens({'additional_special_tokens': [cfg.pseudo_token]})

        # if 'megatron' in self.args.model_name:
        #     self.pseudo_token_id = self.tokenizer.tokenizer.convert_tokens_to_ids(
        #         self.args.pseudo_token)
        #     self.pad_token_id = self.tokenizer.eod
        # else:
        self.pseudo_token_id = self.tokenizer.tokenizer.get_vocab()[cfg.pseudo_token]
        self.pad_token_id = self.tokenizer.tokenizer.pad_token_id if self.tokenizer.tokenizer.pad_token_id is not None else self.tokenizer.tokenizer.unk_token_id
        self.spell_length = sum(self.template)

    def embed_input(self, queries):
        bz = queries.shape[0]
        queries_for_embedding = queries.clone()

        queries_for_embedding[(queries == self.pseudo_token_id)] = self.pad_token_id
        raw_embeds = self.embeddings(queries_for_embedding)

        blocked_indices = (queries == self.pseudo_token_id).nonzero().reshape((bz, self.spell_length, 2))[:, :, 1]  # bz
        replace_embeds = self.prompt_encoder()
        for bidx in range(bz):
            for i in range(self.prompt_encoder.spell_length):
                raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]
        return raw_embeds

    def get_query(self, x_h, prompt_tokens, x_t=None):
        max_seq_len = self.model._cfg.encoder_seq_length
        input_token_ids = self.tokenizer.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenizer.tokenize(' ' + x_h))
        cut = 0
        if len(input_token_ids) + sum(self.template) > max_seq_len:
            logging.warning("Input sequence is longer than the LM model max seq, will cut it off to fit")
            cut = len(input_token_ids) + sum(self.template) - max_seq_len
        return [prompt_tokens * self.template[0]
                + input_token_ids[cut:]  # head entity
                + prompt_tokens * self.template[1]
                + (self.tokenizer.tokenizer.convert_tokens_to_ids(
                   self.tokenizer.tokenize(' ' + x_t)) if x_t is not None else [])
                ]

    def forward(self, x_hs, x_ts, return_candidates=False):
        bz = len(x_hs)

        # construct query ids
        prompt_tokens = [self.pseudo_token_id]
        x_ts = [token_wrapper(x_t) for x_t in x_ts]
        queries = [torch.LongTensor(self.get_query(x_hs[i], prompt_tokens)).squeeze(0) for i in range(bz)]
        queries = pad_sequence(queries, True, padding_value=self.pad_token_id).long().to(self.device)

        # construct label ids
        label_ids = torch.LongTensor(self.tokenizer.tokenizer.convert_tokens_to_ids(x_ts)).reshape(
            (bz, -1)).to(self.device)
        attention_mask = queries != self.pad_token_id
        # get embedded input
        inputs_embeds = self.embed_input(queries)

        def megatron_out():
            bz, seq_len, _ = inputs_embeds.shape
            labels = torch.empty_like(queries).fill_(-100).long()  # bz * seq_len
            label_mask = (attention_mask.long().sum(dim=1) - 1).unsqueeze(1)
            labels = labels.scatter_(1, label_mask, label_ids)

            causal_mask = torch.tril(
                torch.ones((bz, seq_len, seq_len),
                           device=self.device)).view(bz, 1,
                                                     seq_len, seq_len)
            r = causal_mask.permute((1, 2, 0, 3)) * attention_mask.int()
            new_atten = r.permute((2, 0, 1, 3))
            new_atten = new_atten < 0.5

            position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device)
            position_ids = position_ids.unsqueeze(0).expand_as(inputs_embeds[:, :, 0])
            position_embeddings = self.model.model.language_model.embedding.position_embeddings(position_ids)
            encoder_input = inputs_embeds + position_embeddings

            output = self.model.model(None, None, encoder_input=encoder_input.half(),
                                      attention_mask=new_atten,
                                      labels=labels)
            loss, logits = output
            floss = (loss[(labels != -100)]).mean()

            pred_ids = torch.argsort(logits, dim=2, descending=True)
            hit1 = 0
            top10 = []
            returned_pred = []
            returned_label = []
            for i in range(bz):
                top10.append([])
                pred_seq = pred_ids[i, label_mask[i, 0]].tolist()
                for pred in pred_seq:
                    if pred in self.allowed_vocab_ids:
                        top10[-1].append(pred)
                        if len(top10[-1]) >= 10:
                            break
                pred = top10[-1][0]
                returned_pred.append(self.allowed_vocab[pred])
                returned_label.append(self.allowed_vocab[label_ids[i, 0].item()])
                if pred == label_ids[i, 0]:
                    hit1 += 1
            if return_candidates:
                return floss, hit1, top10
            return floss, hit1, torch.tensor(returned_pred).to(self.device), torch.tensor(returned_label).to(self.device)
        return megatron_out()

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        xs, ts  = batch
        train_loss, hit1, pred_ids, label_ids = self.forward(xs, ts)

        lr = self._optimizer.param_groups[0]['lr']
        self.log('train_loss', train_loss)
        self.log('lr', lr, prog_bar=True)

        return {
            'loss': train_loss,
            'lr': lr,
        }

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        xs, ts  = batch
        val_loss, hit1 , preds, labels = self.forward(xs, ts)

        tp, fn, fp, _ = self.classification_report(preds, labels)

        return {'val_loss': val_loss, 'tp': tp, 'fn': fn, 'fp': fp}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        if not outputs:
            return {}
        if self.trainer.testing:
            prefix = 'test'
        else:
            prefix = 'val'

        avg_loss = torch.stack([x[f'val_loss'] for x in outputs]).mean()

        # calculate metrics and classification report
        precision, recall, f1, report = self.classification_report.compute()

        logging.info(f'{prefix}_report: {report}')

        self.log(f'{prefix}_loss', avg_loss, prog_bar=True)
        self.log(f'{prefix}_precision', precision)
        self.log(f'{prefix}_f1', f1)
        self.log(f'{prefix}_recall', recall)

        self.classification_report.reset()

    def test_step(self, batch, batch_idx):
        """
        Lightning calls this inside the test loop with the data from the test dataloader
        passed in as `batch`.
        """
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        """
        Called at the end of test to aggregate outputs.
        :param outputs: list of individual outputs of each test step.
        """
        return self.validation_epoch_end(outputs)

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        if not train_data_config or not train_data_config.file_path:
            logging.info(
                f"Dataloader config or file_path for the train is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

        # we need to create/update the loss module by using the weights calculated from the training data
        self.create_loss_module()

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        if not val_data_config or not val_data_config.file_path:
            logging.info(
                f"Dataloader config or file_path for the validation is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config)

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        if not test_data_config or not test_data_config.file_path:
            logging.info(
                f"Dataloader config or file_path for the test is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config)

    def _setup_dataloader_from_config(self, cfg: Dict) -> 'torch.utils.data.DataLoader':
        input_file = cfg.file_path
        if not os.path.exists(input_file):
            raise FileNotFoundError(
                f'{input_file} not found! The data should be be stored in TAB-separated files \n\
                "validation_ds.file_path" and "train_ds.file_path" for train and evaluation respectively. \n\
                Each line of the files contains text sequences, where words are separated with spaces. \n\
                The label of the example is separated with TAB at the end of each line. \n\
                Each line of the files should follow the format: \n\
                [WORD][SPACE][WORD][SPACE][WORD][...][TAB][LABEL]'
            )

        dataset = BankPTextClassificationDataset(
            input_file,
            self._cfg.dataset.classes
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=cfg.get("num_workers", 0),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=cfg.get("drop_last", False),
            collate_fn=dataset.collate_fn,
        )

    @torch.no_grad()
    def classifytext(self, queries: List[str], batch_size: int = 1, max_seq_length: int = -1) -> List[int]:
        """
        Get prediction for the queries
        Args:
            queries: text sequences
            batch_size: batch size to use during inference
            max_seq_length: sequences longer than max_seq_length will get truncated. default -1 disables truncation.
        Returns:
            all_preds: model predictions
        """
        # store predictions for all queries in a single list
        all_preds = []
        mode = self.training
        device = next(self.parameters()).device
        try:
            # Switch model to evaluation mode
            self.eval()
            logging_level = logging.get_verbosity()
            logging.set_verbosity(logging.WARNING)
            dataloader_cfg = {"batch_size": batch_size, "num_workers": 3, "pin_memory": False}
            infer_datalayer = self._setup_infer_dataloader(dataloader_cfg, queries, max_seq_length)

            for i, batch in enumerate(infer_datalayer):
                input_ids, input_type_ids, input_mask, subtokens_mask = batch

                logits = self.forward(
                    input_ids=input_ids.to(device),
                    token_type_ids=input_type_ids.to(device),
                    attention_mask=input_mask.to(device),
                )

                preds = tensor2list(torch.argmax(logits, axis=-1))
                all_preds.extend(preds)
        finally:
            # set mode back to its original value
            self.train(mode=mode)
            logging.set_verbosity(logging_level)
        return all_preds

    def _setup_infer_dataloader(
        self, cfg: Dict, queries: List[str], max_seq_length: int = -1
    ) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a infer data loader.

        Args:
            cfg: config dictionary containing data loader params like batch_size, num_workers and pin_memory
            queries: text
            max_seq_length: maximum length of queries, default is -1 for no limit
        Returns:
            A pytorch DataLoader.
        """
        pass
        # dataset = BankPTextClassificationDataset()
        # return torch.utils.data.DataLoader(
        #     dataset=dataset,
        #     batch_size=cfg["batch_size"],
        #     shuffle=False,
        #     num_workers=cfg.get("num_workers", 0),
        #     pin_memory=cfg.get("pin_memory", False),
        #     drop_last=False,
        #     collate_fn=dataset.collate_fn,
        # )

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass

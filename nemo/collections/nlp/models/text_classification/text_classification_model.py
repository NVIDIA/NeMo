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
from tqdm import tqdm

from nemo.collections.common.losses import CrossEntropyLoss
from nemo.collections.nlp.data.text_classification import TextClassificationDataset, calc_class_weights
from nemo.collections.nlp.metrics.classification_report import ClassificationReport
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common import SequenceClassifier
from nemo.collections.nlp.parts.utils_funcs import tensor2list
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.utils import logging

__all__ = ['TextClassificationModel']


class TextClassificationModel(NLPModel, Exportable):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """Initializes the BERTTextClassifier model."""
        # shared params for dataset and data loaders
        self.dataset_cfg = cfg.dataset
        self.class_weights = None

        super().__init__(cfg=cfg, trainer=trainer)

        self.classifier = SequenceClassifier(
            hidden_size=self.hidden_size,
            num_classes=cfg.dataset.num_classes,
            num_layers=cfg.classifier_head.num_output_layers,
            activation='relu',
            log_softmax=False,
            dropout=cfg.classifier_head.fc_dropout,
            use_transformer_init=True,
            idx_conditioned_on=0,
        )

        self.create_loss_module()

        # setup to track metrics
        self.classification_report = ClassificationReport(
            num_classes=cfg.dataset.num_classes, mode='micro', dist_sync_on_step=True
        )

        # register the file containing the labels into the artifacts to get stored in the '.nemo' file later
        if 'class_labels' in cfg and 'class_labels_file' in cfg.class_labels and cfg.class_labels.class_labels_file:
            self.register_artifact('class_labels.class_labels_file', cfg.class_labels.class_labels_file)

    def create_loss_module(self):
        # create the loss module if it is not yet created by the training data loader
        if not hasattr(self, 'loss'):
            if hasattr(self, 'class_weights') and self.class_weights:
                # You may need to increase the number of epochs for convergence when using weighted_loss
                self.loss = CrossEntropyLoss(weight=self.class_weights)
            else:
                self.loss = CrossEntropyLoss()

    @typecheck()
    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        No special modification required for Lightning, define it as you normally would
        in the `nn.Module` in vanilla PyTorch.
        """
        hidden_states = self.bert_model(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        logits = self.classifier(hidden_states=hidden_states)
        return logits.float()

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        input_ids, input_type_ids, input_mask, labels = batch
        logits = self.forward(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)

        train_loss = self.loss(logits=logits, labels=labels)

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
        input_ids, input_type_ids, input_mask, labels = batch
        logits = self.forward(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)

        val_loss = self.loss(logits=logits, labels=labels)

        preds = torch.argmax(logits, axis=-1)

        tp, fn, fp, _ = self.classification_report(preds, labels)

        return {'val_loss': val_loss, 'tp': tp, 'fn': fn, 'fp': fp}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
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

        # calculate the class weights to be used in the loss function
        if self.cfg.dataset.class_balancing == 'weighted_loss':
            self.class_weights = calc_class_weights(train_data_config.file_path, self.cfg.dataset.num_classes)
        else:
            self.class_weights = None
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

        dataset = TextClassificationDataset(
            tokenizer=self.tokenizer,
            input_file=input_file,
            max_seq_length=self.dataset_cfg.max_seq_length,
            num_samples=cfg.get("num_samples", -1),
            shuffle=cfg.shuffle,
            use_cache=self.dataset_cfg.use_cache,
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
        if True:
            # Switch model to evaluation mode
            self.eval()
            logging_level = logging.get_verbosity()
            logging.set_verbosity(logging.WARNING)
            dataloader_cfg = {"batch_size": batch_size, "num_workers": 20, "pin_memory": False}
            infer_datalayer = self._setup_infer_dataloader(dataloader_cfg, queries, max_seq_length)
            softmax_layer = torch.nn.Softmax(dim=-1)
            for i, batch in tqdm(enumerate(infer_datalayer), total=len(queries)//batch_size + 1):
                input_ids, input_type_ids, input_mask, subtokens_mask = batch
                # logits = self.forward(
                #     input_ids=input_ids.to(device),
                #     token_type_ids=input_type_ids.to(device),
                #     attention_mask=input_mask.to(device),
                # )
                
                # logits_softmax = softmax_layer(logits)
                # toxic_prob = logits_softmax[:, 1]
                all_trunc_toxic_prob = []
                #all_trunc_input_mask_sum = []


                stride = 511
                if input_ids.size(-1) == 1 and False:
                    logits = self.forward(
                        input_ids=input_ids.to(device),
                        token_type_ids=input_type_ids.to(device),
                        attention_mask=input_mask.to(device),
                    )
                    logits_softmax = softmax_layer(logits)
                    toxic_prob = logits_softmax[:, 1]
                else:
                    for j in range(1, input_ids.size(-1), stride):
                        #add one for the CLS token
                        trunc_input_ids = torch.cat((input_ids[:, :1], input_ids[:, j:j+stride]), dim=1)
                        trunc_input_type_ids = torch.cat((input_type_ids[:, :1], input_type_ids[:, j:j+stride]), dim=1)
                        trunc_input_mask = torch.cat((input_mask[:, :1], input_mask[:,  j:j+stride]), dim=1)
                        # print("j: ", j)
                        # print("trunc_input_type_ids: ", trunc_input_type_ids.size())
                        # print("trunc_input_mask: ", trunc_input_mask.size())
                        

                        logits = self.forward(
                            input_ids=trunc_input_ids.to(device),
                            token_type_ids=trunc_input_type_ids.to(device),
                            attention_mask=trunc_input_mask.to(device),
                        )
                        
                        trunc_logits_softmax = softmax_layer(logits)
                        trunc_toxic_prob = trunc_logits_softmax[:, 1]

                        trunc_input_mask_sum = torch.sum(trunc_input_mask,dim=1)
                        trunc_toxic_prob[trunc_input_mask_sum == 1] = float('nan')
                        # print("logits_softmax: ", trunc_logits_softmax)
                        # print("toxic_prob: ", trunc_toxic_prob)
                        all_trunc_toxic_prob.append(trunc_toxic_prob)
                    
                    #mean
                    #toxic_prob = torch.nanmean(torch.stack(all_trunc_toxic_prob, dim=1), dim=1)
                    #max
                    toxic_prob = torch.max(torch.nan_to_num(torch.stack(all_trunc_toxic_prob, dim=1)), dim=1).values
                    # raise ValueError
                # logits_softmax = softmax_layer(logits)
                # toxic_prob = logits_softmax[:, 1]
                
                
                preds = tensor2list(toxic_prob)
                all_preds.extend(preds)
       
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
        dataset = TextClassificationDataset(tokenizer=self.tokenizer, queries=queries, max_seq_length=max_seq_length)
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg.get("num_workers", 0),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=False,
            collate_fn=dataset.collate_fn,
        )

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass

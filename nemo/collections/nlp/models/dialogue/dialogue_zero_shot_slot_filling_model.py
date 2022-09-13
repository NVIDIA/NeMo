# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import csv
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from transformers import BertModel

from nemo.collections.common.losses import AggregatorLoss, CrossEntropyLoss
from nemo.collections.common.parts import MultiLayerPerceptron
from nemo.collections.nlp.data.dialogue.data_processor.assistant_data_processor import DialogueAssistantDataProcessor
from nemo.collections.nlp.data.dialogue.dataset.dialogue_zero_shot_slot_filling_dataset import (
    DialogueZeroShotSlotFillingDataset,
)
from nemo.collections.nlp.data.intent_slot_classification import IntentSlotDataDesc
from nemo.collections.nlp.metrics.classification_report import ClassificationReport
from nemo.collections.nlp.metrics.dialogue_metrics import DialogueClassificationMetrics
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.core import PretrainedModelInfo
from nemo.core.classes import typecheck
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
from nemo.utils import logging


class DialogueZeroShotSlotFillingModel(NLPModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """
        Zero Shot Slot Filling Model
        based on part of the ReFinED paper
        https://aclanthology.org/2022.naacl-industry.24.pdf
        """
        self.max_seq_length = cfg.dataset.max_seq_length
        self.cfg = cfg
        # Check the presence of data_dir.
        if not cfg.dataset.data_dir or not os.path.exists(cfg.dataset.data_dir):
            # Set default values of data_desc.
            self._set_defaults_data_desc(cfg)
        else:
            self.data_dir = cfg.dataset.data_dir
            # Update configuration of data_desc.
            self._set_data_desc_to_cfg(cfg, cfg.dataset.data_dir, cfg.train_ds, cfg.validation_ds)
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        # Initialize MultiLayerPerceptron for predicting IOB class
        self.bio_mlp = MultiLayerPerceptron(
            hidden_size=self.hidden_size, num_classes=3, num_layers=2, activation='relu', log_softmax=True,
        )

        self.mention_projection_mlp = torch.nn.Linear(self.hidden_size, 300, bias=False, device=self.device)
        self.description_projection_mlp = torch.nn.Linear(self.hidden_size, 300, bias=False, device=self.device)

        # Initialize slot description
        self._set_slot_descriptions(cfg.dataset.data_dir)

        # Initialize description embeddings
        self.description_embeddings = self.get_description_embeddings(self.slot_descriptions)

        # Set-up losses and classification report
        self._setup_losses_and_classfication_report()

        # Set-up label ID for empty slot
        self._set_label_id_for_empty_slot(cfg.dataset.data_dir)

    def _set_label_id_for_empty_slot(self, data_dir):
        """
        Set label id for empty slot using the majority class (first line) in train_slot_stats.tsv
        """
        file_path_for_stats = os.path.join(data_dir, "train_slot_stats.tsv")
        with open(file_path_for_stats) as f:
            self.label_id_for_empty_slot = int(next(f).strip().split('\t')[0])

    def _set_slot_descriptions(self, data_dir):
        """Method read slot description file"""
        description_file_name = os.path.join(data_dir, "description.slots.csv")
        with open(description_file_name) as f:
            descriptions = [line.strip() for line in f.readlines()]

        self.slot_descriptions = descriptions

    @staticmethod
    def _set_defaults_data_desc(cfg):
        """
        Method makes sure that cfg.data_desc params are set.
        If not, set's them to "dummy" defaults.
        """
        if not hasattr(cfg, "data_desc"):
            OmegaConf.set_struct(cfg, False)
            cfg.data_desc = {}
            # Slots.
            cfg.data_desc.slot_labels = " "
            cfg.data_desc.slot_label_ids = {" ": 0}
            cfg.data_desc.slot_weights = [1]

            cfg.data_desc.pad_label = "O"
            OmegaConf.set_struct(cfg, True)

    def _set_data_desc_to_cfg(self, cfg, data_dir, train_ds, validation_ds):
        """Method creates IntentSlotDataDesc and copies generated values to cfg.data_desc."""
        # Save data from data desc to config - so it can be reused later, e.g. in inference.
        data_desc = IntentSlotDataDesc(data_dir=data_dir, modes=[train_ds.prefix, validation_ds.prefix])
        OmegaConf.set_struct(cfg, False)
        if not hasattr(cfg, "data_desc") or cfg.data_desc is None:
            cfg.data_desc = {}
        # Slots.
        cfg.data_desc.slot_labels = list(data_desc.slots_label_ids.keys())
        cfg.data_desc.slot_label_ids = data_desc.slots_label_ids
        cfg.data_desc.slot_weights = data_desc.slot_weights

        cfg.data_desc.pad_label = data_desc.pad_label

        # for older(pre - 1.0.0.b3) configs compatibility
        if not hasattr(cfg, "class_labels") or cfg.class_labels is None:
            cfg.class_labels = {}
            cfg.class_labels = OmegaConf.create(
                {'intent_labels_file': 'intent_labels.csv', 'slot_labels_file': 'slot_labels.csv'}
            )

        slot_labels_file = os.path.join(data_dir, cfg.class_labels.slot_labels_file)
        intent_labels_file = os.path.join(data_dir, cfg.class_labels.intent_labels_file)
        self._save_label_ids(data_desc.slots_label_ids, slot_labels_file)
        self._save_label_ids(data_desc.intents_label_ids, intent_labels_file)

        self.register_artifact('class_labels.intent_labels_file', intent_labels_file)
        self.register_artifact('class_labels.slot_labels_file', slot_labels_file)
        OmegaConf.set_struct(cfg, True)

    @staticmethod
    def _save_label_ids(label_ids: Dict[str, int], filename: str) -> None:
        """This method saves label ids map to a file"""
        with open(filename, 'w') as out:
            labels, _ = zip(*sorted(label_ids.items(), key=lambda x: x[1]))
            out.write('\n'.join(labels))
            logging.info(f'Labels: {label_ids}')
            logging.info(f'Labels mapping saved to : {out.name}')

    def _setup_losses_and_classfication_report(self):
        """Method reconfigures the classifier depending on the settings of model cfg.data_desc"""

        self.bio_slot_loss = CrossEntropyLoss(logits_ndim=3)
        self.slot_loss = CrossEntropyLoss(logits_ndim=3)
        self.total_loss = AggregatorLoss(
            num_inputs=2, weights=[self.cfg.bio_slot_loss_weight, 1 - self.cfg.bio_slot_loss_weight]
        )

        # setup to track metrics
        self.bio_slot_classification_report = ClassificationReport(
            num_classes=len([0, 1, 2]), label_ids={0: 0, 1: 1, 2: 2}, dist_sync_on_step=True, mode='micro',
        )

        self.slot_similarity_classification_report = ClassificationReport(
            num_classes=len(self.slot_descriptions),
            label_ids={slot_class.split("\t")[0]: idx for idx, slot_class in enumerate(self.slot_descriptions)},
            dist_sync_on_step=True,
            mode='micro',
        )

        self.overall_slot_classification_report = ClassificationReport(
            num_classes=len(self.slot_descriptions),
            label_ids={slot_class.split("\t")[0]: idx for idx, slot_class in enumerate(self.slot_descriptions)},
            dist_sync_on_step=True,
            mode='micro',
        )

    def update_data_dir_for_training(self, data_dir: str, train_ds, validation_ds) -> None:
        """
        Update data directory and get data stats with Data Descriptor.
        Also, reconfigures the classifier - to cope with data with e.g. different number of slots.

        Args:
            data_dir: path to data directory
        """
        logging.info(f'Setting data_dir to {data_dir}.')
        self.data_dir = data_dir
        # Update configuration with new data.
        self._set_data_desc_to_cfg(self.cfg, data_dir, train_ds, validation_ds)
        self._setup_losses_and_classfication_report()

    def update_data_dir_for_testing(self, data_dir) -> None:
        """
        Update data directory.

        Args:
            data_dir: path to data directory
        """
        logging.info(f'Setting data_dir to {data_dir}.')
        self.data_dir = data_dir

    def get_description_embeddings(self, types_descriptions):
        """
        Generate one description's embedding

        Args:
            types_descriptions: list of slot description
            eg. ["food type\tdrinks menu vegetarian main desserts sides"]
        Returns:
            description embedding by taking final layer embedding for the [CLS] token,
            which is then projected to shared embedding space
        """

        model = BertModel.from_pretrained("bert-base-uncased")
        reader = csv.reader(types_descriptions, delimiter="\t")
        types, descriptions = zip(*reader)
        inputs = self.tokenizer.tokenizer(
            types, descriptions, return_tensors="pt", max_length=128, truncation="only_second", padding="max_length"
        )
        with torch.no_grad():
            output = model(**inputs)
            return output.last_hidden_state[:, 0, :].to(self.device)

    @staticmethod
    def get_entity_embedding_from_hidden_states(bio_slot_labels, hidden_states):
        '''
        Generate mean pooling embedding per entity mention from hidden_states
        Args:
            bio_slot_labels: tensor of size (batch_size, num_tokens, num_entities), that maps tokens to entity matches using BIO slot class of each word token
            hidden_states: tensor of size (batch_size, num_tokens, hidden_state_dim)

            eg,
            send me a  wake up alert at seven am tomorrow morning
            0    0  0  1    2  0     0  1     2  1        1
            0: 'O' other
            1: 'B' Begin
            2: 'I' Inside
            bio_slot_labels = [[1, 0, 0, 1, 2, 0, 1]]
            hidden_states = torch.ones((1,7,3))
            [1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]
        '''
        _, max_token_len, hidden_states_dim = hidden_states.size()
        mention_hidden_states = []

        for one_bio_slot_labels, one_hidden_states in zip(bio_slot_labels, hidden_states):
            mention_states = []
            for idx, one_mask_value in enumerate(one_bio_slot_labels):
                if one_mask_value == 1:
                    mention_states.append([])
                if mention_states and one_mask_value in [1, 2]:
                    mention_states[-1].append(one_hidden_states[idx])

            mention_states = [torch.mean(torch.stack(mention_state), 0) for mention_state in mention_states]
            mention_states += [
                torch.zeros(hidden_states_dim).to(hidden_states.device)
                for i in range(max_token_len - len(mention_states))
            ]
            mention_hidden_states.append(torch.stack(mention_states))

        return torch.stack(mention_hidden_states)

    @typecheck()
    def forward(self, input_ids, attention_mask, token_type_ids, bio_slot_labels):
        """
        No special modification required for Lightning, define it as you normally would
        in the `nn.Module` in vanilla PyTorch.
        """
        if self._cfg.tokenizer.get('library', '') == 'megatron':
            hidden_states, _ = self.bert_model(input_ids, attention_mask, tokentype_ids=token_type_ids, lm_labels=None)
        else:
            hidden_states = self.bert_model(
                input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
            )

        bio_slot_logits = self.bio_mlp(hidden_states)

        # mention_hidden_states_pad: batch_size * max_token_len * hiddenstate_dim
        mention_hidden_states_pad = DialogueZeroShotSlotFillingModel.get_entity_embedding_from_hidden_states(
            bio_slot_labels, hidden_states
        )
        dot_product_score = self.get_entity_description_score(mention_hidden_states_pad)
        dot_product_score_log_softmax = torch.log_softmax(dot_product_score, dim=-1)

        predicted_bio_slot = torch.argmax(bio_slot_logits, dim=-1)
        predicted_mention_hidden_states_pad = DialogueZeroShotSlotFillingModel.get_entity_embedding_from_hidden_states(
            predicted_bio_slot, hidden_states
        )
        predicted_dot_product_score = self.get_entity_description_score(predicted_mention_hidden_states_pad)
        predicted_dot_product_score_log_softmax = torch.log_softmax(predicted_dot_product_score, dim=-1)

        return bio_slot_logits, dot_product_score_log_softmax, predicted_dot_product_score_log_softmax

    def predict(
        self, texts: Union[str, List[str]], slot_descriptions: List[str] = None
    ) -> Tuple[torch.tensor, List[List[str]], torch.tensor]:
        """
        Method for performing inference on text given a list of slot types and descriptions
        Args:
            texts: A single string or a list of strings on which inference needs to be performed
            slot_descriptions: A list of slot types and descriptions separated by a tab.
                e.g. drink type\tThe type of drink
        """
        if isinstance(texts, str):
            texts = [texts]
        (
            input_ids,
            token_type_ids,
            attention_masks,
            loss_mask,
            subtokens_masks,
            _,
        ) = DialogueZeroShotSlotFillingDataset.get_features(
            texts,
            self.max_seq_length,
            self.tokenizer,
            pad_label=self.cfg.data_desc.pad_label,
            word_level_slots=None,
            ignore_extra_tokens=True,
            ignore_start_end=True,
        )
        input_ids = torch.tensor(input_ids, dtype=torch.int32, device=self.device)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.int32, device=self.device)
        attention_masks = torch.tensor(attention_masks, dtype=torch.int32, device=self.device)
        subtokens_masks = torch.tensor(subtokens_masks, dtype=torch.int32, device=self.device)

        hidden_states = self.bert_model(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_masks
        )
        bio_slot_logits = self.bio_mlp(hidden_states)
        predicted_bio_slot = torch.argmax(bio_slot_logits, dim=-1)
        predicted_mention_hidden_states_pad = DialogueZeroShotSlotFillingModel.get_entity_embedding_from_hidden_states(
            predicted_bio_slot, hidden_states
        )
        predicted_dot_product_score = self.get_entity_description_score(
            predicted_mention_hidden_states_pad, slot_descriptions
        )
        predicted_slot_similarity_preds = torch.argmax(predicted_dot_product_score, dim=-1)

        predicted_slot_class_batch = DialogueZeroShotSlotFillingModel.align_mention_to_tokens(
            predicted_bio_slot, predicted_slot_similarity_preds, label_id_for_empty_slot=self.label_id_for_empty_slot
        )

        utterance_tokens = [
            self.get_utterance_tokens(single_input_ids, single_subtokens_masks)
            for single_input_ids, single_subtokens_masks in zip(input_ids, subtokens_masks)
        ]

        return predicted_slot_class_batch, utterance_tokens, predicted_bio_slot

    def calculate_loss(
        self,
        bio_slot_logits,
        dot_product_score_log_softmax,
        loss_mask,
        bio_slot_labels,
        mention_labels,
        mention_loss_mask,
    ):
        # calculate combined loss for bio and slots
        bio_slot_loss = self.bio_slot_loss(logits=bio_slot_logits, labels=bio_slot_labels, loss_mask=loss_mask)

        # otherwise cross-entropy function returns error if all sentences in the batch have no entity
        if torch.sum(mention_loss_mask).item() == 0.0:
            slot_loss_mention_and_description = 0.0
        else:
            slot_loss_mention_and_description = self.slot_loss(
                logits=dot_product_score_log_softmax, labels=mention_labels, loss_mask=mention_loss_mask
            )

        loss = self.total_loss(loss_1=bio_slot_loss, loss_2=slot_loss_mention_and_description)
        return loss

    def on_fit_start(self) -> None:
        self.description_embeddings = self.description_embeddings.to(self.device)

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        (
            input_ids,
            token_type_ids,
            attention_mask,
            loss_mask,
            subtokens_mask,
            slot_labels,
            bio_slot_labels,
            mention_labels,
            mention_loss_mask,
        ) = batch
        bio_slot_logits, dot_product_score_log_softmax, _ = self(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            bio_slot_labels=bio_slot_labels,
        )

        train_loss = self.calculate_loss(
            bio_slot_logits,
            dot_product_score_log_softmax,
            loss_mask,
            bio_slot_labels,
            mention_labels,
            mention_loss_mask,
        )

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
        (
            input_ids,
            token_type_ids,
            attention_mask,
            loss_mask,
            subtokens_mask,
            slot_labels,
            bio_slot_labels,
            mention_labels,
            mention_loss_mask,
        ) = batch
        bio_slot_logits, dot_product_score_log_softmax, predicted_dot_product_score_log_softmax = self(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            bio_slot_labels=bio_slot_labels,
        )
        val_loss = self.calculate_loss(
            bio_slot_logits,
            dot_product_score_log_softmax,
            loss_mask,
            bio_slot_labels,
            mention_labels,
            mention_loss_mask,
        )

        # bio slots prediction
        loss_mask = loss_mask > 0.5
        bio_slot_preds = torch.argmax(bio_slot_logits, axis=-1)
        self.bio_slot_classification_report.update(bio_slot_preds[loss_mask], bio_slot_labels[loss_mask])

        # mention class prediction based on ground truth slots, at mention level
        mention_loss_mask = mention_loss_mask > 0.5
        slot_similarity_preds = torch.argmax(dot_product_score_log_softmax, axis=-1)
        self.slot_similarity_classification_report.update(
            slot_similarity_preds[mention_loss_mask], mention_labels[mention_loss_mask]
        )

        # mention class prediction based on predicted bio slots, at token level
        subtokens_mask = subtokens_mask > 0.5
        predicted_slot_similarity_preds = torch.argmax(predicted_dot_product_score_log_softmax, axis=-1)
        predicted_slot_class_batch = DialogueZeroShotSlotFillingModel.align_mention_to_tokens(
            bio_slot_preds, predicted_slot_similarity_preds, label_id_for_empty_slot=self.label_id_for_empty_slot
        )
        self.overall_slot_classification_report.update(
            predicted_slot_class_batch[subtokens_mask], slot_labels[subtokens_mask]
        )

        return {
            'val_loss': val_loss,
            'bio_slot_tp': self.bio_slot_classification_report.tp,
            'bio_slot_fn': self.bio_slot_classification_report.fn,
            'bio_slot_fp': self.bio_slot_classification_report.fp,
            'slot_similarity_tp': self.slot_similarity_classification_report.tp,
            'slot_similarity_fn': self.slot_similarity_classification_report.fn,
            'slot_similarity_fp': self.slot_similarity_classification_report.fp,
            'overall_slot_tp': self.overall_slot_classification_report.tp,
            'overall_slot_fn': self.overall_slot_classification_report.fn,
            'overall_slot_fp': self.overall_slot_classification_report.fp,
            'bio_slot_preds': bio_slot_preds,
            'bio_slot_labels': bio_slot_labels,
            'slot_similarity_preds': slot_similarity_preds,
            'overall_slot_preds': predicted_slot_class_batch,
            'slot_labels': slot_labels,
            'input': input_ids,
            'subtokens_mask': subtokens_mask,
        }

    @staticmethod
    def align_mention_to_tokens(bio_labels, mention_labels, label_id_for_empty_slot=0):
        """
        Align the mentions label to token level based on bio labels.

        Args:
            bio_labels: (batch_size*max_token_length) [[1, 0, 0, 1, 2], [1, 2, 2, 0, 0]]
            mention_labels: [[2, 4, 0, 0, 0], [52, 0, 0, 0, 0]]

        Returns:
            predicted_slot_class: [[2, 0, 0, 4, 4], [52, 52, 52, 0, 0]]
        """
        slot_class = []
        for one_bio_labels, one_mention_labels in zip(bio_labels, mention_labels):
            start_and_end = DialogueZeroShotSlotFillingModel.get_start_and_end_for_bio(one_bio_labels)
            one_slot_class = torch.ones(len(one_mention_labels), dtype=torch.int32) * label_id_for_empty_slot

            for idx, one_start_and_end in enumerate(start_and_end):
                start, exclusive_end = one_start_and_end
                for idy in range(start, exclusive_end):
                    one_slot_class[idy] = one_mention_labels[idx]
            slot_class.append(one_slot_class)

        slot_class = torch.stack(slot_class).to(bio_labels.device)
        return slot_class

    @staticmethod
    def get_start_and_end_for_bio(one_bio_labels):
        """
        Getting the start and end of the mention from BIO(0, 1, 2) sequence.
        Mention is start with B(with label 1) and follow by I(with label 2).
        Args:
            one_bio_labels: tensor with size (max_token_length, )
            eg. [1, 0, 0, 1, 2]

        Returns:
            start_and_end: list of mention's start and end.
            List length is the number of mentions in one_bio_labels.
            eg. [[0, 0], [3, 4]]
        """
        start_and_end = []
        i = 0
        while i < len(one_bio_labels):
            # if encounter the first B (with label 1),
            # increment counter while encountering I (with label 2)
            if one_bio_labels[i] == 1:
                start = i
                i += 1
                while i < len(one_bio_labels) and one_bio_labels[i] == 2:
                    i += 1
                exclusive_end = i
                start_and_end.append([start, exclusive_end])
            # if encounter O (with label 0) or I (with label 2) without a preceding B (with label 1)
            # increment counter
            elif one_bio_labels[i] in [0, 2]:
                i += 1
        return start_and_end

    def get_entity_description_score(self, mention_hidden_states_pad, entity_types_descriptions=None):
        """
        Score entity description based on the dot product similarity between mention and description
        Args:
            mention_hidden_states_pad: mention embedding of size (batch_size * max_num_mention * hidden_state_dim)
            entity_types_descriptions: entity types and descriptions List[types\tdescriptions]

        Returns:
            dot product score matrix of given mention embedding and description embedding (batch_size * max_num_mention * num_slot_class)

        """
        if entity_types_descriptions:
            entity_type_embeddings = self.get_description_embeddings(entity_types_descriptions)
            projected_description_embedding = self.description_projection_mlp(entity_type_embeddings)
        else:
            projected_description_embedding = self.description_projection_mlp(self.description_embeddings)

        projected_mention_embedding = self.mention_projection_mlp(mention_hidden_states_pad)
        dot_product_score = torch.matmul(projected_mention_embedding, torch.t(projected_description_embedding))

        return dot_product_score

    @staticmethod
    def get_continuous_slots(slot_ids, utterance_tokens):
        """
        Extract continuous spans of slot_ids
        Args:
            slot_ids: list of str representing slot of each word token
            For instance, 'O', 'email_address', 'email_address', 'email_address', 'O', 'O', 'O', 'O']
            Corresponds to ['enter', 'atdfd@yahoo', 'dot', 'com', 'into', 'my', 'contact', 'list']
            utterance_tokens: A list of utterance tokens
        Returns:
            list of str where each element is a slot name-value pair
            e.g. ['email_address(atdfd@yahoo dot com)']

        """
        slot_id_stack = []
        position_stack = []
        for i, slot_id in enumerate(slot_ids):
            if not slot_id_stack or slot_id != slot_id_stack[-1]:
                slot_id_stack.append(slot_id)
                position_stack.append([])
            position_stack[-1].append(i)

        slot_id_to_start_and_exclusive_end = {
            slot_id_stack[i]: [position_stack[i][0], position_stack[i][-1] + 1]
            for i in range(len(position_stack))
            if slot_id_stack[i] != 'O'
        }

        slot_to_words = {
            slot: ' '.join(utterance_tokens[position[0] : position[1]])
            for slot, position in slot_id_to_start_and_exclusive_end.items()
        }

        slot_name_and_values = ["{}({})".format(slot, value) for slot, value in slot_to_words.items()]

        return slot_name_and_values

    def get_unified_metrics(self, outputs):
        slot_preds = []
        slot_labels = []
        subtokens_mask = []
        input_ids = []

        for output in outputs:
            subtokens_mask += output["subtokens_mask"]
            input_ids += output["input"]
            slot_labels += output["slot_labels"]
            slot_preds += output['overall_slot_preds']

        predicted_slots = self.mask_unused_subword_slots(slot_preds, subtokens_mask)
        ground_truth_slots = self.mask_unused_subword_slots(slot_labels, subtokens_mask)

        all_generated_slots = []
        all_ground_truth_slots = []
        all_utterances = []

        for i in range(len(predicted_slots)):
            utterance_tokens = self.get_utterance_tokens(input_ids[i], subtokens_mask[i])
            ground_truth_slot_names = ground_truth_slots[i].split()
            predicted_slot_names = predicted_slots[i].split()

            processed_ground_truth_slots = DialogueZeroShotSlotFillingModel.get_continuous_slots(
                ground_truth_slot_names, utterance_tokens
            )
            processed_predicted_slots = DialogueZeroShotSlotFillingModel.get_continuous_slots(
                predicted_slot_names, utterance_tokens
            )

            all_generated_slots.append(processed_predicted_slots)
            all_ground_truth_slots.append(processed_ground_truth_slots)
            all_utterances.append(' '.join(utterance_tokens))

        os.makedirs(self.cfg.dataset.dialogues_example_dir, exist_ok=True)
        filename = os.path.join(self.cfg.dataset.dialogues_example_dir, "predictions.jsonl")

        DialogueClassificationMetrics.save_slot_predictions(
            filename, all_generated_slots, all_ground_truth_slots, all_utterances,
        )

        (
            slot_precision,
            slot_recall,
            slot_f1,
            slot_joint_goal_accuracy,
        ) = DialogueClassificationMetrics.get_slot_filling_metrics(all_generated_slots, all_ground_truth_slots)

        return slot_precision, slot_recall, slot_f1, slot_joint_goal_accuracy

    def get_utterance_tokens(self, token_ids, token_masks):
        """
        Get utterance tokens based on initial utterance tokenization using token_masks,
        which shows the starting subtoken of each utterance token.
        Args:
            token_ids: IntTensor of size (max_seq_len, )
            token_masks: BoolTensor of size (max_seq_len, )
        Returns
            token_list: List of Str (list of tokens with len <= max_seq_len)
        """
        tokens_stack = []
        tokens = self.tokenizer.tokenizer.convert_ids_to_tokens(token_ids)

        for token_idx, token in enumerate(tokens):
            if token_masks[token_idx].item():
                tokens_stack.append([token])
            elif tokens_stack:
                clean_token = (
                    token.replace("##", '')
                    .replace(self.tokenizer.tokenizer.sep_token, '')
                    .replace(self.tokenizer.tokenizer.pad_token, '')
                )
                tokens_stack[-1].append(clean_token)
        token_list = [''.join(token) for token in tokens_stack]
        return token_list

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """

        (
            unified_slot_precision,
            unified_slot_recall,
            unified_slot_f1,
            unified_slot_joint_goal_accuracy,
        ) = self.get_unified_metrics(outputs)

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        # calculate metrics and log classification report
        (
            bio_slot_precision,
            bio_slot_recall,
            bio_slot_f1,
            bio_slot_report,
        ) = self.bio_slot_classification_report.compute()
        logging.info(f'BIO slot report: {bio_slot_report}')

        (
            slot_similarity_precision,
            slot_similarity_recall,
            slot_similarity_f1,
            slot_similarity_report,
        ) = self.slot_similarity_classification_report.compute()
        logging.info(f'Slot similarity report: {slot_similarity_report}')

        (
            overall_slot_precision,
            overall_slot_recall,
            overall_slot_f1,
            overall_slot_report,
        ) = self.overall_slot_classification_report.compute()
        logging.info(f'Overall slot report: {overall_slot_report}')

        self.log('val_loss', avg_loss)
        self.log('bio_slot_precision', bio_slot_precision)
        self.log('bio_slot_recall', bio_slot_recall)
        self.log('bio_slot_f1', bio_slot_f1)
        self.log('slot_similarity_precision', slot_similarity_precision)
        self.log('slot_similarity_recall', slot_similarity_recall)
        self.log('slot_similarity_f1', slot_similarity_f1)
        self.log('overall_slot_precision', overall_slot_precision)
        self.log('overall_slot_recall', overall_slot_recall)
        self.log('overall_slot_f1', overall_slot_f1)
        self.log('unified_slot_precision', unified_slot_precision)
        self.log('unified_slot_recall', unified_slot_recall)
        self.log('unified_slot_f1', unified_slot_f1)
        self.log('unified_slot_joint_goal_accuracy', unified_slot_joint_goal_accuracy)

        self.bio_slot_classification_report.reset()
        self.slot_similarity_classification_report.reset()
        self.overall_slot_classification_report.reset()

        return {
            'val_loss': avg_loss,
            'slot_similarity_precision': slot_similarity_precision,
            'slot_similarity_recall': slot_similarity_recall,
            'slot_similarity_f1': slot_similarity_f1,
            'overall_slot_precision': overall_slot_precision,
            'overall_slot_recall': overall_slot_recall,
            'overall_slot_f1': overall_slot_f1,
            'unified_slot_precision': unified_slot_precision,
            'unified_slot_recall': unified_slot_recall,
            'unified_slot_f1': unified_slot_f1,
            'unified_slot_joint_goal_accuracy': unified_slot_joint_goal_accuracy,
        }

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
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config, dataset_split='train')

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config, dataset_split='dev')

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config, dataset_split='test')

    def _setup_dataloader_from_config(self, cfg: DictConfig, dataset_split: str):
        data_processor = DialogueAssistantDataProcessor(self.data_dir, self.tokenizer, cfg=self.cfg.dataset)

        dataset = DialogueZeroShotSlotFillingDataset(
            dataset_split,
            data_processor,
            self.tokenizer,
            self.cfg.dataset,  # this is the model.dataset cfg, which is diff from train_ds cfg etc
        )

        return DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=cfg.drop_last,
            collate_fn=dataset.collate_fn,
        )

    @classmethod
    def restore_from(
        cls,
        restore_path: str,
        override_config_path: Optional[Union[OmegaConf, str]] = None,
        map_location: Optional[torch.device] = None,
        strict: bool = True,
        return_config: bool = False,
        save_restore_connector: SaveRestoreConnector = None,
        trainer: Optional[Trainer] = None,
    ):
        instance: DialogueZeroShotSlotFillingModel = super().restore_from(
            restore_path, override_config_path, map_location, strict, return_config, save_restore_connector, trainer
        )
        instance.description_embeddings = instance.description_embeddings.to(instance.device)
        return instance

    def update_data_dirs(self, data_dir: str, dialogues_example_dir: str):
        """
        Update data directories

        Args:
            data_dir: path to data directory
            dialogues_example_dir: path to preprocessed dialogues example directory, if not exists will be created.
        """
        if not os.path.exists(data_dir):
            raise ValueError(f"{data_dir} is not found")
        self.cfg.dataset.data_dir = data_dir
        self.cfg.dataset.dialogues_example_dir = dialogues_example_dir
        logging.info(f'Setting model.dataset.data_dir to {data_dir}.')
        logging.info(f'Setting model.dataset.dialogues_example_dir to {dialogues_example_dir}.')

    def mask_unused_subword_slots(self, slots, subtokens_mask):
        slot_labels = self.cfg.data_desc.slot_labels

        if 'B-' in slot_labels[1] or 'I-' in slot_labels[1]:
            inference_slot_labels = []
            for label_name in slot_labels:
                inference_label_name = label_name.split('-')[-1]
                if inference_label_name not in inference_slot_labels:
                    inference_slot_labels.append(inference_label_name)
        else:
            inference_slot_labels = slot_labels

        predicted_slots = []
        for slots_query, mask_query in zip(slots, subtokens_mask):
            query_slots = ''
            for slot, mask in zip(slots_query, mask_query):
                if mask == 1:
                    query_slots += inference_slot_labels[int(slot)] + ' '
            predicted_slots.append(query_slots.strip())
        return predicted_slots

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        return

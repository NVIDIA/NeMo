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
import pickle
from typing import Dict, List, Optional

import torch
from transformers import BertTokenizer, BertModel
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from nemo.collections.common.losses import AggregatorLoss, CrossEntropyLoss
from nemo.collections.common.parts import MultiLayerPerceptron
from nemo.collections.nlp.data.dialogue.data_processor.assistant_data_processor import DialogueAssistantDataProcessor
from nemo.collections.nlp.data.dialogue.dataset.dialogue_bert_dataset import DialogueBERTDataset
from nemo.collections.nlp.data.intent_slot_classification import IntentSlotDataDesc, IntentSlotInferenceDataset
from nemo.collections.nlp.metrics.classification_report import ClassificationReport
from nemo.collections.nlp.metrics.dialogue_metrics import DialogueClassificationMetrics
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common import SequenceTokenClassifier
from nemo.collections.nlp.parts.utils_funcs import tensor2list
from nemo.core.classes import typecheck
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging

UNIQUE_COUNTER = 0
UNIQUE_EPOCH_COUNTER = 0
class IntentBIOTypeClassificationModel(NLPModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """ Initializes BERT Joint Intent and Slot model.
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

        # Initialize Classifier.
        self._reconfigure_classifier()
        
        # Initialize MultiLayerPerceptron
        self.slot_mlp = MultiLayerPerceptron(
            hidden_size=self.hidden_size,
            num_classes=len(self.cfg.data_desc.slot_labels),
            num_layers=2,
            activation='relu',
            log_softmax=True,
        )

        # Project mention or description embedding
        self.project_mlp = MultiLayerPerceptron(
            hidden_size=self.hidden_size,
            num_classes=300,
            num_layers=2,
            activation='relu',
            log_softmax=True,
        )
         
        # Initialize slot description
        self._get_slot_description(cfg.dataset.data_dir)

    def _get_slot_description(self, data_dir):
        """ Method read slot description file """
        description_file_name = data_dir + "/description.slots.csv"
        with open(description_file_name) as f:
            descriptions = f.read().strip().split('\n')
        # print(len(descriptions))
        # print(descriptions)
        self.slot_descriptions = descriptions

    def _set_defaults_data_desc(self, cfg):
        """
        Method makes sure that cfg.data_desc params are set.
        If not, set's them to "dummy" defaults.
        """
        if not hasattr(cfg, "data_desc"):
            OmegaConf.set_struct(cfg, False)
            cfg.data_desc = {}
            # Intents.
            cfg.data_desc.intent_labels = " "
            cfg.data_desc.intent_label_ids = {" ": 0}
            cfg.data_desc.intent_weights = [1]
            # Slots.
            cfg.data_desc.slot_labels = " "
            cfg.data_desc.slot_label_ids = {" ": 0}
            cfg.data_desc.slot_weights = [1]

            cfg.data_desc.pad_label = "O"
            OmegaConf.set_struct(cfg, True)

    def _set_data_desc_to_cfg(self, cfg, data_dir, train_ds, validation_ds):
        """ Method creates IntentSlotDataDesc and copies generated values to cfg.data_desc. """
        # Save data from data desc to config - so it can be reused later, e.g. in inference.
        data_desc = IntentSlotDataDesc(data_dir=data_dir, modes=[train_ds.prefix, validation_ds.prefix])
        OmegaConf.set_struct(cfg, False)
        if not hasattr(cfg, "data_desc") or cfg.data_desc is None:
            cfg.data_desc = {}
        # Intents.
        cfg.data_desc.intent_labels = list(data_desc.intents_label_ids.keys())
        cfg.data_desc.intent_label_ids = data_desc.intents_label_ids
        cfg.data_desc.intent_weights = data_desc.intent_weights
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

    def _save_label_ids(self, label_ids: Dict[str, int], filename: str) -> None:
        """ Saves label ids map to a file """
        with open(filename, 'w') as out:
            labels, _ = zip(*sorted(label_ids.items(), key=lambda x: x[1]))
            out.write('\n'.join(labels))
            logging.info(f'Labels: {label_ids}')
            logging.info(f'Labels mapping saved to : {out.name}')

    def _reconfigure_classifier(self):
        """ Method reconfigures the classifier depending on the settings of model cfg.data_desc """

        self.classifier = SequenceTokenClassifier(
            hidden_size=self.hidden_size,
            num_intents=len(self.cfg.data_desc.intent_labels),
            num_slots=3,
            # num_slots=len(self.cfg.data_desc.slot_labels),
            dropout=self.cfg.classifier_head.fc_dropout,
            num_layers=self.cfg.classifier_head.num_output_layers,
            log_softmax=False,
        )

        # define losses
        if self.cfg.class_balancing == 'weighted_loss':
            # You may need to increase the number of epochs for convergence when using weighted_loss
            self.intent_loss = CrossEntropyLoss(logits_ndim=2, weight=self.cfg.data_desc.intent_weights)
            self.slot_loss = CrossEntropyLoss(logits_ndim=3, weight=self.cfg.data_desc.slot_weights)
        else:
            self.intent_loss = CrossEntropyLoss(logits_ndim=2)
            self.slot_loss = CrossEntropyLoss(logits_ndim=3)
            self.bio_slot_loss = CrossEntropyLoss(logits_ndim=3)

        slot_loss_weight = 1 - self.cfg.intent_loss_weight - self.cfg.bio_slot_loss_weight
        self.total_loss = AggregatorLoss(
            num_inputs=3, weights=[self.cfg.intent_loss_weight, slot_loss_weight, self.cfg.bio_slot_loss_weight]
        )

        # setup to track metrics
        self.intent_classification_report = ClassificationReport(
            num_classes=len(self.cfg.data_desc.intent_labels),
            label_ids=self.cfg.data_desc.intent_label_ids,
            dist_sync_on_step=True,
            mode='micro',
        )
        self.slot_classification_report = ClassificationReport(
            num_classes=len(self.cfg.data_desc.slot_labels),
            label_ids=self.cfg.data_desc.slot_label_ids,
            dist_sync_on_step=True,
            mode='micro',
        )

        self.bio_slot_classification_report = ClassificationReport(
            num_classes=len([0, 1, 2]),
            label_ids={0:0, 1:1, 2:2},
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
        # Reconfigure the classifier for different settings (number of intents, slots etc.).
        self._reconfigure_classifier()

    def update_data_dir_for_testing(self, data_dir) -> None:
        """
        Update data directory.

        Args:
            data_dir: path to data directory
        """
        logging.info(f'Setting data_dir to {data_dir}.')
        self.data_dir = data_dir
    
    def get_description_embedding(self, one_description):
        """
        Generate one description's embedding

        Args:
            one_description: each slot description
            eg. food type\tdrinks menu vegetarian main desserts sides
        Returns:
            description embedding by taking final layer embedding for the [CLS] token.
        """

        model = BertModel.from_pretrained("bert-base-uncased")

        tokens_of_entity_label = one_description.split('\t')[0] # tokens of the entity label
        entity_description = one_description.split('\t')[1] # entity description in DB

        inputs = self.tokenizer.tokenizer(tokens_of_entity_label, 
                        entity_description, 
                        return_tensors="pt",
                        max_length=128,
                        truncation="only_second",
                        padding="max_length",)
        outputs = model(**inputs) # the actual input will be a little different 
        last_hidden_states = outputs.last_hidden_state
        
        return last_hidden_states[:, 0]
        
    @staticmethod
    def get_entity_embedding_from_hidden_states(mention_mask, hidden_states):
        '''
        Generate mean pooling embedding per entity mention from hidden_states
        Args:
            mention_mask: tensor of size (batch_size, num_tokens, num_entities), that maps tokens to entity matches using BIO slot class of each word token
            hidden_states: tensor of size (batch_size, num_tokens, hidden_state_dim)

            eg,
            send me a  wake up alert at seven am tomorrow morning
            0    0  0  1    2  0     0  1     2  1        1
            0: 'O' other
            1: 'B' Begin
            2: 'I' Inside
            mention_mask = [[1, 0, 0, 1, 2, 0, 1]]
            hidden_states = torch.ones((1,7,3))
            [1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]


        '''
        _, max_token_len, hidden_states_dim = hidden_states.size()
        mention_hidden_states=[]

        for one_mention_mask, one_hidden_states in zip(mention_mask, hidden_states):
            mention_states = []
            for idx, one_mask_value in enumerate(one_mention_mask):
                if one_mask_value == 1:
                    mention_states.append([])
                if one_mask_value in [1, 2]:
                    mention_states[-1].append(one_hidden_states[idx])
            
            mention_states = [torch.mean(torch.stack(mention_state),0) for mention_state in mention_states]
            mention_states += [torch.zeros((hidden_states_dim)).to(hidden_states.device) for i in range(max_token_len-len(mention_states))]
            mention_hidden_states.append(torch.stack(mention_states))
        
        return torch.stack(mention_hidden_states)

    @typecheck()
    def forward(self, input_ids, attention_mask, token_type_ids, mention_mask, mention_loss_mask):
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

        from os.path import exists
        global UNIQUE_EPOCH_COUNTER

        # print("START DUMP hidden_states")
        # path_to_file="/home/lilee/pickle/hidden_states"+str(UNIQUE_EPOCH_COUNTER)+".pickle"
        # if exists(path_to_file)==False:
        #     with open(path_to_file, 'wb') as f:
        #         # Pickle the 'data' dictionary using the highest protocol available.
        #         pickle.dump(hidden_states, f, pickle.HIGHEST_PROTOCOL)        
        # print("END DUMP hidden_states")


        # intent_logits, slot_logits = self.classifier(hidden_states=hidden_states)
        intent_logits, bio_slot_logits = self.classifier(hidden_states=hidden_states)

        # print("START DUMP bio_slot_logits")
        # path_to_file="/home/lilee/pickle/bio_slot_logits"+str(UNIQUE_EPOCH_COUNTER)+".pickle"
        # if exists(path_to_file)==False:
        #     with open(path_to_file, 'wb') as f:
        #         # Pickle the 'data' dictionary using the highest protocol available.
        #         pickle.dump(bio_slot_logits, f, pickle.HIGHEST_PROTOCOL)
        # print("END DUMP bio_slot_logits")

        # current:bug; hidden_states#(batch*128*768) zzz
        
        # mention_hidden_states_pad: b*128*hiddenstate_dim
        mention_hidden_states_pad = IntentBIOTypeClassificationModel.get_entity_embedding_from_hidden_states(mention_mask, hidden_states)
        
        """
        RUN TIME error cause here (may because too many for loop, may have better way to implement):
            try to implement 3.7 entity description score from ReFinED paper
            input is:
                mention_hidden_states_pad: b*7*hiddenstate_dim  (mention's embedding)
                    eg. [[1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
                mention_loss_mask: b*7 (mention's mask)
                    eg. [[1, 1, 1, 0, 0, 0, 0]]
                self.slot_descriptions: every slot class's description
                self.get_description_embedding: a method to get one description's embedding from BERT Model
        """
        
        score_matrix = [] # this is the matrix store all dot score between mention and every description embedding
        for one_input_sentence, one_input_sentence_mask in zip(mention_hidden_states_pad, mention_loss_mask):
            one_sentence_score = []
            for one_mention_emb, every_mention_mask in zip(one_input_sentence, one_input_sentence_mask):
                if every_mention_mask == True: #only calculate dot product for unmask mention, save time 
                    # print("one_mention_emb")
                    # print(one_mention_emb.size())
                    one_mention_score = []
                    for one_description in self.slot_descriptions:
                        one_description_embedding = self.project_mlp(self.get_description_embedding(one_description).to(hidden_states.device))[0] #(1*300)[0] --> (300)
                        one_mention_embedding = self.project_mlp(one_mention_emb) #(300)
                        # print("one_description_embedding size")
                        # print(one_description_embedding.size())
                        # print("one_mention_embedding size")
                        # print(one_mention_embedding.size())
                        one_score = torch.dot(one_description_embedding, one_mention_embedding)
                        one_mention_score.append(one_score)
                    one_mention_score = torch.stack(one_mention_score)
                    one_sentence_score.append(one_mention_score)
                else:
                    one_sentence_score.append(torch.zeros(len(self.slot_descriptions)).to(hidden_states.device))
            one_sentence_score = torch.stack(one_sentence_score)
            score_matrix.append(one_sentence_score)

        score_matrix = torch.stack(score_matrix)
        print("score_matrix SIZE!!!")
        print(score_matrix.size())

        

        # dump
        # mention_hidden_states
        # hidden_states
        # mention_mask

        # print("START DUMP mention_hidden_states_pad, hidden_states, mention_mask")
        # path_to_file="/home/lilee/pickle/mention_hidden_states_pad"+str(UNIQUE_EPOCH_COUNTER)+".pickle"
        # if exists(path_to_file)==False:
        #     with open(path_to_file, 'wb') as f:
        #         # Pickle the 'data' dictionary using the highest protocol available.
        #         pickle.dump(mention_hidden_states_pad, f, pickle.HIGHEST_PROTOCOL)
        
        # path_to_file="/home/lilee/pickle/mention_mask"+str(UNIQUE_EPOCH_COUNTER)+".pickle"
        # if exists(path_to_file)==False:
        #     with open(path_to_file, 'wb') as f:
        #         # Pickle the 'data' dictionary using the highest protocol available.
        #         pickle.dump(mention_mask, f, pickle.HIGHEST_PROTOCOL)
        
        # print("END DUMP mention_hidden_states_pad, hidden_states, mention_mask")


        slot_logits = self.slot_mlp(mention_hidden_states_pad)

        # print("START DUMP slot_logits")
        # path_to_file="/home/lilee/pickle/slot_logits"+str(UNIQUE_EPOCH_COUNTER)+".pickle"
        # if exists(path_to_file)==False:
        #     with open(path_to_file, 'wb') as f:
        #         # Pickle the 'data' dictionary using the highest protocol available.
        #         pickle.dump(slot_logits, f, pickle.HIGHEST_PROTOCOL)
        # print("END DUMP slot_logits")

        UNIQUE_EPOCH_COUNTER +=1

        return intent_logits, slot_logits, bio_slot_logits

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        input_ids, input_type_ids, input_mask, loss_mask, subtokens_mask, intent_labels, slot_labels, bio_slot_labels, bio_mention_labels, mention_loss_mask = batch
        intent_logits, slot_logits, bio_slot_logits = self(
            input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask, mention_mask=bio_slot_labels, mention_loss_mask=mention_loss_mask
        )

        # calculate combined loss for intents and slots
        intent_loss = self.intent_loss(logits=intent_logits, labels=intent_labels)
        # slot_loss = self.slot_loss(logits=slot_logits, labels=slot_labels, loss_mask=loss_mask)
        slot_loss = self.slot_loss(logits=slot_logits, labels=bio_mention_labels, loss_mask=mention_loss_mask)
        bio_slot_loss = self.bio_slot_loss(logits=bio_slot_logits, labels=bio_slot_labels, loss_mask=loss_mask)

        train_loss = self.total_loss(loss_1=intent_loss, loss_2=slot_loss, loss_3=bio_slot_loss)
        # train_loss = self.total_loss(loss_1=intent_loss, loss_2=slot_loss)
        
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
        input_ids, input_type_ids, input_mask, loss_mask, subtokens_mask, intent_labels, slot_labels, bio_slot_labels, bio_mention_labels, mention_loss_mask = batch
        intent_logits, slot_logits, bio_slot_logits = self(
            input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask, mention_mask=bio_slot_labels
        )


        # calculate combined loss for intents and slots
        intent_loss = self.intent_loss(logits=intent_logits, labels=intent_labels)
        # slot_loss = self.slot_loss(logits=slot_logits, labels=slot_labels, loss_mask=loss_mask)
        slot_loss = self.slot_loss(logits=slot_logits, labels=bio_mention_labels, loss_mask=mention_loss_mask)
        bio_slot_loss = self.bio_slot_loss(logits=bio_slot_logits, labels=bio_slot_labels, loss_mask=loss_mask)

        val_loss = self.total_loss(loss_1=intent_loss, loss_2=slot_loss, loss_3=bio_slot_loss)
        # val_loss = self.total_loss(loss_1=intent_loss, loss_2=slot_loss)

        # calculate accuracy metrics for intents and slot reporting
        # intents
        intent_preds = torch.argmax(intent_logits, axis=-1)
        self.intent_classification_report.update(intent_preds, intent_labels)
        # slots

        # subtokens_mask = subtokens_mask > 0.5
        mention_loss_mask = mention_loss_mask > 0.5
        slot_preds = torch.argmax(slot_logits, axis=-1)
        # self.slot_classification_report.update(slot_preds[subtokens_mask], slot_labels[subtokens_mask])
        self.slot_classification_report.update(slot_preds[mention_loss_mask], bio_mention_labels[mention_loss_mask])

        loss_mask = loss_mask > 0.5
        bio_slot_preds = torch.argmax(bio_slot_logits, axis=-1)
        self.bio_slot_classification_report.update(bio_slot_preds[loss_mask], bio_slot_labels[loss_mask])

        return {
            'val_loss': val_loss,
            'intent_tp': self.intent_classification_report.tp,
            'intent_fn': self.intent_classification_report.fn,
            'intent_fp': self.intent_classification_report.fp,
            'slot_tp': self.slot_classification_report.tp,
            'slot_fn': self.slot_classification_report.fn,
            'slot_fp': self.slot_classification_report.fp,
            'bio_slot_tp': self.bio_slot_classification_report.tp,
            'bio_slot_fn': self.bio_slot_classification_report.fn,
            'bio_slot_fp': self.bio_slot_classification_report.fp,
            'intent_preds': intent_preds,
            'intent_labels': intent_labels,
            'slot_preds': slot_preds,
            'slot_labels': slot_labels,
            'bio_slot_preds': bio_slot_preds,
            'bio_slot_labels': bio_slot_labels,
            'input': input_ids,
            'subtokens_mask': subtokens_mask,
        }

    @staticmethod
    def get_continuous_slots(slot_ids, utterance_tokens):
        """
        Extract continuous spans of slot_ids
        Args:
            Slot_ids: list of str representing slot of each word token
            For instance, 'O', 'email_address', 'email_address', 'email_address', 'O', 'O', 'O', 'O']
            Corresponds to ['enter', 'atdfd@yahoo', 'dot', 'com', 'into', 'my', 'contact', 'list']
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
        inputs = []
        intent_preds = []
        intent_labels = []

        for output in outputs:
            slot_preds += output['slot_preds']
            slot_labels += output["slot_labels"]
            subtokens_mask += output["subtokens_mask"]
            inputs += output["input"]
            intent_preds += output["intent_preds"]
            intent_labels += output["intent_labels"]

        ground_truth_labels = self.convert_intent_ids_to_intent_names(intent_labels)
        generated_labels = self.convert_intent_ids_to_intent_names(intent_preds)

        predicted_slots = self.mask_unused_subword_slots(slot_preds, subtokens_mask)
        ground_truth_slots = self.mask_unused_subword_slots(slot_labels, subtokens_mask)

        all_generated_slots = []
        all_ground_truth_slots = []
        all_utterances = []

        for i in range(len(predicted_slots)):
            utterance = self.tokenizer.tokenizer.decode(inputs[i], skip_special_tokens=True)
            utterance_tokens = utterance.split()
            ground_truth_slot_names = ground_truth_slots[i].split()
            predicted_slot_names = predicted_slots[i].split()
            if len(utterance_tokens) != len(ground_truth_slot_names):
                # fix the bug that abc@xyz get tokenized to 3 tokens and @xyz to 2 tokens
                utterance_tokens = IntentBIOTypeClassificationModel.join_tokens_containing_at_sign(
                    utterance_tokens, ground_truth_slot_names
                )
            processed_ground_truth_slots = IntentBIOTypeClassificationModel.get_continuous_slots(
                ground_truth_slot_names, utterance_tokens
            )
            processed_predicted_slots = IntentBIOTypeClassificationModel.get_continuous_slots(
                predicted_slot_names, utterance_tokens
            )

            all_generated_slots.append(processed_predicted_slots)
            all_ground_truth_slots.append(processed_ground_truth_slots)
            all_utterances.append(' '.join(utterance_tokens))

        os.makedirs(self.cfg.dataset.dialogues_example_dir, exist_ok=True)
        filename = os.path.join(self.cfg.dataset.dialogues_example_dir, "predictions.jsonl")

        DialogueClassificationMetrics.save_predictions(
            filename,
            generated_labels,
            all_generated_slots,
            ground_truth_labels,
            all_ground_truth_slots,
            ['' for i in range(len(generated_labels))],
            ['' for i in range(len(generated_labels))],
            all_utterances,
        )

        (
            slot_precision,
            slot_recall,
            slot_f1,
            slot_joint_goal_accuracy,
        ) = DialogueClassificationMetrics.get_slot_filling_metrics(all_generated_slots, all_ground_truth_slots)

        return slot_precision, slot_recall, slot_f1, slot_joint_goal_accuracy

    @staticmethod
    def join_tokens_containing_at_sign(utterance_tokens, slot_names):
        """
        assumes utterance contains only one @ sign
        """
        target_length = len(slot_names)
        current_length = len(utterance_tokens)
        diff = current_length - target_length
        at_sign_positions = [index for index, token in enumerate(utterance_tokens) if token == "@"]
        try:
            if len(at_sign_positions) > 1:
                raise ValueError(
                    "Current method does not support utterances with more than 1 @ sign ({} encountered), please extend this method for utterance {} with slot names {}".format(
                        len(at_sign_positions), utterance_tokens, slot_names
                    )
                )
            elif diff == 1:
                new_tokens = []
                for index, token in enumerate(utterance_tokens):
                    if utterance_tokens[index - 1] == "@":
                        new_tokens[-1] += token
                    else:
                        new_tokens.append(token)

            elif diff == 2:
                new_tokens = []
                for index, token in enumerate(utterance_tokens[:-1]):
                    if utterance_tokens[index - 1] == "@" or token == "@":
                        new_tokens[-1] += token
                    else:
                        new_tokens.append(token)

            elif diff == 3:
                new_tokens = []
                for index, token in enumerate(utterance_tokens[:-1]):
                    if utterance_tokens[index + 1] == "@" or utterance_tokens[index - 1] == "@" or token == "@":
                        new_tokens[-1] += token
                    else:
                        new_tokens.append(token)
            return new_tokens
        except:
            new_tokens = []
            # print(
            #     "Difference of more than 3 ({}, utterance has {}, predicted slots has {}) encountered. please extend this method for utterance {} with slots {}".format(
            #         diff, len(utterance_tokens), len(slot_names), utterance_tokens, slot_names
            #     )
            # )
            new_tokens = utterance_tokens[:len(slot_names)]
            global UNIQUE_COUNTER
            UNIQUE_COUNTER +=1
            # Append-adds at last
            file1 = open("train_log/conll2003_log.txt", "a")  # append mode
            file1.write("Case {} Difference of more than 3 ({}, utterance has {}, predicted slots has {}) encountered. please extend this method for utterance {} with slots {}\n".format(
                    UNIQUE_COUNTER,  diff, len(utterance_tokens), len(slot_names), utterance_tokens, slot_names)
                    )
            file1.close()
            return new_tokens


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

        # calculate metrics and log classification report (separately for intents and slots)
        intent_precision, intent_recall, intent_f1, intent_report = self.intent_classification_report.compute()
        logging.info(f'Intent report: {intent_report}')

        slot_precision, slot_recall, slot_f1, slot_report = self.slot_classification_report.compute()
        logging.info(f'Slot report: {slot_report}')

        bio_slot_precision, bio_slot_recall, bio_slot_f1, bio_slot_report = self.bio_slot_classification_report.compute()
        logging.info(f'BIO Slot report: {bio_slot_report}')


        self.log('val_loss', avg_loss)
        self.log('intent_precision', intent_precision)
        self.log('intent_recall', intent_recall)
        self.log('intent_f1', intent_f1)
        self.log('slot_precision', slot_precision)
        self.log('slot_recall', slot_recall)
        self.log('slot_f1', slot_f1)
        self.log('bio_slot_precision', bio_slot_precision)
        self.log('bio_slot_recall', bio_slot_recall)
        self.log('bio_slot_f1', bio_slot_f1)
        self.log('unified_slot_precision', unified_slot_precision)
        self.log('unified_slot_recall', unified_slot_recall)
        self.log('unified_slot_f1', unified_slot_f1)
        self.log('unified_slot_joint_goal_accuracy', unified_slot_joint_goal_accuracy)

        self.intent_classification_report.reset()
        self.slot_classification_report.reset()
        self.bio_slot_classification_report.reset()

        return {
            'val_loss': avg_loss,
            'intent_precision': intent_precision,
            'intent_recall': intent_recall,
            'intent_f1': intent_f1,
            'slot_precision': slot_precision,
            'slot_recall': slot_recall,
            'slot_f1': slot_f1,
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

        dataset = DialogueBERTDataset(
            dataset_split,
            data_processor,
            self.tokenizer,
            self.cfg.dataset,  # this is the model.dataset cfg, which is diff from train_ds cfg etc
        )

        print("DUMP dataset after DialogueBERTDataset before dataloader")
        with open('/home/lilee/pickle/dataset.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

        print("END dumping dataset after DialogueBERTDataset before dataloader")

        return DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=cfg.drop_last,
            collate_fn=dataset.collate_fn,
        )

    def _setup_infer_dataloader(self, queries: List[str], test_ds) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a infer data loader.
        Args:
            queries: text
            batch_size: batch size to use during inference
        Returns:
            A pytorch DataLoader.
        """

        dataset = IntentSlotInferenceDataset(
            tokenizer=self.tokenizer, queries=queries, max_seq_length=-1, do_lower_case=False
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            batch_size=test_ds.batch_size,
            shuffle=test_ds.shuffle,
            num_workers=test_ds.num_workers,
            pin_memory=test_ds.pin_memory,
            drop_last=test_ds.drop_last,
        )

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

    def predict_from_examples(self, queries: List[str], test_ds) -> List[List[str]]:
        """
        Get prediction for the queries (intent and slots)
        Args:
            queries: text sequences
            test_ds: Dataset configuration section.
        Returns:
            predicted_intents, predicted_slots: model intent and slot predictions
        """

        predicted_intents = []
        predicted_slots = []
        mode = self.training

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Switch model to evaluation mode
        self.eval()
        self.to(device)

        # Dataset.
        infer_datalayer = self._setup_infer_dataloader(queries, test_ds)

        for batch in infer_datalayer:
            input_ids, input_type_ids, input_mask, loss_mask, subtokens_mask = batch

            intent_logits, slot_logits = self.forward(
                input_ids=input_ids.to(device),
                token_type_ids=input_type_ids.to(device),
                attention_mask=input_mask.to(device),
            )

            # predict intents
            intent_preds = tensor2list(torch.argmax(intent_logits, axis=-1))
            predicted_intents += self.convert_intent_ids_to_intent_names(intent_preds)

            # predict slots
            slot_preds = torch.argmax(slot_logits, axis=-1)
            predicted_slots += self.mask_unused_subword_slots(slot_preds, subtokens_mask)
            # predicted_slots += self.mask_unused_subword_slots(slot_preds, subtokens_mask)

        # set mode back to its original value
        self.train(mode=mode)

        return predicted_intents, predicted_slots

    def convert_intent_ids_to_intent_names(self, intent_preds):
        # Retrieve intent and slot vocabularies from configuration.
        intent_labels = self.cfg.data_desc.intent_labels

        predicted_intents = []

        # convert numerical outputs to Intent and Slot labels from the dictionaries
        for intent_num in intent_preds:
            # if intent_num < len(intent_labels):
            predicted_intents.append(intent_labels[int(intent_num)])
            # else:
            #     # should not happen
            #     predicted_intents.append("Unknown Intent")
        return predicted_intents

    def mask_unused_subword_slots(self, slot_preds, subtokens_mask):
        # Retrieve intent and slot vocabularies from configuration.
        slot_labels = self.cfg.data_desc.slot_labels
        predicted_slots = []
        for slot_preds_query, mask_query in zip(slot_preds, subtokens_mask):
            query_slots = ''
            for slot, mask in zip(slot_preds_query, mask_query):
                if mask == 1:
                    # if slot < len(slot_labels):
                    query_slots += slot_labels[int(slot)] + ' '
                    # else:
                    #     query_slots += 'Unknown_slot '
            predicted_slots.append(query_slots.strip())
        return predicted_slots

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        result = []
        model = PretrainedModelInfo(
            pretrained_model_name="Joint_Intent_Slot_Assistant",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemonlpmodels/versions/1.0.0a5/files/Joint_Intent_Slot_Assistant.nemo",
            description="This models is trained on this https://github.com/xliuhw/NLU-Evaluation-Data dataset which includes 64 various intents and 55 slots. Final Intent accuracy is about 87%, Slot accuracy is about 89%.",
        )
        result.append(model)
        return result

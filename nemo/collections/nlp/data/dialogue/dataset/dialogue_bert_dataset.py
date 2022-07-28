# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright 2019 The Google Research Authors.
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

import pickle
from typing import Dict, Optional

import numpy as np

from nemo.collections.nlp.data.data_utils import get_stats
from nemo.collections.nlp.data.dialogue.dataset.dialogue_dataset import DialogueDataset
from nemo.core.neural_types import ChannelType, LabelsType, MaskType, NeuralType
from nemo.utils import logging

__all__ = ['DialogueBERTDataset', 'DialogueIntentSlotInferenceDataset']


class DialogueBERTDataset(DialogueDataset):

    """
    Creates a dataset to use for the task of joint intent
    and slot classification with pretrained model.

    For a dataset to use during inference without labels, see
    IntentSlotDataset.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'input_ids': NeuralType(('B', 'T'), ChannelType()),
            'segment_ids': NeuralType(('B', 'T'), ChannelType()),
            'input_mask': NeuralType(('B', 'T'), MaskType()),
            'loss_mask': NeuralType(('B', 'T'), MaskType()),
            'subtokens_mask': NeuralType(('B', 'T'), MaskType()),
            'intent_labels': NeuralType(('B'), LabelsType()),
            'slot_labels': NeuralType(('B', 'T'), LabelsType()),
            'bio_slot_labels': NeuralType(('B', 'T'), LabelsType()),
            'bio_mention_labels': NeuralType(('B', 'T'), LabelsType()),
            'mention_loss_mask': NeuralType(('B', 'T'), LabelsType()),
        }

    def __init__(self, dataset_split: str, dialogues_processor: object, tokenizer, cfg):
        """
        Args:
            dataset_split: dataset split
            dialogues_processor: Data generator for dialogues
            tokenizer: tokenizer
            cfg: config container for dataset
        """
        self.cfg = cfg
        self.all_possible_labels = dialogues_processor.intents
        self.label_to_label_id = {self.all_possible_labels[i]: i for i in range(len(self.all_possible_labels))}
        self.all_possible_slots = dialogues_processor.slots
        self.slot_name_to_slot_id = {self.all_possible_slots[i]: i for i in range(len(self.all_possible_slots))}
        self.empty_slot_name = 'O'

        self.features = dialogues_processor.get_dialog_examples(dataset_split)
        self.features = self.features if self.cfg.num_samples == -1 else self.features[: self.cfg.num_samples]

        queries = [feature.data["utterance"] for feature in self.features]
        if self.cfg.do_lowercase:
            queries = [query.lower() for query in queries]
        intents = [self.label_to_label_id[feature.data["labels"]["intent"]] for feature in self.features]
        word_level_slots = [self.convert_slot_position_to_slot_ids(feature.data) for feature in self.features]

        features = DialogueBERTDataset.get_features(
            queries,
            self.cfg.max_seq_length,
            tokenizer,
            pad_label=self.cfg.pad_label,
            word_level_slots=word_level_slots,
            ignore_extra_tokens=self.cfg.ignore_extra_tokens,
            ignore_start_end=self.cfg.ignore_start_end,
        )
        
        self.all_input_ids = features[0]
        self.all_segment_ids = features[1]
        self.all_input_mask = features[2]
        self.all_loss_mask = features[3]
        self.all_subtokens_mask = features[4]
        self.all_slots = features[5]


        
        # dict_slots = {}
        # bio_transfer= {} #0 indicates 'O', 1 indicates 'B', 2 indicates 'I' 
        # bio_transfer['0'] = '0'

        # for idx,s in enumerate(self.all_possible_slots):
        #     dict_slots[str(idx)]=s

        # for k,v in dict_slots.items():
        #     if 'B' in v:
        #         bio_transfer[k] = '1'
        #     elif 'I' in v:
        #         bio_transfer[k] = '2'

        def transfer(input_list, other_label):
            bio_list = []
            start = 0
            idx = 0
            
            
            while idx<len(input_list):
                label=input_list[idx]
                # print(idx)
                
                if label!=other_label and label!=-1:
                    while idx<len(input_list) and label != other_label and label!=-1:
                        # print(idx)
                        if start==0:
                            start = 1
                            bio_list.append(1)
                        elif label!=input_list[idx-1]:
                            start = 1
                            bio_list.append(1)
                        else:
                            bio_list.append(2)
                        idx+=1
                        if idx<len(input_list):
                            label=input_list[idx]
                    start = 0
                    idx-=1
                else:
                    # bio_list.append(label)
                    bio_list.append(0)
                idx +=1
            return bio_list


        # bio_train_slots = []
        # for t in features[5]:
        #     bio_train_slots.append(' '.join([bio_transfer[s] for s in t.split()]))

        # drive through dataset: other_label=0; assistant dataset: other_label='54'
        file_path_for_stats = cfg.data_dir+"/train_slot_stats.tsv"
        # print("============== file_path_for_stats ==========")
        # print("============== file_path_for_stats ==========")
        # print("============== file_path_for_stats ==========")
        # print("============== file_path_for_stats ==========")
        # print("============== file_path_for_stats ==========")
        # print("============== file_path_for_stats ==========")
        # print("============== file_path_for_stats ==========")
        # print("============== file_path_for_stats ==========")
        print(file_path_for_stats)
        with open(file_path_for_stats) as f:
            other_label = int(f.read().strip().split('\n')[0].split()[0])

        bio_train_slots = [transfer(t, other_label) for t in features[5]]
        
        

        self.all_bio_slots = bio_train_slots

        def get_bio_mention_labels(slot_list, bio_list):
            # slot_list [-1  0  0  0  3  0  0  0 -1 -1 -1]
            # bio_list  [-1  0  0  0  1  0  0  0 -1 -1 -1]
            # the function will generate mention's entity lables
            # bio_train_slots [3]+[0 0 ... 0 0]

            bio_train_slots=[]
            for idx_ in range(len(slot_list)):
                if bio_list[idx_]==1:
                    bio_train_slots.append(slot_list[idx_])
            add_zero=len(slot_list)-len(bio_train_slots)
            
            bio_train_slots = bio_train_slots+[-1]*(add_zero)
            # bio_train_slots = [0]+bio_train_slots
            # bio_train_slots = bio_train_slots+[0]*(add_zero-1)
            return bio_train_slots

        bio_mention_labels = [get_bio_mention_labels(t, bio_list) for t, bio_list in zip(features[5], bio_train_slots)]

        self.bio_mention_labels = bio_mention_labels

        # this may be wrong, we need to create new mention_loss_mask, as the next floowing function
        def get_mention_loss_mask(bio_mention_labels_list):
            # bio_mention_labels_list [3]+[0 0 ... 0 0]
            # mention_loss_mask_list  [1]+[0 0 ... 0 0 ]
            mention_loss_mask_list= [1 if b!=-1 else 0 for b in bio_mention_labels_list]
            return mention_loss_mask_list

        # def get_mention_loss_mask(slot_list):
        #     # slot_list              [-1  0  0  0  3  0  0  0 -1 -1 -1]
        #     # the function will generate mention's mask
        #     # mention_loss_mask_list [ 0  0  0  0  3  0  0  0  0  0  0]
        #     # s>0 which is not -1 , and not 0, would be other classes 1, 2, 3, ...
        #     mention_loss_mask_list= [1 if s>0 else 0 for s in slot_list]
        #     return mention_loss_mask_list

        #this may be wrong, check out the next line
        mention_loss_mask = [get_mention_loss_mask(b_list) for b_list in bio_mention_labels]
        # mention_loss_mask = [get_mention_loss_mask(t) for t in features[5]]


        self.mention_loss_mask = mention_loss_mask

        self.all_intents = intents

        print("============== DUMP PICKLE =============")
        with open('/home/lilee/pickle/features.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(features, f, pickle.HIGHEST_PROTOCOL)

        with open('/home/lilee/pickle/bio_train_slots.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(bio_train_slots, f, pickle.HIGHEST_PROTOCOL)

        with open('/home/lilee/pickle/bio_mention_labels.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(bio_mention_labels, f, pickle.HIGHEST_PROTOCOL)
        
        with open('/home/lilee/pickle/mention_loss_mask.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(mention_loss_mask, f, pickle.HIGHEST_PROTOCOL)        

        print("============== DUMP PICKLE END =============")

    def convert_slot_position_to_slot_ids(self, feature):
        slot_ids = [self.slot_name_to_slot_id[self.empty_slot_name] for i in range(len(feature["utterance"].split()))]
        slot_name_to_positions = feature["label_positions"]["slots"]

        for slot_name in slot_name_to_positions:
            slot_id = self.slot_name_to_slot_id[slot_name]
            start = slot_name_to_positions[slot_name]["start"]
            exclusive_end = slot_name_to_positions[slot_name]["exclusive_end"]
            for to_replace_position in range(start, min(exclusive_end, len(slot_ids))):
                slot_ids[to_replace_position] = slot_id

        return slot_ids

    def __len__(self):
        return len(self.all_input_ids)

    def __getitem__(self, idx):
        return (
            np.array(self.all_input_ids[idx]),
            np.array(self.all_segment_ids[idx]),
            np.array(self.all_input_mask[idx], dtype=np.long),
            np.array(self.all_loss_mask[idx]),
            np.array(self.all_subtokens_mask[idx]),
            self.all_intents[idx],
            np.array(self.all_slots[idx]),
            np.array(self.all_bio_slots[idx]),
            np.array(self.bio_mention_labels[idx]),
            np.array(self.mention_loss_mask[idx]),
        )

    @staticmethod
    def truncate_and_pad(
        max_seq_length,
        ignore_start_end,
        with_label,
        pad_label,
        tokenizer,
        all_slots,
        all_subtokens,
        all_input_mask,
        all_loss_mask,
        all_subtokens_mask,
        all_input_ids,
        all_segment_ids,
    ):

        too_long_count = 0

        for i, subtokens in enumerate(all_subtokens):
            if len(subtokens) > max_seq_length:
                subtokens = [tokenizer.cls_token] + subtokens[-max_seq_length + 1 :]
                all_input_mask[i] = [1] + all_input_mask[i][-max_seq_length + 1 :]
                all_loss_mask[i] = [1 - ignore_start_end] + all_loss_mask[i][-max_seq_length + 1 :]
                all_subtokens_mask[i] = [0] + all_subtokens_mask[i][-max_seq_length + 1 :]

                if with_label:
                    all_slots[i] = [pad_label] + all_slots[i][-max_seq_length + 1 :]
                too_long_count += 1

            all_input_ids.append([tokenizer.tokens_to_ids(t) for t in subtokens])

            if len(subtokens) < max_seq_length:
                extra = max_seq_length - len(subtokens)
                all_input_ids[i] = all_input_ids[i] + [0] * extra
                all_loss_mask[i] = all_loss_mask[i] + [0] * extra
                all_subtokens_mask[i] = all_subtokens_mask[i] + [0] * extra
                all_input_mask[i] = all_input_mask[i] + [0] * extra

                if with_label:
                    all_slots[i] = all_slots[i] + [pad_label] * extra

            all_segment_ids.append([0] * max_seq_length)

        logging.info(f'{too_long_count} are longer than {max_seq_length}')
        return (
            all_slots,
            all_subtokens,
            all_input_mask,
            all_loss_mask,
            all_subtokens_mask,
            all_input_ids,
            all_segment_ids,
        )

    @staticmethod
    def get_features(
        queries,
        max_seq_length,
        tokenizer,
        pad_label=128,
        word_level_slots=None,
        ignore_extra_tokens=False,
        ignore_start_end=False,
    ):
        """
        Convert queries (utterance, intent label and slot labels) to BERT input format 
        """

        all_subtokens = []
        all_loss_mask = []
        all_subtokens_mask = []
        all_segment_ids = []
        all_input_ids = []
        all_input_mask = []
        sent_lengths = []
        all_slots = []

        with_label = word_level_slots is not None

        for i, query in enumerate(queries):
            words = query.strip().split()
            subtokens = [tokenizer.cls_token]
            loss_mask = [1 - ignore_start_end]
            subtokens_mask = [0]
            if with_label:
                slots = [pad_label]

            for j, word in enumerate(words):
                word_tokens = tokenizer.text_to_tokens(word)

                # to handle emojis that could be neglected during tokenization
                if len(word.strip()) > 0 and len(word_tokens) == 0:
                    word_tokens = [tokenizer.ids_to_tokens(tokenizer.unk_id)]

                subtokens.extend(word_tokens)
                # mask all sub-word tokens except the first token in a word
                # use the label for the first sub-word token as the label for the entire word to eliminate need for disambiguation
                loss_mask.append(1)
                loss_mask.extend([int(not ignore_extra_tokens)] * (len(word_tokens) - 1))

                subtokens_mask.append(1)
                subtokens_mask.extend([0] * (len(word_tokens) - 1))

                if with_label:
                    slots.extend([word_level_slots[i][j]] * len(word_tokens))

            subtokens.append(tokenizer.sep_token)
            loss_mask.append(1 - ignore_start_end)
            subtokens_mask.append(0)
            sent_lengths.append(len(subtokens))
            all_subtokens.append(subtokens)
            all_loss_mask.append(loss_mask)
            all_subtokens_mask.append(subtokens_mask)
            all_input_mask.append([1] * len(subtokens))
            if with_label:
                slots.append(pad_label)
                all_slots.append(slots)
        max_seq_length_data = max(sent_lengths)
        max_seq_length = min(max_seq_length, max_seq_length_data) if max_seq_length > 0 else max_seq_length_data
        logging.info(f'Setting max length to: {max_seq_length}')
        get_stats(sent_lengths)

        # truncate and pad samples
        (
            all_slots,
            all_subtokens,
            all_input_mask,
            all_loss_mask,
            all_subtokens_mask,
            all_input_ids,
            all_segment_ids,
        ) = DialogueBERTDataset.truncate_and_pad(
            max_seq_length,
            ignore_start_end,
            with_label,
            pad_label,
            tokenizer,
            all_slots,
            all_subtokens,
            all_input_mask,
            all_loss_mask,
            all_subtokens_mask,
            all_input_ids,
            all_segment_ids,
        )

        # log examples for debugging
        logging.debug("*** Some Examples of Processed Data ***")
        for i in range(min(len(all_input_ids), 5)):
            logging.debug("i: %s" % (i))
            logging.debug("subtokens: %s" % " ".join(list(map(str, all_subtokens[i]))))
            logging.debug("loss_mask: %s" % " ".join(list(map(str, all_loss_mask[i]))))
            logging.debug("input_mask: %s" % " ".join(list(map(str, all_input_mask[i]))))
            logging.debug("subtokens_mask: %s" % " ".join(list(map(str, all_subtokens_mask[i]))))
            if with_label:
                logging.debug("slots_label: %s" % " ".join(list(map(str, all_slots[i]))))

        return (all_input_ids, all_segment_ids, all_input_mask, all_loss_mask, all_subtokens_mask, all_slots)


class DialogueIntentSlotInferenceDataset(DialogueBERTDataset):
    """
    Creates dataset to use for the task of joint intent
    and slot classification with pretrained model.
    This is to be used during inference only.
    It uses list of queries as the input.

    Args:
        queries (list): list of queries to run inference on
        max_seq_length (int): max sequence length minus 2 for [CLS] and [SEP]
        tokenizer (Tokenizer): such as NemoBertTokenizer
        pad_label (int): pad value use for slot labels.
            by default, it's the neutral label.

    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """
            Returns definitions of module output ports.
        """
        return {
            'input_ids': NeuralType(('B', 'T'), ChannelType()),
            'segment_ids': NeuralType(('B', 'T'), ChannelType()),
            'input_mask': NeuralType(('B', 'T'), MaskType()),
            'loss_mask': NeuralType(('B', 'T'), MaskType()),
            'subtokens_mask': NeuralType(('B', 'T'), MaskType()),
        }

    def __init__(self, queries, max_seq_length, tokenizer, do_lower_case):
        if do_lower_case:
            queries = [query.lower() for query in queries]

        features = DialogueBERTDataset.get_features(queries, max_seq_length, tokenizer)

        self.all_input_ids = features[0]
        self.all_segment_ids = features[1]
        self.all_input_mask = features[2]
        self.all_loss_mask = features[3]
        self.all_subtokens_mask = features[4]

    def __len__(self):
        return len(self.all_input_ids)

    def __getitem__(self, idx):
        return (
            np.array(self.all_input_ids[idx]),
            np.array(self.all_segment_ids[idx]),
            np.array(self.all_input_mask[idx], dtype=np.long),
            np.array(self.all_loss_mask[idx]),
            np.array(self.all_subtokens_mask[idx]),
        )

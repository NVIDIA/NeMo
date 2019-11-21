import json
import os
import pickle

from torch.utils.data import Dataset


class Vocab:
    """
    PAD_token = 1
    SOS_token = 3
    EOS_token = 2
    UNK_token = 0
    """

    def __init__(self):
        self.word2idx = {'UNK': 0, 'PAD': 1, 'EOS': 2, 'BOS': 3}
        self.idx2word = ['UNK', 'PAD', 'EOS', 'BOS']
        self.unk_id = self.word2idx['UNK']
        self.pad_id = self.word2idx['PAD']
        self.eos_id = self.word2idx['EOS']
        self.bos_id = self.word2idx['BOS']
        self.unk, self.pad, self.eos, self.bos = 'UNK', 'PAD', 'EOS', 'BOS'

    def __len__(self):
        return len(self.idx2word)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)

    def add_words(self, sent, level):
        """
        level == 'utterance': sent is a string
        level == 'slot': sent is a list
        level == 'belief': sent is a dictionary
        """
        if level == 'utterance':
            for word in sent.split():
                self.add_word(word)
        elif level == 'slot':
            for slot in sent:
                domain, info = slot.split('-')
                self.add_word(domain)
                for subslot in info.split(' '):
                    self.add_word(subslot)
        elif level == 'belief':
            for slot, value in sent.items():
                domain, info = slot.split('-')
                self.add_word(domain)
                for subslot in info.split(' '):
                    self.add_word(subslot)
                for val in value.split(' '):
                    self.add_word(val)

    def tokens2ids(self, tokens):
        """Converts list of tokens to list of ids."""
        return [self.word2idx[w]
                if w in self.word2idx else self.unk_id for w in tokens]


class WOZDSTDataset(Dataset):
    """
    By default, use only vocab from training
    Need to modify the code a little bit to use for all_vocab
    """

    def __init__(self,
                 data_dir,
                 domains,
                 mode='train'):
        self.data_dir = data_dir
        self.mode = mode
        self.gating_dict = {'ptr': 0, 'dontcare': 1, 'none': 2}
        self.domains = domains
        self.vocab, self.mem_vocab = Vocab(), Vocab()

        ontology_file = open(f'{self.data_dir}/ontology.json', 'r')
        self.ontology = json.load(ontology_file)

        self.get_slots()
        self.get_vocab()
        self.features, self.max_len = self.get_features()
        print(vars(self).keys())
        print('One feature', self.features[0])

    def get_vocab(self):
        self.vocab_file = f'{self.data_dir}/vocab.pkl'
        self.mem_vocab_file = f'{self.data_dir}/mem_vocab.pkl'

        if self.mode != 'train' and (not os.path.exists(self.vocab_file) or
                                     not os.path.exists(self.mem_vocab_file)):
            raise ValueError(f"{self.vocab_file} and {self.mem_vocab_file}"
                             " don't exist!")

        if os.path.exists(self.vocab_file) and \
                os.path.exists(self.mem_vocab_file):
            print(f'Loading vocab and mem_vocab from {self.data_dir}')
            self.vocab = pickle.load(open(self.vocab_file, 'rb'))
            self.mem_vocab = pickle.load(open(self.mem_vocab_file, 'rb'))
        else:
            self.create_vocab()

        print('Mem vocab size', len(self.mem_vocab))
        print('Vocab size', len(self.vocab))

    def get_slots(self):
        used_domains = [
            key for key in self.ontology if key.split('-')[0] in self.domains]
        self.slots = [k.replace(' ', '').lower() if 'book' not in k
                      else k.lower() for k in used_domains]

    def create_vocab(self):
        print('Creating vocab from train files')
        self.vocab.add_words(self.slots, 'slot')
        self.mem_vocab.add_words(self.slots, 'slot')

        filename = f'{self.data_dir}/train_dialogs.json'
        print(f'Building vocab from {filename}')
        dialogs = json.load(open(filename, 'r'))

        max_value_len = 0

        for dialog_dict in dialogs:
            for turn in dialog_dict['dialog']:
                self.vocab.add_words(turn['sys_transcript'], 'utterance')
                self.vocab.add_words(turn['transcript'], 'utterance')

                turn_beliefs = fix_general_label_error(turn['belief_state'],
                                                       self.slots)
                self.mem_vocab.add_words(turn_beliefs, 'belief')

                lengths = [len(turn_beliefs[slot])
                           for slot in self.slots if slot in turn_beliefs]
                lengths.append(max_value_len)
                max_value_len = max(lengths)

        if f'f{max_value_len-1}' not in self.mem_vocab.word2idx:
            for time_i in range(max_value_len):
                self.mem_vocab.add_words(f't{time_i}', 'utterance')

        print(f'Saving vocab and mem_vocab to {self.data_dir}')
        with open(self.vocab_file, 'wb') as handle:
            pickle.dump(self.vocab, handle)
        with open(self.mem_vocab_file, 'wb') as handle:
            pickle.dump(self.mem_vocab, handle)

    def get_features(self):
        filename = f'{self.data_dir}/{self.mode}_dialogs.json'
        print(f'Reading from {filename}')
        dialogs = json.load(open(filename, 'r'))

        domain_count = {}
        data = []
        max_resp_len, max_value_len = 0, 0

        for dialog_dict in dialogs:
            dialog_histories = []
            for domain in dialog_dict['domains']:
                if domain not in self.domains:
                    continue
                if domain not in domain_count:
                    domain_count[domain] = 0
                domain_count[domain] += 1

            for turn in dialog_dict['dialog']:
                turn_uttr = turn['sys_transcript'] + ' ; ' + turn['transcript']
                turn_uttr = turn_uttr.strip()
                dialog_histories.append(turn_uttr)
                turn_beliefs = fix_general_label_error(turn['belief_state'],
                                                       self.slots)

                turn_belief_list = [f'{k}-{v}'
                                    for k, v in turn_beliefs.items()]

                gating_label, responses = [], []

                for slot in self.slots:
                    gating_slot = slot
                    if gating_slot not in ['dontcare', 'none']:
                        gating_slot = 'ptr'

                    if slot in turn_beliefs:
                        responses.append(turn_beliefs[slot])
                    else:
                        responses.append('none')
                        gating_slot = 'none'
                    gating_label.append(self.gating_dict[gating_slot])

                sample = {'ID': dialog_dict['dialog_idx'],
                          'domains': dialog_dict['domains'],
                          'turn_domain': turn['domain'],
                          'turn_id': turn['turn_idx'],
                          'dialog_history': ' ; '.join(dialog_histories),
                          'turn_belief': turn_belief_list,
                          'gating_label': gating_label,
                          'turn_uttr': turn_uttr,
                          'responses': responses}
                data.append(sample)

                resp_len = len(sample['dialog_history'].split())
                max_resp_len = max(max_resp_len, resp_len)

        print('Domain count', domain_count)
        print('Max response length', max_resp_len)
        print(f'Processing {len(data)} samples')
        return data, max_resp_len

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = self.features[idx]
        context_ids = self.vocab.tokens2ids(item['dialog_history'].split())
        # feature['context_ids'] = torch.Tensor(context_ids)
        item['context_ids'] = context_ids
        y_ids = [self.vocab.tokens2ids(y.split() + [self.vocab.eos])
                 for y in item['responses']]
        item['responses'] = y_ids
        item['turn_domain'] = self.domains[item['turn_domain']]

        return (item['ID'],
                item['turn_id'],
                item['turn_belief'],
                item['gating_label'],
                item['context_ids'],
                item['turn_domain'],
                item['responses'])


def fix_general_label_error(labels, slots):
    label_dict = dict([label['slots'][0] for label in labels])
    GENERAL_TYPO = {
        # type
        "guesthouse": "guest house",
        "guesthouses": "guest house",
        "guest": "guest house",
        "mutiple sports": "multiple sports",
        "sports": "multiple sports",
        "mutliple sports": "multiple sports",
        "swimmingpool": "swimming pool",
        "concerthall": "concert hall",
        "concert": "concert hall",
        "pool": "swimming pool",
        "night club": "nightclub",
        "mus": "museum",
        "ol": "architecture",
        "colleges": "college",
        "coll": "college",
        "architectural": "architecture",
        "musuem": "museum",
        "churches": "church",
        # area
        "center": "centre",
        "center of town": "centre",
        "near city center": "centre",
        "in the north": "north",
        "cen": "centre",
        "east side": "east",
        "east area": "east",
        "west part of town": "west",
        "ce": "centre",
        "town center": "centre",
        "centre of cambridge": "centre",
        "city center": "centre",
        "the south": "south",
        "scentre": "centre",
        "town centre": "centre",
        "in town": "centre",
        "north part of town": "north",
        "centre of town": "centre",
        "cb30aq": "none",
        # price
        "mode": "moderate",
        "moderate -ly": "moderate",
        "mo": "moderate",
        # day
        "next friday": "friday",
        "monda": "monday",
        # parking
        "free parking":
        "free",
        # internet
        "free internet": "yes",
        # star
        "4 star": "4",
        "4 stars": "4",
        "0 star rarting": "none",
        # others
        "y": "yes",
        "any": "dontcare",
        "n": "no",
        "does not care": "dontcare",
        "not men": "none",
        "not": "none",
        "not mentioned": "none",
        '': "none",
        "not mendtioned": "none",
        "3 .": "3",
        "does not": "no",
        "fun": "none",
        "art": "none",
    }

    hotel_ranges = ["nigh", "moderate -ly priced", "bed and breakfast",
                    "centre", "venetian", "intern", "a cheap -er hotel"]
    locations = ["gastropub", "la raza", "galleria", "gallery", "science", "m"]
    detailed_hotels = ["hotel with free parking and free wifi",
                       "4",
                       "3 star hotel"]
    areas = ["stansted airport", "cambridge", "silver street"]
    attr_areas = ["norwich", "ely", "museum", "same area as hotel"]

    for slot in slots:
        if slot in label_dict.keys():
            # general typos
            if label_dict[slot] in GENERAL_TYPO.keys():
                label_dict[slot] = label_dict[slot].replace(
                    label_dict[slot], GENERAL_TYPO[label_dict[slot]])

            # miss match slot and value
            if (slot == "hotel-type" and label_dict[slot] in hotel_ranges) or \
                (slot == "hotel-internet" and label_dict[slot] == "4") or \
                (slot == "hotel-pricerange" and label_dict[slot] == "2") or \
                (slot == "attraction-type" and
                    label_dict[slot] in locations) or \
                ("area" in slot and label_dict[slot] in ["moderate"]) or \
                    ("day" in slot and label_dict[slot] == "t"):
                label_dict[slot] = "none"
            elif slot == "hotel-type" and label_dict[slot] in detailed_hotels:
                label_dict[slot] = "hotel"
            elif slot == "hotel-star" and label_dict[slot] == "3 star hotel":
                label_dict[slot] = "3"
            elif "area" in slot:
                if label_dict[slot] == "no":
                    label_dict[slot] = "north"
                elif label_dict[slot] == "we":
                    label_dict[slot] = "west"
                elif label_dict[slot] == "cent":
                    label_dict[slot] = "centre"
            elif "day" in slot:
                if label_dict[slot] == "we":
                    label_dict[slot] = "wednesday"
                elif label_dict[slot] == "no":
                    label_dict[slot] = "none"
            elif "price" in slot and label_dict[slot] == "ch":
                label_dict[slot] = "cheap"
            elif "internet" in slot and label_dict[slot] == "free":
                label_dict[slot] = "yes"

            # some out-of-define classification slot values
            if (slot == "restaurant-area" and label_dict[slot] in areas) or \
                    (slot == "attraction-area" and
                        label_dict[slot] in attr_areas):
                label_dict[slot] = "none"

    return label_dict

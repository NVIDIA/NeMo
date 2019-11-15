import json
import os

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


class DSTDataset(Dataset):
    """
    By default, use all vocab
    """

    def __init__(self, data_dir, domains):
        self.data_dir = data_dir
        self.gating_dict = {'ptr': 0, 'dontcare': 1, 'none': 2}
        self.domains = domains

        ontology_file = open(f'{self.data_dir}/ontology.json', 'r')
        self.ontology = json.load(ontology_file)

        self.vocab, self.mem_vocab = Vocab(), Vocab()
        self.get_slots()
        self.vocab.add_words(self.slots, 'slot')
        self.mem_vocab.add_words(self.slots, 'slot')
        self.vocab_file = f'{self.data_dir}/vocab.pkl'
        self.mem_vocab_file = f'{self.data_dir}/mem_vocab.pkl'

    def get_slots(self):
        used_domains = [
            key for key in self.ontology if key.split('-')[0] in self.domains]
        self.slots = [k.replace(' ', '').lower() if 'book' not in k
                      else k.lower() for k in used_domains]

    def read_vocab(self, mode='train', training=True):
        filename = f'{self.data_dir}/{mode}_dialogs.json'
        print(f'Reading from {filename}')
        dialogs = json.load(open(filename, 'r'))
        
        for dialog_dict in dialogs:
            if mode == 'train' and training:
                for ti, turn in enumerate(dialog_dict['dialog']):
                    self.vocab.add_words(turn['sys_transcript'], 'utterance')
                    self.vocab.add_words(turn['transcript'], 'utterance')

        if training 

    def prepare_data(self, training=True):
        if training:
            train_pair, train_max_len, train_slot = self.read_vocab(
                mode='train', training)

        if os.path.exists(vocab_file):
            vocab = pickle.load(open(vocab_file, 'rb'))

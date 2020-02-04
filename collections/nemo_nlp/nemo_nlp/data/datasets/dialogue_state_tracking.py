import json
import random

from nemo_nlp.data.datasets.utils import fix_general_label_error_multiwoz
from torch.utils.data import Dataset


class MultiWOZDataset(Dataset):
    """
    By default, use only vocab from training
    Need to modify the code a little bit to use for all_vocab
    """

    def __init__(self, data_dir, mode, domains, all_domains, vocab, gating_dict, slots, num_samples=-1, shuffle=False):

        print(f'Processing {mode} data')
        self.data_dir = data_dir
        self.mode = mode
        self.gating_dict = gating_dict
        self.domains = domains
        self.all_domains = all_domains
        self.vocab = vocab
        self.slots = slots

        self.features, self.max_len = self.get_features(num_samples, shuffle)
        print("Sample 0: " + str(self.features[0]))

    def get_features(self, num_samples, shuffle):

        if num_samples == 0:
            raise ValueError("num_samples has to be positive", num_samples)

        filename = f'{self.data_dir}/{self.mode}_dials.json'
        print(f'Reading from {filename}')
        dialogs = json.load(open(filename, 'r'))

        domain_count = {}
        data = []
        max_resp_len, max_value_len = 0, 0

        for dialog_dict in dialogs:
            if num_samples > 0 and len(data) >= num_samples:
                break

            dialog_history = ""
            for domain in dialog_dict['domains']:
                if domain not in self.domains:
                    continue
                if domain not in domain_count:
                    domain_count[domain] = 0
                domain_count[domain] += 1

            for turn in dialog_dict['dialogue']:
                if num_samples > 0 and len(data) >= num_samples:
                    break

                turn_uttr = turn['system_transcript'] + ' ; ' + turn['transcript']
                turn_uttr_strip = turn_uttr.strip()
                dialog_history += turn["system_transcript"] + " ; " + turn["transcript"] + " ; "
                source_text = dialog_history.strip()

                turn_beliefs = fix_general_label_error_multiwoz(turn['belief_state'], self.slots)

                turn_belief_list = [f'{k}-{v}' for k, v in turn_beliefs.items()]

                gating_label, responses = [], []
                for slot in self.slots:
                    if slot in turn_beliefs:
                        responses.append(str(turn_beliefs[slot]))
                        if turn_beliefs[slot] == "dontcare":
                            gating_label.append(self.gating_dict["dontcare"])
                        elif turn_beliefs[slot] == "none":
                            gating_label.append(self.gating_dict["none"])
                        else:
                            gating_label.append(self.gating_dict["ptr"])
                    else:
                        responses.append("none")
                        gating_label.append(self.gating_dict["none"])

                sample = {
                    'ID': dialog_dict['dialogue_idx'],
                    'domains': dialog_dict['domains'],
                    'turn_domain': turn['domain'],
                    'turn_id': turn['turn_idx'],
                    'dialogue_history': source_text,
                    'turn_belief': turn_belief_list,
                    'gating_label': gating_label,
                    'turn_uttr': turn_uttr_strip,
                    'responses': responses,
                }

                sample['context_ids'] = self.vocab.tokens2ids(sample['dialogue_history'].split())
                sample['responses_ids'] = [
                    self.vocab.tokens2ids(y.split() + [self.vocab.eos]) for y in sample['responses']
                ]
                sample['turn_domain'] = self.all_domains[sample['turn_domain']]

                data.append(sample)

                resp_len = len(sample['dialogue_history'].split())
                max_resp_len = max(max_resp_len, resp_len)

        print('Domain count', domain_count)
        print('Max response length', max_resp_len)
        print(f'Processing {len(data)} samples')

        if shuffle:
            print(f'Shuffling samples.')
            random.shuffle(data)

        return data, max_resp_len

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = self.features[idx]

        return {
            'dialog_id': item['ID'],
            'turn_id': item['turn_id'],
            'turn_belief': item['turn_belief'],
            'gating_label': item['gating_label'],
            'context_ids': item['context_ids'],
            'turn_domain': item['turn_domain'],
            'responses_ids': item['responses_ids'],
        }

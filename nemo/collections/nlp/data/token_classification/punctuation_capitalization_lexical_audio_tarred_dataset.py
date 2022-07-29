import os
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Type, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from nemo.collections.common.tokenizers import TokenizerSpec, AutoTokenizer
from nemo.collections.nlp.data.token_classification.punctuation_capitalization_dataset import \
    create_masks_and_segment_ids
from nemo.collections.nlp.data.token_classification.punctuation_capitalization_tarred_dataset import \
    BertPunctuationCapitalizationTarredDataset
from nemo.core.neural_types import ChannelType, LabelsType, MaskType, NeuralType, AudioSignal, LengthsType


class BertPunctuationCapitalizationLexicalAudioTarredDataset(BertPunctuationCapitalizationTarredDataset):
    def __init__(self, metadata_file: Union[os.PathLike, str], tokenizer: TokenizerSpec, pad_label: str,
                 label_info_save_dir: Optional[Union[os.PathLike, str]] = None, ignore_extra_tokens: bool = False,
                 ignore_start_end: bool = True, world_size: int = 1, global_rank: int = 0, shuffle_n: int = 1,
                 shard_strategy: str = "scatter"):
        super().__init__(metadata_file, tokenizer, pad_label, label_info_save_dir, ignore_extra_tokens,
                         ignore_start_end, world_size, global_rank, shuffle_n, shard_strategy)

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports. """
        return {
            'input_ids': NeuralType(('B', 'T'), ChannelType()),
            'segment_ids': NeuralType(('B', 'T'), ChannelType()),
            'input_mask': NeuralType(('B', 'T'), MaskType()),
            'subtokens_mask': NeuralType(('B', 'T'), MaskType()),
            'loss_mask': NeuralType(('B', 'T'), MaskType()),
            'punct_labels': NeuralType(('B', 'T'), LabelsType()),
            'capit_labels': NeuralType(('B', 'T'), LabelsType()),
            'features': NeuralType(('B', 'T'), AudioSignal()),
            'features_length': NeuralType(('B', 'T'), LengthsType()),
        }

    def collate_fn(self, batches: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        for batch in batches:
            batch_segment_ids, batch_input_mask, batch_loss_mask = create_masks_and_segment_ids(batch['input_ids'],
                                                                                                batch['subtokens_mask'],
                                                                                                self.tokenizer.pad_id,
                                                                                                self.tokenizer.cls_id,
                                                                                                self.tokenizer.sep_id,
                                                                                                self.ignore_start_end,
                                                                                                self.ignore_extra_tokens, )
            batch['segment_ids'] = torch.as_tensor(batch_segment_ids, dtype=torch.int)
            batch['input_mask'] = torch.as_tensor(batch_input_mask)
            batch['loss_mask'] = torch.as_tensor(batch_loss_mask)
            batch['input_ids'] = torch.as_tensor(batch['input_ids'], dtype=torch.int)
            batch['subtokens_mask'] = torch.as_tensor(batch['subtokens_mask'])
            batch['punct_labels'] = torch.as_tensor(batch['punct_labels'], dtype=torch.long)
            batch['capit_labels'] = torch.as_tensor(batch['capit_labels'], dtype=torch.long)

        segment_ids = pad_sequence([batch['segment_ids'] for batch in batches])
        input_mask = pad_sequence([batch['input_mask'] for batch in batches])
        loss_mask = pad_sequence([batch['loss_mask'] for batch in batches])
        input_ids = pad_sequence([batch['input_ids'] for batch in batches], padding_value=self.tokenizer.pad_id)
        subtokens_mask = pad_sequence([batch['subtokens_mask'] for batch in batches], padding_value=False)
        punct_labels = pad_sequence([batch['punct_labels'] for batch in batches], padding_value=0)
        capit_labels = pad_sequence([batch['capit_labels'] for batch in batches], padding_value=0)
        features = pad_sequence([batch['features'] for batch in batches], padding_value=0.0)
        features_length = torch.tensor([batch['features_length'] for batch in batches])
        return {
            'input_ids': input_ids.T,
            'subtokens_mask': subtokens_mask.T,
            'punct_labels': punct_labels.T,
            'capit_labels': capit_labels.T,
            'features': features.T,
            'features_length': features_length,
            'segment_ids': segment_ids.T,
            'input_mask': input_mask.T,
            'loss_mask': loss_mask.T
        }


# a = BertPunctuationCapitalizationLexicalAudioTarredDataset(
#     '/mnt/sda/FL_tarred/metadata.punctuation_capitalization.tokens15000.max_seq_length512.bert-base-uncased.json',
#     AutoTokenizer('bert-base-uncased'), 'O')
# f = 0
# pb = tqdm(a, total=len(a))
# for b in pb:
#     if b['features_length'] == 0:
#         f += 1
#         # print(b)
#         pb.set_description(f'F={f}')
# print(f, len(a))
# #
# a = BertPunctuationCapitalizationLexicalAudioTarredDataset(
#     '/mnt/sda/FL_tarred_4/metadata.punctuation_capitalization.tokens15000.max_seq_length512.bert-base-uncased.json',
#     AutoTokenizer('bert-base-uncased'), 'O')
# s = 0
# pb = tqdm(a, total=len(a))
# for b in pb:
#     if b['features_length'] == 0:
#         s += 1
#         pb.set_description(f'S={s}')
# print(s, len(a))

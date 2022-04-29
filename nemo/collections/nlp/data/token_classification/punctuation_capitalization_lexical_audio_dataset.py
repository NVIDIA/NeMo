from typing import Union, Optional, Dict, Callable, List

import os
import multiprocessing as mp
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from nemo.collections.common.parts.preprocessing.parsers import make_parser

from omegaconf import DictConfig

from nemo.collections.asr.data.audio_to_text import _AudioTextDataset
from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.nlp.data.token_classification.punctuation_capitalization_dataset import \
    BertPunctuationCapitalizationDataset, create_masks_and_segment_ids
from nemo.core.neural_types import NeuralType, ChannelType, MaskType, LabelsType, AudioSignal, LengthsType


class PunctuationCapitalizationLexicalAudioDataset(BertPunctuationCapitalizationDataset):

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

    def __init__(self, text_file: Union[str, os.PathLike], labels_file: Union[str, os.PathLike], max_seq_length: int,
                 tokenizer: TokenizerSpec, num_samples: int = -1, tokens_in_batch: int = 5000, pad_label: str = 'O',
                 punct_label_ids: Optional[Union[Dict[str, int], DictConfig]] = None,
                 capit_label_ids: Optional[Union[Dict[str, int], DictConfig]] = None, ignore_extra_tokens: bool = False,
                 ignore_start_end: bool = True, use_cache: bool = True,
                 cache_dir: Optional[Union[str, os.PathLike]] = None, get_label_frequencies: bool = False,
                 label_info_save_dir: Optional[Union[str, os.PathLike]] = None,
                 punct_label_vocab_file: Optional[Union[str, os.PathLike]] = None,
                 capit_label_vocab_file: Optional[Union[str, os.PathLike]] = None,
                 add_masks_and_segment_ids_to_batch: bool = True, verbose: bool = True, n_jobs: Optional[int] = 0,
                 tokenization_progress_queue: Optional[mp.Queue] = None,
                 batch_mark_up_progress_queue: Optional[mp.Queue] = None,
                 batch_building_progress_queue: Optional[mp.Queue] = None,
                 manifest_filepath: str = None,
                 parser: Union[str, Callable] = None,
                 sample_rate: int = 0,
                 int_values: bool = False,
                 augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
                 max_duration: Optional[int] = None,
                 min_duration: Optional[int] = None,
                 max_utts: int = 0,
                 trim: bool = False,
                 bos_id: Optional[int] = None,
                 eos_id: Optional[int] = None,
                 pad_id: int = 0,
                 return_sample_id: bool = False,
                 ):
        super().__init__(text_file=text_file, labels_file=labels_file, max_seq_length=max_seq_length,
                         tokenizer=tokenizer, num_samples=num_samples, tokens_in_batch=tokens_in_batch,
                         pad_label=pad_label, punct_label_ids=punct_label_ids, capit_label_ids=capit_label_ids,
                         ignore_extra_tokens=ignore_extra_tokens, ignore_start_end=ignore_start_end,
                         use_cache=use_cache, cache_dir=cache_dir, get_label_frequencies=get_label_frequencies,
                         label_info_save_dir=label_info_save_dir, punct_label_vocab_file=punct_label_vocab_file,
                         capit_label_vocab_file=capit_label_vocab_file,
                         add_masks_and_segment_ids_to_batch=False, verbose=verbose,
                         n_jobs=n_jobs, tokenization_progress_queue=tokenization_progress_queue,
                         batch_mark_up_progress_queue=batch_mark_up_progress_queue,
                         batch_building_progress_queue=batch_building_progress_queue, use_features=False)
        parser = make_parser()
        self._audio_dataset = _AudioTextDataset(manifest_filepath=manifest_filepath, parser=parser,
                                                sample_rate=sample_rate, int_values=int_values, augmentor=augmentor,
                                                max_duration=max_duration, min_duration=min_duration, max_utts=max_utts,
                                                trim=trim, bos_id=bos_id, eos_id=eos_id, pad_id=pad_id,
                                                return_sample_id=return_sample_id)

    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, torch.LongTensor]]:
        text_batch = super().__getitem__(idx)
        audio_batch = self._audio_dataset.__getitem__(idx)
        batch = text_batch.copy()
        batch.update({'features': audio_batch[0], 'features_length': audio_batch[1]})
        return batch

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

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

import copy
from collections import OrderedDict
from typing import Dict, List, Optional, Union
from operator import attrgetter
from torch import Tensor

import time
import torch
from torch import nn
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict
from pytorch_lightning import Trainer
from tqdm import tqdm
import itertools
from dataclasses import dataclass
import random

from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.asr.data.audio_to_eesd_label_lhotse import LhotseSpeechToDiarizationLabelDataset
from nemo.collections.asr.data.audio_to_eesd_label import AudioToSpeechMSDDTrainDataset
from nemo.collections.asr.metrics.multi_binary_acc import MultiBinaryAccuracy
from nemo.collections.asr.models.asr_model import ExportableEncDecModel
from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer

from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.neural_types import AudioSignal, LengthsType, NeuralType
from nemo.core.neural_types.elements import ProbsType
from nemo.utils import logging

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


torch.backends.cudnn.enabled = False 

__all__ = ['EncDecDiarLabelModel']

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class SortformerEncLabelModel(ModelPT, ExportableEncDecModel):
    """
    Encoder decoder class for multiscale diarization decoder (MSDD). Model class creates training, validation methods for setting
    up data performing model forward pass.

    This model class expects config dict for:
        * preprocessor
        * msdd_model
        * speaker_model
    """

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        result = []
        return result

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """
        Initialize an MSDD model and the specified speaker embedding model. 
        In this init function, training and validation datasets are prepared.
        """
        random.seed(42)
        self._trainer = trainer if trainer else None
        self._cfg = cfg
        
        if self._trainer:
            self.world_size = trainer.num_nodes * trainer.num_devices
        else:
            self.world_size = 1

        if self._trainer is not None and self._cfg.get('augmentor', None) is not None:
            self.augmentor = process_augmentations(self._cfg.augmentor)
        else:
            self.augmentor = None
        super().__init__(cfg=self._cfg, trainer=trainer)
        window_length_in_sec = self._cfg.diarizer.speaker_embeddings.parameters.window_length_in_sec
        if isinstance(window_length_in_sec, int) or len(window_length_in_sec) <= self._cfg.interpolated_scale:
            raise ValueError("window_length_in_sec should be a list containing multiple segment (window) lengths")
        else:
            self._cfg.diarizer_module.scale_n = self._cfg.scale_n
        # self._cfg.preprocessor.normalize = "NA"
        self.preprocessor = EncDecSpeakerLabelModel.from_config_dict(self._cfg.preprocessor)

        if hasattr(self._cfg, 'spec_augment') and self._cfg.spec_augment is not None:
            self.spec_augmentation = SortformerEncLabelModel.from_config_dict(self._cfg.spec_augment)
        else:
            self.spec_augmentation = None

        self.n_feat_fr = self._cfg.get("num_featframe_per_diarframe", 8)
        self.use_raw_encoder = self._cfg.get("use_raw_encoder", False)
        self.use_raw_encoder_only = self._cfg.get("use_raw_encoder_only", False)
        
        if self.use_raw_encoder:
            self.encoder = SortformerEncLabelModel.from_config_dict(self._cfg.encoder)
            if self._cfg.encoder.d_model != self._cfg.diarizer_module.emb_dim:
                self.encoder_proj = nn.Linear(self._cfg.encoder.d_model, self._cfg.diarizer_module.emb_dim)
            if not self.use_raw_encoder_only:
                self.emb_combine = nn.Linear(2*self._cfg.diarizer_module.emb_dim, self._cfg.diarizer_module.emb_dim)

        self.sortformer_diarizer = SortformerEncLabelModel.from_config_dict(self._cfg.diarizer_module)
        self.sortformer_encoder = SortformerEncLabelModel.from_config_dict(self._cfg.sortformer_encoder)
        self.transformer_encoder = SortformerEncLabelModel.from_config_dict(self._cfg.transformer_encoder)
        if self._cfg.get('position_embedding', None) is not None:
            self.position_embedding = SortformerEncLabelModel.from_config_dict(self._cfg.position_embedding)
        self.sort_preds = self._cfg.get("sort_preds", False)
        self.sort_accum_frames = self._cfg.get("sort_accum_frames", 1)
        self.subsegment_check_frames = self._cfg.get("subsegment_check_frames", 5)
        self.use_new_pil = self._cfg.get("use_new_pil", False)
        self.pil_noise_level = self._cfg.get("pil_noise_level", 0.0)
        self._init_loss_weights()
        self.min_sample_duration = self._cfg.get("min_sample_duration", -1)

        self.original_audio_offsets = {}
        self.eps = 1e-3
        self.emb_dim = self._cfg.diarizer_module.emb_dim

        if trainer is not None:
            self.loss = instantiate(self._cfg.loss)
            self.affinity_loss = instantiate(self._cfg.affinity_loss)
        else:
            self.loss = instantiate(self._cfg.loss)
            self.multichannel_mixing = self._cfg.get('multichannel_mixing', True)

        self.streaming_mode = self._cfg.get("streaming_mode", False)
        self.use_positional_embedding = self._cfg.get("use_positional_embedding", False)
        self.use_roformer = self._cfg.get("use_roformer", False)
        if self.use_roformer and self.use_positional_embedding and self._cfg.sortformer_encoder.num_layers > 0:
            raise ValueError("`use_roformer` with sortformer layers and `use_positional_embedding` are mutually exclusive. Choose only one of them.")
        self.alpha = self._cfg.alpha
        self.affinity_weighting = self._cfg.get('affinity_weighting', True)
        self.save_hyperparameters("cfg")
        self._init_eval_metrics()
        
        speaker_inds = list(range(self._cfg.max_num_of_spks))
        self.spk_perm = torch.tensor(list(itertools.permutations(speaker_inds))) # Get all permutations
    
    def _init_loss_weights(self):
        pil_weight = self._cfg.get("pil_weight", 0.0)
        ats_weight = self._cfg.get("ats_weight", 1.0)
        if pil_weight + ats_weight == 0:
            raise ValueError(f"weights for PIL {pil_weight} and ATS {ats_weight} cannot sum to 0")
        self.pil_weight = pil_weight/(pil_weight + ats_weight)
        self.ats_weight = ats_weight/(pil_weight + ats_weight)
        logging.info(f"Normalized weights for PIL {self.pil_weight} and ATS {self.ats_weight}")
        
    def _init_eval_metrics(self):
        self._accuracy_test = MultiBinaryAccuracy()
        self._accuracy_train = MultiBinaryAccuracy()
        self._accuracy_valid = MultiBinaryAccuracy()
        self._accuracy_train_vad= MultiBinaryAccuracy()
        self._accuracy_valid_vad= MultiBinaryAccuracy()
        self._accuracy_test_vad= MultiBinaryAccuracy()
        self._accuracy_train_ovl= MultiBinaryAccuracy()
        self._accuracy_valid_ovl= MultiBinaryAccuracy()
        self._accuracy_test_ovl= MultiBinaryAccuracy()
        self.max_f1_acc = 0.0
        self.time_flag = 0.0
        self.time_flag_end = 0.0


    def __setup_dataloader_from_config(self, config):
        # Switch to lhotse dataloader if specified in the config
        if config.get("use_lhotse"):
            return get_lhotse_dataloader_from_config(
                config,
                global_rank=self.global_rank,
                local_rank=self.local_rank,
                world_size=self.world_size,
                dataset=LhotseSpeechToDiarizationLabelDataset(cfg=config),
            )

        featurizer = WaveformFeaturizer(
            sample_rate=config['sample_rate'], int_values=config.get('int_values', False), augmentor=self.augmentor
        )

        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
            return None

        logging.info(f"Loading dataset from {config.manifest_filepath}")

        if self._trainer is not None:
            global_rank = self._trainer.global_rank
        else:
            global_rank = 0
        time_flag = time.time()
        logging.info("AAB: Starting Dataloader Instance loading... Step A")
        
        AudioToSpeechDiarTrainDataset = AudioToSpeechMSDDTrainDataset
        
        preprocessor = EncDecSpeakerLabelModel.from_config_dict(self._cfg.preprocessor)
        dataset = AudioToSpeechDiarTrainDataset(
            manifest_filepath=config.manifest_filepath,
            preprocessor=preprocessor,
            soft_label_thres=config.soft_label_thres,
            session_len_sec=config.session_len_sec,
            num_spks=config.num_spks,
            featurizer=featurizer,
            window_stride=self._cfg.preprocessor.window_stride,
            global_rank=global_rank,
            soft_targets=config.soft_targets if 'soft_targets' in config else False,
        )
        logging.info(f"AAB: Dataloader dataset is created, starting torch.utils.data.Dataloader step B: {time.time() - time_flag}")

        self.data_collection = dataset.collection
        self.collate_ds = dataset
         
        dataloader_instance = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            collate_fn=self.collate_ds.msdd_train_collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=False,
            num_workers=config.get('num_workers', 1),
            pin_memory=config.get('pin_memory', False),
        )
        print(f"AAC: Dataloader Instance loading is done ETA Step B done: {time.time() - time_flag}")
        return dataloader_instance
    
    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        self._train_dl = self.__setup_dataloader_from_config(config=train_data_config,)

    def setup_validation_data(self, val_data_layer_config: Optional[Union[DictConfig, Dict]]):
        self._validation_dl = self.__setup_dataloader_from_config(config=val_data_layer_config,)
    
    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        self._test_dl = self.__setup_dataloader_from_config(config=test_data_config,)

    def test_dataloader(self):
        if self._test_dl is not None:
            return self._test_dl

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            audio_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            audio_eltype = AudioSignal()
        return {
            "audio_signal": NeuralType(('B', 'T'), audio_eltype),
            "audio_signal_length": NeuralType(('B',), LengthsType()),
            "ms_seg_timestamps": NeuralType(('B', 'C', 'T', 'D'), LengthsType()),
            "ms_seg_counts": NeuralType(('B', 'C'), LengthsType()),
            "scale_mapping": NeuralType(('B', 'C', 'T'), LengthsType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        return OrderedDict(
            {
                "probs": NeuralType(('B', 'T', 'C'), ProbsType()),
                "scale_weights": NeuralType(('B', 'T', 'C', 'D'), ProbsType()),
                "batch_affinity_mat": NeuralType(('B', 'T', 'T'), ProbsType()),
            }
        )

    def length_to_mask(self, context_embs):
        """
        Convert length values to encoder mask input tensor.

        Args:
            lengths (torch.Tensor): tensor containing lengths of sequences
            max_len (int): maximum sequence length

        Returns:
            mask (torch.Tensor): tensor of shape (batch_size, max_len) containing 0's
                                in the padded region and 1's elsewhere
        """
        lengths=torch.tensor([context_embs.shape[1]] * context_embs.shape[0]) 
        batch_size = context_embs.shape[0]
        max_len=context_embs.shape[1]
        # create a tensor with the shape (batch_size, 1) filled with ones
        row_vector = torch.arange(max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
        # create a tensor with the shape (batch_size, max_len) filled with lengths
        length_matrix = lengths.unsqueeze(1).expand(-1, max_len).to(lengths.device)
        # create a mask by comparing the row vector and length matrix
        mask = row_vector < length_matrix
        return mask.float().to(context_embs.device)
    
    def frontend_encoder(self, processed_signal, processed_signal_length, pre_encode_input=False):
        """ 
        Generate encoder outputs from frontend encoder.
        
        Args:
            process_signal (torch.Tensor): tensor containing audio signal
            processed_signal_length (torch.Tensor): tensor containing lengths of audio signal

        Returns:
            emb_seq (torch.Tensor): tensor containing encoder outputs
            emb_seq_length (torch.Tensor): tensor containing lengths of encoder outputs
        """
        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)
        self.encoder = self.encoder.to(self.device)
        emb_seq, emb_seq_length = self.encoder(audio_signal=processed_signal, length=processed_signal_length, pre_encode_input=pre_encode_input)
        emb_seq = emb_seq.transpose(1, 2)
        if self._cfg.encoder.d_model != self._cfg.diarizer_module.emb_dim:
            self.encoder_proj = self.encoder_proj.to(self.device)
            emb_seq = self.encoder_proj(emb_seq)   
        return emb_seq, emb_seq_length

    def forward_infer(self, emb_seq, start_pos=0):
        """
        The main forward pass for diarization inference.

        Args:
            emb_seq (torch.Tensor): tensor containing embeddings of multiscale embedding vectors
                Dimension: (batch_size, max_seg_count, msdd_scale_n, emb_dim)
        
        Returns:
            preds (torch.Tensor): Sorted tensor containing predicted speaker labels
                Dimension: (batch_size, max. diar frame count, num_speakers)
            encoder_states_list (list): List containing total speaker memory for each step for debugging purposes
                Dimension: [(batch_size, max. diar frame count, inner dim), ]
        """
        encoder_states_list = []
        encoder_mask = self.length_to_mask(emb_seq)
        trans_emb_seq = self.transformer_encoder(encoder_states=emb_seq, encoder_mask=encoder_mask)
        _preds = self.sortformer_diarizer.forward_speaker_sigmoids(trans_emb_seq)
        if self.sort_preds:
            _preds = self.sort_probs_and_labels(_preds, discrete=False)
        if self.sortformer_encoder.sort_bin_order and (self._cfg.sortformer_encoder.num_layers > 0 and not self.use_roformer):
            preds = self.alpha * _preds + (1 - self.alpha) * preds_mean
            preds = self.sort_probs_and_labels(preds, discrete=False)
        else:
            preds = _preds
        return preds, encoder_states_list
    
    def process_signal(self, audio_signal, audio_signal_length):
        audio_signal = audio_signal.to(self.device)
        self.sortformer_encoder = self.sortformer_encoder.to(self.device)
        audio_signal = (1/(audio_signal.max()+self.eps)) * audio_signal 
        processed_signal, processed_signal_length = self.preprocessor(input_signal=audio_signal, length=audio_signal_length) 
        return processed_signal, processed_signal_length
    
    def forward(
        self, 
        audio_signal, 
        audio_signal_length, 
    ):
        """
        Forward pass for training.
        
        Args:
            audio_signal (torch.Tensor): tensor containing audio waveform
                Dimension: (batch_size, num_samples)
            audio_signal_length (torch.Tensor): tensor containing lengths of audio waveforms
                Dimension: (batch_size,)
            
        Returns:
            preds (torch.Tensor): Sorted tensor containing predicted speaker labels
                Dimension: (batch_size, max. diar frame count, num_speakers)
            encoder_states_list (list): List containing total speaker memory for each step for debugging purposes
                Dimension: [(batch_size, max. diar frame count, inner dim), ]
        """
        processed_signal, processed_signal_length = self.process_signal(audio_signal=audio_signal, audio_signal_length=audio_signal_length)
        processed_signal = processed_signal[:, :, :processed_signal_length.max()]
        if self.streaming_mode:
            preds, encoder_states_list = self.forward_streaming_infer(processed_signal)
        else:
            emb_seq, _ = self.frontend_encoder(processed_signal=processed_signal, processed_signal_length=processed_signal_length)
            preds, encoder_states_list = self.forward_infer(emb_seq)
        return preds, encoder_states_list
    
    def _streaming_feat_loader(self, feat_seq, chunk_n):
        """
        Load a chunk of feature sequence for streaming inference.

        Args:
            feat_seq (torch.Tensor): Tensor containing feature sequence
                Dimension: (batch_size, feat_dim, feat frame count)
            
        Yields:
            step_idx (int): Index of the current step
            chunk_feat_seq (torch.Tensor): Tensor containing the chunk of feature sequence
                Dimension: (batch_size, diar frame count, feat_dim)
            feat_lengths (torch.Tensor): Tensor containing lengths of the chunk of feature sequence
                Dimension: (batch_size,)
        """
        for step_idx in tqdm(range(0, chunk_n), desc="memory and steps diar", leave=False): 
            if step_idx == 0:
                stt_feat = 0
                end_feat = (self.sortformer_diarizer.mem_len*self.n_feat_fr)
            else:
                stt_feat = (self.sortformer_diarizer.mem_len + (step_idx-1) * self.sortformer_diarizer.step_len) * self.n_feat_fr
                end_feat = (self.sortformer_diarizer.mem_len + (step_idx)* self.sortformer_diarizer.step_len) * self.n_feat_fr

            chunk_feat_seq = feat_seq[:, :, stt_feat:end_feat]
            feat_lengths = torch.tensor(chunk_feat_seq.shape[-1]).repeat(chunk_feat_seq.shape[0])
            chunk_feat_seq_t = torch.transpose(chunk_feat_seq, 1, 2)
            yield step_idx, chunk_feat_seq_t, feat_lengths
        
    def forward_streaming_infer(self, feat_seq, streaming_mode=True):
        """ 
        Streaming inference.

        Args:
            feat_seq (torch.Tensor): Tensor containing feature sequence
                Dimension: (batch_size, feat_dim, feat frame count)

        Returns:
            preds (torch.Tensor): Sorted tensor containing predicted speaker labels
                Dimension: (batch_size, max. diar frame count, num_speakers)
            encoder_states_list (list): List containing total speaker memory for each step for debugging purposes
                Dimension: [(batch_size, max. diar frame count, inner dim), ]
        """
        total_pred_list, total_memory_list = [], []
        encoder_states_list, chunk_emb_seq, memory_buff, memory_label = None, None, None, None
        chunk_n = torch.ceil(torch.tensor((feat_seq.shape[2]- self.n_feat_fr*self.sortformer_diarizer.mem_len)/(self.n_feat_fr*self.sortformer_diarizer.step_len))).int().item() + 1
        for (step_idx, chunk_feat_seq_t, feat_lengths) in self._streaming_feat_loader(feat_seq=feat_seq, chunk_n=chunk_n):
            chunk_pre_encoded, _ = self.encoder.pre_encode(x=chunk_feat_seq_t, lengths=feat_lengths)
            if step_idx == 0:
                memory_buff = new_context_embs = chunk_pre_encoded
            elif step_idx > 0:
                chunk_pre_encoded, _ = self.encoder.pre_encode(x=chunk_feat_seq_t, lengths=feat_lengths)
                new_context_embs = torch.cat([memory_buff, chunk_pre_encoded], dim=1)
                chunk_emb_seq = chunk_pre_encoded
              
            # Specify `pre_encode_input=True` to skip pre-encode layers  
            org_feat_lengths = torch.tensor(new_context_embs.shape[1]*self.n_feat_fr).repeat(new_context_embs.shape[0]).to(self.device)
            nest_embs, _ = self.frontend_encoder(processed_signal=new_context_embs, processed_signal_length=org_feat_lengths, pre_encode_input=True)

            preds_mem_new, encoder_states_list = self.forward_infer(nest_embs)
            if step_idx == 0:
                total_pred_list.append(preds_mem_new)
            else:
                total_pred_list.append(preds_mem_new[:, self.sortformer_diarizer.mem_len:]) 
            
            # Binarize the predictions
            preds_mem_new_binary = preds_mem_new.round() 
            if step_idx < chunk_n - 1:
                memory_buff, memory_label = self.sortformer_diarizer.memory_compressor(step_idx=step_idx, 
                                                                                       chunk_emb_seq=chunk_emb_seq, 
                                                                                       prev_preds=preds_mem_new_binary, 
                                                                                       memory_buff=memory_buff, 
                                                                                       memory_label=memory_label
                                                                                    )
            total_memory_list.append(memory_label.detach().cpu())
            del chunk_emb_seq, preds_mem_new_binary, new_context_embs
            torch.cuda.empty_cache()
        preds = torch.cat(total_pred_list, dim=1)
        return preds, encoder_states_list  
 
    def find_first_nonzero(self, mat, max_cap_val=-1):
        # non zero values mask
        non_zero_mask = mat != 0
        # operations on the mask to find first nonzero values in the rows
        mask_max_values, mask_max_indices = torch.max(non_zero_mask, dim=1)
        # if the max-mask is zero, there is no nonzero value in the row
        mask_max_indices[mask_max_values == 0] = max_cap_val
        return mask_max_indices

    def sort_probs_and_labels(self, labels, discrete=True, thres=0.5, return_inds=False, accum_frames=1):
        max_cap_val = labels.shape[1] + 1
        labels_discrete = labels.clone()
        if not discrete:
            labels_discrete[labels_discrete < thres] = 0
            labels_discrete[labels_discrete >= thres] = 1
        modi =torch.ones(labels.shape[1],labels.shape[1]).triu().to(labels.device)
        labels_accum = torch.matmul(labels_discrete.permute(0,2,1),modi).permute(0,2,1)
        labels_accum[labels_accum < accum_frames] = 0
        label_fz = self.find_first_nonzero(labels_accum, max_cap_val)
        label_fz[label_fz == -1] = max_cap_val
        sorted_inds = torch.sort(label_fz)[1]
        sorted_labels = labels.transpose(0,1)[:, torch.arange(labels.shape[0]).unsqueeze(1), sorted_inds].transpose(0, 1)
        if return_inds:
            return sorted_labels, sorted_inds
        else:
            return sorted_labels

    def sort_targets_with_preds(self, labels, preds):
        """
        Sorts labels and predictions to get optimal permutation
        """
        perm_size = self.spk_perm.shape[0]
        permed_labels = labels[:, :, self.spk_perm]
        preds_rep = torch.unsqueeze(preds, 2).repeat(1,1, self.spk_perm.shape[0],1)
        match_score = torch.sum(permed_labels * preds_rep, axis=1).sum(axis=2)
        batch_best_perm = torch.argmax(match_score, axis=1)
        rep_spk_perm = self.spk_perm.repeat(batch_best_perm.shape[0],1) # (batch_size * perm_size, max_num_of_spks)
        global_inds_vec = torch.arange(0, perm_size*batch_best_perm.shape[0], perm_size).to(batch_best_perm.device) + batch_best_perm
        batch_perm_inds = rep_spk_perm[global_inds_vec.to(rep_spk_perm.device), :] # (batch_size, max_num_of_spks)
        max_score_permed_labels = torch.vstack([ labels[k, :, batch_perm_inds[k]].unsqueeze(0) for k in range(batch_perm_inds.shape[0])])
        return max_score_permed_labels

    def sort_targets_with_preds_new(self, labels, preds, target_lens, noise=0):
        """
        Sorts labels and predictions to get optimal permutation
        New implementation based on losses, supports noising
        """
        perm_size = self.spk_perm.shape[0]
        permed_labels = labels[:, :, self.spk_perm]
        match_loss = torch.zeros(labels.shape[0], perm_size)
        for k in range(labels.shape[0]):
            for i in range(perm_size):
                match_loss[k,i] = self.loss(probs=preds[k,:,:].unsqueeze(0), labels=permed_labels[k,:,i,:].unsqueeze(0), target_lens=target_lens)
        if noise > 0:
            min_loss = torch.min(match_loss, dim=1).values.reshape(-1,1).repeat(1,perm_size)
            match_loss = match_loss + torch.rand(labels.shape[0], self.spk_perm.shape[0])*min_loss*noise

        batch_best_perm = torch.argmin(match_loss, axis=1)
        rep_spk_perm = self.spk_perm.repeat(batch_best_perm.shape[0],1) # (batch_size * perm_size, max_num_of_spks)
        global_inds_vec = torch.arange(0, perm_size*batch_best_perm.shape[0], perm_size).to(batch_best_perm.device) + batch_best_perm
        batch_perm_inds = rep_spk_perm[global_inds_vec.to(rep_spk_perm.device), :] # (batch_size, max_num_of_spks)
        max_score_permed_labels = torch.vstack([ labels[k, :, batch_perm_inds[k]].unsqueeze(0) for k in range(batch_perm_inds.shape[0])])
        return max_score_permed_labels

    def compute_aux_f1(self, preds, targets):
        preds_bin = (preds > 0.5).to(torch.int64).detach()
        targets_ovl_mask = (targets.sum(dim=2) >= 2)
        preds_vad_mask = (preds_bin.sum(dim=2) > 0)
        targets_vad_mask = (targets.sum(dim=2) > 0)
        preds_ovl = preds[targets_ovl_mask, :].unsqueeze(0)
        targets_ovl = targets[targets_ovl_mask, :].unsqueeze(0)
        preds_vad_mask_ = preds_vad_mask.int().unsqueeze(0)
        targets_vad_mask_ = targets_vad_mask.int().unsqueeze(0) 
        return preds_vad_mask_, preds_ovl, targets_vad_mask_, targets_ovl
    
    def _reset_train_f1_accs(self):
        self._accuracy_train.reset() 
        self._accuracy_train_vad.reset()
        self._accuracy_train_ovl.reset()
        
    def training_step(self, batch: list, batch_idx: int):
        audio_signal, audio_signal_length, targets, target_lens = batch
        # apply random cutting of subsegments 
        if self.min_sample_duration > 0:
            audio_signal, audio_signal_length, targets, target_lens = get_sample_frames(audio_signal=audio_signal, 
                                                                                        audio_signal_length=audio_signal_length, 
                                                                                        targets=targets, 
                                                                                        target_lens=targets_lens, 
                                                                                        min_sample_duration=self.min_sample_duration,
                                                                                        subsegment_check_frames=self.subsegment_check_frames,
                                                                                        cfg=self._cfg,
                                                                                        preprocessor=self.preprocessor,
                                                                                        )

        preds, encoder_states_list = self.forward(audio_signal=audio_signal, audio_signal_length=audio_signal_length)
        # Arrival-time sorted (ATS) targets
        targets_ats = self.sort_probs_and_labels(targets.clone(), discrete=False, accum_frames=self.sort_accum_frames)
        # Optimally permuted targets for Permutation-Invariant Loss (PIL)
        if self.use_new_pil:
            targets_pil = self.sort_targets_with_preds_new(targets=targets.clone(), preds=preds, target_lens=target_lens, noise=self.pil_noise_level)
        else:
            targets_pil = self.sort_targets_with_preds(targets.clone(), preds)

        ats_loss = self.loss(probs=preds, labels=targets_ats, target_lens=target_lens)
        pil_loss = self.loss(probs=preds, labels=targets_pil, target_lens=target_lens)
        loss = self.ats_weight * ats_loss + self.pil_weight * pil_loss
        self.log('loss', loss, sync_dist=True)
        self.log('ats_loss', ats_loss, sync_dist=True)
        self.log('pil_loss', pil_loss, sync_dist=True)
        self.log('learning_rate', self._optimizer.param_groups[0]['lr'], sync_dist=True)

        self._reset_train_f1_accs()
        preds_vad, preds_ovl, targets_vad, targets_ovl = self.compute_aux_f1(preds, targets_pil)
        self._accuracy_train_vad(preds_vad, targets_vad, target_lens)
        self._accuracy_train_ovl(preds_ovl, targets_ovl, target_lens)
        train_f1_vad = self._accuracy_train_vad.compute()
        train_f1_ovl = self._accuracy_train_ovl.compute()
        self._accuracy_train(preds, targets_pil, target_lens)
        f1_acc = self._accuracy_train.compute()
        precision, recall = self._accuracy_train.compute_pr()

        self.log('train_f1_acc', f1_acc, sync_dist=True)
        self.log('train_precision', precision, sync_dist=True)
        self.log('train_recall', recall, sync_dist=True)
        self.log('train_f1_vad_acc', train_f1_vad, sync_dist=True)
        self.log('train_f1_ovl_acc', train_f1_ovl, sync_dist=True)

        self._reset_train_f1_accs()
        preds_vad, preds_ovl, targets_vad, targets_ovl = self.compute_aux_f1(preds, targets_ats)
        self._accuracy_train_vad(preds_vad, targets_vad, target_lens)
        self._accuracy_train_ovl(preds_ovl, targets_ovl, target_lens)
        train_f1_vad = self._accuracy_train_vad.compute()
        train_f1_ovl = self._accuracy_train_ovl.compute()
        self._accuracy_train(preds, targets_ats, target_lens)
        f1_acc = self._accuracy_train.compute()

        self.log('train_f1_acc_ats', f1_acc, sync_dist=True)
        self.log('train_f1_vad_acc_ats', train_f1_vad, sync_dist=True)
        self.log('train_f1_ovl_acc_ats', train_f1_ovl, sync_dist=True)
        self._accuracy_train.reset()
        return {'loss': loss}
    
    def _reset_valid_f1_accs(self):
        self._accuracy_valid.reset() 
        self._accuracy_valid_vad.reset()
        self._accuracy_valid_ovl.reset()
    
    def _reset_test_f1_accs(self):
        self._accuracy_valid.reset() 
        self._accuracy_test_vad.reset()
        self._accuracy_test_ovl.reset()
        
    def _cumulative_test_set_eval(self, score_dict: Dict[str, float], batch_idx: int, sample_count: int):
        if batch_idx == 0:
            self._reset_test_f1_accs()
            self.total_sample_counts = 0
            self.cumulative_f1_acc_sum = 0
            self.cumulative_f1_vad_acc_sum = 0
            self.cumulative_f1_ovl_acc_sum = 0
            
        self.total_sample_counts += sample_count
        self.cumulative_f1_acc_sum += score_dict['f1_acc'] * sample_count
        self.cumulative_f1_vad_acc_sum += score_dict['f1_vad_acc'] * sample_count
        self.cumulative_f1_ovl_acc_sum += score_dict['f1_ovl_acc'] * sample_count
        
        cumulative_f1_acc = self.cumulative_f1_acc_sum / self.total_sample_counts
        cumulative_f1_vad_acc = self.cumulative_f1_vad_acc_sum / self.total_sample_counts
        cumulative_f1_ovl_acc = self.cumulative_f1_ovl_acc_sum / self.total_sample_counts
        return {"cum_test_f1_acc": cumulative_f1_acc,
                "cum_test_f1_vad_acc": cumulative_f1_vad_acc,
                "cum_test_f1_ovl_acc": cumulative_f1_ovl_acc,
        }
    
    def validation_step(self, batch: list, batch_idx: int, dataloader_idx: int = 0):
        audio_signal, audio_signal_length, targets, target_lens = batch
        # sequence_lengths = torch.tensor([x[-1] for x in target_lens])
        preds, encoder_states_list = self.forward(
            audio_signal=audio_signal,
            audio_signal_length=audio_signal_length,
        )

        # Arrival-time sorted (ATS) targets
        targets_ats = self.sort_probs_and_labels(targets.clone(), discrete=False, accum_frames=self.sort_accum_frames)
        # Optimally permuted targets for Permutation-Invariant Loss (PIL)
        if self.use_new_pil:
            targets_pil = self.sort_targets_with_preds_new(targets=targets.clone(), preds=preds, target_lens=target_lens)
        else:
            targets_pil = self.sort_targets_with_preds(targets.clone(), preds)

        ats_loss = self.loss(probs=preds, labels=targets_ats, target_lens=target_lens)
        pil_loss = self.loss(probs=preds, labels=targets_pil, target_lens=target_lens)
        loss = self.ats_weight * ats_loss + self.pil_weight * pil_loss

        self._reset_valid_f1_accs()
        preds_vad, preds_ovl, targets_vad, targets_ovl = self.compute_aux_f1(preds, targets_pil)
        self._accuracy_valid_vad(preds_vad, targets_vad, target_lens)
        valid_f1_vad = self._accuracy_valid_vad.compute()
        self._accuracy_valid_ovl(preds_ovl, targets_ovl, target_lens)
        valid_f1_ovl = self._accuracy_valid_ovl.compute()
        self._accuracy_valid(preds, targets_pil, target_lens)
        f1_acc = self._accuracy_valid.compute()
        precision, recall = self._accuracy_valid.compute_pr()

        self._reset_valid_f1_accs()
        preds_vad, preds_ovl, targets_vad, targets_ovl = self.compute_aux_f1(preds, targets_ats)
        self._accuracy_valid_vad(preds_vad, targets_vad, target_lens)
        self._accuracy_valid_ovl(preds_ovl, targets_ovl, target_lens)
        valid_f1_ovl_ats = self._accuracy_valid_ovl.compute()
        self._accuracy_valid(preds, targets_ats, target_lens)
        f1_acc_ats = self._accuracy_valid.compute()

        metrics = {
            'val_loss': loss,
            'val_ats_loss': ats_loss,
            'val_pil_loss': pil_loss,
            'val_f1_acc': f1_acc,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1_vad_acc': valid_f1_vad,
            'val_f1_ovl_acc': valid_f1_ovl,
            'val_f1_acc_ats': f1_acc_ats,
            'val_f1_ovl_acc_ats': valid_f1_ovl_ats
        }

        if isinstance(self.trainer.val_dataloaders, list) and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(metrics)
        else:
            self.validation_step_outputs.append(metrics)
        return metrics

    def multi_validation_epoch_end(self, outputs: list, dataloader_idx: int = 0):
        if not outputs:
            logging.warning(f"MAYBE A BUG: empty outputs for dataloader={dataloader_idx}")
            return
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_ats_loss_mean = torch.stack([x['val_ats_loss'] for x in outputs]).mean()
        val_pil_loss_mean = torch.stack([x['val_pil_loss'] for x in outputs]).mean()
        val_f1_acc_mean = torch.stack([x['val_f1_acc'] for x in outputs]).mean()
        val_precision_mean = torch.stack([x['val_precision'] for x in outputs]).mean()
        val_recall_mean = torch.stack([x['val_recall'] for x in outputs]).mean()
        val_f1_vad_acc_mean = torch.stack([x['val_f1_vad_acc'] for x in outputs]).mean()
        val_f1_ovl_acc_mean = torch.stack([x['val_f1_ovl_acc'] for x in outputs]).mean()
        val_f1_acc_ats_mean = torch.stack([x['val_f1_acc_ats'] for x in outputs]).mean()
        val_f1_ovl_acc_ats_mean = torch.stack([x['val_f1_ovl_acc_ats'] for x in outputs]).mean()

        metrics = {
            'val_loss': val_loss_mean,
            'val_ats_loss': val_ats_loss_mean,
            'val_pil_loss': val_pil_loss_mean,
            'val_f1_acc': val_f1_acc_mean,
            'val_precision': val_precision_mean,
            'val_recall': val_recall_mean,
            'val_f1_vad_acc': val_f1_vad_acc_mean,
            'val_f1_ovl_acc': val_f1_ovl_acc_mean,
            'val_f1_acc_ats': val_f1_acc_ats_mean,
            'val_f1_ovl_acc_ats': val_f1_ovl_acc_ats_mean
        }
        return {'log': metrics}

    def multi_test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]], dataloader_idx: int = 0):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        f1_acc = self._accuracy_test.compute()
        self._accuracy_test.reset()
        self.log('test_f1_acc', f1_acc, sync_dist=True)
        return {
            'test_loss': test_loss_mean,
            'test_f1_acc': f1_acc,
        }
   
    def test_step(self, batch: list, batch_idx: int, dataloader_idx: int = 0):
        audio_signal, audio_signal_length, targets, target_lens = batch
        batch_size = audio_signal.shape[0]
        target_lens = self.target_lens.unsqueeze(0).repeat(batch_size, 1).to(audio_signal.device)
        preds, encoder_states_list = self.forward(
            audio_signal=audio_signal,
            audio_signal_length=audio_signal_length,
        )
        # Arrival-time sorted (ATS) targets
        targets_ats = self.sort_probs_and_labels(targets.clone(), discrete=False, accum_frames=self.sort_accum_frames)
        # Optimally permuted targets for Permutation-Invariant Loss (PIL)
        if self.use_new_pil:
            targets_pil = self.sort_targets_with_preds_new(targets.clone(), preds, target_lens)
        else:
            targets_pil = self.sort_targets_with_preds(targets.clone(), preds)

        preds_vad, preds_ovl, targets_vad, targets_ovl = self.compute_aux_f1(preds, targets_pil)
        self._accuracy_test_vad(preds_vad, targets_vad, target_lens, cumulative=True)
        test_f1_vad = self._accuracy_test_vad.compute()
        self._accuracy_test_ovl(preds_ovl, targets_ovl, target_lens, cumulative=True)
        test_f1_ovl = self._accuracy_test_ovl.compute()
        self._accuracy_test(preds, targets_pil, target_lens, cumulative=True)
        f1_acc = self._accuracy_test.compute()
        self.max_f1_acc = max(self.max_f1_acc, f1_acc)
        batch_score_dict = {"f1_acc": f1_acc, "f1_vad_acc": test_f1_vad, "f1_ovl_acc": test_f1_ovl}
        cum_score_dict = self._cumulative_test_set_eval(score_dict=batch_score_dict, batch_idx=batch_idx, sample_count=len(sequence_lengths))
        return self.preds_all

    def test_batch(self,):
        self.preds_total_list, self.batch_f1_accs_list, self.batch_precision_list, self.batch_recall_list = [], [], [], []
        self.batch_f1_accs_ats_list, self.batch_precision_ats_list, self.batch_recall_ats_list = [], [], []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self._test_dl)):
                audio_signal, audio_signal_length, targets, target_lens = batch
                audio_signal = audio_signal.to(self.device)
                audio_signal_length = audio_signal_length.to(self.device)
                preds, encoder_states_list = self.forward(
                    audio_signal=audio_signal,
                    audio_signal_length=audio_signal_length,
                )
                preds = preds.detach().to('cpu')
                if preds.shape[0] == 1: # Batch size = 1
                    self.preds_total_list.append(preds)
                else:
                    self.preds_total_list.extend(torch.split(preds, [1] * preds.shape[0]))
                torch.cuda.empty_cache()
                # Batch-wise evaluation: Arrival-time sorted (ATS) targets
                targets_ats = self.sort_probs_and_labels(targets.clone(), discrete=False, accum_frames=self.sort_accum_frames)
                # Optimally permuted targets for Permutation-Invariant Loss (PIL)
                if self.use_new_pil:
                    targets_pil = self.sort_targets_with_preds_new(targets.clone(), preds, target_lens)
                else:
                    targets_pil = self.sort_targets_with_preds(targets.clone(), preds)
                preds_vad, preds_ovl, targets_vad, targets_ovl = self.compute_aux_f1(preds, targets_pil)
                self._accuracy_valid(preds, targets_pil, target_lens)
                f1_acc = self._accuracy_valid.compute()
                precision, recall = self._accuracy_valid.compute_pr()
                self.batch_f1_accs_list.append(f1_acc)
                self.batch_precision_list.append(precision)
                self.batch_recall_list.append(recall)
                logging.info(f"batch {batch_idx}: f1_acc={f1_acc}, precision={precision}, recall={recall}")

                self._reset_valid_f1_accs()
                self._accuracy_valid(preds, targets_ats, target_lens)
                f1_acc = self._accuracy_valid.compute()
                precision, recall = self._accuracy_valid.compute_pr()
                self.batch_f1_accs_ats_list.append(f1_acc)
                self.batch_precision_ats_list.append(precision)
                self.batch_recall_ats_list.append(recall)
                logging.info(f"batch {batch_idx}: f1_acc_ats={f1_acc}, precision_ats={precision}, recall_ats={recall}")
                
        logging.info(f"Batch F1Acc. MEAN: {torch.mean(torch.tensor(self.batch_f1_accs_list))}")
        logging.info(f"Batch Precision MEAN: {torch.mean(torch.tensor(self.batch_precision_list))}")
        logging.info(f"Batch Recall MEAN: {torch.mean(torch.tensor(self.batch_recall_list))}")
        logging.info(f"Batch ATS F1Acc. MEAN: {torch.mean(torch.tensor(self.batch_f1_accs_ats_list))}")
        logging.info(f"Batch ATS Precision MEAN: {torch.mean(torch.tensor(self.batch_precision_ats_list))}")
        logging.info(f"Batch ATS Recall MEAN: {torch.mean(torch.tensor(self.batch_recall_ats_list))}")

    def diarize(self,):
        raise NotImplementedError

    def compute_accuracies(self):
        """
        Calculate F1 score and accuracy of the predicted sigmoid values.

        Returns:
            f1_score (float):
                F1 score of the estimated diarized speaker label sequences.
            simple_acc (float):
                Accuracy of predicted speaker labels: (total # of correct labels)/(total # of sigmoid values)
        """
        f1_score = self._accuracy_test.compute()
        num_correct = torch.sum(self._accuracy_test.true.bool())
        total_count = torch.prod(torch.tensor(self._accuracy_test.targets.shape))
        simple_acc = num_correct / total_count
        return f1_score, simple_acc
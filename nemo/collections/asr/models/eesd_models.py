
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

from collections import OrderedDict
from typing import Dict, List, Optional, Union
import time
import torch
from torch import nn
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from tqdm import tqdm
import itertools
import random
import math
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.asr.data.audio_to_eesd_label_lhotse import LhotseSpeechToDiarizationLabelDataset
from nemo.collections.asr.data.audio_to_eesd_label import AudioToSpeechMSDDTrainDataset
from nemo.collections.asr.metrics.multi_binary_acc import MultiBinaryAccuracy
from nemo.collections.asr.models.asr_model import ExportableEncDecModel
from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.utils.asr_multispeaker_utils import get_pil_target, get_ats_targets
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

torch.set_printoptions(sci_mode=False)
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
        * Transformer Encoder
        * FastConformer Encoder
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
        self.preprocessor = SortformerEncLabelModel.from_config_dict(self._cfg.preprocessor)

        if hasattr(self._cfg, 'spec_augment') and self._cfg.spec_augment is not None:
            self.spec_augmentation = SortformerEncLabelModel.from_config_dict(self._cfg.spec_augment)
        else:
            self.spec_augmentation = None

        self.encoder = SortformerEncLabelModel.from_config_dict(self._cfg.encoder)
        self.sortformer_modules = SortformerEncLabelModel.from_config_dict(self._cfg.sortformer_modules)
        self.transformer_encoder = SortformerEncLabelModel.from_config_dict(self._cfg.transformer_encoder)

        self.transformer_memory_compressor = SortformerEncLabelModel.from_config_dict(self._cfg.transformer_memory_compressor)
        if self.sortformer_modules.use_memory_pe:
            self.memory_position_embedding = SortformerEncLabelModel.from_config_dict(self._cfg.memory_position_embedding)

        self._init_loss_weights()

        self.eps = 1e-3
        self.loss = instantiate(self._cfg.loss)

        self.streaming_mode = self._cfg.get("streaming_mode", False)
        self.save_hyperparameters("cfg")
        self._init_eval_metrics()
        
        speaker_inds = list(range(self._cfg.max_num_of_spks))
        self.speaker_permutations = torch.tensor(list(itertools.permutations(speaker_inds))) # Get all permutations

        self.fifo_len = 0

        self.mem_refresh_rate = 0
        
    
    def _init_loss_weights(self):
        pil_weight = self._cfg.get("pil_weight", 0.0)
        ats_weight = self._cfg.get("ats_weight", 1.0)
        if pil_weight + ats_weight == 0:
            raise ValueError(f"weights for PIL {pil_weight} and ATS {ats_weight} cannot sum to 0")
        self.pil_weight = pil_weight/(pil_weight + ats_weight)
        self.ats_weight = ats_weight/(pil_weight + ats_weight)
        logging.info(f"Normalized weights for PIL {self.pil_weight} and ATS {self.ats_weight}")
        
    def _init_eval_metrics(self):
        """ 
        If there is no label, then the evaluation metrics will be based on Permutation Invariant Loss (PIL).
        """
        self._accuracy_test = MultiBinaryAccuracy()
        self._accuracy_train = MultiBinaryAccuracy()
        self._accuracy_valid = MultiBinaryAccuracy()
        
        self._accuracy_test_ats = MultiBinaryAccuracy()
        self._accuracy_train_ats = MultiBinaryAccuracy()
        self._accuracy_valid_ats = MultiBinaryAccuracy()

    def _reset_train_metrics(self):
        self._accuracy_train.reset()
        self._accuracy_train_ats.reset()
        
    def _reset_valid_metrics(self):
        self._accuracy_valid.reset()
        self._accuracy_valid_ats.reset()
        
    def __setup_dataloader_from_config(self, config):
        # Switch to lhotse dataloader if specified in the config
        if config.get("use_lhotse"):
            return get_lhotse_dataloader_from_config(
                config,
                global_rank=self.global_rank,
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
        logging.info(f"AAC: Dataloader Instance loading is done ETA Step B done: {time.time() - time_flag}")
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
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        return OrderedDict(
            {
                "preds": NeuralType(('B', 'T', 'C'), ProbsType()),
            }
        )

    def _streaming_feat_loader(self, feat_seq):
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
        feat_len = feat_seq.shape[2]
        num_chunks = math.ceil(feat_len / (self.sortformer_modules.step_len * self.encoder.subsampling_factor))
        logging.info(f"feat_len={feat_len}, num_chunks={num_chunks}")
        stt_feat = 0
        for step_idx in range(num_chunks):
            left_offset = min(self.sortformer_modules.step_left_context * self.encoder.subsampling_factor, stt_feat)
            end_feat = min(stt_feat + self.sortformer_modules.step_len * self.encoder.subsampling_factor, feat_len)
            right_offset = min(self.sortformer_modules.step_right_context * self.encoder.subsampling_factor, feat_len-end_feat)
            chunk_feat_seq = feat_seq[:, :, stt_feat-left_offset:end_feat+right_offset]
            stt_feat = end_feat

            feat_lengths = torch.tensor(chunk_feat_seq.shape[-1]).repeat(chunk_feat_seq.shape[0])
            chunk_feat_seq_t = torch.transpose(chunk_feat_seq, 1, 2)
            logging.info(f"step_idx: {step_idx}, chunk_feat_seq_t shape: {chunk_feat_seq_t.shape}")
            yield step_idx, chunk_feat_seq_t, feat_lengths, left_offset, right_offset
    
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
        if self._cfg.encoder.d_model != self._cfg.tf_d_model:
            self.sortformer_modules.encoder_proj = self.sortformer_modules.encoder_proj.to(self.device)
            emb_seq = self.sortformer_modules.encoder_proj(emb_seq)   
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
        encoder_mask = self.sortformer_modules.length_to_mask(emb_seq)
        trans_emb_seq = self.transformer_encoder(encoder_states=emb_seq, encoder_mask=encoder_mask)
        preds = self.sortformer_modules.forward_speaker_sigmoids(trans_emb_seq)
        return preds
    
    def process_signal(self, audio_signal, audio_signal_length):
        audio_signal = audio_signal.to(self.device)
        audio_signal = (1/(audio_signal.max()+self.eps)) * audio_signal 
        processed_signal, processed_signal_length = self.preprocessor(input_signal=audio_signal, length=audio_signal_length) 
        return processed_signal, processed_signal_length
    
    def forward(
        self, 
        audio_signal, 
        audio_signal_length, 
    ):
        """
        Forward pass for training and inference.
        
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
        if self._cfg.get("streaming_mode", False):
            if self.fifo_len > 0:
                logging.info("Using FIFO queue for streaming inference")
                preds = self.forward_streaming_FIFO(processed_signal, processed_signal_length)
            else:
                logging.info("Using Ivan's method for streaming inference")
                preds = self.forward_streaming(processed_signal, processed_signal_length)
        else:
            emb_seq, _ = self.frontend_encoder(processed_signal=processed_signal, processed_signal_length=processed_signal_length, pre_encode_input=False)
            preds = self.forward_infer(emb_seq)
        return preds

    def forward_streaming_FIFO(
        self,
        processed_signal,
        processed_signal_length,
    ):
        batch_size = processed_signal.shape[0]
        total_pred_list = []
        MEM = torch.empty(batch_size, 0, 512).to(self.device)
        FIFO_QUEUE = torch.empty(batch_size, 0, 512).to(self.device)
        MAX_FIFO_LEN = self.fifo_len
        MEM_REFRESH_RATE = self.mem_refresh_rate

        for (step_idx, chunk_feat_seq_t, feat_lengths, left_offset, right_offset) in self._streaming_feat_loader(feat_seq=processed_signal):
            #get pre_encode embs for current chunk
            
            chunk_pre_encode_embs, _ = self.encoder.pre_encode(x=chunk_feat_seq_t, lengths=feat_lengths)
            # logging.info(f"step_idx: {step_idx}, full chunk_pre_encode_embs shape: {chunk_pre_encode_embs.shape}")

            lc = round(left_offset / self.encoder.subsampling_factor)
            rc = round(right_offset / self.encoder.subsampling_factor)
            # chunk_pre_encode_embs = chunk_pre_encode_embs[:, left_context_size:left_context_size+self.sortformer_modules.step_len]
            logging.info(f"step_idx: {step_idx}, chunk_pre_encode_embs shape: {chunk_pre_encode_embs.shape}")
            
            mem_chunk_pre_encode_embs = torch.cat([MEM, FIFO_QUEUE, chunk_pre_encode_embs], dim=1)
            mem_chunk_len = mem_chunk_pre_encode_embs.shape[1]
            org_feat_lengths = torch.tensor(mem_chunk_len*self.encoder.subsampling_factor).repeat(batch_size).to(self.device)
            mem_chunk_encoder_embs, _ = self.frontend_encoder(processed_signal=mem_chunk_pre_encode_embs, processed_signal_length=org_feat_lengths, pre_encode_input=True)
            logging.info(f"step_idx: {step_idx}, mem_chunk_encoder_embs shape: {mem_chunk_encoder_embs.shape}")

            mem_chunk_preds = self.forward_infer(mem_chunk_encoder_embs)
            MEM_LEN, FIFO_LEN, CHUNK_LEN = MEM.size(1), FIFO_QUEUE.size(1), chunk_pre_encode_embs.size(1)
            chunk_preds = mem_chunk_preds[:, -CHUNK_LEN:][:, lc:lc+self.sortformer_modules.step_len]
            chunk_embs = mem_chunk_pre_encode_embs[:, -CHUNK_LEN:][:, lc:lc+self.sortformer_modules.step_len]
            if FIFO_LEN == 0:
                mem_chunk_preds = chunk_preds
                mem_chunk_pre_encode_embs = chunk_embs
            else:
                fifo_preds = mem_chunk_preds[:, -FIFO_LEN-CHUNK_LEN:-CHUNK_LEN]
                fifo_embs = mem_chunk_pre_encode_embs[:, -FIFO_LEN-CHUNK_LEN:-CHUNK_LEN]
                if MEM_LEN == 0:
                    mem_preds = torch.zeros(batch_size, 0, self._cfg.max_num_of_spks).to(self.device)
                    mem_embs = torch.zeros(batch_size, 0, 512).to(self.device)
                else:
                    mem_preds = mem_chunk_preds[:, :-FIFO_LEN-CHUNK_LEN]
                    mem_embs = mem_chunk_pre_encode_embs[:, :-FIFO_LEN-CHUNK_LEN]
                mem_chunk_preds = torch.cat([mem_preds, fifo_preds, chunk_preds], dim=1)
                mem_chunk_pre_encode_embs = torch.cat([mem_embs, fifo_embs, chunk_embs], dim=1)
            total_pred_list.append(chunk_preds)

            # update MEM and FIFO_QUEUE
            if MAX_FIFO_LEN > 0:
                FIFO_QUEUE = torch.cat([FIFO_QUEUE, chunk_embs], dim=1)
                
                # logging.info(f"FIFO_QUEUE shape: {FIFO_QUEUE.shape}")
                if MEM_REFRESH_RATE == 0 and FIFO_QUEUE.size(1) > MAX_FIFO_LEN:
                    MEM = self._compress_memory(emb_seq=mem_chunk_pre_encode_embs, preds=mem_chunk_preds)
                    FIFO_QUEUE = torch.empty(batch_size, 0, 512).to(self.device)
                    # FIFO_QUEUE = FIFO_QUEUE[:, -MAX_FIFO_LEN:]
                    
                elif MEM_REFRESH_RATE > 0:
                    if FIFO_QUEUE.size(1) > MAX_FIFO_LEN:
                        pop_out_embs = FIFO_QUEUE[:, :-MAX_FIFO_LEN]
                        pop_out_preds = fifo_preds[:, :pop_out_embs.size(1)]
                        MEM = torch.cat([MEM, pop_out_embs], dim=1)
                        if step_idx % MEM_REFRESH_RATE == 0:                     

                            if MEM.size(1) >= self.sortformer_modules.mem_len:
                                MEM = self._compress_memory(emb_seq=MEM, preds=torch.cat([mem_preds, pop_out_preds], dim=1))

                    FIFO_QUEUE = FIFO_QUEUE[:, -MAX_FIFO_LEN:]

            else:
                MEM = self._compress_memory(emb_seq=mem_chunk_pre_encode_embs, preds=mem_chunk_preds)      
            logging.info(f"step_idx: {step_idx}, FIFO_QUEUE shape: {FIFO_QUEUE.shape}")
            logging.info(f"step_idx: {step_idx}, MEM shape: {MEM.shape}")
       

        preds = torch.cat(total_pred_list, dim=1)
        # logging.info(f"preds shape: {preds.shape}")
        del MEM, FIFO_QUEUE
        torch.cuda.empty_cache()
        return preds

    def forward_streaming(
        self,
        processed_signal,
        processed_signal_length,
    ):
        batch_size = processed_signal.shape[0]
        total_pred_list = []
        memory_buff = torch.empty(batch_size, 0, self.encoder.d_model).to(self.device)

        for (step_idx, chunk_feat_seq_t, feat_lengths, left_offset, right_offset) in self._streaming_feat_loader(feat_seq=processed_signal):
            #get pre_encode embs for current chunk
            chunk_pre_encode_embs, _ = self.encoder.pre_encode(x=chunk_feat_seq_t, lengths=feat_lengths)
#            logging.info(f"step_idx: {step_idx}, chunk_pre_encode_embs shape: {chunk_pre_encode_embs.shape}")

            #get (mem+chunk) pre_encode embs
            #memory_buff = memory_buff.detach()
            mem_chunk_pre_encode_embs = torch.cat([memory_buff, chunk_pre_encode_embs], dim=1)
            logging.info(f"step_idx: {step_idx}, mem_chunk_pre_encode_embs shape: {mem_chunk_pre_encode_embs.shape}")

            #get preds for (mem+chunk)
            lc = round(left_offset / self.encoder.subsampling_factor)
            rc = round(right_offset / self.encoder.subsampling_factor)
            mem_chunk_len = mem_chunk_pre_encode_embs.shape[1]
            current_mem_len = memory_buff.shape[1]
            chunk_len = mem_chunk_len - current_mem_len - lc - rc

            org_feat_lengths = torch.tensor(mem_chunk_len*self.encoder.subsampling_factor).repeat(batch_size).to(self.device)
            mem_chunk_encoder_embs, _ = self.frontend_encoder(processed_signal=mem_chunk_pre_encode_embs, processed_signal_length=org_feat_lengths, pre_encode_input=True)
            logging.info(f"step_idx: {step_idx}, mem_chunk_encoder_embs shape: {mem_chunk_encoder_embs.shape}")

            mem_chunk_preds = self.forward_infer(mem_chunk_encoder_embs)

            mem_preds = mem_chunk_preds[:, :current_mem_len, :]
            chunk_preds = mem_chunk_preds[:, current_mem_len+lc:current_mem_len+lc+chunk_len, :] #drop left and right context
            total_pred_list.append(chunk_preds) #append chunk preds to the list

            #update memory buffer
            mem_chunk_preds = torch.cat([mem_preds, chunk_preds], dim=1)
            mem_chunk_pre_encode_embs = torch.cat([memory_buff, chunk_pre_encode_embs[:, lc:lc+chunk_len, :]], dim=1) #drop left and right context
            if current_mem_len + chunk_len <= self.sortformer_modules.mem_len:
                memory_buff = mem_chunk_pre_encode_embs
            else:
                memory_buff = self._compress_memory(emb_seq=mem_chunk_pre_encode_embs, preds=mem_chunk_preds)

        preds = torch.cat(total_pred_list, dim=1)
        logging.info(f"preds shape: {preds.shape}")
        del memory_buff
        torch.cuda.empty_cache()
        return preds

    def _compress_memory(self, emb_seq, preds):
        """.
        Compresses memory for streaming inference
        Keeps mem_len most important frames out of input n_frames, based on speaker sigmoid scores and positional information

        Args:
            emb_seq (torch.Tensor): Tensor containing n_frames > mem_len (mem+chunk) embeddings
                Dimension: (batch_size, n_frames, emb_dim)
            preds (torch.Tensor): Tensor containing n_frames > mem_len (mem+chunk) speaker sigmoid outputs
                Dimension: (batch_size, n_frames, max_num_spk)

        Returns:
            memory_buff (torch.Tensor): concatenation of num_spk subtensors of emb_seq corresponding to each speaker
            each of subtensors contains (mem_len//num_spk) frames out of n_frames
                Dimension: (batch_size, mem_len, emb_dim)
        """
        B, n_frames, n_spk = preds.shape
        emb_dim = emb_seq.shape[2]
        mem_len_per_spk = self.sortformer_modules.mem_len // n_spk
        last_n_sil_per_spk = 5

        #condition for frame being silence
        is_sil = preds.sum(dim=2) < 0.1 # Shape: (B, n_frames)
        is_sil = is_sil.unsqueeze(-1) # Shape: (B, n_frames, 1)
        #get mean silence embedding tensor
        emb_seq_sil = torch.where(is_sil, emb_seq, torch.tensor(0.0)) # Shape: (B, n_frames, emb_dim)
        emb_seq_sil_sum = emb_seq_sil.sum(dim=1) # Shape: (B, emb_dim)
        sil_count = is_sil.sum(dim=1).clamp(min=1) # Shape: (B)
        emb_seq_sil_mean = emb_seq_sil_sum / sil_count # Shape: (B, emb_dim)
        emb_seq_sil_mean = emb_seq_sil_mean.unsqueeze(1).expand(-1, n_spk*mem_len_per_spk, -1) # Shape: (B, n_spk*mem_len_for_spk, emb_dim)

        if self.sortformer_modules.use_memory_pe:
            #add position embeddings
            start_pos=0
            position_ids = torch.arange(start=start_pos, end=start_pos + n_frames, dtype=torch.long, device=preds.device)
            position_ids = position_ids.unsqueeze(0).repeat(preds.size(0), 1)
            preds = preds + self.memory_position_embedding(position_ids)

        #get frame importance scores
        encoder_mask = self.sortformer_modules.length_to_mask(preds)
        scores = self.transformer_memory_compressor(encoder_states=preds, encoder_mask=encoder_mask) # Shape: (B, n_frames, n_spk)

#        logging.info(f"MC scores: {scores[0,:,:]}")

        #normalized scores (non-overlapped frames are more preferable for memory)
        scores_norm = 2*scores - torch.sum(scores, dim=2).unsqueeze(-1).expand(-1, -1, n_spk)
#        logging.info(f"MC scores normalized: {scores_norm[0,:,:]}")

        #cumsum-normalized scores: this is to avoid speakers appearing in memory buffer before their block
        # as a result, for speaker i={0,1,2,...,n_spk-1}, scores_csnorm_i = 2*scores_i - sum_{j=i}^{n_spk-1}(scores_j)
        scores_csnorm = 2*scores - scores.flip(dims=[2]).cumsum(dim=2).flip(dims=[2])
#        logging.info(f"MC scores cumsum-normalized: {scores_csnorm[0,:,:]}")

        #scores thresholding: set -inf if cumsum-normalized score is less than 0.5.
        # This exclude non-speech frames and also doesn't allow speakers to appear in memory before their block
        is_good = scores_csnorm > 0.5
        scores = torch.where(is_good, scores_norm, torch.tensor(float('-inf'))) # Shape: (B, n_frames, n_spk)

        #ALTERNATIVE: thresholding, then using frame index as a score to keep latest possible frames in memory
#        scores = torch.where(is_good, torch.arange(n_frames, device=scores.device).view(1, n_frames, 1), torch.tensor(float('-inf'))) # Shape: (B, n_frames, n_spk)

#        logging.info(f"MC scores final: {scores[0,:,:]}")

        #get mem_len_per_spk most important indices for each speaker
        topk_values, topk_indices = torch.topk(scores, mem_len_per_spk-last_n_sil_per_spk, dim=1, largest=True, sorted=False) # Shape: (B, mem_len_per_spk-last_n_sil_per_spk, n_spk)
        valid_topk_mask = topk_values != float('-inf')
        topk_indices = torch.where(valid_topk_mask, topk_indices, torch.tensor(9999))  # Replace invalid indices with 9999

        if last_n_sil_per_spk > 0: #add number of silence frames in the end of each block
            topk_indices = torch.cat([topk_indices, torch.full((B, last_n_sil_per_spk, n_spk), 9999, device=topk_indices.device)], dim=1) # Shape: (B, mem_len_for_spk, n_spk)

        topk_indices = topk_indices.permute(0, 2, 1) # Shape: (B, n_spk, mem_len_for_spk)

        # sort indices to preserve original order of frames
        topk_indices_sorted, _ = torch.sort(topk_indices, dim=2) # Shape: (B, n_spk, mem_len_for_spk)
        topk_indices_flatten = topk_indices_sorted.reshape(B, n_spk*mem_len_per_spk) # Shape: (B, n_spk*mem_len_for_spk)
#        logging.info(f"MC topk indices: {topk_indices_sorted[0,:,:]}")

        # condition of being invalid index
        is_inf = topk_indices_flatten == 9999
        topk_indices_flatten[is_inf] = 0 # set a placeholder index instead of 9999 to make gather work

        # expand topk indices to emb_dim in last dimension to use gather
        topk_indices_expanded = topk_indices_flatten.unsqueeze(-1).expand(-1, -1, emb_dim) # Shape: (B, n_spk*mem_len_for_spk, emb_dim)

        # gather memory buffer including placeholder embeddings for silence frames
        emb_seq_gathered = torch.gather(emb_seq, 1, topk_indices_expanded) # Shape: (B, n_spk*mem_len_for_spk, emb_dim)

        # replace placeholder embeddings with actual mean silence embedding
        memory_buff = torch.where(is_inf.unsqueeze(-1), emb_seq_sil_mean, emb_seq_gathered)

        return memory_buff
    
    def _get_aux_train_evaluations(self, preds, targets, target_lens):
        # Arrival-time sorted (ATS) targets
        targets_ats = get_ats_targets(targets.clone(), preds, speaker_permutations=self.speaker_permutations)
        # Optimally permuted targets for Permutation-Invariant Loss (PIL)
        targets_pil = get_pil_target(targets.clone(), preds, speaker_permutations=self.speaker_permutations)
        ats_loss = self.loss(probs=preds, labels=targets_ats, target_lens=target_lens)
        pil_loss = self.loss(probs=preds, labels=targets_pil, target_lens=target_lens)
        loss = self.ats_weight * ats_loss + self.pil_weight * pil_loss

        self._accuracy_train(preds, targets_pil, target_lens)
        train_f1_acc, train_precision, train_recall = self._accuracy_train.compute()

        self._accuracy_train_ats(preds, targets_ats, target_lens)
        train_f1_acc_ats, _, _ = self._accuracy_train_ats.compute()

        train_metrics = {
            'loss': loss,
            'ats_loss': ats_loss,
            'pil_loss': pil_loss,
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'train_f1_acc': train_f1_acc,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1_acc_ats': train_f1_acc_ats,
        } 
        return train_metrics

    def training_step(self, batch: list, batch_idx: int):
        audio_signal, audio_signal_length, targets, target_lens = batch
        preds = self.forward(audio_signal=audio_signal, audio_signal_length=audio_signal_length)
        train_metrics = self._get_aux_train_evaluations(preds, targets, target_lens)
        self._reset_train_metrics()
        self.log_dict(train_metrics, sync_dist=True, on_step=True, on_epoch=False, logger=True)
        return {'loss': train_metrics['loss']}
        
    def _cumulative_test_set_eval(self, score_dict: Dict[str, float], batch_idx: int, sample_count: int):
        if batch_idx == 0:
            self.total_sample_counts = 0
            self.cumulative_f1_acc_sum = 0
            
        self.total_sample_counts += sample_count
        self.cumulative_f1_acc_sum += score_dict['f1_acc'] * sample_count
        
        cumulative_f1_acc = self.cumulative_f1_acc_sum / self.total_sample_counts
        return {"cum_test_f1_acc": cumulative_f1_acc}

    def _get_aux_validation_evaluations(self, preds, targets, target_lens):
        targets_ats = get_ats_targets(targets.clone(), preds, speaker_permutations=self.speaker_permutations)
        targets_pil = get_pil_target(targets.clone(), preds, speaker_permutations=self.speaker_permutations)

        val_ats_loss = self.loss(probs=preds, labels=targets_ats, target_lens=target_lens)
        val_pil_loss = self.loss(probs=preds, labels=targets_pil, target_lens=target_lens)
        val_loss = self.ats_weight * val_ats_loss + self.pil_weight * val_pil_loss

        self._accuracy_valid(preds, targets_pil, target_lens)
        val_f1_acc, val_precision, val_recall = self._accuracy_valid.compute()

        self._accuracy_valid_ats(preds, targets_ats, target_lens)
        valid_f1_acc_ats, _, _ = self._accuracy_valid_ats.compute()

        self._accuracy_valid.reset()
        self._accuracy_valid_ats.reset()

        val_metrics = {
            'val_loss': val_loss,
            'val_ats_loss': val_ats_loss,
            'val_pil_loss': val_pil_loss,
            'val_f1_acc': val_f1_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1_acc_ats': valid_f1_acc_ats,
        }
        return val_metrics

    def validation_step(self, batch: list, batch_idx: int, dataloader_idx: int = 0):
        audio_signal, audio_signal_length, targets, target_lens = batch
        preds = self.forward(
            audio_signal=audio_signal,
            audio_signal_length=audio_signal_length,
        )
        val_metrics = self._get_aux_validation_evaluations(preds, targets, target_lens)
        if isinstance(self.trainer.val_dataloaders, list) and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(val_metrics)
        else:
            self.validation_step_outputs.append(val_metrics)
        return val_metrics

    def multi_validation_epoch_end(self, outputs: list, dataloader_idx: int = 0):
        if not outputs:
            logging.warning(f"`outputs` is None; empty outputs for dataloader={dataloader_idx}")
            return None
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_ats_loss_mean = torch.stack([x['val_ats_loss'] for x in outputs]).mean()
        val_pil_loss_mean = torch.stack([x['val_pil_loss'] for x in outputs]).mean()
        val_f1_acc_mean = torch.stack([x['val_f1_acc'] for x in outputs]).mean()
        val_precision_mean = torch.stack([x['val_precision'] for x in outputs]).mean()
        val_recall_mean = torch.stack([x['val_recall'] for x in outputs]).mean()
        val_f1_acc_ats_mean = torch.stack([x['val_f1_acc_ats'] for x in outputs]).mean()

        self._reset_valid_metrics()
        
        multi_val_metrics = {
            'val_loss': val_loss_mean,
            'val_ats_loss': val_ats_loss_mean,
            'val_pil_loss': val_pil_loss_mean,
            'val_f1_acc': val_f1_acc_mean,
            'val_precision': val_precision_mean,
            'val_recall': val_recall_mean,
            'val_f1_acc_ats': val_f1_acc_ats_mean,
        }
        return {'log': multi_val_metrics}
    
    def multi_test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]], dataloader_idx: int = 0):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        f1_acc, _, _ = self._accuracy_test.compute()
        self._accuracy_test.reset()
        multi_test_metrics = {
            'test_loss': test_loss_mean,
            'test_f1_acc': f1_acc,
        }
        self.log_dict(multi_test_metrics, sync_dist=True, on_step=True, on_epoch=False, logger=True)
        return multi_test_metrics
   
    def test_step(self, batch: list, batch_idx: int, dataloader_idx: int = 0):
        audio_signal, audio_signal_length, targets, target_lens = batch
        batch_size = audio_signal.shape[0]
        target_lens = self.target_lens.unsqueeze(0).repeat(batch_size, 1).to(audio_signal.device)
        preds = self.forward(
            audio_signal=audio_signal,
            audio_signal_length=audio_signal_length,
        )
        targets_pil = get_pil_target(targets.clone(), preds, speaker_permutations=self.speaker_permutations)
        self._accuracy_test(preds, targets_pil, target_lens, cumulative=True)
        f1_acc, _, _ = self._accuracy_test.compute()
        batch_score_dict = {"f1_acc": f1_acc}
        cum_score_dict = self._cumulative_test_set_eval(score_dict=batch_score_dict, batch_idx=batch_idx, sample_count=len(sequence_lengths))
        return self.preds_all

    def _get_aux_test_batch_evaluations(self, batch_idx, preds, targets, target_lens):
        targets_ats = get_ats_targets(targets.clone(), preds, speaker_permutations=self.speaker_permutations)
        targets_pil = get_pil_target(targets.clone(), preds, speaker_permutations=self.speaker_permutations)
        self._accuracy_test(preds, targets_pil, target_lens)
        f1_acc, precision, recall = self._accuracy_test.compute()
        self.batch_f1_accs_list.append(f1_acc)
        self.batch_precision_list.append(precision)
        self.batch_recall_list.append(recall)
        logging.info(f"batch {batch_idx}: f1_acc={f1_acc}, precision={precision}, recall={recall}")

        self._accuracy_test_ats(preds, targets_ats, target_lens)
        f1_acc_ats, precision_ats, recall_ats = self._accuracy_test_ats.compute()
        self.batch_f1_accs_ats_list.append(f1_acc_ats)
        logging.info(f"batch {batch_idx}: f1_acc_ats={f1_acc_ats}, precision_ats={precision_ats}, recall_ats={recall_ats}")

        self._accuracy_test.reset()
        self._accuracy_test_ats.reset()

    def test_batch(self,):
        self.preds_total_list, self.batch_f1_accs_list, self.batch_precision_list, self.batch_recall_list, self.batch_f1_accs_ats_list = [], [], [], [], []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self._test_dl)):
                audio_signal, audio_signal_length, targets, target_lens = batch
                audio_signal = audio_signal.to(self.device)
                audio_signal_length = audio_signal_length.to(self.device)
                preds = self.forward(
                    audio_signal=audio_signal,
                    audio_signal_length=audio_signal_length,
                )
                preds = preds.detach().to('cpu')
                if preds.shape[0] == 1: # batch size = 1
                    self.preds_total_list.append(preds)
                else:
                    self.preds_total_list.extend(torch.split(preds, [1] * preds.shape[0]))
                torch.cuda.empty_cache()
                self._get_aux_test_batch_evaluations(batch_idx, preds, targets, target_lens)

        logging.info(f"Batch F1Acc. MEAN: {torch.mean(torch.tensor(self.batch_f1_accs_list))}")
        logging.info(f"Batch Precision MEAN: {torch.mean(torch.tensor(self.batch_precision_list))}")
        logging.info(f"Batch Recall MEAN: {torch.mean(torch.tensor(self.batch_recall_list))}")
        logging.info(f"Batch ATS F1Acc. MEAN: {torch.mean(torch.tensor(self.batch_f1_accs_ats_list))}")

    def forward_batch_streaming(
        self, 
        audio_signal, 
        audio_signal_length, 
    ):
        processed_signal, processed_signal_length = self.process_signal(audio_signal=audio_signal, audio_signal_length=audio_signal_length)
        processed_signal = processed_signal[:, :, :processed_signal_length.max()]
        if self._cfg.get("streaming_mode", False):
            preds = self.forward_streaming(processed_signal, processed_signal_length)
        else:
            emb_seq, _ = self.frontend_encoder(processed_signal=processed_signal, processed_signal_length=processed_signal_length, pre_encode_input=False)
            preds = self.forward_infer(emb_seq)
        return preds
        

    def diarize(self,):
        raise NotImplementedError

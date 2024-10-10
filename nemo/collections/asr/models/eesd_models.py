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
        self.preprocessor = EncDecSpeakerLabelModel.from_config_dict(self._cfg.preprocessor)

        if hasattr(self._cfg, 'spec_augment') and self._cfg.spec_augment is not None:
            self.spec_augmentation = SortformerEncLabelModel.from_config_dict(self._cfg.spec_augment)
        else:
            self.spec_augmentation = None

        self.encoder = SortformerEncLabelModel.from_config_dict(self._cfg.encoder)
        self.sortformer_modules = SortformerEncLabelModel.from_config_dict(self._cfg.diarizer_module)
        self.transformer_encoder = SortformerEncLabelModel.from_config_dict(self._cfg.transformer_encoder)
        self._init_loss_weights()

        self.eps = 1e-3
        if trainer is not None:
            self.loss = instantiate(self._cfg.loss)
        else:
            self.loss = instantiate(self._cfg.loss)

        self.streaming_mode = self._cfg.get("streaming_mode", False)
        self.save_hyperparameters("cfg")
        self._init_eval_metrics()
        
        speaker_inds = list(range(self._cfg.max_num_of_spks))
        self.speaker_permutations = torch.tensor(list(itertools.permutations(speaker_inds))) # Get all permutations
    
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
        # The main F1 accuracies 
        self._accuracy_test = MultiBinaryAccuracy()
        self._accuracy_train = MultiBinaryAccuracy()
        self._accuracy_valid = MultiBinaryAccuracy()
        
        self._accuracy_test_ats = MultiBinaryAccuracy()
        self._accuracy_train_ats = MultiBinaryAccuracy()
        self._accuracy_valid_ats = MultiBinaryAccuracy()
        
        # VAD accuracy is not related to the loss types  
        self._accuracy_train_vad = MultiBinaryAccuracy()
        self._accuracy_valid_vad = MultiBinaryAccuracy()
        self._accuracy_test_vad = MultiBinaryAccuracy()
        
        self._accuracy_train_ovl = MultiBinaryAccuracy()
        self._accuracy_valid_ovl = MultiBinaryAccuracy()
        self._accuracy_test_ovl = MultiBinaryAccuracy()
        
        # Overlapping regions accuracy should be separately computed for ATS 
        self._accuracy_train_ovl_ats = MultiBinaryAccuracy()
        self._accuracy_valid_ovl_ats = MultiBinaryAccuracy()
        self._accuracy_test_ovl_ats = MultiBinaryAccuracy()
        
        self.max_f1_acc = 0.0
        self.time_flag = 0.0
        self.time_flag_end = 0.0

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
        lengths = torch.tensor([context_embs.shape[1]] * context_embs.shape[0]) 
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
        emb_seq, emb_seq_length = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
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
        encoder_mask = self.length_to_mask(emb_seq)
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
            raise NotImplementedError("Streaming mode is not implemented yet.")
        else:
            emb_seq, _ = self.frontend_encoder(processed_signal=processed_signal, processed_signal_length=processed_signal_length)
            preds = self.forward_infer(emb_seq)
        return preds

    def _get_ovl_and_vad(self, preds, targets):
        preds_bin = (preds > 0.5).to(torch.int64).detach()
        targets_ovl_mask = (targets.sum(dim=2) >= 2)
        preds_vad_mask = (preds_bin.sum(dim=2) > 0)
        targets_vad_mask = (targets.sum(dim=2) > 0)
        preds_ovl = preds[targets_ovl_mask, :].unsqueeze(0)
        targets_ovl = targets[targets_ovl_mask, :].unsqueeze(0)
        preds_vad_mask_ = preds_vad_mask.int().unsqueeze(0)
        targets_vad_mask_ = targets_vad_mask.int().unsqueeze(0) 
        return preds_vad_mask_, preds_ovl, targets_vad_mask_, targets_ovl
    
    def _get_aux_train_evaluations(self, preds, targets, target_lens):
        # Arrival-time sorted (ATS) targets
        targets_ats = get_ats_targets(targets.clone(), preds, speaker_permutations=self.speaker_permutations)
        # Optimally permuted targets for Permutation-Invariant Loss (PIL)
        targets_pil = get_pil_target(targets.clone(), preds, speaker_permutations=self.speaker_permutations)
        ats_loss = self.loss(probs=preds, labels=targets_ats, target_lens=target_lens)
        pil_loss = self.loss(probs=preds, labels=targets_pil, target_lens=target_lens)
        loss = self.ats_weight * ats_loss + self.pil_weight * pil_loss

        preds_vad, preds_ovl, targets_vad, targets_ovl = self._get_ovl_and_vad(preds, targets_pil)
        self._accuracy_train_vad(preds_vad, targets_vad, target_lens)
        self._accuracy_train_ovl(preds_ovl, targets_ovl, target_lens)
        train_f1_vad, _, _ = self._accuracy_train_vad.compute()
        train_f1_ovl, _, _ = self._accuracy_train_ovl.compute()
        self._accuracy_train(preds, targets_pil, target_lens)
        train_f1_acc, train_precision, train_recall = self._accuracy_train.compute()

        preds_vad_ats, preds_ovl_ats, targets_vad_ats, targets_ovl_ats = self._get_ovl_and_vad(preds, targets_ats)
        self._accuracy_train_ovl_ats(preds_ovl_ats, targets_ovl_ats, target_lens)
        train_f1_ovl_ats, _, _ = self._accuracy_train_ovl_ats.compute()
        self._accuracy_train_ats(preds, targets_ats, target_lens)
        train_f1_acc_ats, _, _ = self._accuracy_train_ats.compute()

        self._accuracy_train.reset()
        self._accuracy_train_ats.reset()
        self._accuracy_train_ovl.reset()
        self._accuracy_train_ovl_ats.reset()
        self._accuracy_train_vad.reset()
         
        train_metrics = {
            'loss': loss,
            'ats_loss': ats_loss,
            'pil_loss': pil_loss,
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'train_f1_acc': train_f1_acc,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1_vad_acc': train_f1_vad,
            'train_f1_ovl_acc': train_f1_ovl,
            'train_f1_acc_ats': train_f1_acc_ats,
            'train_f1_ovl_acc_ats': train_f1_ovl_ats,
        } 
        return train_metrics

    def training_step(self, batch: list, batch_idx: int):
        audio_signal, audio_signal_length, targets, target_lens = batch
        preds = self.forward(audio_signal=audio_signal, audio_signal_length=audio_signal_length)
        train_metrics = self._get_aux_train_evaluations(preds, targets, target_lens)
        self.log_dict(train_metrics, sync_dist=True, on_step=True, on_epoch=False, logger=True)
        return {'loss': train_metrics['loss']}
        
    def _cumulative_test_set_eval(self, score_dict: Dict[str, float], batch_idx: int, sample_count: int):
        if batch_idx == 0:
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

    def _get_aux_validation_evaluations(self, preds, targets, target_lens):
        targets_ats = get_ats_targets(targets.clone(), preds, speaker_permutations=self.speaker_permutations)
        targets_pil = get_pil_target(targets.clone(), preds, speaker_permutations=self.speaker_permutations)

        val_ats_loss = self.loss(probs=preds, labels=targets_ats, target_lens=target_lens)
        val_pil_loss = self.loss(probs=preds, labels=targets_pil, target_lens=target_lens)
        val_loss = self.ats_weight * val_ats_loss + self.pil_weight * val_pil_loss

        preds_vad, preds_ovl, targets_vad, targets_ovl = self._get_ovl_and_vad(preds, targets_pil)
        self._accuracy_valid_vad(preds_vad, targets_vad, target_lens)
        valid_f1_vad, _, _ = self._accuracy_valid_vad.compute()
        self._accuracy_valid_ovl(preds_ovl, targets_ovl, target_lens)
        valid_f1_ovl, _, _ = self._accuracy_valid_ovl.compute()
        self._accuracy_valid(preds, targets_pil, target_lens)
        val_f1_acc, val_precision, val_recall = self._accuracy_valid.compute()

        preds_vad, preds_ovl, targets_vad, targets_ovl = self._get_ovl_and_vad(preds, targets_ats)
        self._accuracy_valid_ovl_ats(preds_ovl, targets_ovl, target_lens)
        valid_f1_ovl_ats, _, _ = self._accuracy_valid_ovl_ats.compute()
        self._accuracy_valid_ats(preds, targets_ats, target_lens)
        valid_f1_acc_ats, _, _ = self._accuracy_valid_ats.compute()

        self._accuracy_valid.reset()
        self._accuracy_valid_ats.reset()
        self._accuracy_valid_ovl.reset()
        self._accuracy_valid_ovl_ats.reset()
        self._accuracy_valid_vad.reset()

        val_metrics = {
            'val_loss': val_loss,
            'val_ats_loss': val_ats_loss,
            'val_pil_loss': val_pil_loss,
            'val_f1_acc': val_f1_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1_vad_acc': valid_f1_vad,
            'val_f1_ovl_acc': valid_f1_ovl,
            'val_f1_acc_ats': valid_f1_acc_ats,
            'val_f1_ovl_acc_ats': valid_f1_ovl_ats
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
        val_f1_vad_acc_mean = torch.stack([x['val_f1_vad_acc'] for x in outputs]).mean()
        val_f1_ovl_acc_mean = torch.stack([x['val_f1_ovl_acc'] for x in outputs]).mean()
        val_f1_acc_ats_mean = torch.stack([x['val_f1_acc_ats'] for x in outputs]).mean()
        val_f1_ovl_acc_ats_mean = torch.stack([x['val_f1_ovl_acc_ats'] for x in outputs]).mean()

        self._accuracy_valid.reset()
        self._accuracy_valid_vad.reset()
        self._accuracy_valid_ats.reset()
        self._accuracy_valid_ovl.reset()
        self._accuracy_valid_ovl_ats.reset()
        
        multi_val_metrics = {
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
        preds_vad, preds_ovl, targets_vad, targets_ovl = self._get_ovl_and_vad(preds, targets_pil)
        self._accuracy_test_vad(preds_vad, targets_vad, target_lens, cumulative=True)
        test_f1_vad, _, _ = self._accuracy_test_vad.compute()
        self._accuracy_test_ovl(preds_ovl, targets_ovl, target_lens, cumulative=True)
        test_f1_ovl, _, _ = self._accuracy_test_ovl.compute()
        self._accuracy_test(preds, targets_pil, target_lens, cumulative=True)
        f1_acc, _, _ = self._accuracy_test.compute()
        self.max_f1_acc = max(self.max_f1_acc, f1_acc)
        batch_score_dict = {"f1_acc": f1_acc, "f1_vad_acc": test_f1_vad, "f1_ovl_acc": test_f1_ovl}
        cum_score_dict = self._cumulative_test_set_eval(score_dict=batch_score_dict, batch_idx=batch_idx, sample_count=len(sequence_lengths))
        return self.preds_all

    def _get_aux_test_batch_evaluations(self, batch_idx, preds, targets, target_lens):
        targets_ats = get_ats_targets(targets.clone(), preds, speaker_permutations=self.speaker_permutations)
        targets_pil = get_pil_target(targets.clone(), preds, speaker_permutations=self.speaker_permutations)
        preds_vad, preds_ovl, targets_vad, targets_ovl = self._get_ovl_and_vad(preds, targets_pil)
        self._accuracy_test(preds, targets_pil, target_lens)
        f1_acc, precision, recall = self._accuracy_test.compute()
        self.batch_f1_accs_list.append(f1_acc)
        self.batch_precision_list.append(precision)
        self.batch_recall_list.append(recall)
        logging.info(f"batch {batch_idx}: f1_acc={f1_acc}, precision={precision}, recall={recall}")

        self._accuracy_test_ats(preds, targets_ats, target_lens)
        f1_acc, precision, recall = self._accuracy_test_ats.compute()
        self.batch_f1_accs_ats_list.append(f1_acc)
        self.batch_precision_ats_list.append(precision)
        self.batch_recall_ats_list.append(recall)
        logging.info(f"batch {batch_idx}: f1_acc_ats={f1_acc}, precision_ats={precision}, recall_ats={recall}")

        self._accuracy_test.reset()
        self._accuracy_test_ats.reset()

    def test_batch(self,):
        self.preds_total_list, self.batch_f1_accs_list, self.batch_precision_list, self.batch_recall_list = [], [], [], []
        self.batch_f1_accs_ats_list, self.batch_precision_ats_list, self.batch_recall_ats_list = [], [], []

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
        logging.info(f"Batch ATS Precision MEAN: {torch.mean(torch.tensor(self.batch_precision_ats_list))}")
        logging.info(f"Batch ATS Recall MEAN: {torch.mean(torch.tensor(self.batch_recall_ats_list))}")

    def diarize(self,):
        raise NotImplementedError

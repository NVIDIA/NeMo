# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import itertools
import os, math
import random
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from tqdm import tqdm

from nemo.collections.asr.data.audio_to_diar_label import AudioToSpeechE2ESpkDiarDataset
from nemo.collections.asr.data.audio_to_diar_label_lhotse import LhotseAudioToSpeechE2ESpkDiarDataset
from nemo.collections.asr.metrics.multi_binary_acc import MultiBinaryAccuracy
from nemo.collections.asr.models.asr_model import ExportableEncDecModel
from nemo.collections.asr.parts.mixins.diarization import DiarizeConfig, SpkDiarizationMixin
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.asr.parts.utils.asr_multispeaker_utils import get_ats_targets, get_pil_targets
from nemo.collections.asr.parts.utils.speaker_utils import generate_diarization_output_lines
from nemo.collections.asr.parts.utils.vad_utils import ts_vad_post_processing
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.neural_types import AudioSignal, LengthsType, NeuralType
from nemo.core.neural_types.elements import ProbsType
from nemo.utils import logging

__all__ = ['SortformerEncLabelModel']

def concat_and_pad(embs, lengths):
    """Concatenates lengths[i] first embeddings of embs[i], and pads the rest elements with zeros.
    Args:
        embs: List of embeddings Tensors of (B, T_i, D) shape
        lengths: List of lengths Tensors of (B,) shape

    Returns:
        output: concatenated embeddings Tensor of (B, T, D) shape
        total_lengths: output lengths Tensor of (B,) shape
    """

    assert len(embs) == len(lengths)
    device = embs[0].device
    dtype = embs[0].dtype
    B, D = embs[0].shape[0], embs[0].shape[2]

    total_lengths = torch.sum(torch.stack(lengths), dim=0)
    max_length = total_lengths.max().item()

    output = torch.zeros(B, max_length, D, device=device, dtype=dtype)
    start_indices = torch.zeros(B, dtype=torch.int64, device=device)

    for E, L in zip(embs, lengths):
        end_indices = start_indices + L
        for b in range(B):
            output[b, start_indices[b]:end_indices[b]] = E[b, :L[b]]
        start_indices = end_indices

    return output, total_lengths

class SortformerEncLabelModel(ModelPT, ExportableEncDecModel, SpkDiarizationMixin):
    """
    Encoder class for Sortformer diarization model.
    Model class creates training, validation methods for setting up data performing model forward pass.

    This model class expects config dict for:
        * preprocessor
        * Transformer Encoder
        * FastConformer Encoder
        * Sortformer Modules
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
        Initialize an Sortformer Diarizer model and a pretrained NEST encoder.
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

        self.encoder = SortformerEncLabelModel.from_config_dict(self._cfg.encoder).to(self.device)
        self.sortformer_modules = SortformerEncLabelModel.from_config_dict(self._cfg.sortformer_modules).to(
            self.device
        )
        self.transformer_encoder = SortformerEncLabelModel.from_config_dict(self._cfg.transformer_encoder).to(
            self.device
        )
        if self._cfg.encoder.d_model != self._cfg.model_defaults.tf_d_model:
            self.sortformer_modules.encoder_proj = self.sortformer_modules.encoder_proj.to(self.device)
        else:
            self.sortformer_modules.encoder_proj = None
        self._init_loss_weights()

        self.eps = 1e-3
        self.loss = instantiate(self._cfg.loss)

        self.pad_front = self._cfg.get("pad_front", False)
        self.async_streaming = self._cfg.get("async_streaming", False)

        self.streaming_mode = self._cfg.get("streaming_mode", False)
        self.save_hyperparameters("cfg")
        self._init_eval_metrics()
        speaker_inds = list(range(self._cfg.max_num_of_spks))
        self.speaker_permutations = torch.tensor(list(itertools.permutations(speaker_inds)))  # Get all permutations

        self.concat_and_pad_script = torch.jit.script(concat_and_pad)

    def _init_loss_weights(self):
        pil_weight = self._cfg.get("pil_weight", 0.0)
        ats_weight = self._cfg.get("ats_weight", 1.0)
        if pil_weight + ats_weight == 0:
            raise ValueError(f"weights for PIL {pil_weight} and ATS {ats_weight} cannot sum to 0")
        self.pil_weight = pil_weight / (pil_weight + ats_weight)
        self.ats_weight = ats_weight / (pil_weight + ats_weight)

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
                dataset=LhotseAudioToSpeechE2ESpkDiarDataset(cfg=config),
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

        dataset = AudioToSpeechE2ESpkDiarDataset(
            manifest_filepath=config.manifest_filepath,
            soft_label_thres=config.soft_label_thres,
            session_len_sec=config.session_len_sec,
            num_spks=config.num_spks,
            featurizer=featurizer,
            window_stride=self._cfg.preprocessor.window_stride,
            global_rank=global_rank,
            soft_targets=config.soft_targets if 'soft_targets' in config else False,
            device=self.device,
        )

        self.data_collection = dataset.collection
        self.collate_ds = dataset

        dataloader_instance = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            collate_fn=self.collate_ds.eesd_train_collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=False,
            num_workers=config.get('num_workers', 1),
            pin_memory=config.get('pin_memory', False),
        )
        return dataloader_instance

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        self._train_dl = self.__setup_dataloader_from_config(
            config=train_data_config,
        )

    def setup_validation_data(self, val_data_layer_config: Optional[Union[DictConfig, Dict]]):
        self._validation_dl = self.__setup_dataloader_from_config(
            config=val_data_layer_config,
        )

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        self._test_dl = self.__setup_dataloader_from_config(
            config=test_data_config,
        )

    def test_dataloader(self):
        if self._test_dl is not None:
            return self._test_dl
        return None

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

    def frontend_encoder(self, processed_signal, processed_signal_length, pre_encode_input: bool=False):
        """
        Generate encoder outputs from frontend encoder.

        Args:
            processed_signal (torch.Tensor): tensor containing audio-feature (mel spectrogram, mfcc, etc.)
            processed_signal_length (torch.Tensor): tensor containing lengths of audio signal in integers

        Returns:
            emb_seq (torch.Tensor): tensor containing encoder outputs
            emb_seq_length (torch.Tensor): tensor containing lengths of encoder outputs
        """
        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)
        emb_seq, emb_seq_length = self.encoder(audio_signal=processed_signal, length=processed_signal_length, pre_encode_input=pre_encode_input)
        emb_seq = emb_seq.transpose(1, 2)
        if self.sortformer_modules.encoder_proj is not None:
            emb_seq = self.sortformer_modules.encoder_proj(emb_seq)
        return emb_seq, emb_seq_length

    def forward_infer(self, emb_seq, emb_seq_length):
        """
        The main forward pass for diarization for offline diarization inference.

        Args:
            emb_seq (torch.Tensor): tensor containing FastConformer encoder states (embedding vectors).
                Dimension: (batch_size, diar_frame_count, emb_dim)
            emb_seq_length (torch.Tensor): tensor containing lengths of FastConformer encoder states.
                Dimension: (batch_size,)

        Returns:
            preds (torch.Tensor): Sorted tensor containing Sigmoid values for predicted speaker labels.
                Dimension: (batch_size, diar_frame_count, num_speakers)
        """
        encoder_mask = self.sortformer_modules.length_to_mask(emb_seq_length, emb_seq.shape[1])
        trans_emb_seq = self.transformer_encoder(encoder_states=emb_seq, encoder_mask=encoder_mask)
        preds = self.sortformer_modules.forward_speaker_sigmoids(trans_emb_seq)
        return preds

    def _diarize_forward(self, batch: Any):
        """
        A counterpart of `_transcribe_forward` function in ASR.
        This function is a wrapper for forward pass functions for compataibility
        with the existing classes.

        Args:
            batch (Any): The input batch containing audio signal and audio signal length.

        Returns:
            preds (torch.Tensor): Sorted tensor containing Sigmoid values for predicted speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
        """
        with torch.no_grad():
            preds = self.forward(audio_signal=batch[0], audio_signal_length=batch[1])
            preds = preds.to('cpu')
            torch.cuda.empty_cache()
        return preds

    def _diarize_output_processing(
        self, outputs, uniq_ids, diarcfg: DiarizeConfig
    ) -> Union[List[List[str]], Tuple[List[List[str]], List[torch.Tensor]]]:
        """
        Processes the diarization outputs and generates RTTM (Real-time Text Markup) files.
        TODO: Currently, this function is not included in mixin test because of
              `ts_vad_post_processing` function.
              (1) Implement a test-compatible function
              (2) `vad_utils.py` has `predlist_to_timestamps` function that is close to this function.
                  Needs to consolute differences and implement the test-compatible function.

        Args:
            outputs (torch.Tensor): Sorted tensor containing Sigmoid values for predicted speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
            uniq_ids (List[str]): List of unique identifiers for each audio file.
            diarcfg (DiarizeConfig): Configuration object for diarization.

        Returns:
            diar_output_lines_list (List[List[str]]): A list of lists, where each inner list contains
                                                      the RTTM lines for a single audio file.
            preds_list (List[torch.Tensor]): A list of tensors containing the diarization outputs
                                             for each audio file.

        """
        preds_list, diar_output_lines_list = [], []
        if outputs.shape[0] == 1:  # batch size = 1
            preds_list.append(outputs)
        else:
            preds_list.extend(torch.split(outputs, [1] * outputs.shape[0]))

        for sample_idx, uniq_id in enumerate(uniq_ids):
            offset = self._diarize_audio_rttm_map[uniq_id]['offset']
            speaker_assign_mat = preds_list[sample_idx].squeeze(dim=0)
            speaker_timestamps = [[] for _ in range(speaker_assign_mat.shape[-1])]
            for spk_id in range(speaker_assign_mat.shape[-1]):
                ts_mat = ts_vad_post_processing(
                    speaker_assign_mat[:, spk_id],
                    cfg_vad_params=diarcfg.postprocessing_params,
                    unit_10ms_frame_count=int(self._cfg.encoder.subsampling_factor),
                    bypass_postprocessing=False,
                )
                ts_mat = ts_mat + offset
                ts_seg_raw_list = ts_mat.tolist()
                ts_seg_list = [[round(stt, 2), round(end, 2)] for (stt, end) in ts_seg_raw_list]
                speaker_timestamps[spk_id].extend(ts_seg_list)

            diar_output_lines = generate_diarization_output_lines(
                speaker_timestamps=speaker_timestamps, model_spk_num=len(speaker_timestamps)
            )
            diar_output_lines_list.append(diar_output_lines)
        if diarcfg.include_tensor_outputs:
            return (diar_output_lines_list, preds_list)
        else:
            return diar_output_lines_list

    def _setup_diarize_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a temporary data loader which wraps the provided audio file.

        Args:
            config: A python dictionary which contains the following keys:
            - manifest_filepath: Path to the manifest file containing audio file paths
              and corresponding speaker labels.

        Returns:
            A pytorch DataLoader for the given audio file(s).
        """
        if 'manifest_filepath' in config:
            manifest_filepath = config['manifest_filepath']
            batch_size = config['batch_size']
        else:
            manifest_filepath = os.path.join(config['temp_dir'], 'manifest.json')
            batch_size = min(config['batch_size'], len(config['paths2audio_files']))

        dl_config = {
            'manifest_filepath': manifest_filepath,
            'sample_rate': self.preprocessor._sample_rate,
            'num_spks': config.get('num_spks', self._cfg.max_num_of_spks),
            'batch_size': batch_size,
            'shuffle': False,
            'soft_label_thres': 0.5,
            'session_len_sec': config['session_len_sec'],
            'num_workers': config.get('num_workers', min(batch_size, os.cpu_count() - 1)),
            'pin_memory': True,
        }
        temporary_datalayer = self.__setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer

    def process_signal(self, audio_signal, audio_signal_length):
        """
        Extract audio features from time-series signal for further processing in the model.

        This function performs the following steps:
        1. Moves the audio signal to the correct device.
        2. Normalizes the time-series audio signal.
        3. Extrac audio feature from from the time-series audio signal using the model's preprocessor.

        Args:
            audio_signal (torch.Tensor): The input audio signal.
                Shape: (batch_size, num_samples)
            audio_signal_length (torch.Tensor): The length of each audio signal in the batch.
                Shape: (batch_size,)

        Returns:
            tuple: A tuple containing:
                - processed_signal (torch.Tensor): The preprocessed audio signal.
                    Shape: (batch_size, num_features, num_frames)
                - processed_signal_length (torch.Tensor): The length of each processed signal.
                    Shape: (batch_size,)
        """
        audio_signal, audio_signal_length = audio_signal.to(self.device), audio_signal_length.to(self.device)
        if not self.streaming_mode:
            audio_signal = (1 / (audio_signal.max() + self.eps)) * audio_signal
        processed_signal, processed_signal_length = self.preprocessor(
            input_signal=audio_signal, length=audio_signal_length
        )
        del audio_signal, audio_signal_length
        if not self.training:
            torch.cuda.empty_cache()
        return processed_signal, processed_signal_length

    def shift_signal(
        self,
        processed_signal,
        offsets,
    ):
        B, C, T = processed_signal.shape
        shifted_signal = torch.stack([torch.cat([processed_signal[b, :, offsets[b].item():], processed_signal[b, :, :offsets[b].item()]], dim=1) for b in range(B)])
        return shifted_signal

    def shift_emb(
        self,
        emb_seq,
        offsets,
    ):
        B, T, C = emb_seq.shape
        shifted_emb = torch.stack([torch.cat([emb_seq[b, offsets[b].item():, :], emb_seq[b, :offsets[b].item(), :]], dim=0) for b in range(B)])
        return shifted_emb

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
        if self.streaming_mode:
            preds = self.forward_streaming(processed_signal, processed_signal_length)
        else:
            emb_seq, emb_seq_length = self.frontend_encoder(processed_signal=processed_signal, processed_signal_length=processed_signal_length)
            preds = self.forward_infer(emb_seq, emb_seq_length)
        return preds

    @property
    def input_names(self):
        return ["chunk", "chunk_lengths", "mem", "mem_lengths", "fifo", "fifo_lengths"]

    @property
    def output_names(self):
        return ["mem_fifo_chunk_preds", "chunk_pre_encode_embs", "chunk_pre_encode_lengths"]

    def streaming_input_examples(self):
        """
        Input tensor examples for exporting streaming version of model.
        """
        bs = 4
        chunk = torch.rand([bs, 120, 80]).to(self.device)
        chunk_lengths = torch.tensor([120] * bs).to(self.device)
        mem = torch.randn([bs, 188, 512]).to(self.device)
        mem_lengths = torch.tensor([40, 188, 0, 68]).to(self.device)
        fifo = torch.randn([bs, 188, 512]).to(self.device)
        fifo_lengths = torch.tensor([50, 88, 0, 90]).to(self.device)
        return chunk, chunk_lengths, mem, mem_lengths, fifo, fifo_lengths

    def streaming_export(self, output: str):
        input_example = self.streaming_input_examples()
        export_out = self.export(output, input_example=input_example)
        return export_out

    def forward_for_export(self, chunk, chunk_lengths, mem, mem_lengths, fifo, fifo_lengths):
        """
        This forward pass is for ONNX model export.

        Args:
            chunk (torch.Tensor): tensor containing audio waveform
                Dimension: (batch_size, feature frame count, dimension)
            chunk_lengths (torch.Tensor): tensor containing lengths of audio waveforms
                Dimension: (batch_size,)
            mem (torch.Tensor): tensor containing memory for the embeddings from start
                Dimension: (batch_size, mem_len, emb_dim)
            mem_lengths (torch.Tensor): tensor containing lengths of memory embeddings
                Dimension: (batch_size,)
            fifo (torch.Tensor): tensor containing memory for the embeddings from latest chunks
                Dimension: (batch_size, fifo_len, emb_dim)
            fifo_lengths (torch.Tensor): tensor containing lengths of FIFO queue embeddings
                Dimension: (batch_size,)

        Returns:
            mem_fifo_chunk_preds (torch.Tensor): Sorted tensor containing predicted speaker labels
                Dimension: (batch_size, max. diar frame count, num_speakers)
            chunk_pre_encode_embs (torch.Tensor): tensor containing pre-encoded embeddings from the chunk
                Dimension: (batch_size, num_frames, emb_dim)
            chunk_pre_encode_lengths (torch.Tensor): tensor containing lengths of pre-encoded embeddings from the chunk
                Dimension: (batch_size,)
        """
        # pre-encode the chunk
        chunk_pre_encode_embs, chunk_pre_encode_lengths = self.encoder.pre_encode(x=chunk, lengths=chunk_lengths)
        chunk_pre_encode_lengths = chunk_pre_encode_lengths.to(torch.int64)
        print(f"chunk_pre_encode_embs: {chunk_pre_encode_embs.shape}, chunk_pre_encode_lengths: {chunk_pre_encode_lengths}")

        # concat the embeddings from the memory, FIFO and the chunk
        mem_fifo_chunk_pre_encode_embs, mem_fifo_chunk_pre_encode_lengths = self.concat_and_pad_script([mem, fifo, chunk_pre_encode_embs], [mem_lengths, fifo_lengths, chunk_pre_encode_lengths])
        print(f"mem_fifo_chunk_pre_encode_embs: {mem_fifo_chunk_pre_encode_embs.shape}, mem_fifo_chunk_pre_encode_lengths: {mem_fifo_chunk_pre_encode_lengths}")

        # encode the concatenated embeddings
        mem_fifo_chunk_fc_encoder_embs, mem_fifo_chunk_fc_encoder_lengths  = self.frontend_encoder(processed_signal=mem_fifo_chunk_pre_encode_embs, processed_signal_length=mem_fifo_chunk_pre_encode_lengths, pre_encode_input=True)
        print(f"mem_fifo_chunk_fc_encoder_embs: {mem_fifo_chunk_fc_encoder_embs.shape}, mem_fifo_chunk_fc_encoder_lengths: {mem_fifo_chunk_fc_encoder_lengths}")

        # forward pass for inference
        mem_fifo_chunk_preds = self.forward_infer(mem_fifo_chunk_fc_encoder_embs, mem_fifo_chunk_fc_encoder_lengths)
        print(f"mem_fifo_chunk_preds: {mem_fifo_chunk_preds.shape}")

        return mem_fifo_chunk_preds, chunk_pre_encode_embs, chunk_pre_encode_lengths

    def forward_streaming(
        self,
        processed_signal,
        processed_signal_length,
    ):
        """
        The main forward pass for diarization inference in streaming mode.

        Args:
            processed_signal (torch.Tensor): tensor containing audio waveform
                Dimension: (batch_size, num_samples)
            processed_signal_length (torch.Tensor): tensor containing lengths of audio waveforms
                Dimension: (batch_size,)

        Returns:
            total_pred (torch.Tensor): tensor containing predicted speaker labels for the current chunk and all previous chunks
                Dimension: (batch_size, pred_len, num_speakers)
        """

        MEM = None # memory to save the embeddings from start
        MEM_LENS = None
        MEM_PREDS = None # memory predictions
        FIFO_QUEUE = None # memory to save the embedding from the latest chunks
        FIFO_LENS = None
        total_pred = None
        spk_perm = None

        B, C, T = processed_signal.shape
        if self.pad_front:
            deltas = (T - processed_signal_length) % (self.sortformer_modules.step_len*self.sortformer_modules.subsampling_factor)
            shifts = processed_signal_length + deltas
            processed_signal = self.shift_signal(processed_signal, shifts)
            processed_signal_offset = T - shifts
        else:
            processed_signal_offset = torch.zeros((B,), dtype=torch.long, device=self.device)

        if dist.is_available() and dist.is_initialized():
            local_tensor = torch.tensor([T], device=processed_signal.device)
            dist.all_reduce(local_tensor, op=dist.ReduceOp.MAX, async_op=False) # get max feature length across all GPUs
            max_T = local_tensor.item()
            if dist.get_rank() == 0:
                logging.info(f"Maximum feature length across all GPUs: {max_T}")
        else:
            max_T = T

        if T < max_T: # need padding to have the same feature length for all GPUs
            pad_tensor = torch.full((B, C, max_T-T), -99, dtype=processed_signal.dtype, device=processed_signal.device)
            processed_signal = torch.cat([processed_signal, pad_tensor], dim=2)

        att_mod = False
        if self.training:
            r = random.random()
            if r < self.sortformer_modules.causal_attn_rate:
                self.encoder.att_context_size=[-1, self.sortformer_modules.causal_attn_rc]
                self.transformer_encoder.diag = self.sortformer_modules.causal_attn_rc
                att_mod = True
        elif self.sortformer_modules.use_causal_eval:
            self.encoder.att_context_size=[-1, self.sortformer_modules.causal_attn_rc]
            self.transformer_encoder.diag = self.sortformer_modules.causal_attn_rc
            att_mod = True

        feat_len = processed_signal.shape[2]
        num_chunks = math.ceil(feat_len / (self.sortformer_modules.step_len * self.sortformer_modules.subsampling_factor))
        for (step_idx, chunk_feat_seq_t, feat_lengths, left_offset, right_offset) in tqdm(self.sortformer_modules.streaming_feat_loader(feat_seq=processed_signal, feat_seq_length=processed_signal_length, feat_seq_offset=processed_signal_offset), total=num_chunks, desc="Streaming Steps", disable=self.training):
            MEM, MEM_LENS, FIFO_QUEUE, FIFO_LENS, MEM_PREDS, _, total_pred, spk_perm = self.forward_streaming_step(
                processed_signal=chunk_feat_seq_t,
                processed_signal_length=feat_lengths,
                fifo_last_time=FIFO_QUEUE,
                fifo_lengths=FIFO_LENS,
                mem_last_time=MEM,
                mem_lengths=MEM_LENS,
                mem_preds_last_time=MEM_PREDS,
                previous_pred_out=total_pred,
                previous_spk_perm=spk_perm,
                left_offset=left_offset,
                right_offset=right_offset,
            )

        if att_mod:
            self.encoder.att_context_size=[-1, -1]
            self.transformer_encoder.diag = None

        del processed_signal, processed_signal_length, MEM, FIFO_QUEUE
        if not self.training:
            torch.cuda.empty_cache()

        if self.pad_front:
            total_offset = processed_signal_offset // self.sortformer_modules.subsampling_factor
            total_pred = self.shift_emb(total_pred, total_offset)

        if T < max_T: #discard preds corresponding to padding
            n_frames = math.ceil(T / self.encoder.subsampling_factor)
            total_pred = total_pred[:,:n_frames,:]
        return total_pred

    def forward_streaming_step(
        self,
        processed_signal,
        processed_signal_length,
        fifo_last_time=None,
        fifo_lengths=None,
        mem_last_time=None,
        mem_lengths=None,
        mem_preds_last_time=None,
        previous_pred_out=None,
        previous_spk_perm=None,
        left_offset=0,
        right_offset=0,
    ):
        """
        One-step forward pass for diarization inference in streaming mode.

        Args:
            processed_signal (torch.Tensor): tensor containing audio waveform
                Dimension: (batch_size, num_samples)
            processed_signal_length (torch.Tensor): tensor containing lengths of audio waveforms
                Dimension: (batch_size,)
            fifo_last_time (torch.Tensor): tensor containing memory for the latest chunks
                Dimension: (batch_size, fifo_len, emb_dim)
            mem_last_time (torch.Tensor): tensor containing memory for the embeddings from start
                Dimension: (batch_size, mem_len, emb_dim)
            mem_preds_last_time (torch.Tensor): tensor containing original predictions for memory
                Dimension: (batch_size, mem_len, num_speakers)
            previous_pred_out (torch.Tensor): tensor containing previous predicted speaker labels
                Dimension: (batch_size, pred_len, num_speakers)
            left_offset (int): left offset for the current chunk
            right_offset (int): right offset for the current chunk

        Returns:
            mem (torch.Tensor): tensor containing memory for the pre-encode embeddings from start
                Dimension: (batch_size, mem_len, emb_dim)
            fifo (torch.Tensor): tensor containing memory for the pre-encode embeddings from latest chunks
                Dimension: (batch_size, fifo_len, emb_dim)
            mem_preds (torch.Tensor): tensor containing predicted speaker labels for mem
                Dimension: (batch_size, mem_len, num_speakers)
            fifo_preds (torch.Tensor): tensor containing predicted speaker labels for fifo
                Dimension: (batch_size, fifo_len, num_speakers)
            total_step_preds (torch.Tensor): tensor containing predicted speaker labels for the current chunk and all previous chunks
                Dimension: (batch_size, total_pred_len, num_speakers)
            
        """

        B = processed_signal.shape[0]

        if self.async_streaming:
            if mem_last_time is None:
                mem_last_time = torch.zeros((B, self.sortformer_modules.mem_len, self.sortformer_modules.fc_d_model), device=self.device)
                mem_preds_last_time = torch.full((B, self.sortformer_modules.mem_len, self.sortformer_modules.unit_n_spks), -0.1, device=self.device)
                mem_lengths = torch.zeros((B,), dtype=torch.long, device=self.device) #zero offsets
            if fifo_last_time is None:
                fifo_last_time = torch.zeros((B, self.sortformer_modules.fifo_len, self.sortformer_modules.fc_d_model), device=self.device)
                fifo_lengths = torch.zeros((B,), dtype=torch.long, device=self.device) #zero offsets
        else:
            if mem_last_time is None:
                mem_last_time = self.sortformer_modules.init_memory(batch_size=B, d_model=self.sortformer_modules.fc_d_model, device=self.device)# memory to save the embeddings from start
            if fifo_last_time is None:
                fifo_last_time = self.sortformer_modules.init_memory(batch_size=B, d_model=self.sortformer_modules.fc_d_model, device=self.device)# memory to save the embedding from the latest chunks

        if previous_pred_out is None:
            previous_pred_out = self.sortformer_modules.init_memory(batch_size=B, d_model=self.sortformer_modules.unit_n_spks, device=self.device)

        chunk_pre_encode_embs, chunk_pre_encode_lengths = self.encoder.pre_encode(x=processed_signal, lengths=processed_signal_length)

        if self.async_streaming:
            mem_fifo_chunk_pre_encode_embs, mem_fifo_chunk_pre_encode_lengths = concat_and_pad([mem_last_time, fifo_last_time, chunk_pre_encode_embs], [mem_lengths, fifo_lengths, chunk_pre_encode_lengths])
        else:
            mem_fifo_chunk_pre_encode_embs = self.sortformer_modules.concat_embs([mem_last_time, fifo_last_time, chunk_pre_encode_embs], dim=1, device=self.device)
            mem_fifo_chunk_pre_encode_lengths = mem_last_time.shape[1] + fifo_last_time.shape[1] + chunk_pre_encode_lengths

        mem_fifo_chunk_fc_encoder_embs, mem_fifo_chunk_fc_encoder_lengths = self.frontend_encoder(processed_signal=mem_fifo_chunk_pre_encode_embs, processed_signal_length=mem_fifo_chunk_pre_encode_lengths, pre_encode_input=True)
        mem_fifo_chunk_preds = self.forward_infer(mem_fifo_chunk_fc_encoder_embs, mem_fifo_chunk_fc_encoder_lengths)

        B, T, C  = mem_fifo_chunk_preds.shape
        preds_mask = torch.arange(T, device=self.device).view(1, -1, 1).expand(B,-1,C) < mem_fifo_chunk_fc_encoder_lengths.view(-1, 1, 1).expand(-1,T,C)
        mem_fifo_chunk_preds = torch.where(preds_mask, mem_fifo_chunk_preds, torch.tensor(0.1))

        if self.async_streaming:
            mem, mem_lengths, fifo, fifo_lengths, mem_preds, fifo_preds, chunk_preds, spk_perm = self.sortformer_modules.update_memory_FIFO_async(
                mem=mem_last_time,
                mem_lengths=mem_lengths,
                mem_preds=mem_preds_last_time,
                fifo=fifo_last_time,
                fifo_lengths=fifo_lengths,
                chunk=chunk_pre_encode_embs,
                chunk_lengths=chunk_pre_encode_lengths,
                preds=mem_fifo_chunk_preds,
                spk_perm=previous_spk_perm,
                chunk_left_offset=round(left_offset / self.encoder.subsampling_factor),
                chunk_right_offset=math.ceil(right_offset / self.encoder.subsampling_factor),
            )
        else:
            mem, fifo, mem_preds, fifo_preds, chunk_preds, spk_perm = self.sortformer_modules.update_memory_FIFO(
                mem=mem_last_time,
                mem_preds=mem_preds_last_time,
                fifo=fifo_last_time,
                chunk=chunk_pre_encode_embs,
                preds=mem_fifo_chunk_preds,
                spk_perm=previous_spk_perm,
                chunk_left_offset=round(left_offset / self.encoder.subsampling_factor),
                chunk_right_offset=math.ceil(right_offset / self.encoder.subsampling_factor),
            )

        total_step_preds = torch.cat([previous_pred_out, chunk_preds], dim=1)

        if not self.training and self.sortformer_modules.visualization:
            self.chunk_preds_list.append(chunk_preds.detach().cpu().numpy())
            self.fifo_preds_list.append(fifo_preds.detach().cpu().numpy())
            self.mem_preds_list.append(mem_preds.detach().cpu().numpy())

        return mem, mem_lengths, fifo, fifo_lengths, mem_preds, fifo_preds, total_step_preds, spk_perm

    def _get_aux_train_evaluations(self, preds, targets, target_lens) -> dict:
        """
        Compute auxiliary training evaluations including losses and metrics.

        This function calculates various losses and metrics for the training process,
        including Arrival Time Sort (ATS) Loss and Permutation Invariant Loss (PIL)
        based evaluations.

        Args:
            preds (torch.Tensor): Predicted speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
            targets (torch.Tensor): Ground truth speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
            target_lens (torch.Tensor): Lengths of target sequences.
                Shape: (batch_size,)

        Returns:
            (dict): A dictionary containing the following training metrics.
        """
        targets_ats = get_ats_targets(targets.clone(), preds, speaker_permutations=self.speaker_permutations)
        targets_pil = get_pil_targets(targets.clone(), preds, speaker_permutations=self.speaker_permutations)
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

    def training_step(self, batch: list, batch_idx: int) -> dict:
        """
        Performs a single training step.

        Args:
            batch (list): A list containing the following elements:
                - audio_signal (torch.Tensor): The input audio signal in time-series format.
                - audio_signal_length (torch.Tensor): The length of each audio signal in the batch.
                - targets (torch.Tensor): The target labels for the batch.
                - target_lens (torch.Tensor): The length of each target sequence in the batch.
            batch_idx (int): The index of the current batch.

        Returns:
            (dict): A dictionary containing the 'loss' key with the calculated loss value.
        """
        audio_signal, audio_signal_length, targets, target_lens = batch
        preds = self.forward(audio_signal=audio_signal, audio_signal_length=audio_signal_length)
        train_metrics = self._get_aux_train_evaluations(preds, targets, target_lens)
        self._reset_train_metrics()
        self.log_dict(train_metrics, sync_dist=True, on_step=True, on_epoch=False, logger=True)
        return {'loss': train_metrics['loss']}

    def _get_aux_validation_evaluations(self, preds, targets, target_lens) -> dict:
        """
        Compute auxiliary validation evaluations including losses and metrics.

        This function calculates various losses and metrics for the training process,
        including Arrival Time Sort (ATS) Loss and Permutation Invariant Loss (PIL)
        based evaluations.

        Args:
            preds (torch.Tensor): Predicted speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
            targets (torch.Tensor): Ground truth speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
            target_lens (torch.Tensor): Lengths of target sequences.
                Shape: (batch_size,)

        Returns:
            val_metrics (dict): A dictionary containing the following validation metrics
        """
        targets_ats = get_ats_targets(targets.clone(), preds, speaker_permutations=self.speaker_permutations)
        targets_pil = get_pil_targets(targets.clone(), preds, speaker_permutations=self.speaker_permutations)

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
        """
        Performs a single validation step.

        This method processes a batch of data during the validation phase. It forward passes
        the audio signal through the model, computes various validation metrics, and stores
        these metrics for later aggregation.

        Args:
            batch (list): A list containing the following elements:
                - audio_signal (torch.Tensor): The input audio signal.
                - audio_signal_length (torch.Tensor): The length of each audio signal in the batch.
                - targets (torch.Tensor): The target labels for the batch.
                - target_lens (torch.Tensor): The length of each target sequence in the batch.
            batch_idx (int): The index of the current batch.
            dataloader_idx (int, optional): The index of the dataloader in case of multiple
                                            validation dataloaders. Defaults to 0.

        Returns:
            dict: A dictionary containing various validation metrics for this batch.
        """
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

    def _get_aux_test_batch_evaluations(self, batch_idx: int, preds, targets, target_lens):
        """
        Compute auxiliary validation evaluations including losses and metrics.

        This function calculates various losses and metrics for the training process,
        including Arrival Time Sort (ATS) Loss and Permutation Invariant Loss (PIL)
        based evaluations.

        Args:
            preds (torch.Tensor): Predicted speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
            targets (torch.Tensor): Ground truth speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
            target_lens (torch.Tensor): Lengths of target sequences.
                Shape: (batch_size,)
        """
        targets_ats = get_ats_targets(targets.clone(), preds, speaker_permutations=self.speaker_permutations)
        targets_pil = get_pil_targets(targets.clone(), preds, speaker_permutations=self.speaker_permutations)
        self._accuracy_test(preds, targets_pil, target_lens)
        f1_acc, precision, recall = self._accuracy_test.compute()
        self.batch_f1_accs_list.append(f1_acc)
        self.batch_precision_list.append(precision)
        self.batch_recall_list.append(recall)
        logging.info(f"batch {batch_idx}: f1_acc={f1_acc}, precision={precision}, recall={recall}")

        self._accuracy_test_ats(preds, targets_ats, target_lens)
        f1_acc_ats, precision_ats, recall_ats = self._accuracy_test_ats.compute()
        self.batch_f1_accs_ats_list.append(f1_acc_ats)
        logging.info(
            f"batch {batch_idx}: f1_acc_ats={f1_acc_ats}, precision_ats={precision_ats}, recall_ats={recall_ats}"
        )

        self._accuracy_test.reset()
        self._accuracy_test_ats.reset()

    def test_batch(
        self,
    ):
        """
        Perform batch testing on the model.

        This method iterates through the test data loader, making predictions for each batch,
        and calculates various evaluation metrics. It handles both single and multi-sample batches.
        """
        (
            self.preds_total_list,
            self.batch_f1_accs_list,
            self.batch_precision_list,
            self.batch_recall_list,
            self.batch_f1_accs_ats_list,
        ) = ([], [], [], [], [])

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self._test_dl)):
                audio_signal, audio_signal_length, targets, target_lens = batch
                audio_signal = audio_signal.to(self.device)
                audio_signal_length = audio_signal_length.to(self.device)
                targets = targets.to(self.device)
                preds = self.forward(
                    audio_signal=audio_signal,
                    audio_signal_length=audio_signal_length,
                )
                self._get_aux_test_batch_evaluations(batch_idx, preds, targets, target_lens)
                preds = preds.detach().to('cpu')
                if preds.shape[0] == 1:  # batch size = 1
                    self.preds_total_list.append(preds)
                else:
                    self.preds_total_list.extend(torch.split(preds, [1] * preds.shape[0]))
                torch.cuda.empty_cache()

        logging.info(f"Batch F1Acc. MEAN: {torch.mean(torch.tensor(self.batch_f1_accs_list))}")
        logging.info(f"Batch Precision MEAN: {torch.mean(torch.tensor(self.batch_precision_list))}")
        logging.info(f"Batch Recall MEAN: {torch.mean(torch.tensor(self.batch_recall_list))}")
        logging.info(f"Batch ATS F1Acc. MEAN: {torch.mean(torch.tensor(self.batch_f1_accs_ats_list))}")

    def on_validation_epoch_end(self) -> Optional[dict[str, dict[str, torch.Tensor]]]:
        """Run validation with sync_dist=True."""
        return super().on_validation_epoch_end(sync_metrics=True)

    @torch.no_grad()
    def diarize(
        self,
        audio: Union[str, List[str], np.ndarray, DataLoader],
        batch_size: int = 1,
        include_tensor_outputs: bool = False,
        postprocessing_yaml: Optional[str] = None,
        num_workers: int = 0,
        verbose: bool = True,
        override_config: Optional[DiarizeConfig] = None,
    ) -> Union[List[List[str]], Tuple[List[List[str]], List[torch.Tensor]]]:
        """One-click runner function for diarization.

        Args:
            audio: (a single or list) of paths to audio files or path to a manifest file.
            batch_size: (int) Batch size to use during inference.
                Bigger will result in better throughput performance but would use more memory.
            include_tensor_outputs: (bool) Include raw speaker activity probabilities to the output.
                See Returns: for more details.
            postprocessing_yaml: Optional(str) Path to .yaml file with postprocessing parameters.
            num_workers: (int) Number of workers for DataLoader.
            verbose: (bool) Whether to display tqdm progress bar.
            override_config: (Optional[DiarizeConfig]) A config to override the default config.

        Returns:
            *if include_tensor_outputs is False: A list of lists of speech segments with a corresponding speaker index,
                in format "[begin_seconds, end_seconds, speaker_index]".
            *if include_tensor_outputs is True: A tuple of the above list
                and list of tensors of raw speaker activity probabilities.
        """
        return super().diarize(
            audio=audio,
            batch_size=batch_size,
            include_tensor_outputs=include_tensor_outputs,
            postprocessing_yaml=postprocessing_yaml,
            num_workers=num_workers,
            verbose=verbose,
            override_config=override_config,
        )
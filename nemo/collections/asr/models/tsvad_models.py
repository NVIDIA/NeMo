# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import itertools
import json
import os
import pickle as pkl
from typing import Dict, List, Optional, Union

import librosa
import torch
from omegaconf import DictConfig
from omegaconf.omegaconf import open_dict
from pytorch_lightning import Trainer
from torch.utils.data import ChainDataset
from collections import OrderedDict
from nemo.collections.asr.models import ClusteringDiarizer
from nemo.collections.asr.data.audio_to_label import AudioToSpeechLabelDataset, AudioToSpeechTSVADDataset
# tsvad_collate_fn
from nemo.collections.asr.data.audio_to_label_dataset import get_tarred_speech_label_dataset
from nemo.collections.asr.data.audio_to_text_dataset import convert_to_config_list
from nemo.collections.asr.losses.angularloss import AngularSoftmaxLoss
from nemo.collections.asr.models.asr_model import ExportableEncDecModel
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.asr.parts.utils.speaker_utils import embedding_normalize
from nemo.collections.common.losses import CrossEntropyLoss as CELoss
from nemo.collections.common.metrics import TopKClassificationAccuracy
from nemo.collections.common.parts.preprocessing.collections import ASRSpeechLabel

from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import *
from nemo.utils import logging
from torchmetrics import Metric
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    LengthsType,
    LogitsType,
    LogprobsType,
    EncodedRepresentation,
    NeuralType,
    SpectrogramType,
)
from nemo.core.neural_types.elements import ProbsType


__all__ = ['EncDecDiarLabelModel', 'MultiBinaryAcc', 'ClusterEmbedding']

# def tsvad_accuracy(self, preds, targets):
# """
# TSVAD
# Loss function for multispeaker loss
# """

class MultiBinaryAcc(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.correct_counts_k = torch.sum(preds.round().bool() == targets.round().bool())
            self.total_counts_k = torch.prod(torch.tensor(x.shape))

    def compute(self):
        return self.correct_counts_k.float() / self.total_counts_k

class ClusterEmbedding:
    def __init__(self, cfg_clus: DictConfig):
        self._cfg = cfg_clus
        self._cfg_tsvad = cfg_clus.ts_vad_model
        self.max_num_of_spks = cfg_clus.ts_vad_model.max_num_of_spks
        self.clus_emb_path = 'speaker_outputs/embeddings/clus_emb_info.pkl'
    
    def prepare_cluster_embs(self):
        """
        TSVAD
        Prepare embeddings from clustering diarizer for TS-VAD style diarizer.
        """
        self.emb_sess_train_dict = self.run_clustering_diarizer(self._cfg_tsvad.train_ds.manifest_filepath,
                                                                self._cfg_tsvad.train_ds.emb_dir)
        self.emb_sess_dev_dict = self.run_clustering_diarizer(self._cfg_tsvad.validation_ds.manifest_filepath,
                                                              self._cfg_tsvad.validation_ds.emb_dir)
        self.emb_sess_test_dict = self.run_clustering_diarizer(self._cfg_tsvad.test_ds.manifest_filepath,
                                                               self._cfg_tsvad.test_ds.emb_dir)
    
    def get_clus_emb(self, emb_dict, clus_label, mapping_dict):
        """
        TSVAD
        Get an average embedding vector for each cluster (speaker).
        """
        label_dict = { key: [] for key in emb_dict.keys() }
        emb_sess_dict = { key: [] for key in emb_dict.keys() }
        for line in clus_label:
            uniq_id = line.split()[0]
            label = int(line.split()[-1].split('_')[-1])
            label_dict[uniq_id].append(label)
        dim = emb_dict[uniq_id][0].shape[0]
        for uniq_id, emb_list in emb_dict.items():
            spk_set = sorted(list(set(label_dict[uniq_id])))
            label_array = torch.Tensor(label_dict[uniq_id])
            emb_array = torch.Tensor(emb_list)
            avg_embs = torch.zeros(dim, self.max_num_of_spks)
            for spk_idx in spk_set:
                selected_embs = emb_array[label_array == spk_idx]
                avg_embs[:, spk_idx] = torch.mean(selected_embs, dim=0)
            inv_map = {v: k for k, v in mapping_dict[uniq_id].items()}
            emb_sess_dict[uniq_id] = {'mapping': inv_map, 'avg_embs': avg_embs}
        return emb_sess_dict

    def run_clustering_diarizer(self, manifest_filepath, out_dir):
        """
        TSVAD
        Run clustering diarizer to get initial clustering results.
        """
        if os.path.exists(f'{out_dir}/{self.clus_emb_path}'):
            emb_sess_dict = self.load_dict_from_pkl(out_dir) 
        else:
            self._cfg.diarizer.manifest_filepath = manifest_filepath
            self._cfg.diarizer.out_dir = out_dir
            sd_model = ClusteringDiarizer(cfg=self._cfg)
            score = sd_model.diarize()
            metric, mapping_dict = score 
            emb_sess_dict = self.load_embeddings_from_pickle(out_dir, mapping_dict)
        return emb_sess_dict
    
    def load_dict_from_pkl(self, out_dir): 
        with open(f'{out_dir}/{self.clus_emb_path}', 'rb') as handle:
            emb_sess_dict = pkl.load(handle)
        return emb_sess_dict

    def save_dict_as_pkl(self, out_dir, emb_sess_dict):
        with open(f'{out_dir}/{self.clus_emb_path}', 'wb') as handle:
            pkl.dump(emb_sess_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)
    
    def load_embeddings_from_pickle(self, out_dir, mapping_dict):
        """
        TSVAD
        Load embeddings from diarization result folder.
        """
        scale_index = 0
        pickle_path = os.path.join(out_dir, 'speaker_outputs', 'embeddings', f'subsegments_scale{scale_index}_embeddings.pkl')
        clus_label_path = os.path.join(out_dir, 'speaker_outputs',f'subsegments_cluster.label')
        with open(pickle_path, "rb") as input_file:
            emb_dict = pkl.load(input_file)
        with open(clus_label_path) as f:
            clus_label = f.readlines()
        emb_sess_dict = self.get_clus_emb(emb_dict, clus_label, mapping_dict)
        self.save_dict_as_pkl(out_dir, emb_sess_dict)
        return emb_sess_dict

class EncDecDiarLabelModel(ModelPT, ExportableEncDecModel):
    """Encoder decoder class for speaker label models.
    Model class creates training, validation methods for setting up data
    performing model forward pass.
    Expects config dict for
    * preprocessor
    * Jasper/Quartznet Encoder
    * Speaker Decoder
    """

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        result = []

        model = PretrainedModelInfo(
            pretrained_model_name="speakerverification_speakernet",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/speakerverification_speakernet/versions/1.0.0rc1/files/speakerverification_speakernet.nemo",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:speakerverification_speakernet",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="ecapa_tdnn",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/ecapa_tdnn/versions/v1/files/ecapa_tdnn.nemo",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:ecapa_tdnn",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="titanet_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/titanet_large/versions/v0/files/titanet-l.nemo",
            description="For details about this model, please visit https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_large",
        )
        result.append(model)

        return result

    def __init__(self, cfg: DictConfig, emb_clus: Dict, trainer: Trainer = None):
        self.get_emb_clus(emb_clus)
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_gpus

        super().__init__(cfg=cfg, trainer=trainer)
        self.preprocessor = EncDecDiarLabelModel.from_config_dict(cfg.preprocessor)
        # self.encoder = EncDecDiarLabelModel.from_config_dict(cfg.encoder)
        self.tsvad = EncDecDiarLabelModel.from_config_dict(cfg.tsvad_module)
        # if 'angular' in cfg.decoder and cfg.decoder['angular']:
            # logging.info("Training with Angular Softmax Loss")
            # scale = cfg.loss.scale
            # margin = cfg.loss.margin
            # self.loss = AngularSoftmaxLoss(scale=scale, margin=margin)
        # else:
            # logging.info("Training with Softmax-CrossEntropy loss")
        self.loss = self.multispeaker_loss()
        self.task = None
        self._accuracy = MultiBinaryAcc()
        self.labels = None
        self.max_num_of_spks = 8

    def multispeaker_loss(self):
        """
        TSVAD
        Loss function for multispeaker loss
        """
        return torch.nn.BCELoss(reduction='sum')
    
    def get_emb_clus(self, emb_clus):
        self.emb_sess_train_dict = emb_clus.emb_sess_train_dict
        self.emb_sess_dev_dict = emb_clus.emb_sess_dev_dict
        self.emb_sess_test_dict = emb_clus.emb_sess_test_dict

    @staticmethod
    def extract_labels(data_layer_config):
        labels = set()
        manifest_filepath = data_layer_config.get('manifest_filepath', None)
        if manifest_filepath is None:
            logging.warning("No manifest_filepath was provided, no labels got extracted!")
            return None
        manifest_filepaths = convert_to_config_list(data_layer_config['manifest_filepath'])

        for manifest_filepath in itertools.chain.from_iterable(manifest_filepaths):
            collection = ASRSpeechLabel(
                manifests_files=manifest_filepath,
                min_duration=data_layer_config.get("min_duration", None),
                max_duration=data_layer_config.get("max_duration", None),
                index_by_file_id=True,  # Must set this so the manifest lines can be indexed by file ID
            )
            labels.update(collection.uniq_labels)
        labels = list(labels)
        logging.warning(f"Total number of {len(labels)} found in all the manifest files.")
        return labels

    def __setup_dataloader_from_config(self, config: Optional[Dict], emb_dict: Dict):
        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        featurizer = WaveformFeaturizer(
            sample_rate=config['sample_rate'], int_values=config.get('int_values', False), augmentor=augmentor
        )
        shuffle = config.get('shuffle', False)
        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
            return None

        dataset = AudioToSpeechTSVADDataset(
            manifest_filepath=config['manifest_filepath'],
            emb_dict=emb_dict,
            featurizer=featurizer,
        )
        s0 = dataset.item_sim(0)
        s1 = dataset.item_sim(1)



        collate_ds = dataset
        collate_fn = collate_ds.tsvad_collate_fn
        # collate_fn = __tsvad_collate_fn
        packed_batch = list(zip(s0, s1))
        batch_size = config['batch_size']
        # _dataloader = torch.utils.data.DataLoader(
            # dataset=dataset,
            # batch_size=batch_size,
            # collate_fn=collate_fn,
            # drop_last=config.get('drop_last', False),
            # shuffle=shuffle,
            # num_workers=config.get('num_workers', 0),
            # pin_memory=config.get('pin_memory', False),
        # )
        # ff, ffl, tt, iiv = next(iter(_dataloader))
        # import ipdb; ipdb.set_trace()
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        # self.labels = self.extract_labels(train_data_config)
        # train_data_config['labels'] = self.labels
        # if 'shuffle' not in train_data_config:
            # train_data_config['shuffle'] = True
        self._train_dl = self.__setup_dataloader_from_config(config=train_data_config, emb_dict=self.emb_sess_train_dict)

    def setup_validation_data(self, val_data_layer_config: Optional[Union[DictConfig, Dict]]):
        # val_data_layer_config['labels'] = self.labels
        # self.task = 'identification'
        self._validation_dl = self.__setup_dataloader_from_config(config=val_data_layer_config, emb_dict=self.emb_sess_dev_dict)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        # if hasattr(self, 'dataset'):
            # test_data_config['labels'] = self.labels
        self._test_dl = self.__setup_dataloader_from_config(config=test_data_config, emb_dict=self.emb_sess_test_dict)

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
            "input_signal": NeuralType(('B', 'T'), audio_eltype),
            "input_signal_length": NeuralType(tuple('B'), LengthsType()),
            "ivectors": NeuralType(('B', 'D', 'C'), EncodedRepresentation()),
        }

    @property
    def output_types(self):
        return OrderedDict({"probs": NeuralType(('B', 'T', 'C'), LogprobsType())})

    @typecheck()
    def forward(self, input_signal, input_signal_length, ivectors):
        processed_signal, processed_signal_len = self.preprocessor(
            input_signal=input_signal, length=input_signal_length,
        )
        preds = self.tsvad(feats=processed_signal, signal_lengths=processed_signal_len, ivectors=ivectors)
        return preds

    # PTL-specific methods
    def training_step(self, batch, batch_idx):
        signals, signal_lengths, targets, ivectors = batch
        preds = self.forward(input_signal=signals, 
                             input_signal_length=signal_lengths, 
                             ivectors=ivectors)
        loss = self.loss(preds, targets)

        self.log('loss', loss)
        self.log('learning_rate', self._optimizer.param_groups[0]['lr'])

        self._accuracy(preds, targets)
        acc = self._accuracy.compute()
        self._accuracy.reset()
        self.log(f'training_batch_accuracy', acc)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        signals, signal_lengths, targets, ivectors = batch
        preds = self.forward(input_signal=signals, 
                             input_signal_length=signal_lengths, 
                             ivectors=ivectors)
        loss_value = self.loss(preds, targets)
        self._accuracy(preds, targets)
        acc = self._accuracy.compute()
        correct_counts, total_counts = self._accuracy.correct_counts_k, self._accuracy.total_counts_k

        return {
            'val_loss': loss_value,
            'val_correct_counts': correct_counts,
            'val_total_counts': total_counts,
            'val_acc': acc,
        }

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        correct_counts = torch.stack([x['val_correct_counts'] for x in outputs]).sum(axis=0)
        total_counts = torch.stack([x['val_total_counts'] for x in outputs]).sum(axis=0)

        self._accuracy.correct_counts_k = correct_counts
        self._accuracy.total_counts_k = total_counts
        topk_scores = self._accuracy.compute()
        self._accuracy.reset()

        logging.info("val_loss: {:.3f}".format(val_loss_mean))
        self.log('val_loss', val_loss_mean)
        self.log('training_batch_accuracy', acc)

        return {
            'val_loss': val_loss_mean,
            'val_acc': topk_scores,
        }

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        signals, signal_lengths, targets, ivectors = batch
        preds = self.forward(input_signal=signals, 
                             input_signal_length=signal_lengths, 
                             ivectors=ivectors)
        loss_value = self.loss(preds, targets)
        self._accuracy(preds, targets)
        acc = self._accuracy.compute()
        correct_counts, total_counts = self._accuracy.correct_counts_k, self._accuracy.total_counts_k

        return {
            'test_loss': loss_value,
            'test_correct_counts': correct_counts,
            'test_total_counts': total_counts,
            'test_acc_top_k': acc,
        }

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        correct_counts = torch.stack([x['test_correct_counts'] for x in outputs]).sum(axis=0)
        total_counts = torch.stack([x['test_total_counts'] for x in outputs]).sum(axis=0)

        self._accuracy.correct_counts_k = correct_counts
        self._accuracy.total_counts_k = total_counts
        topk_scores = self._accuracy.compute()
        self._accuracy.reset()

        logging.info("test_loss: {:.3f}".format(test_loss_mean))
        self.log('test_loss', test_loss_mean)

        return {
            'test_loss': test_loss_mean,
            'test_acc_top_k': topk_scores,
        }

    # def setup_finetune_model(self, model_config: DictConfig):
        # """
        # setup_finetune_model method sets up training data, validation data and test data with new
        # provided config, this checks for the previous labels set up during training from scratch, if None,
        # it sets up labels for provided finetune data from manifest files
        # Args:
        # model_config: cfg which has train_ds, optional validation_ds, optional test_ds and
        # mandatory encoder and decoder model params
        # make sure you set num_classes correctly for finetune data
        # Returns: None
        # """
        # logging.info("Setting up data loaders with manifests provided from model_config")

        # if 'train_ds' in model_config and model_config.train_ds is not None:
            # self.setup_training_data(model_config.train_ds)
        # else:
            # raise KeyError("train_ds is not found in model_config but you need it for fine tuning")

        # if self.labels is None or len(self.labels) == 0:
            # raise ValueError(f'New labels must be non-empty list of labels. But I got: {self.labels}')

        # if 'validation_ds' in model_config and model_config.validation_ds is not None:
            # self.setup_multiple_validation_data(model_config.validation_ds)

        # if 'test_ds' in model_config and model_config.test_ds is not None:
            # self.setup_multiple_test_data(model_config.test_ds)

        # if self.labels is not None:  # checking for new finetune dataset labels
            # logging.warning(
                # "Trained dataset labels are same as finetune dataset labels -- continuing change of decoder parameters"
            # )
        # else:
            # logging.warning(
                # "Either you provided a dummy manifest file during training from scratch or you restored from a pretrained nemo file"
            # )

        # decoder_config = model_config.decoder
        # new_decoder_config = copy.deepcopy(decoder_config)
        # if new_decoder_config['num_classes'] != len(self.labels):
            # raise ValueError(
                # "number of classes provided {} is not same as number of different labels in finetuning data: {}".format(
                    # new_decoder_config['num_classes'], len(self.labels)
                # )
            # )

        # del self.decoder
        # self.decoder = EncDecDiarLabelModel.from_config_dict(new_decoder_config)

        # with open_dict(self._cfg.decoder):
            # self._cfg.decoder = new_decoder_config

        # logging.info(f"Changed decoder output to # {self.decoder._num_classes} classes.")

    # @torch.no_grad()
    # def get_embedding(self, path2audio_file):
        # audio, sr = librosa.load(path2audio_file, sr=None)
        # target_sr = self._cfg.train_ds.get('sample_rate', 16000)
        # if sr != target_sr:
            # audio = librosa.core.resample(audio, sr, target_sr)
        # audio_length = audio.shape[0]
        # device = self.device
        # audio_signal, audio_signal_len = (
            # torch.tensor([audio], device=device),
            # torch.tensor([audio_length], device=device),
        # )
        # mode = self.training
        # self.freeze()

        # _, embs = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)

        # self.train(mode=mode)
        # if mode is True:
            # self.unfreeze()
        # del audio_signal, audio_signal_len
        # return embs


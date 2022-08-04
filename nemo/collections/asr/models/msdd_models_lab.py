# cOPYRight (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import logging as opt_logging
import math
import os
import pickle as pkl
import shutil
import subprocess
import time
from collections import OrderedDict
from statistics import mode
from typing import Dict, List, Optional, Union

import librosa
import numpy as np
import optuna
import torch
from omegaconf import DictConfig
from omegaconf.omegaconf import open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_only

# from pytorch_lightning.utilities.rank_zero import _get_rank
from torch.utils.data import ChainDataset
from torchmetrics import Metric
from tqdm import tqdm

from nemo.collections.asr.data.audio_to_diar_label import AudioToSpeechMSDDInferDataset, AudioToSpeechMSDDTrainDataset
from nemo.collections.asr.data.audio_to_text_dataset import convert_to_config_list
from nemo.collections.asr.losses.angularloss import AngularSoftmaxLoss
from nemo.collections.asr.losses.bce_loss import BCELoss
from nemo.collections.asr.metrics.multi_binary_acc import MultiBinaryAccuracy
from nemo.collections.asr.models import ClusteringDiarizer
from nemo.collections.asr.models.asr_model import ExportableEncDecModel
from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
from nemo.collections.asr.models.msdd_models import (
    ClusterEmbedding,
    EncDecDiarLabelModel,
    OverlapAwareDiarizer,
    compute_accuracies,
    generate_speaker_timestamps,
    get_id_tup_dict,
    get_overlap_stamps,
    get_uniq_id_from_manifest_line,
    get_uniq_id_list_from_manifest,
    getScaleMappingArgmat,
    make_rttm_with_overlap,
)
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.asr.parts.utils.nmesc_clustering import get_argmin_mat, getCosAffinityMatrix
from nemo.collections.asr.parts.utils.speaker_utils import (
    audio_rttm_map,
    embedding_normalize,
    get_contiguous_stamps,
    get_embs_and_timestamps,
    get_uniq_id_with_dur,
    get_uniqname_from_filepath,
    labels_to_pyannote_object,
    labels_to_rttmfile,
    merge_stamps,
    parse_scale_configs,
    perform_clustering,
    rttm_to_labels,
    score_labels,
    segments_manifest_to_subsegments_manifest,
    write_rttm2manifest,
)
from nemo.collections.common.losses import CrossEntropyLoss as CELoss
from nemo.collections.common.metrics import TopKClassificationAccuracy
from nemo.collections.common.parts.preprocessing.collections import ASRSpeechLabel
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import *
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    EncodedRepresentation,
    LengthsType,
    LogitsType,
    NeuralType,
    ProbsType,
    SpectrogramType,
)
from nemo.core.neural_types.elements import ProbsType
from nemo.utils import logging, model_utils

__all__ = ['EncDecDiarLabelModelLab', 'ClusterEmbeddingTest']

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


def get_audio_rttm_map(manifest):
    """
    This function creates AUDIO_RTTM_MAP which is used by all diarization components to extract embeddings,
    cluster and unify time stamps
    input: manifest file that contains keys audio_filepath, rttm_filepath if exists, text, num_speakers if known and uem_filepath if exists

    returns:
        AUDIO_RTTM_MAP (dict) : A dictionary with keys of uniq id, which is being used to map audio files and corresponding rttm files
    """

    AUDIO_RTTM_MAP = {}
    with open(manifest, 'r') as inp_file:
        lines = inp_file.readlines()
        logging.info("Number of files to diarize: {}".format(len(lines)))
        for line in lines:
            line = line.strip()
            dic = json.loads(line)

            meta = {
                'audio_filepath': dic['audio_filepath'],
                'rttm_filepath': dic.get('rttm_filepath', None),
                'offset': dic.get('offset', None),
                'duration': dic.get('duration', None),
                'text': dic.get('text', None),
                'num_speakers': dic.get('num_speakers', None),
                'uem_filepath': dic.get('uem_filepath', None),
                'ctm_filepath': dic.get('ctm_filepath', None),
            }

            uniqname = get_uniq_id_with_dur(meta)
            if uniqname not in AUDIO_RTTM_MAP:
                AUDIO_RTTM_MAP[uniqname] = meta
            else:
                raise KeyError(f"Unique name:{uniqname} already exists in the AUDIO_RTTM_MAP dictionary.")

    return AUDIO_RTTM_MAP


class ClusterEmbeddingTest(ClusterEmbedding):
    def __init__(self, cfg_base: DictConfig, cfg_msdd_model: DictConfig):
        super().__init__(cfg_base, cfg_msdd_model)

    # def prepare_cluster_embs_infer(self):
    # self.emb_sess_test_dict, self.emb_seq_test, self.clus_test_label_dict, metric= self.prepare_datasets(
    # self._cfg_msdd.test_ds
    # )
    def sw(self, r, K):
        return [r - kvar * (r - 1) / (K - 1) for kvar in range(K)]

    def prepare_cluster_embs_infer_clus_opt(self, trial):
        self.set_optuna_trial_clus(trial)
        self.emb_sess_test_dict, self.emb_seq_test, self.clus_test_label_dict, metric = self.prepare_datasets(
            self._cfg_msdd.test_ds
        )
        DER = round(1 - (metric.accumulated_['correct'] / metric.accumulated_['total']), 6)
        return DER

    def set_optuna_trial_clus(self, trial):
        r_value = round(trial.suggest_float("r_value", 0.0, 2.25, step=0.01), 4)
        max_rp_threshold = round(trial.suggest_float("max_rp_threshold", 0.05, 0.35, step=0.01), 2)
        sparse_search_volume = trial.suggest_int("sparse_search_volume", 2, 100, log=True)
        max_num_speakers = trial.suggest_int("max_num_speakers", 8, 27, step=1)
        scale_n = 5
        self.cfg_base.diarizer.clustering.parameters.max_num_speakers = max_num_speakers
        self.cfg_base.diarizer.clustering.parameters.sparse_search_volume = sparse_search_volume
        self.cfg_base.diarizer.clustering.parameters.max_rp_threshold = max_rp_threshold
        msw = self.sw(r_value, scale_n)
        self.cfg_base.diarizer.speaker_embeddings.parameters.multiscale_weights = msw

        print(
            f"=========== Sampler picked max_rp_threshold: {max_rp_threshold}, sparse_search_volume: {sparse_search_volume}, max_num_speakers: {max_num_speakers} r_value: {r_value}, multi-scale weights: {msw}"
        )
        return None

    def set_optuna_trial(self, trial):
        stt = 1.0
        end = 0.5
        scale_n = trial.suggest_int("scale_n", 2, 11)
        # stt, end = 1.5, 0.5
        _step = -1 * (stt - end) / (scale_n - 1)
        # shift_ratio = trial.suggest_float("shift_ratio", 0.1, 0.5, step=0.1)
        shift_ratio = 0.5
        window_mat = np.arange(stt, end - 0.01, _step).round(5)
        self.cfg_base.diarizer.speaker_embeddings.parameters.window_length_in_sec = window_mat.tolist()
        self.cfg_base.diarizer.speaker_embeddings.parameters.shift_length_in_sec = (
            shift_ratio * window_mat / 2
        ).tolist()
        self.cfg_base.diarizer.speaker_embeddings.parameters.multiscale_weights = [1] * scale_n
        # import ipdb; ipdb.set_trace()
        print(
            f"=========== Sampler picked stt: {stt} end: {end}, shift_ratio: {shift_ratio} step:{_step}, scale_n:{scale_n}"
        )
        return None

    def prepare_cluster_embs_infer_scale_opt(self, trial):
        """
        Launch clustering diarizer to prepare embedding vectors and clustering results.
        """
        self.set_optuna_trial(trial)
        try:
            metric = self._run_clustering_diarizer_opt(
                self._cfg_msdd.test_ds.manifest_filepath, self._cfg_msdd.test_ds.emb_dir
            )
            del self.clusdiar_model
            torch.cuda.empty_cache()
            DER = round(1 - (metric.accumulated_['correct'] / metric.accumulated_['total']), 6)
        except:
            DER = 1.0
        torch.cuda.empty_cache()
        # import ipdb; ipdb.set_trace()
        return DER

    def prepare_cluster_embs_infer(self):
        """
        Launch clustering diarizer to prepare embedding vectors and clustering results.
        """
        self.emb_sess_test_dict, self.emb_seq_test, self.clus_test_label_dict, metrics = self.run_clustering_diarizer(
            self._cfg_msdd.test_ds.manifest_filepath, self._cfg_msdd.test_ds.emb_dir
        )

    def get_longest_scale_clus_avg_emb(self, emb_vectors, longest_scale_index=0):
        """
        Use the cluster-average embedding vector from the longest scale. Using the cluster-average from the longest scale
        can improve the performance.

        Args:
            emb_vectors (Tensor):
                Tensor containing the cluster-average embeddings of each scale.
                Dimension: batch size x number of scales x embedding dimension x number of speakers
            longest_scale_index (int):
                The index of the longest scale. Default is `longest_scale_index=0`.

        Returns:
            repeated_emb_vectors (Tensor):
                Tensor containing the repeated cluster-average embeddings from the longest scale (scale index = 0)
                Dimension: batch size x number of scales x embedding dimension x number of speakers
        """
        batch_size, scale_n, emb_dim, n_spks = emb_vectors.shape
        repeated_emb_vectors = torch.repeat_interleave(
            emb_vectors[:, longest_scale_index, :, :], scale_n, dim=0
        ).reshape(batch_size, scale_n, emb_dim, n_spks)
        return repeated_emb_vectors

    def get_manifest_uniq_ids(self, manifest_filepath):
        manifest_lines = []
        with open(manifest_filepath) as f:
            manifest_lines = f.readlines()
            for jsonObj in f:
                student_dict = json.loads(jsonObj)
                manifest_lines.append(student_dict)
        uniq_id_list, json_dict_list = [], []
        for json_string in manifest_lines:
            json_dict = json.loads(json_string)
            json_dict_list.append(json_dict)
            uniq_id = get_uniqname_from_filepath(json_dict['audio_filepath'])
            uniq_id_list.append(uniq_id)
        return uniq_id_list, json_dict_list

    def get_uniq_id(self, rttm_path):
        return rttm_path.split('/')[-1].split('.rttm')[0]

    def read_rttm_file(self, rttm_path):
        return open(rttm_path).readlines()

    def s2n(self, x, ROUND=2):
        return round(float(x), ROUND)

    def parse_rttm(self, rttm_path):
        rttm_lines = self.read_rttm_file(rttm_path)
        speaker_list = []
        for line in rttm_lines:
            rttm = line.strip().split()
            start, end, speaker = self.s2n(rttm[3]), self.s2n(rttm[4]) + self.s2n(rttm[3]), rttm[7]
            speaker = rttm[7]
            speaker_list.append(speaker)
        return set(speaker_list)

    def check_embedding_and_RTTM(self, manifest_filepath, multiscale_data):
        emb_sess_avg_dict, emb_scale_seq_dict, base_clus_label_dict = multiscale_data
        uniq_id_list, json_lines_list = self.get_manifest_uniq_ids(manifest_filepath)
        for scale_index in emb_sess_avg_dict.keys():
            for uniq_id, json_dict in zip(uniq_id_list, json_lines_list):
                rttm_filepath = json_dict['rttm_filepath']
                rttm_speaker_set = self.parse_rttm(rttm_filepath)
                dict_speaker_set = set(list(emb_sess_avg_dict[scale_index][uniq_id]['mapping'].keys()))
                dict_speaker_value_set = set(list(emb_sess_avg_dict[scale_index][uniq_id]['mapping'].values()))
                if rttm_speaker_set != dict_speaker_set:
                    remainder_rttm_keys = rttm_speaker_set - dict_speaker_set
                    total_spk_set = set(['speaker_' + str(x) for x in range(len(rttm_speaker_set))])
                    remainder_dict_keys = total_spk_set - dict_speaker_value_set
                    for rttm_key, dict_key in zip(remainder_rttm_keys, remainder_dict_keys):
                        emb_sess_avg_dict[scale_index][uniq_id]['mapping'][rttm_key] = dict_key
                    dict_speaker_set = set(list(emb_sess_avg_dict[scale_index][uniq_id]['mapping'].keys()))
                    assert rttm_speaker_set == dict_speaker_set

    def check_loaded_clus_data(self, loaded_clus_data, manifest_filepath):
        """
        Check if all the unique IDs in the manifest file has corresponding clustering outputs and embedding sequences.

        Args:

        Returns:
        """
        uniq_id_list, _ = self.get_manifest_uniq_ids(manifest_filepath)
        condA = set(uniq_id_list).issubset(set(loaded_clus_data[0][self.base_scale_index].keys()))
        condB = set(uniq_id_list).issubset(set(loaded_clus_data[1].keys()))
        condC = set(uniq_id_list).issubset(set(loaded_clus_data[2].keys()))
        return all([condA, condB, condC])

    def check_prepared_dataset(self, manifest_filepath, emb_dir):
        if os.path.exists(f'{emb_dir}/speaker_outputs/embeddings'):
            logging.info(f"Embedding path already exists: {emb_dir}/speaker_outputs/embeddings")
            file_exist_checks = [
                os.path.exists(f'{emb_dir}/{self.cluster_avg_emb_path}'),
                os.path.exists(f'{emb_dir}/{self.speaker_mapping_path}'),
                os.path.exists(f'{emb_dir}/{self.scale_map_path}'),
            ]
            if all(file_exist_checks):
                loaded_clus_data = self.load_clustering_info_dictionaries(emb_dir)
                isClusteringInfoReady = self.check_loaded_clus_data(loaded_clus_data, manifest_filepath)
                if not isClusteringInfoReady:
                    loaded_clus_data = None
            else:
                isClusteringInfoReady, loaded_clus_data = False, None

            # clus_label_path = os.path.join(emb_dir, 'speaker_outputs', f'subsegments_scale{self.base_scale_index}_cluster.label')
            # scales_subsegment_exist_list = [os.path.exists(clus_label_path)]
            scales_subsegment_exist_list = []
            for scale_index in range(self.scale_n):
                pickle_path = os.path.join(
                    emb_dir, 'speaker_outputs', 'embeddings', f'subsegments_scale{scale_index}_embeddings.pkl'
                )
                scales_subsegment_exist_list.append(os.path.exists(pickle_path))
            if all(scales_subsegment_exist_list):
                isScaleEmbReady = True
            else:
                isScaleEmbReady = False
        else:
            isClusteringInfoReady, isScaleEmbReady, loaded_clus_data = False, False, []
        return isScaleEmbReady, isClusteringInfoReady, loaded_clus_data

    def load_existing_data(self, manifest_filepath, emb_dir, loaded_clus_data):
        """
        Load the existing scale-wise speaker embedding sequence, cluster-average embeddings, scale mapping and base
        scale clustering labels.

        Args:
            manifest_filepath (str):
                Path of the input manifest file
            embed_dir (str):
                Path of the folder where the extracted data will be stored
            loaded_clus_data (list):
                List containing emb_sess_avg_dict, emb_scale_seq_dict and base_clus_label_dict
        Returns:
            loaded_clus_data (list):
                List containing emb_sess_avg_dict, emb_scale_seq_dict and base_clus_label_dict
        """
        emb_sess_avg_dict, session_scale_mapping_dict, speaker_mapping_dict = loaded_clus_data
        cluster_label_file_exists, _ = self.check_clustering_labels(emb_dir)
        metric = None
        if self.cfg_base.diarizer.msdd_model.parameters.run_clus_from_loaded_emb or not cluster_label_file_exists:
            (
                emb_sess_avg_dict,
                emb_scale_seq_dict,
                base_clus_label_dict,
                metric,
            ) = self.run_clustering_from_existing_data(manifest_filepath, emb_dir)
        else:
            emb_scale_seq_dict = self.load_emb_scale_seq_dict(emb_dir)
            clus_labels = self.load_clustering_labels(emb_dir)
            if not self.clusdiar_model:
                self.clusdiar_model = ClusteringDiarizer(cfg=self.cfg_base)
            self.load_embs_and_timestamps(emb_dir, emb_scale_seq_dict)
            base_clus_label_dict, emb_dim = self.get_base_clus_label_dict(clus_labels, emb_scale_seq_dict)
            emb_sess_avg_dict, base_clus_label_dict = self.get_cluster_avg_embs(
                emb_scale_seq_dict, clus_labels, speaker_mapping_dict, session_scale_mapping_dict
            )
            emb_scale_seq_dict['session_scale_mapping'] = session_scale_mapping_dict
        if metric is None:
            # import ipdb; ipdb.set_trace()
            raise ValueError("metric is None")
        return emb_sess_avg_dict, emb_scale_seq_dict, base_clus_label_dict, metric

    def run_clustering_from_existing_data(self, manifest_filepath, emb_dir):
        """
        This function is executed when there are extracted speaker embeddings and clustering labels.

        Args:
            manifest_filepath (str):
                Path of the input manifest file
            embed_dir (str):
                Path of the folder where the extracted data will be stored
        Returns:
            return emb_sess_avg_dict (dict):

            emb_scale_seq_dict (dict):

            base_clus_label_dict (dict):
        """
        emb_scale_seq_dict = self.load_emb_scale_seq_dict(emb_dir)
        metric, speaker_mapping_dict = self.run_multiscale_clustering(
            manifest_filepath, emb_dir, emb_scale_seq_dict, run_clustering=True
        )
        # self.multiscale_args_dict['use_single_scale_clustering'] = False
        self.multiscale_args_dict = parse_scale_configs(
            self._cfg_msdd.base.diarizer.speaker_embeddings.parameters.window_length_in_sec,
            self._cfg_msdd.base.diarizer.speaker_embeddings.parameters.shift_length_in_sec,
            self._cfg_msdd.base.diarizer.speaker_embeddings.parameters.multiscale_weights,
        )

        # _embs_and_timestamps should be stored again in case of use_single_scale_clus=True.
        # All scales are required to prepare data even in case of single scale clustering.
        self._embs_and_timestamps = get_embs_and_timestamps(
            self.clusdiar_model.multiscale_embeddings_and_timestamps, self.multiscale_args_dict
        )
        session_scale_mapping_dict = self.get_scale_map(self._embs_and_timestamps)
        clus_labels = self.load_clustering_labels(emb_dir)
        emb_sess_avg_dict, base_clus_label_dict = self.get_cluster_avg_embs(
            emb_scale_seq_dict, clus_labels, speaker_mapping_dict, session_scale_mapping_dict
        )
        self.save_dict_as_pkl(emb_dir, [emb_sess_avg_dict, speaker_mapping_dict, session_scale_mapping_dict])
        emb_scale_seq_dict['session_scale_mapping'] = session_scale_mapping_dict
        return emb_sess_avg_dict, emb_scale_seq_dict, base_clus_label_dict, metric

    def run_clustering_diarizer(self, manifest_filepath, emb_dir):
        """
        If no pre-existing data is provided, run clustering diarizer from scratch. This will create scale-wise speaker embedding
        sequence, cluster-average embeddings, scale mapping and base scale clustering labels.

        Returns:
            emb_sess_avg_dict (dict):
                Dictionary containing cluster-average embeddings for each session.
            emb_scale_seq_dict (dict):
                Dictionary containing embedding tensors which are indexed by scale numbers.
            base_clus_label_dict (dict):
                Dictionary containing clustering results. Clustering results are cluster labels for the base scale segments.
        """
        self.cfg_base.diarizer.manifest_filepath = manifest_filepath
        self.cfg_base.diarizer.out_dir = emb_dir

        # Run ClusteringDiarizer which includes system VAD or oracle VAD.
        self.clusdiar_model = ClusteringDiarizer(cfg=self.cfg_base)
        self.clusdiar_model._cluster_params = self.cfg_base.diarizer.clustering.parameters
        self.clusdiar_model.multiscale_args_dict[
            "multiscale_weights"
        ] = self.cfg_base.diarizer.speaker_embeddings.parameters.multiscale_weights
        self.clusdiar_model._diarizer_params.speaker_embeddings.parameters = (
            self.cfg_base.diarizer.speaker_embeddings.parameters
        )
        clustering_params_str = json.dumps(dict(self.clusdiar_model._cluster_params), indent=4)

        # Since only one scale is used, base_scale_index should be 0.
        if self.cfg_base.diarizer.msdd_model.parameters.use_single_scale_clustering:
            self.clusdiar_model.multiscale_args_dict['use_single_scale_clustering'] = True
            self.base_scale_index = 0

        logging.info(f"Multiscale Weights: {self.clusdiar_model.multiscale_args_dict['multiscale_weights']}")
        logging.info(f"Clustering Parameters: {clustering_params_str}")
        metric, speaker_mapping_dict = self.clusdiar_model.diarize(batch_size=self.cfg_base.batch_size)

        # Get the mapping between segments in different scales.
        self.clusdiar_model.multiscale_args_dict['use_single_scale_clustering'] = False
        self._embs_and_timestamps = get_embs_and_timestamps(
            self.clusdiar_model.multiscale_embeddings_and_timestamps, self.clusdiar_model.multiscale_args_dict
        )
        session_scale_mapping_dict = self.get_scale_map(self._embs_and_timestamps)
        emb_scale_seq_dict = self.load_emb_scale_seq_dict(emb_dir)
        clus_labels = self.load_clustering_labels(emb_dir)
        emb_sess_avg_dict, base_clus_label_dict = self.get_cluster_avg_embs(
            emb_scale_seq_dict, clus_labels, speaker_mapping_dict, session_scale_mapping_dict
        )
        emb_scale_seq_dict['session_scale_mapping'] = session_scale_mapping_dict
        return emb_sess_avg_dict, emb_scale_seq_dict, base_clus_label_dict, metric

    def _run_clustering_diarizer_opt(self, manifest_filepath, emb_dir):
        """
        RUN CLUSTER DIAR for Optuna:

        If no pre-existing data is provided, run clustering diarizer from scratch. This will create scale-wise speaker embedding
        sequence, cluster-average embeddings, scale mapping and base scale clustering labels.

        Returns:
            emb_sess_avg_dict (dict):
                Dictionary containing cluster-average embeddings for each session.
            emb_scale_seq_dict (dict):
                Dictionary containing embedding tensors which are indexed by scale numbers.
            base_clus_label_dict (dict):
                Dictionary containing clustering results. Clustering results are cluster labels for the base scale segments.
        """
        self.cfg_base.diarizer.manifest_filepath = manifest_filepath
        self.cfg_base.diarizer.out_dir = emb_dir

        # Run ClusteringDiarizer which includes system VAD or oracle VAD.
        self.clusdiar_model = ClusteringDiarizer(cfg=self.cfg_base)
        self.clusdiar_model._cluster_params = self.cfg_base.diarizer.clustering.parameters
        self.clusdiar_model.multiscale_args_dict[
            "multiscale_weights"
        ] = self.cfg_base.diarizer.speaker_embeddings.parameters.multiscale_weights
        self.clusdiar_model._diarizer_params.speaker_embeddings.parameters = (
            self.cfg_base.diarizer.speaker_embeddings.parameters
        )
        clustering_params_str = json.dumps(dict(self.clusdiar_model._cluster_params), indent=4)

        # Since only one scale is used, base_scale_index should be 0.
        if self.cfg_base.diarizer.msdd_model.parameters.use_single_scale_clustering:
            self.clusdiar_model.multiscale_args_dict['use_single_scale_clustering'] = True
            self.base_scale_index = 0

        logging.info(f"Multiscale Weights: {self.clusdiar_model.multiscale_args_dict['multiscale_weights']}")
        logging.info(f"Clustering Parameters: {clustering_params_str}")
        metric, speaker_mapping_dict = self.clusdiar_model.diarize(batch_size=self.cfg_base.batch_size)

        # Get the mapping between segments in different scales.
        # self.clusdiar_model.multiscale_args_dict['use_single_scale_clustering'] = False
        # self._embs_and_timestamps = get_embs_and_timestamps(
        # self.clusdiar_model.multiscale_embeddings_and_timestamps, self.clusdiar_model.multiscale_args_dict
        # )
        # session_scale_mapping_dict = self.get_scale_map(self._embs_and_timestamps)
        # emb_scale_seq_dict = self.load_emb_scale_seq_dict(emb_dir)
        # clus_labels = self.load_clustering_labels(emb_dir)
        # emb_sess_avg_dict, base_clus_label_dict = self.get_cluster_avg_embs(
        # emb_scale_seq_dict, clus_labels, speaker_mapping_dict, session_scale_mapping_dict
        # )
        # emb_scale_seq_dict['session_scale_mapping'] = session_scale_mapping_dict
        # return emb_sess_avg_dict, emb_scale_seq_dict, base_clus_label_dict,
        return metric

    def assign_labels_to_shorter_segs(self, base_clus_label_dict, session_scale_mapping_dict):
        """

        """
        new_clus_label_dict = {scale_index: {} for scale_index in range(self.scale_n)}
        for uniq_id, uniq_scale_mapping_dict in session_scale_mapping_dict.items():
            base_scale_clus_label = np.array([x[-1] for x in base_clus_label_dict[uniq_id]])
            new_clus_label_dict[self.base_scale_index][uniq_id] = base_scale_clus_label
            for scale_index in range(self.scale_n):
                new_clus_label = []
                seq_length = max(uniq_scale_mapping_dict[scale_index])
                for seg_idx in range(seq_length + 1):
                    if seg_idx in uniq_scale_mapping_dict[scale_index]:
                        index_bool = uniq_scale_mapping_dict[scale_index] == seg_idx
                        clus_spk_label = torch.tensor(
                            base_scale_clus_label[uniq_scale_mapping_dict[self.base_scale_index][index_bool]]
                        )
                        seg_clus_label = torch.mode(clus_spk_label)[0].item()
                    else:
                        seg_clus_label = 0 if len(new_clus_label) == 0 else new_clus_label[-1]
                    new_clus_label.append(seg_clus_label)
                new_clus_label_dict[scale_index][uniq_id] = new_clus_label
        return new_clus_label_dict

    def replace_output_clus_label_dict(self, all_scale_clus_label_dict):
        """
        In case of using single sclae clustering, the output label should be replaced: (longest -> shortest).
        After converting longest scale results to shortest scale results, this function updates the clustering result file.

        Args:
            all_scale_clus_label_dict (dict):
                Dictionary containing the estimated speaker labels for all scales.
        Returns:
            base_clus_label_dict (dict):
                Dictionary containing clustering results. Clustering results are cluster labels for the base scale segments.
        """
        # Temporarily turn off `use_single_scale_clustering` variable to get all the scales.
        self.clusdiar_model.multiscale_args_dict['use_single_scale_clustering'] = False
        self._embs_and_timestamps = get_embs_and_timestamps(
            self.clusdiar_model.multiscale_embeddings_and_timestamps, self.clusdiar_model.multiscale_args_dict
        )
        base_clus_label_dict = {}
        for uniq_id, data_dict in self._embs_and_timestamps.items():
            label_scale = max(list(data_dict["scale_dict"].keys()))
            stt_end_lines = data_dict["scale_dict"][label_scale]["time_stamps"]
            spk_label_list = all_scale_clus_label_dict[label_scale][uniq_id]
            base_clus_label_dict[uniq_id] = []
            for k, line in enumerate(stt_end_lines):
                label = spk_label_list[k]
                stt, end = [round(float(x), 2) for x in line.split()[:2]]
                base_clus_label_dict[uniq_id].append([stt, end, label])
        return base_clus_label_dict

    def get_cluster_avg_embs(self, emb_scale_seq_dict, clus_labels, speaker_mapping_dict, session_scale_mapping_dict):
        """
        MSDD requires cluster-average speaker embedding vectors for each scale. This function calculates an average embedding vector for each cluster (speaker)
        and each scale.

        Args:
            emb_scale_seq_dict (dict):
                Dictionary containing embedding sequence for each scale. Keys are scale index in integer.
            clus_labels (list):
                Clustering results from clustering diarizer including all the sessions provided in input manifest files.
            speaker_mapping_dict (dict):
                Speaker mapping dictionary in case RTTM files are provided. This is mapping between integer based speaker index and
                speaker ID tokens in RTTM files.
                Example:
                    {'en_0638': {'speaker_0': 'en_0638_A', 'speaker_1': 'en_0638_B'},
                     'en_4065': {'speaker_0': 'en_4065_B', 'speaker_1': 'en_4065_A'}, ...,}
            session_scale_mapping_dict (dict):

        Returns:
            emb_sess_avg_dict (dict):
                Dictionary containing speaker mapping information and cluster-average speaker embedding vector.
                Each session-level dictionary is indexed by scale index in integer.
            output_clus_label_dict (dict):
                Subegmentation timestamps in float type and Clustering result in integer type. Indexed by `uniq_id` keys.
        """
        self.scale_n = len(emb_scale_seq_dict.keys())
        emb_sess_avg_dict = {
            scale_index: {key: [] for key in emb_scale_seq_dict[self.scale_n - 1].keys()}
            for scale_index in emb_scale_seq_dict.keys()
        }
        output_clus_label_dict, emb_dim = self.get_base_clus_label_dict(clus_labels, emb_scale_seq_dict)
        if self.cfg_base.diarizer.msdd_model.parameters.use_single_scale_clustering:
            # The longest scale has output labels therefore assign the labels to shorter ones.
            all_scale_clus_label_dict = self.assign_labels_to_shorter_segs(
                output_clus_label_dict, session_scale_mapping_dict
            )
            # The output label should be replaced: (longest -> shortest)
            output_clus_label_dict = self.replace_output_clus_label_dict(all_scale_clus_label_dict)
        else:
            all_scale_clus_label_dict = self.assign_labels_to_longer_segs(
                output_clus_label_dict, session_scale_mapping_dict
            )
        for scale_index in emb_scale_seq_dict.keys():
            for uniq_id, _emb_tensor in emb_scale_seq_dict[scale_index].items():
                if type(_emb_tensor) == list:
                    emb_tensor = torch.tensor(np.array(_emb_tensor))
                else:
                    emb_tensor = _emb_tensor
                clus_label_list = all_scale_clus_label_dict[scale_index][uniq_id]
                spk_set = set(clus_label_list)
                # Create a label array which identifies clustering result for each segment.
                num_of_spks = len(spk_set)
                # if num_of_spks > self.max_num_of_spks:
                    # raise ValueError(
                        # f"uniq_id {uniq_id} - self.max_num_of_spks {self.max_num_of_spks} is smaller than the actual number of speakers: {num_of_spks}"
                    # )
                label_array = torch.Tensor(clus_label_list)
                avg_embs = torch.zeros(emb_dim, self.max_num_of_spks)
                for spk_idx in spk_set:
                    selected_embs = emb_tensor[label_array == spk_idx]
                    avg_embs[:, spk_idx] = torch.mean(selected_embs, dim=0)
                inv_map = {clus_key: rttm_key for rttm_key, clus_key in speaker_mapping_dict[uniq_id].items()}
                emb_sess_avg_dict[scale_index][uniq_id] = {'mapping': inv_map, 'avg_embs': avg_embs}
        return emb_sess_avg_dict, output_clus_label_dict

    def prepare_datasets(self, cfg_split_ds):
        """
        Check if the extracted speaker embedding exists and also check if those extracted embeddings are in a right shape.
        If embeddings and cluster average information already exist, load the pickle files. If the embeddings and information
        do not exist, run clustering diarizer to get the initializing cluster results.

        """
        manifest_filepath = cfg_split_ds.manifest_filepath
        emb_dir = cfg_split_ds.emb_dir

        isScaleEmbReady, isClusteringInfoReady, loaded_clus_data = self.check_prepared_dataset(
            manifest_filepath, emb_dir
        )
        # metric = None
        if isClusteringInfoReady:
            multiscale_data_metric = self.load_existing_data(manifest_filepath, emb_dir, loaded_clus_data)
            multiscale_data = multiscale_data_metric[:3]
            metric = multiscale_data_metric[-1]
        elif not isClusteringInfoReady and isScaleEmbReady:
            multiscale_data_metric = self.run_clustering_from_existing_data(manifest_filepath, emb_dir)
            multiscale_data = multiscale_data_metric[:3]
            metric = multiscale_data_metric[-1]
        elif (not isClusteringInfoReady and not isScaleEmbReady) or (isClusteringInfoReady and not isScaleEmbReady):
            multiscale_data_metric = self.run_clustering_diarizer(manifest_filepath, emb_dir)

        # logging.info("Checking clustering results and rttm files. Don't use this for inference. This is for training.")
        # self.check_embedding_and_RTTM(manifest_filepath, loaded_clus_data)
        # import ipdb; ipdb.set_trace()
        logging.info("Clustering results and rttm files test passed.")
        return multiscale_data_metric

    def extract_time_stamps(self, manifest_file):
        """
        Extract timestamps from manifest file and save into a list in a dictionary.

        Args:
            manifest_file (str):

        Returns:
            time_stamps (dict):

        """
        self.time_stamps = {}
        with open(manifest_file, 'r', encoding='utf-8') as manifest:
            for i, line in enumerate(manifest.readlines()):
                line = line.strip()
                dic = json.loads(line)
                uniq_name = get_uniqname_from_filepath(dic['audio_filepath'])
                if uniq_name not in self.time_stamps:
                    self.time_stamps[uniq_name] = []
                start = dic['offset']
                end = start + dic['duration']
                stamp = '{:.3f} {:.3f} '.format(start, end)
                self.time_stamps[uniq_name].append(stamp)
        return self.time_stamps

    def load_embs_and_timestamps(self, emb_dir, emb_scale_seq_dict):
        for scale_idx, (window, shift) in self.clusdiar_model.multiscale_args_dict['scale_dict'].items():
            subsegments_manifest_path = os.path.join(emb_dir, 'speaker_outputs', f'subsegments_scale{scale_idx}.json')
            self.embeddings = emb_scale_seq_dict[scale_idx]
            self.time_stamps = self.extract_time_stamps(subsegments_manifest_path)
            self.clusdiar_model.multiscale_embeddings_and_timestamps[scale_idx] = [self.embeddings, self.time_stamps]

    def run_multiscale_clustering(self, manifest_filepath, emb_dir, emb_scale_seq_dict, run_clustering=False):
        """
        Run multiscale clustering without extracting speaker embedding process. The saved speaker embeddings are used
        for clustering.

        Args:

        Returns:

        """

        if not self.clusdiar_model:
            self.clusdiar_model = ClusteringDiarizer(cfg=self.cfg_base)
        self.clusdiar_model.AUDIO_RTTM_MAP = audio_rttm_map(manifest_filepath)
        self.clusdiar_model._cluster_params = self.cfg_base.diarizer.clustering.parameters
        self.cfg_base.diarizer.out_dir = emb_dir

        self.clusdiar_model._out_dir = emb_dir
        out_rttm_dir = os.path.join(emb_dir, 'pred_rttms')
        self.load_embs_and_timestamps(emb_dir, emb_scale_seq_dict)

        # if self.cfg_base.diarizer.msdd_model.parameters.use_single_scale_clustering:
        # self.clusdiar_model.multiscale_args_dict['use_single_scale_clustering'] = True
        # self.base_scale_index = 0
        if self.cfg_base.diarizer.msdd_model.parameters.use_single_scale_clustering is not False:
            self.clusdiar_model.multiscale_args_dict[
                'use_single_scale_clustering'
            ] = self.cfg_base.diarizer.msdd_model.parameters.use_single_scale_clustering
            self.base_scale_index = self.clusdiar_model.multiscale_args_dict['use_single_scale_clustering'] - 1

        self.clusdiar_model.embs_and_timestamps = get_embs_and_timestamps(
            self.clusdiar_model.multiscale_embeddings_and_timestamps, self.clusdiar_model.multiscale_args_dict
        )

        self.clusdiar_model.multiscale_args_dict[
            "multiscale_weights"
        ] = self.cfg_base.diarizer.speaker_embeddings.parameters.multiscale_weights
        self.clusdiar_model._diarizer_params.speaker_embeddings.parameters = (
            self.cfg_base.diarizer.speaker_embeddings.parameters
        )
        clustering_params_str = json.dumps(dict(self.clusdiar_model._cluster_params), indent=4)
        logging.info(f"Multiscale Weights: {self.clusdiar_model.multiscale_args_dict['multiscale_weights']}")
        logging.info(f"Clustering Parameters: {clustering_params_str}")

        if run_clustering:
            all_reference, all_hypothesis = perform_clustering(
                embs_and_timestamps=self.clusdiar_model.embs_and_timestamps,
                AUDIO_RTTM_MAP=self.clusdiar_model.AUDIO_RTTM_MAP,
                out_rttm_dir=out_rttm_dir,
                clustering_params=self.clusdiar_model._cluster_params,
            )
        else:
            all_hypothesis, all_reference, lines_cluster_labels = [], [], []
            no_references = False
            for uniq_id, value in tqdm(self.clusdiar_model.AUDIO_RTTM_MAP.items()):
                with open(f"{out_rttm_dir}/{uniq_id}.rttm") as f:
                    rttm_lines = f.readlines()
                labels = self.convert_rttm_to_labels(rttm_lines)
                hypothesis = labels_to_pyannote_object(labels, uniq_name=uniq_id)
                all_hypothesis.append([uniq_id, hypothesis])
                rttm_file = value.get('rttm_filepath', None)
                if rttm_file is not None and os.path.exists(rttm_file) and not no_references:
                    ref_labels = rttm_to_labels(rttm_file)
                    reference = labels_to_pyannote_object(ref_labels, uniq_name=uniq_id)
                    all_reference.append([uniq_id, reference])
                else:
                    no_references = True
                    all_reference = []

        logging.info(f"  [Re-clustering results]:")
        # for (collar, ignore_overlap) in [(0.25, True), (0.25, False), (0.0, False)]:
        for (collar, ignore_overlap) in [(0.25, True), (0.0, False), (0.25, False)]:

            self.score = score_labels(
                self.clusdiar_model.AUDIO_RTTM_MAP,
                all_reference,
                all_hypothesis,
                collar=collar,
                ignore_overlap=ignore_overlap,
            )
        return self.score

    def convert_rttm_to_labels(self, rttm_lines):
        labels = []
        for line in rttm_lines:
            ls = line.split()
            start = ls[3]
            end = str(float(ls[3]) + float(ls[4]))
            speaker = ls[7]
            labels.append(f"{start} {end} {speaker}")
        return labels

    def get_scale_map(self, embs_and_timestamps):
        """
        Args:

        Returns:

        """
        session_scale_mapping_dict = {}
        for uniq_id, uniq_embs_and_timestamps in embs_and_timestamps.items():
            scale_mapping_dict = getScaleMappingArgmat(uniq_embs_and_timestamps)
            session_scale_mapping_dict[uniq_id] = scale_mapping_dict
        return session_scale_mapping_dict

    def load_clustering_info_dictionaries(self, out_dir):
        with open(f'{out_dir}/{self.cluster_avg_emb_path}', 'rb') as handle:
            emb_sess_avg_dict = pkl.load(handle)
        with open(f'{out_dir}/{self.scale_map_path}', 'rb') as handle:
            session_scale_mapping_dict = pkl.load(handle)
        with open(f'{out_dir}/{self.speaker_mapping_path}', 'rb') as handle:
            speaker_mapping_dict = pkl.load(handle)
        return emb_sess_avg_dict, session_scale_mapping_dict, speaker_mapping_dict

    def save_dict_as_pkl(self, out_dir, loaded_clus_data):
        logging.info(
            f"Saving clustering results and cluster-average embedding files:\n {out_dir}/{self.cluster_avg_emb_path},\n {out_dir}/{self.speaker_mapping_path},\n {out_dir}/{self.scale_map_path}"
        )
        emb_sess_avg_dict, speaker_mapping_dict, session_scale_mapping_dict = loaded_clus_data
        with open(f'{out_dir}/{self.cluster_avg_emb_path}', 'wb') as handle:
            pkl.dump(emb_sess_avg_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)
        with open(f'{out_dir}/{self.speaker_mapping_path}', 'wb') as handle:
            pkl.dump(speaker_mapping_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)
        with open(f'{out_dir}/{self.scale_map_path}', 'wb') as handle:
            pkl.dump(session_scale_mapping_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

    def load_emb_scale_seq_dict(self, out_dir):
        window_len_list = list(self.cfg_base.diarizer.speaker_embeddings.parameters.window_length_in_sec)
        emb_scale_seq_dict = {scale_index: None for scale_index in range(len(window_len_list))}
        for scale_index in range(len(window_len_list)):
            pickle_path = os.path.join(
                out_dir, 'speaker_outputs', 'embeddings', f'subsegments_scale{scale_index}_embeddings.pkl'
            )
            print(f"Loading embedding pickle file of scale:{scale_index} at {pickle_path}")
            with open(pickle_path, "rb") as input_file:
                emb_dict = pkl.load(input_file)
            for key, val in emb_dict.items():
                emb_dict[key] = val
            emb_scale_seq_dict[scale_index] = emb_dict
        return emb_scale_seq_dict

    def load_clustering_labels(self, out_dir):
        file_exists, clus_label_path = self.check_clustering_labels(out_dir)
        print(f"Loading cluster label file from {clus_label_path}")
        with open(clus_label_path) as f:
            clus_labels = f.readlines()
        return clus_labels

    def check_clustering_labels(self, out_dir):
        if self.cfg_base.diarizer.msdd_model.parameters.use_single_scale_clustering is not False:
            self.base_scale_index = int(self.cfg_base.diarizer.msdd_model.parameters.use_single_scale_clustering) - 1
        clus_label_path = os.path.join(
            out_dir, 'speaker_outputs', f'subsegments_scale{self.base_scale_index}_cluster.label'
        )
        file_exists = os.path.exists(clus_label_path)
        if not file_exists:
            logging.info(f"Clustering label file {clus_label_path} does not exist.")
        return file_exists, clus_label_path


class EncDecDiarLabelModelLab(ExportableEncDecModel, ClusterEmbeddingTest):
    """Encoder decoder class for multiscale speaker diarization decoder.
    Model class creates training, validation methods for setting up data
    performing model forward pass.
    Expects config dict for
    * preprocessor
    * msdd_model
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self.trainer = trainer
        self.pairwise_infer = False
        self.cfg_msdd_model = cfg
        self.cfg_msdd_model.msdd_module.num_spks = self.cfg_msdd_model.max_num_of_spks
        self.cfg_msdd_model.train_ds.num_spks = self.cfg_msdd_model.max_num_of_spks
        self.cfg_msdd_model.validation_ds.num_spks = self.cfg_msdd_model.max_num_of_spks
        super().__init__(cfg=self.cfg_msdd_model, trainer=trainer)
        ClusterEmbeddingTest.__init__(self, cfg_base=self.cfg_msdd_model.base, cfg_msdd_model=self.cfg_msdd_model)


class NeuralDiarizerLab(OverlapAwareDiarizer):
    """Class for inference based on multiscale diarization decoder (MSDD)."""

    def __init__(self, cfg: DictConfig):
        """ """
        self._cfg = cfg
        super().__init__(cfg)
        # Initialize diarization decoder
        msdd_dict, spk_emb_dict = {}, {}

        self.clustering_embedding = ClusterEmbeddingTest(cfg_base=cfg, cfg_msdd_model=self.msdd_model.cfg)
        self.clustering_embedding.run_clus_from_loaded_emb = True

    def diarize(self):
        """
        Launch diarization pipeline which starts from VAD (or a oracle VAD stamp generation), initialization clustering and multiscale diarization decoder (MSDD).
        Note that the result of MSDD can include multiple speakers at the same time. Therefore, RTTM output of MSDD needs to be based on `make_rttm_with_overlap()`
        function that can generate overlapping timestamps. `self.run_overlap_aware_eval()` function performs DER evaluation.
        """
        torch.set_grad_enabled(False)
        if False:
            # Optuna Segmentation Optimization
            logger = opt_logging.getLogger()
            logger.setLevel(opt_logging.INFO)  # Setup the root logger.
            logger.addHandler(opt_logging.FileHandler("Vox_optuna_multiscale_scaleN_ttnL_1.log", mode="w"))

            optuna.logging.enable_propagation()  # Propagate logs to the root logger.
            optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.
            n_trials = 1000
            self.optuna_log = []
            self.study = optuna.create_study()
            logger.info("Start optimization.")
            self.study.optimize(
                lambda trial: self.clustering_embedding.prepare_cluster_embs_infer_scale_opt(trial), n_trials=n_trials
            )
        elif self._cfg.diarizer.msdd_model.parameters.run_optuna:
            # Optuna Clustering Optimization
            logger = opt_logging.getLogger()
            logger.setLevel(opt_logging.INFO)  # Setup the root logger.
            # logger.addHandler(opt_logging.FileHandler("Vox_optuna_multiscale_clusA1_ttnL.log", mode="w"))
            logger.addHandler(opt_logging.FileHandler("Vox_optuna_multiscale_clusB1_ttnM.log", mode="w"))

            optuna.logging.enable_propagation()  # Propagate logs to the root logger.
            optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.
            n_trials = 1000
            self.optuna_log = []
            self.study = optuna.create_study()
            logger.info("Start Clustering Algorithm Optimization.")
            self.study.optimize(
                lambda trial: self.clustering_embedding.prepare_cluster_embs_infer_clus_opt(trial), n_trials=n_trials
            )

        else:
            self.clustering_embedding.prepare_cluster_embs_infer()
            self.msdd_model.pairwise_infer = True
        self.msdd_model.get_emb_clus_infer(self.clustering_embedding)
        preds_list, targets_list, signal_lengths_list = self.run_pairwise_diarization()
        for threshold in list(self._cfg.diarizer.msdd_model.parameters.sigmoid_threshold):
            self.run_overlap_aware_eval(preds_list, threshold)

    def diar_infer(self, test_batch, test_data_collection):
        """
        Launch forward_infer() function by feeding the session-wise embedding sequences to get pairwise speaker prediction values.
        If split_infer is True, the input audio clips are broken into short sequences then cluster average embeddings are calculated
        for inference. Split-infer might result in an improved results if calculating clustering average on the shorter tim-espan can
        help speaker assignment.

        Args:
            test_batch: (list)
                List containing embedding sequences, length of embedding sequences, ground truth labels (if exists) and initializing embedding vectors.
            test_data_collection: (list)
                List containing test-set dataloader contents. test_data_collection includes wav file path, RTTM file path, clustered speaker indices.

        Returns:
            preds (Tensor):
            targets (Tensor):
            signal_lengths (Tensor):
        """
        signals, signal_lengths, _targets, emb_vectors = test_batch
        if self._cfg.diarizer.msdd_model.parameters.split_infer:
            split_count = torch.ceil(torch.tensor(signals.shape[1] / self.diar_window_length)).int()
            sess_emb_vectors, sess_emb_seq, sess_sig_lengths = self.get_range_clus_avg_emb(
                test_batch, test_data_collection, split_count
            )
            if self._cfg.diarizer.msdd_model.parameters.use_longest_scale_clus_avg_emb:
                sess_emb_vectors = self.msdd_model.get_longest_scale_clus_avg_emb(sess_emb_vectors)
            with autocast():
                _preds, scale_weights = self.msdd_model.forward_infer(
                    input_signal=sess_emb_seq,
                    input_signal_length=sess_sig_lengths,
                    emb_vectors=sess_emb_vectors,
                    targets=None,
                )
            _preds = _preds.reshape(len(signal_lengths), split_count * self.diar_window_length, -1)
            _preds = _preds[:, : signals.shape[1], :]
        else:
            if self._cfg.diarizer.msdd_model.parameters.use_longest_scale_clus_avg_emb:
                sess_emb_vectors = self.msdd_model.get_longest_scale_clus_avg_emb(sess_emb_vectors)
            with autocast():
                _preds, scale_weights = self.msdd_model.forward_infer(
                    input_signal=signals, input_signal_length=signal_lengths, emb_vectors=emb_vectors, targets=_targets
                )
        self.max_pred_length = max(_preds.shape[1], self.max_pred_length)
        preds = torch.zeros(_preds.shape[0], self.max_pred_length, _preds.shape[2])
        targets = torch.zeros(_preds.shape[0], self.max_pred_length, _preds.shape[2])
        preds[:, : _preds.shape[1], :] = _preds
        return preds, targets, signal_lengths

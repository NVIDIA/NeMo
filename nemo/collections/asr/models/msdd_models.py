# cOPYRight (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from omegaconf.omegaconf import open_dict
from pytorch_lightning import Trainer

# from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.rank_zero import _get_rank
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

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


__all__ = ['EncDecDiarLabelModel', '_ClusterEmbedding']


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

            # uniqname = get_uniqname_from_filepath(filepath=meta['audio_filepath'])
            uniqname = get_uniq_id_with_dur(meta)
            if uniqname not in AUDIO_RTTM_MAP:
                AUDIO_RTTM_MAP[uniqname] = meta
            else:
                raise KeyError(f"Unique name:{uniqname} already exists in the AUDIO_RTTM_MAP dictionary.")

    return AUDIO_RTTM_MAP


def getScaleMappingArgmat(uniq_embs_and_timestamps):
    """
    Calculate cosine similarity values among speaker embeddings for each scale then
    apply multiscale weights to calculate the fused similarity matrix.

    Args:
        uniq_embs_and_timestamps: (dict)
            The dictionary containing embeddings, timestamps and multiscale weights.
            If uniq_embs_and_timestamps contains only one scale, single scale diarization
            is performed.

    Returns:
        fused_sim_d (np.array):
            This function generates an ffinity matrix that is obtained by calculating
            the weighted sum of the affinity matrices from the different scales.
    """
    uniq_scale_dict = uniq_embs_and_timestamps['scale_dict']
    multiscale_weights = uniq_embs_and_timestamps['multiscale_weights']
    scale_mapping_argmat = {}

    session_scale_mapping_dict = get_argmin_mat(uniq_scale_dict)
    for scale_idx in sorted(uniq_scale_dict.keys()):
        mapping_argmat = session_scale_mapping_dict[scale_idx]
        scale_mapping_argmat[scale_idx] = mapping_argmat
    return scale_mapping_argmat


def get_overlap_stamps(cont_stamps, ovl_spk_idx):
    ovl_spk_cont_list = [[] for _ in range(len(ovl_spk_idx))]
    for spk_idx in range(len(ovl_spk_idx)):
        for idx, cont_a_line in enumerate(cont_stamps):
            start, end, speaker = cont_a_line.split()
            if idx in ovl_spk_idx[spk_idx]:
                ovl_spk_cont_list[spk_idx].append(f"{start} {end} speaker_{spk_idx}")
    total_ovl_cont_list = []
    for ovl_cont_list in ovl_spk_cont_list:
        if len(ovl_cont_list) > 0:
            total_ovl_cont_list.extend(merge_stamps(ovl_cont_list))
    return total_ovl_cont_list


def generate_speaker_timestamps(clus_labels, preds, max_overlap=2, **params):
    '''
    Generate timestamps from the predicted sigmoid values
    '''
    preds.squeeze(0)
    main_speaker_lines = []
    overlap_speaker_list = [[] for _ in range(params['max_num_of_spks'])]
    for seg_idx, cluster_label in enumerate(clus_labels):
        preds.squeeze(0)
        spk_for_seg = (preds[0, seg_idx] > params['threshold']).int().cpu().numpy().tolist()
        sm_for_seg = preds[0, seg_idx].cpu().numpy()

        if params['use_dec']:
            main_spk_idx = np.argsort(preds[0, seg_idx].cpu().numpy())[::-1][0]
        elif params['use_clus'] or sum(spk_for_seg) == 0:
            main_spk_idx = int(cluster_label[2])
        else:
            main_spk_idx = np.argsort(preds[0, seg_idx].cpu().numpy())[::-1][0]

        if sum(spk_for_seg) > 1:
            max_idx = np.argmax(sm_for_seg)
            idx_arr = np.argsort(sm_for_seg)[::-1]
            for ovl_spk_idx in idx_arr[:max_overlap].tolist():
                if ovl_spk_idx != int(main_spk_idx):
                    overlap_speaker_list[ovl_spk_idx].append(seg_idx)
        main_speaker_lines.append(f"{cluster_label[0]} {cluster_label[1]} speaker_{main_spk_idx}")
    cont_stamps = get_contiguous_stamps(main_speaker_lines)
    maj_labels = merge_stamps(cont_stamps)
    ovl_labels = get_overlap_stamps(cont_stamps, overlap_speaker_list)
    return maj_labels, ovl_labels


def get_uniq_id_list_from_manifest(manifest_file):
    uniq_id_list = []
    with open(manifest_file, 'r', encoding='utf-8') as manifest:
        for i, line in enumerate(manifest.readlines()):
            line = line.strip()
            dic = json.loads(line)
            uniq_id = dic['audio_filepath'].split('/')[-1].split('.wav')[-1]
            uniq_id = get_uniqname_from_filepath(dic['audio_filepath'])
            uniq_id_list.append(uniq_id)
    return uniq_id_list


def get_id_tup_dict(uniq_id_list, test_data_collection, preds_list):
    session_dict = {x: [] for x in uniq_id_list}
    for idx, line in enumerate(test_data_collection):
        uniq_id = get_uniqname_from_filepath(line.audio_file)
        session_dict[uniq_id].append([line.target_spks, preds_list[idx], line.clus_spk_digits, line.rttm_spk_digits])
    return session_dict


def compute_accuracies(diar_decoder_model):
    f1_score = diar_decoder_model._accuracy_test.compute()
    num_correct = torch.sum(diar_decoder_model._accuracy_test.true.bool())
    total_count = torch.prod(torch.tensor(diar_decoder_model._accuracy_test.targets.shape))
    simple_acc = num_correct / total_count
    return f1_score, simple_acc


def get_uniq_id_from_manifest_line(line):
    line = line.strip()
    dic = json.loads(line)
    if len(dic['audio_filepath'].split('/')[-1].split('.')) > 2:
        uniq_id = '.'.join(dic['audio_filepath'].split('/')[-1].split('.')[:-1])
    else:
        uniq_id = dic['audio_filepath'].split('/')[-1].split('.')[0]
    return uniq_id


def make_rttm_with_overlap(manifest_file, clus_label_dict, hyp, **params):
    params['max_num_of_spks'] = 8
    AUDIO_RTTM_MAP = audio_rttm_map(manifest_file)

    manifest_file_lengths_list = []
    all_hypothesis, all_reference = [], []
    no_references = False
    with open(manifest_file, 'r', encoding='utf-8') as manifest:
        for i, line in enumerate(manifest.readlines()):
            uniq_id = get_uniq_id_from_manifest_line(line)
            manifest_dic = AUDIO_RTTM_MAP[uniq_id]
            clus_labels = clus_label_dict[uniq_id]
            manifest_file_lengths_list.append(len(clus_labels))
            maj_labels, ovl_labels = generate_speaker_timestamps(clus_labels, hyp[i], **params)
            if params['infer_overlap']:
                hyp_labels = maj_labels + ovl_labels
            else:
                hyp_labels = maj_labels
            hypothesis = labels_to_pyannote_object(hyp_labels, uniq_name=uniq_id)
            all_hypothesis.append([uniq_id, hypothesis])
            rttm_file = manifest_dic.get('rttm_filepath', None)
            if rttm_file is not None and os.path.exists(rttm_file) and not no_references:
                ref_labels = rttm_to_labels(rttm_file)
                reference = labels_to_pyannote_object(ref_labels, uniq_name=uniq_id)
                all_reference.append([uniq_id, reference])
            else:
                no_references = True
                all_reference = []

    return all_reference, all_hypothesis


class ClusterEmbedding:
    def __init__(self, cfg_base: DictConfig, cfg_msdd_model: DictConfig):
        self.cfg_base = cfg_base
        self._cfg_msdd = cfg_msdd_model
        self.max_num_of_spks = int(self.cfg_base.diarizer.clustering.parameters.max_num_speakers)
        self.cluster_avg_emb_path = 'speaker_outputs/embeddings/clus_emb_info.pkl'
        self.speaker_mapping_path = 'speaker_outputs/embeddings/clus_mapping.pkl'
        self.scale_map_path = 'speaker_outputs/embeddings/scale_mapping.pkl'
        self.multiscale_weights_list = self._cfg_msdd.base.diarizer.speaker_embeddings.parameters.multiscale_weights
        self.clusdiar_model = None
        self.msdd_model = None
        self.embs_filtering_thres = None
        self.run_clus_from_loaded_emb = False
        self.scale_window_length_list = list(self.cfg_base.diarizer.speaker_embeddings.parameters.window_length_in_sec)
        self.scale_n = len(self.scale_window_length_list)
        self.base_scale_index = len(self.scale_window_length_list) - 1

    def prepare_cluster_embs_infer(self):
        self.emb_sess_test_dict, self.emb_seq_test, self.clus_test_label_dict = self.run_clustering_diarizer(
            self._cfg_msdd.test_ds.manifest_filepath, self._cfg_msdd.test_ds.emb_dir
        )
        # self.emb_sess_test_dict, self.emb_seq_test, self.clus_test_label_dict = self.prepare_datasets(self._cfg_msdd.test_ds)

    def assign_labels_to_shorter_segs(self, base_clus_label_dict, session_scale_mapping_dict):
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

    def assign_labels_to_longer_segs(self, base_clus_label_dict, session_scale_mapping_dict):
        new_clus_label_dict = {scale_index: {} for scale_index in range(self.scale_n)}
        for uniq_id, uniq_scale_mapping_dict in session_scale_mapping_dict.items():
            base_scale_clus_label = np.array([x[-1] for x in base_clus_label_dict[uniq_id]])
            new_clus_label_dict[self.base_scale_index][uniq_id] = base_scale_clus_label
            for scale_index in range(self.scale_n - 1):
                new_clus_label = []
                assert (
                    uniq_scale_mapping_dict[scale_index].shape[0] == base_scale_clus_label.shape[0]
                ), "The number of base scale labels does not match the segment numbers in uniq_scale_mapping_dict"
                max_index = max(uniq_scale_mapping_dict[scale_index])
                for seg_idx in range(max_index + 1):
                    if seg_idx in uniq_scale_mapping_dict[scale_index]:
                        seg_clus_label = mode(base_scale_clus_label[uniq_scale_mapping_dict[scale_index] == seg_idx])
                    else:
                        seg_clus_label = 0 if len(new_clus_label) == 0 else new_clus_label[-1]
                    new_clus_label.append(seg_clus_label)
                new_clus_label_dict[scale_index][uniq_id] = new_clus_label
        return new_clus_label_dict

    def get_base_clus_label_dict(self, clus_labels, emb_scale_seq_dict):
        """
        Retrieve the base scale clustering labels.

        Args:
            clus_labels
        """
        base_clus_label_dict = {key: [] for key in emb_scale_seq_dict[self.base_scale_index].keys()}
        for line in clus_labels:
            uniq_id = line.split()[0]
            label = int(line.split()[-1].split('_')[-1])
            stt, end = [round(float(x), 2) for x in line.split()[1:3]]
            base_clus_label_dict[uniq_id].append([stt, end, label])
        emb_dim = emb_scale_seq_dict[0][uniq_id][0].shape[0]
        return base_clus_label_dict, emb_dim

    def replace_output_clus_label_dict(self, all_scale_clus_label_dict):
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
        MSDD
        Get an average embedding vector for each cluster (speaker).
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
                if num_of_spks > self.max_num_of_spks:
                    raise ValueError(
                        f"uniq_id {uniq_id} - self.max_num_of_spks {self.max_num_of_spks} is smaller than the actual number of speakers: {num_of_spks}"
                    )
                label_array = torch.Tensor(clus_label_list)
                avg_embs = torch.zeros(emb_dim, self.max_num_of_spks)
                for spk_idx in spk_set:
                    selected_embs = emb_tensor[label_array == spk_idx]
                    avg_embs[:, spk_idx] = torch.mean(selected_embs, dim=0)
                inv_map = {clus_key: rttm_key for rttm_key, clus_key in speaker_mapping_dict[uniq_id].items()}
                emb_sess_avg_dict[scale_index][uniq_id] = {'mapping': inv_map, 'avg_embs': avg_embs}
        return emb_sess_avg_dict, output_clus_label_dict

    def get_longest_scale_clus_avg_emb(self, emb_vectors, longest=0):
        """
        Use the cluster-average embedding vector from the longest scale. Using the cluster-average from the longest scale
        can improve the performance.

        Args:
            emb_vectors (torch.tensor):
                Tensor containing the cluster-average embeddings of each scale.
                Dimension: batch size x number of scales x embedding dimension x number of speakers

        Returns:
            emb_vectors (torch.tensor):
                Tensor containing the repeated cluster-average embeddings from the longest scale (scale index = 0)
                Dimension: batch size x number of scales x embedding dimension x number of speakers
        """
        bs, scale_n, emb_dim, n_spks = emb_vectors.shape
        return torch.repeat_interleave(emb_vectors[:, longest, :, :], scale_n, dim=0).reshape(
            bs, scale_n, emb_dim, n_spks
        )

    def run_clustering_diarizer(self, manifest_filepath, emb_dir):
        """
        If no pre-existing data is provided, run clustering diarizer from scratch. This will create scale-wise speaker embedding
        sequence, cluster-average embeddings, scale mapping and base scale clustering labels.
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
        return emb_sess_avg_dict, emb_scale_seq_dict, base_clus_label_dict

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

    def check_clustering_labels(self, out_dir):
        if self.cfg_base.diarizer.msdd_model.parameters.use_single_scale_clustering:
            self.base_scale_index = 0
        clus_label_path = os.path.join(
            out_dir, 'speaker_outputs', f'subsegments_scale{self.base_scale_index}_cluster.label'
        )
        file_exists = os.path.exists(clus_label_path)
        if not file_exists:
            logging.info(f"Clustering label file {clus_label_path} does not exist.")
        return file_exists, clus_label_path

    def load_clustering_labels(self, out_dir):
        file_exists, clus_label_path = self.check_clustering_labels(out_dir)
        print(f"Loading cluster label file from {clus_label_path}")
        with open(clus_label_path) as f:
            clus_labels = f.readlines()
        return clus_labels

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


class EncDecDiarLabelModel(ModelPT, ExportableEncDecModel, ClusterEmbedding):
    """Encoder decoder class for multiscale speaker diarization decoder.
    Model class creates training, validation methods for setting up data
    performing model forward pass.
    Expects config dict for
    * preprocessor
    * msdd_model
    """

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        return None

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self.trainer = trainer
        self.pairwise_infer = False
        self.cfg_msdd_model = cfg
        self.cfg_msdd_model.msdd_module.num_spks = self.cfg_msdd_model.max_num_of_spks

        self.cfg_msdd_model.train_ds.num_spks = self.cfg_msdd_model.max_num_of_spks
        self.cfg_msdd_model.validation_ds.num_spks = self.cfg_msdd_model.max_num_of_spks
        ClusterEmbedding.__init__(self, cfg_base=self.cfg_msdd_model.base, cfg_msdd_model=self.cfg_msdd_model)
        if trainer:
            if self.cfg_msdd_model.end_to_end_train:
                self._init_segmentation_info()
                self.prepare_train_split()

            self.world_size = trainer.num_nodes * trainer.num_devices
            self.emb_batch_size = self.cfg_msdd_model.emb_batch_size
        else:
            self.world_size = 1
            self.pairwise_infer = True
        super().__init__(cfg=self.cfg_msdd_model, trainer=trainer)

        if type(self.cfg_msdd_model.base.diarizer.speaker_embeddings.parameters.window_length_in_sec) == int:
            raise ValueError("window_length_in_sec should be a list containing multiple segment (window) lengths")
        else:
            self.cfg_msdd_model.scale_n = len(
                self.cfg_msdd_model.base.diarizer.speaker_embeddings.parameters.window_length_in_sec
            )
            self.cfg_msdd_model.msdd_module.scale_n = self.cfg_msdd_model.scale_n
            self.scale_n = self.cfg_msdd_model.scale_n

        self.preprocessor = EncDecSpeakerLabelModel.from_config_dict(self.cfg_msdd_model.preprocessor)
        self.frame_per_sec = int(1 / self.preprocessor._cfg.window_stride)
        self.msdd = EncDecDiarLabelModel.from_config_dict(self.cfg_msdd_model.msdd_module)
        if trainer is not None:
            self._init_speaker_model()
        torch.cuda.empty_cache()
        self.loss = BCELoss(weight=None)
        self.task = None
        self.min_detached_embs = 2
        self._accuracy_test = MultiBinaryAccuracy()
        self._accuracy_train = MultiBinaryAccuracy()
        self._accuracy_valid = MultiBinaryAccuracy()
        self.labels = None

    def _init_segmentation_info(self):
        self._diarizer_params = self.cfg_msdd_model.base.diarizer
        self.multiscale_args_dict = parse_scale_configs(
            self._diarizer_params.speaker_embeddings.parameters.window_length_in_sec,
            self._diarizer_params.speaker_embeddings.parameters.shift_length_in_sec,
            self._diarizer_params.speaker_embeddings.parameters.multiscale_weights,
        )

    def _init_speaker_model(self):
        """
        Initialize speaker embedding model with model name or path passed through config
        """
        model_path = self.cfg_msdd_model.base.diarizer.speaker_embeddings.model_path
        self._diarizer_params = self.cfg_msdd_model.base.diarizer
        gpu_device = torch.device(torch.cuda.current_device())
        if model_path is not None and model_path.endswith('.nemo'):
            rank_id = torch.device(self.trainer.global_rank)
            self.msdd._speaker_model = EncDecSpeakerLabelModel.restore_from(model_path, map_location=rank_id)
            logging.info("Speaker Model restored locally from {}".format(model_path))
        elif model_path.endswith('.ckpt'):
            self._speaker_model = EncDecSpeakerLabelModel.load_from_checkpoint(model_path)
            logging.info("Speaker Model restored locally from {}".format(model_path))
        else:
            if model_path not in get_available_model_names(EncDecSpeakerLabelModel):
                logging.warning(
                    "requested {} model name not available in pretrained models, instead".format(model_path)
                )
                model_path = "ecapa_tdnn"
            logging.info("Loading pretrained {} model from NGC".format(model_path))
            self._speaker_model = EncDecSpeakerLabelModel.from_pretrained(model_name=model_path)
        self.multiscale_embeddings_and_timestamps = {}
        self._speaker_params = self.cfg_msdd_model.base.diarizer.speaker_embeddings.parameters

    def get_emb_clus_infer(self, cluster_embeddings):
        self.emb_sess_test_dict = cluster_embeddings.emb_sess_test_dict
        self.clus_test_label_dict = cluster_embeddings.clus_test_label_dict
        self.emb_seq_test = cluster_embeddings.emb_seq_test

    def prepare_train_split(self):
        device = torch.cuda.current_device()
        self.train_multiscale_timestamp_dict = self.prepare_split_data(
            self.cfg_msdd_model.train_ds.manifest_filepath,
            self.cfg_msdd_model.train_ds.emb_dir,
            self.cfg_msdd_model.train_ds.batch_size,
        )
        self.validation_multiscale_timestamp_dict = self.prepare_split_data(
            self.cfg_msdd_model.validation_ds.manifest_filepath,
            self.cfg_msdd_model.validation_ds.emb_dir,
            self.cfg_msdd_model.validation_ds.batch_size,
        )

    def prepare_split_data(self, manifest_filepath, _out_dir, batch_size):
        _speaker_dir = os.path.join(_out_dir, 'speaker_outputs')
        if not os.path.exists(_speaker_dir):
            os.makedirs(_speaker_dir)
        if not os.path.exists(f'{_out_dir}/speaker_outputs/embeddings'):
            os.makedirs(f'{_out_dir}/speaker_outputs/embeddings')

        split_audio_rttm_map = get_audio_rttm_map(manifest_filepath)

        out_rttm_dir = os.path.join(_out_dir, 'pred_rttms')
        os.makedirs(out_rttm_dir, exist_ok=True)

        # Speech Activity Detection part
        _speaker_manifest_path = os.path.join(_speaker_dir, f'oracle_vad_manifest_rank{self.trainer.global_rank}.json')
        _speaker_manifest_path = write_rttm2manifest(
            split_audio_rttm_map, _speaker_manifest_path, include_uniq_id=True
        )

        multiscale_and_timestamps = {}

        # Segmentation
        for scale_idx, (window, shift) in self.multiscale_args_dict['scale_dict'].items():

            # Segmentation for the current scale (scale_idx)
            subsegments_manifest_path = self.run_segmentation(
                window, shift, _speaker_dir, _speaker_manifest_path, scale_tag=f'_scale{scale_idx}'
            )
            multiscale_timestamps = self._extract_timestamps(subsegments_manifest_path)
            multiscale_and_timestamps[scale_idx] = multiscale_timestamps

        multiscale_timestamps_dict = self.get_timestamps(multiscale_and_timestamps, self.multiscale_args_dict)
        return multiscale_timestamps_dict

    def _extract_timestamps(self, manifest_file: str):
        """
        This method extracts speaker embeddings from segments passed through manifest_file
        Optionally you may save the intermediate speaker embeddings for debugging or any use.
        """
        logging.info("Extracting only timestamps for multiscale segmentation")
        time_stamps = {}
        with open(manifest_file, 'r', encoding='utf-8') as manifest:
            for i, line in enumerate(manifest.readlines()):
                line = line.strip()
                dic = json.loads(line)

                uniq_name = dic['uniq_id']
                if uniq_name not in time_stamps:
                    time_stamps[uniq_name] = []
                start = dic['offset']
                end = start + dic['duration']
                stamp = '{:.3f} {:.3f} '.format(start, end)
                time_stamps[uniq_name].append(stamp)
        return time_stamps

    def get_timestamps(self, multiscale_and_timestamps, multiscale_args_dict):
        """
        The embeddings and timestamps in multiscale_embeddings_and_timestamps dictionary are
        indexed by scale index. This function rearranges the extracted speaker embedding and
        timestamps by unique ID to make the further processing more convenient.

        Args:
            multiscale_embeddings_and_timestamps (dict):
                Dictionary of timestamps for each scale.
            multiscale_args_dict (dict):
                Dictionary of scale information: window, shift and multiscale weights.

        Returns:
            timestamps_dict (dict)
                A dictionary containing embeddings and timestamps of each scale, indexed by unique ID.
        """
        timestamps_dict = {
            uniq_id: {'multiscale_weights': [], 'scale_dict': {}} for uniq_id in multiscale_and_timestamps[0].keys()
        }
        for scale_idx in sorted(multiscale_args_dict['scale_dict'].keys()):
            time_stamps = multiscale_and_timestamps[scale_idx]
            for uniq_id in time_stamps.keys():
                timestamps_dict[uniq_id]['multiscale_weights'] = (
                    torch.tensor(multiscale_args_dict['multiscale_weights']).unsqueeze(0).half()
                )
                timestamps_dict[uniq_id]['scale_dict'][scale_idx] = {
                    'time_stamps': time_stamps[uniq_id],
                }

        return timestamps_dict

    def __setup_dataloader_from_config(self, config: Optional[Dict], multiscale_timestamp_dict):
        featurizer = WaveformFeaturizer(
            sample_rate=config['sample_rate'], int_values=config.get('int_values', False), augmentor=None
        )

        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
            return None
        dataset = AudioToSpeechMSDDTrainDataset(
            manifest_filepath=config['manifest_filepath'],
            multiscale_args_dict=self.multiscale_args_dict,
            multiscale_timestamp_dict=multiscale_timestamp_dict,
            soft_label_thres=config.soft_label_thres,
            featurizer=featurizer,
            window_stride=self.cfg_msdd_model.preprocessor.window_stride,
            emb_batch_size=config['emb_batch_size'],
            pairwise_infer=False,
        )

        self.data_collection = dataset.collection
        collate_ds = dataset
        collate_fn = collate_ds.msdd_train_collate_fn
        batch_size = config['batch_size']
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=False,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def __setup_dataloader_from_config_infer(
        self, config: Optional[Dict], emb_dict: Dict, emb_seq: Dict, clus_label_dict: Dict, pairwise_infer=False
    ):
        featurizer = WaveformFeaturizer(
            sample_rate=config['sample_rate'], int_values=config.get('int_values', False), augmentor=None
        )
        shuffle = config.get('shuffle', False)

        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
            return None

        dataset = AudioToSpeechMSDDInferDataset(
            manifest_filepath=config['manifest_filepath'],
            emb_dict=emb_dict,
            clus_label_dict=clus_label_dict,
            emb_seq=emb_seq,
            soft_label_thres=config.soft_label_thres,
            seq_eval_mode=self.cfg_base.diarizer.msdd_model.parameters.seq_eval_mode,
            window_stride=self._cfg.preprocessor.window_stride,
            use_single_scale_clus=self.cfg_base.diarizer.msdd_model.parameters.use_single_scale_clustering,
            pairwise_infer=pairwise_infer,
        )
        self.data_collection = dataset.collection
        collate_ds = dataset
        collate_fn = collate_ds.msdd_infer_collate_fn
        batch_size = config['batch_size']
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def run_segmentation(
        self, window: float, shift: float, _speaker_dir: str, _speaker_manifest_path: str, scale_tag: str = ''
    ):

        subsegments_manifest_path = os.path.join(_speaker_dir, f'subsegments{scale_tag}_rank{self.trainer.global_rank}.json')
        logging.info(
            f"Subsegmentation for embedding extraction:{scale_tag.replace('_',' ')}, {subsegments_manifest_path}"
        )
        subsegments_manifest_path = segments_manifest_to_subsegments_manifest(
            segments_manifest_file=_speaker_manifest_path,
            subsegments_manifest_file=subsegments_manifest_path,
            window=window,
            shift=shift,
            include_uniq_id=True,
        )
        return subsegments_manifest_path

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        self._train_dl = self.__setup_dataloader_from_config(
            config=train_data_config, multiscale_timestamp_dict=self.train_multiscale_timestamp_dict
        )

    def setup_validation_data(self, val_data_layer_config: Optional[Union[DictConfig, Dict]]):
        self._validation_dl = self.__setup_dataloader_from_config(
            config=val_data_layer_config, multiscale_timestamp_dict=self.validation_multiscale_timestamp_dict
        )

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        if self.pairwise_infer:
            self._test_dl = self.__setup_dataloader_from_config_infer(
                config=test_data_config,
                emb_dict=self.emb_sess_test_dict,
                emb_seq=self.emb_seq_test,
                clus_label_dict=self.clus_test_label_dict,
                pairwise_infer=self.pairwise_infer,
            )

    def setup_multiple_test_data(self, test_data_config):
        """
        MSDD does not use multiple_test_data template. This function is a placeholder for preventing error.
        """
        return None

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
            "features": NeuralType(('B', 'T'), audio_eltype),
            "feature_length": NeuralType(('B'), LengthsType()),
            "ms_seg_timestamps": NeuralType(('B', 'C', 'T', 'D'), LengthsType()),
            "ms_seg_counts": NeuralType(('B', 'C'), LengthsType()),
            "clus_label_index": NeuralType(('B', 'T'), LengthsType()),
            "scale_mapping": NeuralType(('B', 'C', 'T'), LengthsType()),
            "targets": NeuralType(('B', 'T', 'C'), ProbsType()),
        }

    @property
    def output_types(self):
        return OrderedDict(
            {
                "probs": NeuralType(('B', 'T', 'C'), ProbsType()),
                "scale_weights": NeuralType(('B', 'T', 'C', 'D'), ProbsType()),
            }
        )

    def get_ms_emb_seq(self, embs, scale_mapping, ms_seg_counts):
        """
        Reshape the given tensor and organize the embedding sequence based on the original sequence counts.
        Repeat the embeddings according to the scale_mapping information so that the final embedding sequence has
        the identical length for all scales.

        Args:

        Returns:

        """
        device = torch.cuda.current_device()
        scale_n, batch_size = scale_mapping[0].shape[0], scale_mapping.shape[0]
        split_emb_tup = torch.split(embs, ms_seg_counts.view(-1).tolist(), dim=0)
        batch_emb_list = [split_emb_tup[i : i + scale_n] for i in range(0, len(split_emb_tup), scale_n)]
        ms_emb_seq_list = []
        for batch_idx in range(batch_size):
            feats_list = []
            for scale_index in range(scale_n):
                repeat_mat = scale_mapping[batch_idx][scale_index]
                feats_list.append(batch_emb_list[batch_idx][scale_index][repeat_mat, :])
            repp = torch.stack(feats_list).permute(1, 0, 2)
            ms_emb_seq_list.append(repp)
        ms_emb_seq = torch.stack(ms_emb_seq_list)
        return ms_emb_seq

    @torch.no_grad()
    def get_cluster_avg_embs_model(self, embs, clus_label_index, ms_seg_counts, scale_mapping):
        device = torch.cuda.current_device()
        scale_n, batch_size = scale_mapping[0].shape[0], scale_mapping.shape[0]
        split_emb_tup = torch.split(embs, ms_seg_counts.view(-1).tolist(), dim=0)
        batch_emb_list = [split_emb_tup[i : i + scale_n] for i in range(0, len(split_emb_tup), scale_n)]
        ms_avg_embs_list = []
        for batch_idx in range(batch_size):
            oracle_clus_idx = clus_label_index[batch_idx]
            max_seq_len = sum(ms_seg_counts[batch_idx])
            clus_label_index_batch = torch.split(oracle_clus_idx[:max_seq_len], ms_seg_counts[batch_idx].tolist())
            session_avg_emb_set_list = []
            for scale_index in range(scale_n):
                spk_set_list = []
                for idx in range(self.cfg_msdd_model.max_num_of_spks):
                    _where = (clus_label_index_batch[scale_index] == idx).clone().detach()
                    if torch.any(_where) == False:
                        avg_emb = torch.zeros(self.msdd._speaker_model._cfg.decoder.emb_sizes).to(device)
                    else:
                        avg_emb = torch.mean(batch_emb_list[batch_idx][scale_index][_where], dim=0)
                    spk_set_list.append(avg_emb)
                session_avg_emb_set_list.append(torch.stack(spk_set_list))
            session_avg_emb_set = torch.stack(session_avg_emb_set_list)
            ms_avg_embs_list.append(session_avg_emb_set)
        ms_avg_embs = torch.stack(ms_avg_embs_list).permute(0, 1, 3, 2)
        ms_avg_embs = ms_avg_embs.float().detach().to(device)
        if self.cfg_msdd_model.use_longest_scale_clus_avg_emb:
            ms_avg_embs = self.get_longest_scale_clus_avg_emb(ms_avg_embs)
        assert (
            ms_avg_embs.requires_grad == False
        ), "ms_avg_embs.requires_grad = True. ms_avg_embs should be detached from the torch graph."
        return ms_avg_embs

    @torch.no_grad()
    def get_ms_mel_feat(self, processed_signal, processed_signal_len, ms_seg_timestamps, ms_seg_counts):
        """
        Load acoustic feature from audio segments for each scale and save it into a torch.tensor matrix.
        In addition, create variables containing the information of the multiscale subsegmentation information.

        Args:
            processed_signal: (torch.tensor)
            processed_signal_len: (torch.tensor)
            ms_seg_timestamps: (torch.tensor)
            ms_seg_counts: (torch.tensor)


        Returns:
            return ms_mel_feat: (torch.tensor)
            ms_mel_feat_len: (torch.tensor)
            seq_len: (torch.tensor)
            detach_ids: (torch.tensor)
                (attached, detached)

        """
        device = torch.cuda.current_device()
        _emb_batch_size = min(self.emb_batch_size, ms_seg_counts.sum().item() - self.min_detached_embs)
        feat_dim = self.preprocessor._cfg.features
        max_sample_count = int(self.multiscale_args_dict["scale_dict"][0][0] * self.frame_per_sec)
        ms_mel_feat_len_list, sequence_lengths_list, ms_mel_feat_list = [], [], []
        total_seg_count = torch.sum(ms_seg_counts)

        batch_size = processed_signal.shape[0]
        for batch_idx in range(batch_size):
            max_seq_len = sum(ms_seg_counts[batch_idx])
            for scale_idx in range(self.scale_n):
                scale_seg_num = ms_seg_counts[batch_idx][scale_idx]
                for k, (stt, end) in enumerate(ms_seg_timestamps[batch_idx][scale_idx][:scale_seg_num]):
                    stt, end = int(stt.detach().item()), int(end.detach().item())
                    end = min(end, stt + max_sample_count)
                    _features = torch.zeros(feat_dim, max_sample_count).to(torch.float32).to(device)
                    _features[:, : (end - stt)] = processed_signal[batch_idx][:, stt:end]
                    ms_mel_feat_list.append(_features)
                    ms_mel_feat_len_list.append(end - stt)
            sequence_lengths_list.append(ms_seg_counts[batch_idx][-1])
        ms_mel_feat = torch.stack(ms_mel_feat_list).to(device)
        ms_mel_feat_len = torch.tensor(ms_mel_feat_len_list).to(device)
        seq_len = torch.tensor(sequence_lengths_list).to(device)

        torch.manual_seed(self.trainer.current_epoch)
        if _emb_batch_size < self.min_detached_embs:
            attached, _emb_batch_size = torch.tensor([]), 0
            detached = torch.randperm(total_seg_count)
        else:
            attached = torch.randperm(total_seg_count)[:_emb_batch_size]
            detached = torch.randperm(total_seg_count)[_emb_batch_size:]
        detach_ids = (attached, detached)
        return ms_mel_feat, ms_mel_feat_len, seq_len, detach_ids

    def forward_infer(self, input_signal, input_signal_length, emb_vectors, targets):
        """
        Wrapper function for inference case.
        """
        preds, scale_weights = self.msdd(
            ms_emb_seq=input_signal, length=input_signal_length, ms_avg_embs=emb_vectors, targets=targets
        )
        return preds, scale_weights

    @typecheck()
    def forward(
        self, features, feature_length, ms_seg_timestamps, ms_seg_counts, clus_label_index, scale_mapping, targets
    ):
        processed_signal, processed_signal_len = self.msdd._speaker_model.preprocessor(
            input_signal=features, length=feature_length
        )
        processed_signal = processed_signal.detach()
        del features, feature_length
        torch.cuda.empty_cache()
        audio_signal, audio_signal_len, sequence_lengths, detach_ids = self.get_ms_mel_feat(
            processed_signal, processed_signal_len, ms_seg_timestamps, ms_seg_counts
        )
        with torch.no_grad():
            self.msdd._speaker_model.eval()
            logits, embs_d = self.msdd._speaker_model.forward_for_export(
                processed_signal=audio_signal[detach_ids[1]], processed_signal_len=audio_signal_len[detach_ids[1]]
            )
            embs = torch.zeros(audio_signal.shape[0], embs_d.shape[1]).cuda()
            embs[detach_ids[1], :] = embs_d.detach()

        self.msdd._speaker_model.train()
        if len(detach_ids[0]) > 1:
            logits, embs_a = self.msdd._speaker_model.forward_for_export(
                processed_signal=audio_signal[detach_ids[0]], processed_signal_len=audio_signal_len[detach_ids[0]]
            )
            embs[detach_ids[0], :] = embs_a

        ms_emb_seq = self.get_ms_emb_seq(embs, scale_mapping, ms_seg_counts)
        ms_avg_embs = self.get_cluster_avg_embs_model(embs, clus_label_index, ms_seg_counts, scale_mapping)
        preds, scale_weights = self.msdd(
            ms_emb_seq=ms_emb_seq, length=sequence_lengths, ms_avg_embs=ms_avg_embs, targets=targets
        )
        scale_weights = scale_weights.detach()
        embs = embs.detach()
        return preds, scale_weights

    def training_step(self, batch, batch_idx):
        features, feature_length, ms_seg_timestamps, ms_seg_counts, clus_label_index, scale_mapping, targets = batch
        sequence_lengths = torch.tensor([x[-1] for x in ms_seg_counts.detach()])
        preds, _ = self.forward(
            features=features,
            feature_length=feature_length,
            ms_seg_timestamps=ms_seg_timestamps,
            ms_seg_counts=ms_seg_counts,
            clus_label_index=clus_label_index,
            scale_mapping=scale_mapping,
            targets=targets,
        )
        loss = self.loss(probs=preds, labels=targets, signal_lengths=sequence_lengths)
        self._accuracy_train(preds, targets, sequence_lengths)
        torch.cuda.empty_cache()
        f1_acc = self._accuracy_train.compute()
        self.log('loss', loss)
        self.log('learning_rate', self._optimizer.param_groups[0]['lr'])
        self.log('train_f1_acc', f1_acc)
        self._accuracy_train.reset()
        return {'loss': loss}

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        features, feature_length, ms_seg_timestamps, ms_seg_counts, clus_label_index, scale_mapping, targets = batch
        sequence_lengths = torch.tensor([x[-1] for x in ms_seg_counts])
        preds, _ = self.forward(
            features=features,
            feature_length=feature_length,
            ms_seg_timestamps=ms_seg_timestamps,
            ms_seg_counts=ms_seg_counts,
            clus_label_index=clus_label_index,
            scale_mapping=scale_mapping,
            targets=targets,
        )
        loss = self.loss(probs=preds, labels=targets, signal_lengths=sequence_lengths)
        self._accuracy_valid(preds, targets, sequence_lengths)
        f1_acc = self._accuracy_valid.compute()
        return {
            'val_loss': loss,
            'val_f1_acc': f1_acc,
        }

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        f1_acc = self._accuracy_valid.compute()
        self._accuracy_valid.reset()

        self.log('val_loss', val_loss_mean)
        self.log('val_f1_acc', f1_acc)
        return {
            'val_loss': val_loss_mean,
            'val_f1_acc': f1_acc,
        }

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        f1_acc = self._accuracy_test.compute()
        self._accuracy_test.reset()
        self.log('val_f1_acc', f1_acc)
        return {
            'test_loss': test_loss_mean,
            'test_f1_acc': f1_acc,
        }


class NeuralDiarizer:
    """Class for inference based on multiscale diarization decoder (MSDD)."""

    def __init__(self, cfg: DictConfig):
        """ """
        self._cfg = cfg
        # Initialize diarization decoder
        msdd_dict, spk_emb_dict = {}, {}
        loaded = torch.load(cfg.diarizer.msdd_model.model_path)
        _loaded = copy.deepcopy(loaded)
        for name, param in _loaded['state_dict'].items():
            if 'msdd._speaker_model' in name:
                del loaded['state_dict'][name]
                spk_emb_dict[name] = param
            elif 'msdd.' in name:
                msdd_dict[name] = param
        new_model_path = cfg.diarizer.msdd_model.model_path.replace('.ckpt', '_msdd.ckpt')
        torch.save(loaded, new_model_path)
        cfg.diarizer.msdd_model.model_path = new_model_path
        self.msdd_model = self._init_msdd_model(cfg)
        pretrained_dict = self.msdd_model.state_dict()
        pretrained_dict.update(msdd_dict)
        self.msdd_model.load_state_dict(pretrained_dict)
        self.diar_window_length = cfg.diarizer.msdd_model.parameters.diar_window_length
        self.msdd_model.cfg = self.transfer_diar_params_to_model_params(self.msdd_model, cfg)
        self.msdd_model.run_clus_from_loaded_emb = True
        self.manifest_filepath = self.msdd_model.cfg.test_ds.manifest_filepath
        self.use_clus = False
        self.use_dec = True
        self.AUDIO_RTTM_MAP = audio_rttm_map(self.manifest_filepath)

        # Initialize clustering and embedding preparation instance (as a diarization encoder).
        self.clustering_embedding = ClusterEmbedding(cfg_base=cfg, cfg_msdd_model=self.msdd_model.cfg)
        self.clustering_embedding.run_clus_from_loaded_emb = True

        self.use_prime_cluster_avg_emb = True
        self.max_pred_length = 0
        self.eps = 10e-5

    def transfer_diar_params_to_model_params(self, msdd_model, cfg):
        msdd_model.cfg.base.diarizer.out_dir = cfg.diarizer.out_dir
        msdd_model.cfg.test_ds.manifest_filepath = cfg.diarizer.manifest_filepath
        msdd_model.cfg.test_ds.emb_dir = cfg.diarizer.out_dir
        msdd_model.cfg_base = cfg
        msdd_model._cfg.base.diarizer.clustering.parameters.max_num_speakers = (
            cfg.diarizer.clustering.parameters.max_num_speakers
        )
        return msdd_model.cfg

    def _init_msdd_model(self, cfg):
        self.device = 'cuda'
        if not torch.cuda.is_available():
            self.device = 'cpu'
            logging.warning(
                "Running model on CPU, for faster performance it is adviced to use atleast one NVIDIA GPUs"
            )

        if cfg.diarizer.msdd_model.model_path.endswith('.nemo'):
            logging.info(f"Using local speaker model from {cfg.diarizer.msdd_model.model_path}")
            msdd_model = EncDecDiarLabelModel.restore_from(restore_path=cfg.diarizer.msdd_model.model_path)
        elif cfg.diarizer.msdd_model.model_path.endswith('.ckpt'):
            msdd_model = EncDecDiarLabelModel.load_from_checkpoint(checkpoint_path=cfg.diarizer.msdd_model.model_path)
        return msdd_model

    def get_pred_mat(self, data_list):
        all_tups = tuple()
        for data in data_list:
            all_tups += data[0]
        n_est_spks = len(set(all_tups))
        digit_map = dict(zip(sorted(set(all_tups)), range(n_est_spks)))
        total_len = max([sess[1].shape[1] for sess in data_list])
        sum_pred = torch.zeros(total_len, n_est_spks)
        for data in data_list:
            _dim_tup, pred_mat = data[:2]
            dim_tup = [digit_map[x] for x in _dim_tup]
            if len(pred_mat.shape) == 3:
                pred_mat = pred_mat.squeeze(0)
            if n_est_spks in [1, 2]:
                sum_pred = pred_mat
            else:
                _end = pred_mat.shape[0]
                sum_pred[:_end, dim_tup] += pred_mat.cpu().float()
        sum_pred = sum_pred / (n_est_spks - 1)
        return sum_pred

    def get_integrated_preds_list(self, uniq_id_list, test_data_collection, preds_list):
        """
        Merge multiple sequence inference outputs into a session level result.

        """
        session_dict = get_id_tup_dict(uniq_id_list, test_data_collection, preds_list)
        output_dict = {uniq_id: [] for uniq_id in uniq_id_list}
        for uniq_id, data_list in session_dict.items():
            sum_pred = self.get_pred_mat(data_list)
            output_dict[uniq_id] = sum_pred.unsqueeze(0)
        output_list = [output_dict[uniq_id] for uniq_id in uniq_id_list]
        return output_list

    def diarize(self):
        """
        Launch diarization pipeline which starts from VAD (or a oracle VAD stamp generation), initialization clustering and
        multiscale diarization decoder (MSDD). Note that the result of MSDD can include multiple speakers at the same time.
        Therefore, RTTM output of MSDD needs to be based on make_rttm_with_overlap() function that can generate overlapping
        timestamps.
        """
        torch.set_grad_enabled(False)

        self.clustering_embedding.prepare_cluster_embs_infer()
        self.msdd_model.pairwise_infer = True
        self.msdd_model.get_emb_clus_infer(self.clustering_embedding)
        preds_list, targets_list, signal_lengths_list = self.run_pairwise_diarization()
        for threshold in list(self._cfg.diarizer.msdd_model.parameters.sigmoid_threshold):
            self.run_overlap_aware_eval(preds_list, threshold)

    def get_range_average(self, signals, emb_vectors, diar_window_index, test_data_collection):
        emb_vectors_split = torch.zeros_like(emb_vectors)
        uniq_id = os.path.splitext(os.path.basename(test_data_collection.rttm_file))[0]
        clus_label_tensor = torch.tensor([x[-1] for x in self.msdd_model.clus_test_label_dict[uniq_id]])
        for spk_idx in range(len(test_data_collection.target_spks)):
            stt, end = (
                diar_window_index * self.diar_window_length,
                min((diar_window_index + 1) * self.diar_window_length, clus_label_tensor.shape[0]),
            )
            seq_len = end - stt
            if stt < clus_label_tensor.shape[0]:
                target_clus_label_tensor = clus_label_tensor[stt:end]
                emb_seq, seg_length = (
                    signals[stt:end, :, :],
                    min(
                        self.diar_window_length,
                        clus_label_tensor.shape[0] - diar_window_index * self.diar_window_length,
                    ),
                )
                target_clus_label_bool = target_clus_label_tensor == test_data_collection.target_spks[spk_idx]
                
                # There are cases where there is no corresponding speaker in split range, so any(target_clus_label_bool) could be False.
                if any(target_clus_label_bool) == True:
                    emb_vectors_split[:, :, spk_idx] = torch.mean(emb_seq[target_clus_label_bool], dim=0)
                
                # In case when the loop reaches the end of the sequence
                if seq_len < self.diar_window_length:
                    emb_seq = torch.cat(
                        [
                            emb_seq,
                            torch.zeros(self.diar_window_length - seq_len, emb_seq.shape[1], emb_seq.shape[2]).to(
                                self.device
                            ),
                        ],
                        dim=0,
                    )
            else:
                emb_seq = torch.zeros(self.diar_window_length, emb_vectors.shape[0], emb_vectors.shape[1]).to(
                    self.device
                )
                seq_len = 0
        return emb_vectors_split, emb_seq, seq_len

    def get_range_clus_avg_emb(self, test_batch, _test_data_collection, split_count):
        """
        

        split_count

        """
        _signals, signal_lengths, _targets, _emb_vectors = test_batch
        sess_emb_vectors, sess_emb_seq, sess_sig_lengths = [], [], []
        split_count = torch.ceil(torch.tensor(_signals.shape[1] / self.diar_window_length)).int()
        self.max_pred_length = max(self.max_pred_length, self.diar_window_length * split_count)
        for k in range(_signals.shape[0]):
            signals, emb_vectors, test_data_collection = _signals[k], _emb_vectors[k], _test_data_collection[k]
            for diar_window_index in range(split_count):
                emb_vectors_split, emb_seq, seq_len = self.get_range_average(
                    signals, emb_vectors, diar_window_index, test_data_collection
                )
                sess_emb_vectors.append(emb_vectors_split)
                sess_emb_seq.append(emb_seq)
                sess_sig_lengths.append(seq_len)
        sess_emb_vectors = torch.stack(sess_emb_vectors)
        sess_emb_seq = torch.stack(sess_emb_seq)
        sess_sig_lengths = torch.tensor(sess_sig_lengths)
        return sess_emb_vectors, sess_emb_seq, sess_sig_lengths

    def diar_infer(self, test_batch, test_data_collection):
        """
        Launch forward_infer() function by feeding the session-wise embedding sequences to get pairwise speaker prediction values.
        If split_infer is True, the input audio clips are broken into short sequences then cluster average embeddings are calculated
        for inference.

        If split_infer is True, the input embedding sequence is broken down into short sequences to calculate the cluster-average vectors.
        Split-infer might result in an improved results if calculating clustering average on the shorter timespan can help speaker assignment.

        Args:
            test_batch: (list)
                List containing embedding sequences, length of embedding sequences, ground truth labels (if exists) and initializing embedding vectors.
            test_data_collection: (list)
                List containing test-set dataloader contents. test_data_collection includes wav file path, RTTM file path, clustered speaker indices.

        Returns:
            preds: (torch.Tensor)
            targets: (torch.Tensor)
            signal_lengths: (torch.Tensor)
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

    def run_pairwise_diarization(self, embedding_dir='./', device='cuda'):
        """
        Setup the parameters needed for batch inference and run batch inference. Note that each sample is pairwise speaker input.
        The pairwise inference results are reconstructed to make session-wise prediction results.

        Args:
            embedding_dir: (str)
                Embedding directory in string format
            device: (torch.device)
                Torch device variable.

        Returns:
            integrated_preds_list: (list)
                List containing the session-wise speaker predictions in torch.tensor format.
            targets_list: (list)
                List containing the ground-truth labels in matrix format filled with  0 or 1.
            signal_lengths_list: (list)
                List containing the actual length of each sequence in session.
        """
        test_cfg = self.msdd_model.cfg.test_ds
        self.msdd_model.setup_test_data(test_cfg)
        self.msdd_model = self.msdd_model.to(self.device)
        self.msdd_model.eval()
        torch.set_grad_enabled(False)
        cumul_sample_count = [0]
        preds_list, targets_list, signal_lengths_list = [], [], []
        uniq_id_list = get_uniq_id_list_from_manifest(self.manifest_filepath)
        test_data_collection = [d for d in self.msdd_model.data_collection]
        for sidx, test_batch in enumerate(tqdm(self.msdd_model.test_dataloader())):
            test_batch = [x.to(self.device) for x in test_batch]
            signals, signal_lengths, _targets, emb_vectors = test_batch
            cumul_sample_count.append(cumul_sample_count[-1] + signal_lengths.shape[0])
            preds, targets, signal_lengths = self.diar_infer(
                test_batch, test_data_collection[cumul_sample_count[-2] : cumul_sample_count[-1]]
            )
            if self._cfg.diarizer.msdd_model.parameters.seq_eval_mode:
                self.msdd_model._accuracy_test(preds, targets, signal_lengths)

            preds_list.extend(list(torch.split(preds, 1)))
            targets_list.extend(list(torch.split(targets, 1)))
            signal_lengths_list.extend(list(torch.split(signal_lengths, 1)))

        if self._cfg.diarizer.msdd_model.parameters.seq_eval_mode:
            f1_score, simple_acc = compute_accuracies(self.msdd_model)
            logging.info(f"Test Inference F1 score. {f1_score:.4f}, simple Acc. {simple_acc:.4f}")
        integrated_preds_list = self.get_integrated_preds_list(uniq_id_list, test_data_collection, preds_list)
        return integrated_preds_list, targets_list, signal_lengths_list

    def run_overlap_aware_eval(self, preds_list: List[torch.Tensor], threshold: float):
        """
        Based on the predicted sigmoid values, render RTTM files then evaluate the overlap-aware diarization results.

        Args:
            preds_list: (list)
                List containing predicted pairwise speaker labels.
            threshold: (float)
                A floating-point threshold value that determines overlapped speech detection.
                    - If threshold is 1.0, no overlap speech is detected and only detect major speaker.
                    - If threshold is 0.0, all speakers are considered active at any time step.
        """
        for k, (collar, ignore_overlap) in enumerate([(0.25, True), (0.0, False)]):
            logging.info(
                f"     [Threshold: {threshold:.4f}]   [infer_overlap={not ignore_overlap}]   [use_clus={self.use_clus}]    [use_dec={self.use_dec}]"
            )
            all_reference, all_hypothesis = make_rttm_with_overlap(
                self.manifest_filepath,
                self.msdd_model.clus_test_label_dict,
                preds_list,
                threshold=threshold,
                infer_overlap=(not ignore_overlap),
                use_clus=self.use_clus,
                use_dec=self.use_dec,
            )
            score = score_labels(
                self.AUDIO_RTTM_MAP, all_reference, all_hypothesis, collar=collar, ignore_overlap=ignore_overlap,
            )

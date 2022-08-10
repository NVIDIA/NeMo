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

import json
import os
import pickle as pkl
import shutil
from collections import OrderedDict
from statistics import mode
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from tqdm import tqdm

from nemo.collections.asr.data.audio_to_diar_label import AudioToSpeechMSDDInferDataset, AudioToSpeechMSDDTrainDataset
from nemo.collections.asr.losses.bce_loss import BCELoss
from nemo.collections.asr.metrics.multi_binary_acc import MultiBinaryAccuracy
from nemo.collections.asr.models import ClusteringDiarizer
from nemo.collections.asr.models.asr_model import ExportableEncDecModel
from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.utils.nmesc_clustering import get_argmin_mat
from nemo.collections.asr.parts.utils.speaker_utils import (
    audio_rttm_map,
    get_contiguous_stamps,
    get_embs_and_timestamps,
    get_uniqname_from_filepath,
    labels_to_pyannote_object,
    merge_stamps,
    parse_scale_configs,
    rttm_to_labels,
    score_labels,
    segments_manifest_to_subsegments_manifest,
    write_rttm2manifest,
)
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import AudioSignal, LengthsType, NeuralType, ProbsType
from nemo.core.neural_types.elements import ProbsType
from nemo.utils import logging

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


__all__ = ['EncDecDiarLabelModel', 'ClusterEmbedding']


def getScaleMappingArgmat(uniq_embs_and_timestamps: Dict[str, dict]) -> Dict[int, torch.Tensor]:
    """
    Calculate cosine similarity values among speaker embeddings for each scale then
    apply multiscale weights to calculate the fused similarity matrix.

    Args:
        uniq_embs_and_timestamps: (dict)
            The dictionary containing embeddings, timestamps and multiscale weights.
            If uniq_embs_and_timestamps contains only one scale, single scale diarization
            is performed.

    Returns:
        scale_mapping_argmat (dict)
            Dictionary containing scale mapping information matrix for each scale.
    """
    scale_mapping_argmat = {}
    uniq_scale_dict = uniq_embs_and_timestamps['scale_dict']
    session_scale_mapping_dict = get_argmin_mat(uniq_scale_dict)
    for scale_idx in sorted(uniq_scale_dict.keys()):
        mapping_argmat = session_scale_mapping_dict[scale_idx]
        scale_mapping_argmat[scale_idx] = mapping_argmat
    return scale_mapping_argmat


def get_overlap_stamps(cont_stamps: List[str], ovl_spk_idx: List[str]):
    """
    Generate timestamps that include overlap speech. Overlap-including timestamps are created based on the segments that are
    created for clustering diarizer. Overlap speech is assigned to the existing speech segments in `cont_stamps`.

    Args:
        cont_stamps (list):
            Non-overlapping (single speaker per segment) diarization output in string format.
            Each line contains the start and end time of segments and corresponding speaker labels.
        ovl_spk_idx (list):
            List containing segment index of the estimated overlapped speech. The start and end of segments are based on the
            single-speaker (i.e., non-overlap-aware) RTTM generation.
    Returns:
        total_ovl_cont_list (list):
            Rendered diarization output in string format. Each line contains the start and end time of segments and
            corresponding speaker labels. This format is identical to `cont_stamps`.
    """
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


def get_adaptive_threshold(estimated_num_of_spks: int, min_threshold: float, overlap_infer_spk_limit: int):
    """
    This function controls the magnitude of the sigmoid threshold based on the estimated number of speakers. As the number of
    speakers becomes larger, diarization error rate is very sensitive on overlap speech detection. This function linearly increases
    the threshold in proportion to the estimated number of speakers so more confident overlap speech results are reflected when
    the number of estimated speakers are relatively high.

    Args:
        estimated_num_of_spks (int):
            Estimated number of speakers from the clustering result.
        min_threshold (float):
            Sigmoid threshold value from the config file. This threshold value is minimum threshold value when `estimated_num_of_spks=2`
        overlap_infer_spk_limit (int):
            If the `estimated_num_of_spks` is less then `overlap_infer_spk_limit`, overlap speech estimation is skipped.

    Returns:
        adaptive_threshold (float):
            Threshold value that is scaled based on the `estimated_num_of_spks`.
    """
    adaptive_threshold = min_threshold - (estimated_num_of_spks - 2) * (min_threshold - 1) / (
        overlap_infer_spk_limit - 2
    )
    return adaptive_threshold


def generate_speaker_timestamps(
    clus_labels: List[Union[float, int]], msdd_preds: List[torch.Tensor], **params
) -> Tuple[List[str], List[str]]:
    '''
    Generate speaker timestamps from the segmentation information. If `use_clus_as_main=True`, use clustering result for main speaker
    labels and use timestamps from the predicted sigmoid values. In this function, the main speaker labels in `maj_labels` exist for
    every subsegment steps while overlap speaker labels in `ovl_labels` only exist for segments where overlap-speech is occuring.

    Args:
        clus_labels (list):
            List containing integer-valued speaker clustering results.
        msdd_preds (list):
            List containing tensors of the predicted sigmoid values.
            Each tensor has shape of: (Session length, estimated number of speakers).
        params:
            Parameters for generating RTTM output and evaluation. Parameters include:
                infer_overlap (bool): If False, overlap-speech will not be detected.
                use_clus_as_main (bool): Add overlap-speech detection from MSDD to clustering results. If False, only MSDD output
                                         is used for constructing output RTTM files.
                overlap_infer_spk_limit (int): Above this limit, overlap-speech detection is bypassed.
                use_adaptive_thres (bool): Boolean that determines whehther to use adaptive_threshold depending on the estimated
                                           number of speakers.
                max_overlap_spks (int): Maximum number of overlap speakers detected. Default is 2.
                threshold (float): Sigmoid threshold for MSDD output.

    Returns:
        maj_labels (list):
            List containing string-formated single-speaker speech segment timestamps and corresponding speaker labels.
            Example: [..., '551.685 552.77 speaker_1', '552.99 554.43 speaker_0', '554.97 558.19 speaker_0', ...]
        ovl_labels (list):
            List containing string-formated additional overlapping speech segment timestamps and corresponding speaker labels.
            Note that `ovl_labels` includes only overlapping speech that is not included in `maj_labels`.
            Example: [..., '152.495 152.745 speaker_1', '372.71 373.085 speaker_0', '554.97 555.885 speaker_1', ...]
    '''
    msdd_preds.squeeze(0)
    estimated_num_of_spks = msdd_preds.shape[-1]
    overlap_speaker_list = [[] for _ in range(estimated_num_of_spks)]
    infer_overlap = estimated_num_of_spks < int(params['overlap_infer_spk_limit'])
    main_speaker_lines = []
    if params['use_adaptive_thres']:
        threshold = get_adaptive_threshold(
            estimated_num_of_spks, params['threshold'], params['overlap_infer_spk_limit']
        )
    else:
        threshold = params['threshold']
    for seg_idx, cluster_label in enumerate(clus_labels):
        msdd_preds.squeeze(0)
        spk_for_seg = (msdd_preds[0, seg_idx] > threshold).int().cpu().numpy().tolist()
        sm_for_seg = msdd_preds[0, seg_idx].cpu().numpy()

        if params['use_clus_as_main']:
            main_spk_idx = int(cluster_label[2])
        else:
            main_spk_idx = np.argsort(msdd_preds[0, seg_idx].cpu().numpy())[::-1][0]

        if sum(spk_for_seg) > 1 and infer_overlap:
            idx_arr = np.argsort(sm_for_seg)[::-1]
            for ovl_spk_idx in idx_arr[: params['max_overlap_spks']].tolist():
                if ovl_spk_idx != int(main_spk_idx):
                    overlap_speaker_list[ovl_spk_idx].append(seg_idx)
        main_speaker_lines.append(f"{cluster_label[0]} {cluster_label[1]} speaker_{main_spk_idx}")
    cont_stamps = get_contiguous_stamps(main_speaker_lines)
    maj_labels = merge_stamps(cont_stamps)
    ovl_labels = get_overlap_stamps(cont_stamps, overlap_speaker_list)
    return maj_labels, ovl_labels


def get_uniq_id_list_from_manifest(manifest_file: str):
    """Retrieve `uniq_id` values from the given manifest_file and save the IDs to a list.
    """
    uniq_id_list = []
    with open(manifest_file, 'r', encoding='utf-8') as manifest:
        for i, line in enumerate(manifest.readlines()):
            line = line.strip()
            dic = json.loads(line)
            uniq_id = get_uniqname_from_filepath(dic['audio_filepath'])
            uniq_id_list.append(uniq_id)
    return uniq_id_list


def get_id_tup_dict(uniq_id_list: List[str], test_data_collection, preds_list: List[torch.Tensor]):
    """
    Create session-level dictionary containing data needed to construct RTTM diarization output.

    Args:
        uniq_id_list (list):
            List containing the `uniq_id` values.
        test_data_collection (collections.DiarizationLabelEntity):
            Class instance that is containing session information such as targeted speaker indices, audio filepath and RTTM filepath.
        preds_list (list):
            List containing tensors of predicted sigmoid values.

    Returns:
        session_dict (dict):
            Dictionary containing session-level target speakers data and predicted simoid values in tensor format.
    """
    session_dict = {x: [] for x in uniq_id_list}
    for idx, line in enumerate(test_data_collection):
        uniq_id = get_uniqname_from_filepath(line.audio_file)
        session_dict[uniq_id].append([line.target_spks, preds_list[idx]])
    return session_dict


def get_uniq_id_from_manifest_line(line: str) -> str:
    """
    Retrieve `uniq_id` from the `audio_filepath` in manifest file.
    """
    line = line.strip()
    dic = json.loads(line)
    if len(dic['audio_filepath'].split('/')[-1].split('.')) > 2:
        uniq_id = '.'.join(dic['audio_filepath'].split('/')[-1].split('.')[:-1])
    else:
        uniq_id = dic['audio_filepath'].split('/')[-1].split('.')[0]
    return uniq_id


def make_rttm_with_overlap(
    manifest_file_path: str,
    clus_label_dict: Dict[str, List[Union[float, int]]],
    msdd_preds: List[torch.Tensor],
    **params,
):
    """
    Create RTTM files that include detected overlap speech. Note that the effect of overlap detection is only
    notable when RTTM files are evaluated with `ignore_overlap=False` option.

    Args:
        manifest_file_path (str):
            Path to the input manifest file.
        clus_label_dict (dict):
            Dictionary containing subsegment timestamps in float type and cluster labels in integer type.
            Indexed by `uniq_id` string.
        msdd_preds (list):
            List containing tensors of the predicted sigmoid values.
            Each tensor has shape of: (Session length, estimated number of speakers).
        params:
            Parameters for generating RTTM output and evaluation. Parameters include:
                infer_overlap (bool): If False, overlap-speech will not be detected.
            See docstrings of `generate_speaker_timestamps` function for other variables in `params`.

    Returns:
        all_hypothesis (list):
            List containing Pyannote's `Annotation` objects that are created from hypothesis RTTM outputs.
        all_reference
            List containing Pyannote's `Annotation` objects that are created from ground-truth RTTM outputs
    """
    AUDIO_RTTM_MAP = audio_rttm_map(manifest_file_path)
    manifest_file_lengths_list = []
    all_hypothesis, all_reference = [], []
    no_references = False
    with open(manifest_file_path, 'r', encoding='utf-8') as manifest:
        for i, line in enumerate(manifest.readlines()):
            uniq_id = get_uniq_id_from_manifest_line(line)
            manifest_dic = AUDIO_RTTM_MAP[uniq_id]
            clus_labels = clus_label_dict[uniq_id]
            manifest_file_lengths_list.append(len(clus_labels))
            maj_labels, ovl_labels = generate_speaker_timestamps(clus_labels, msdd_preds[i], **params)
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
        self.cluster_avg_emb_path = 'speaker_outputs/embeddings/clus_emb_info.pkl'
        self.speaker_mapping_path = 'speaker_outputs/embeddings/clus_mapping.pkl'
        self.scale_map_path = 'speaker_outputs/embeddings/scale_mapping.pkl'
        self.multiscale_weights_list = self._cfg_msdd.base.diarizer.speaker_embeddings.parameters.multiscale_weights
        self.clusdiar_model = None
        self.msdd_model = None
        self.spk_emb_state_dict = {}

        self.run_clus_from_loaded_emb = False
        self.use_speaker_model_from_MSDD = False
        self.scale_window_length_list = list(self.cfg_base.diarizer.speaker_embeddings.parameters.window_length_in_sec)
        self.scale_n = len(self.scale_window_length_list)
        self.base_scale_index = len(self.scale_window_length_list) - 1

    def prepare_cluster_embs_infer(self):
        """
        Launch clustering diarizer to prepare embedding vectors and clustering results.
        """
        self.max_num_speakers = self.cfg_base.diarizer.clustering.parameters.get('max_num_speakers', 8)
        self.emb_sess_test_dict, self.emb_seq_test, self.clus_test_label_dict, _ = self.run_clustering_diarizer(
            self._cfg_msdd.test_ds.manifest_filepath, self._cfg_msdd.test_ds.emb_dir
        )

    def assign_labels_to_longer_segs(self, base_clus_label_dict: Dict, session_scale_mapping_dict: Dict):
        """
        In multi-scale speaker diarization system, clustering result is solely based on the base-scale (the shortest scale).
        To calculate cluster-average speaker embeddings for each scale that are longer than the base-scale. This function assigns
        clustering results for the base-scale to the longer scales by measuring the distance between subsegment timestamps in the
        base-scale and non-base-scales.

        Args:
            base_clus_label_dict (dict):
                Dictionary containing clustering results for base-scale segments. Indexed by `uniq_id` string.
            session_scale_mapping_dict (dict):
                Dictionary containing multiscale mapping information for each session. Indexed by `uniq_id` string.

        Returns:
            all_scale_clus_label_dict (dict):
                Dictionary containing clustering labels of all scales. Indexed by scale_index in integer format.

        """
        all_scale_clus_label_dict = {scale_index: {} for scale_index in range(self.scale_n)}
        for uniq_id, uniq_scale_mapping_dict in session_scale_mapping_dict.items():
            base_scale_clus_label = np.array([x[-1] for x in base_clus_label_dict[uniq_id]])
            all_scale_clus_label_dict[self.base_scale_index][uniq_id] = base_scale_clus_label
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
                all_scale_clus_label_dict[scale_index][uniq_id] = new_clus_label
        return all_scale_clus_label_dict

    def get_base_clus_label_dict(self, clus_labels: List[str], emb_scale_seq_dict: Dict[int, dict]):
        """
        Retrieve base scale clustering labels from `emb_scale_seq_dict`.

        Args:
            clus_labels (list):
                List containing cluster results generated by clustering diarizer.
            emb_scale_seq_dict (dict):
                Dictionary containing multiscale embedding input sequences.
        Returns:
            base_clus_label_dict (dict):
                Dictionary containing start and end of base scale segments and its cluster label. Indexed by `uniq_id`.
            emb_dim (int):
                Embedding dimension in integer.
        """
        base_clus_label_dict = {key: [] for key in emb_scale_seq_dict[self.base_scale_index].keys()}
        for line in clus_labels:
            uniq_id = line.split()[0]
            label = int(line.split()[-1].split('_')[-1])
            stt, end = [round(float(x), 2) for x in line.split()[1:3]]
            base_clus_label_dict[uniq_id].append([stt, end, label])
        emb_dim = emb_scale_seq_dict[0][uniq_id][0].shape[0]
        return base_clus_label_dict, emb_dim

    def get_cluster_avg_embs(
        self, emb_scale_seq_dict: Dict, clus_labels: List, speaker_mapping_dict: Dict, session_scale_mapping_dict: Dict
    ):
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
                Dictionary containing multiscale mapping information for each session. Indexed by `uniq_id` string.

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
                label_array = torch.Tensor(clus_label_list)
                avg_embs = torch.zeros(emb_dim, self.max_num_speakers)
                for spk_idx in spk_set:
                    selected_embs = emb_tensor[label_array == spk_idx]
                    avg_embs[:, spk_idx] = torch.mean(selected_embs, dim=0)

                if speaker_mapping_dict is not None:
                    inv_map = {clus_key: rttm_key for rttm_key, clus_key in speaker_mapping_dict[uniq_id].items()}
                else:
                    inv_map = None

                emb_sess_avg_dict[scale_index][uniq_id] = {'mapping': inv_map, 'avg_embs': avg_embs}
        return emb_sess_avg_dict, output_clus_label_dict

    def run_clustering_diarizer(self, manifest_filepath: str, emb_dir: str):
        """
        If no pre-existing data is provided, run clustering diarizer from scratch. This will create scale-wise speaker embedding
        sequence, cluster-average embeddings, scale mapping and base scale clustering labels. Note that speaker embedding `state_dict`
        is loaded from the `state_dict` in the provided MSDD checkpoint.

        Args:
            manifest_filepath (str):
                Input manifest file for creating audio-to-RTTM mapping.
            emb_dir (str):
                Output directory where embedding files and timestamp files are saved.

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

        # Load speaker embedding model state_dict which is loaded from the MSDD checkpoint.
        if self.use_speaker_model_from_MSDD:
            self.clusdiar_model._speaker_model.load_state_dict(self.spk_emb_state_dict)

        self.clusdiar_model._cluster_params = self.cfg_base.diarizer.clustering.parameters
        self.clusdiar_model.multiscale_args_dict[
            "multiscale_weights"
        ] = self.cfg_base.diarizer.speaker_embeddings.parameters.multiscale_weights
        self.clusdiar_model._diarizer_params.speaker_embeddings.parameters = (
            self.cfg_base.diarizer.speaker_embeddings.parameters
        )
        clustering_params_str = json.dumps(dict(self.clusdiar_model._cluster_params), indent=4)

        logging.info(f"Multiscale Weights: {self.clusdiar_model.multiscale_args_dict['multiscale_weights']}")
        logging.info(f"Clustering Parameters: {clustering_params_str}")
        scores = self.clusdiar_model.diarize(batch_size=self.cfg_base.batch_size)

        # If RTTM (ground-truth diarization annotation) files do not exist, scores is None.
        if scores is not None:
            metric, speaker_mapping_dict = scores
        else:
            metric, speaker_mapping_dict = None, None

        # Get the mapping between segments in different scales.
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

    def get_scale_map(self, embs_and_timestamps):
        """
        Save multiscale mapping data into dictionary format.

        Args:
            embs_and_timestamps (dict):
                Dictionary containing embedding tensors and timestamp tensors. Indexed by `uniq_id` string.
        Returns:
            session_scale_mapping_dict (dict):
                Dictionary containing multiscale mapping information for each session. Indexed by `uniq_id` string.
        """
        session_scale_mapping_dict = {}
        for uniq_id, uniq_embs_and_timestamps in embs_and_timestamps.items():
            scale_mapping_dict = getScaleMappingArgmat(uniq_embs_and_timestamps)
            session_scale_mapping_dict[uniq_id] = scale_mapping_dict
        return session_scale_mapping_dict

    def check_clustering_labels(self, out_dir):
        """
        Check whether the laoded clustering label file is including clustering results for all sessions.
        This function is used for inference mode of MSDD.

        Args:
            out_dir (str):
                Path to the directory where clustering result files are saved.

        Returns:
            file_exists (bool):
                Boolean that indicates whether clustering result file exists.
            clus_label_path (str):
                Path to the clustering label output file.
        """
        clus_label_path = os.path.join(
            out_dir, 'speaker_outputs', f'subsegments_scale{self.base_scale_index}_cluster.label'
        )
        file_exists = os.path.exists(clus_label_path)
        if not file_exists:
            logging.info(f"Clustering label file {clus_label_path} does not exist.")
        return file_exists, clus_label_path

    def load_clustering_labels(self, out_dir):
        """
        Load clustering labels generated by clustering diarizer. This function is used for inference mode of MSDD.

        Args:
            out_dir (str):
                Path to the directory where clustering result files are saved.
        Returns:
            emb_scale_seq_dict (dict):
                List containing clustering results in string format.
        """
        file_exists, clus_label_path = self.check_clustering_labels(out_dir)
        logging.info(f"Loading cluster label file from {clus_label_path}")
        with open(clus_label_path) as f:
            clus_labels = f.readlines()
        return clus_labels

    def load_emb_scale_seq_dict(self, out_dir):
        """
        Load saved embeddings generated by clustering diarizer. This function is used for inference mode of MSDD.

        Args:
            out_dir (str):
                Path to the directory where embedding pickle files are saved.
        Returns:
            emb_scale_seq_dict (dict):
                Dictionary containing embedding tensors which are indexed by scale numbers.
        """
        window_len_list = list(self.cfg_base.diarizer.speaker_embeddings.parameters.window_length_in_sec)
        emb_scale_seq_dict = {scale_index: None for scale_index in range(len(window_len_list))}
        for scale_index in range(len(window_len_list)):
            pickle_path = os.path.join(
                out_dir, 'speaker_outputs', 'embeddings', f'subsegments_scale{scale_index}_embeddings.pkl'
            )
            logging.info(f"Loading embedding pickle file of scale:{scale_index} at {pickle_path}")
            with open(pickle_path, "rb") as input_file:
                emb_dict = pkl.load(input_file)
            for key, val in emb_dict.items():
                emb_dict[key] = val
            emb_scale_seq_dict[scale_index] = emb_dict
        return emb_scale_seq_dict


class EncDecDiarLabelModel(ModelPT, ExportableEncDecModel, ClusterEmbedding):
    """
    Encoder decoder class for multiscale diarization decoder (MSDD). Model class creates training, validation methods for setting
    up data performing model forward pass.

    This model class expects config dict for:
        * preprocessor
        * msdd_model

    Since MSDD requires initializing clustering result, this model class needs to inherit from `ClusterEmbedding` class to reuse some of
    the embedding post processing functions.
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
        """
        Initialize an MSDD model and the specified speaker embedding model. In this init function, training and validation datasets are prepared.
        """
        self.trainer = trainer
        self.pairwise_infer = False
        self.cfg_msdd_model = cfg
        self.cfg_msdd_model.msdd_module.num_spks = self.cfg_msdd_model.max_num_of_spks

        self.cfg_msdd_model.train_ds.num_spks = self.cfg_msdd_model.max_num_of_spks
        self.cfg_msdd_model.validation_ds.num_spks = self.cfg_msdd_model.max_num_of_spks
        ClusterEmbedding.__init__(self, cfg_base=self.cfg_msdd_model.base, cfg_msdd_model=self.cfg_msdd_model)
        if trainer:
            self._init_segmentation_info()
            self.prepare_splits()
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
        self.min_detached_embs = 2
        self._accuracy_test = MultiBinaryAccuracy()
        self._accuracy_train = MultiBinaryAccuracy()
        self._accuracy_valid = MultiBinaryAccuracy()

    def _init_segmentation_info(self):
        """Initialize segmentation settings: window, shift and multiscale weights.
        """
        self._diarizer_params = self.cfg_msdd_model.base.diarizer
        self.multiscale_args_dict = parse_scale_configs(
            self._diarizer_params.speaker_embeddings.parameters.window_length_in_sec,
            self._diarizer_params.speaker_embeddings.parameters.shift_length_in_sec,
            self._diarizer_params.speaker_embeddings.parameters.multiscale_weights,
        )

    def _init_speaker_model(self):
        """
        Initialize speaker embedding model with model name or path passed through config. Note that speaker embedding model is loaded to
        `self.msdd` to enable multi-gpu and multi-node training. In addition, speaker embedding model is also saved with msdd model when
        `.ckpt` files are saved.
        """
        model_path = self.cfg_msdd_model.base.diarizer.speaker_embeddings.model_path
        self._diarizer_params = self.cfg_msdd_model.base.diarizer
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
            logging.info("Loading pretrained {} model from NGC".format(model_path))
            self._speaker_model = EncDecSpeakerLabelModel.from_pretrained(model_name=model_path)
        self._speaker_params = self.cfg_msdd_model.base.diarizer.speaker_embeddings.parameters

    def get_emb_clus_infer(self, cluster_embeddings):
        """Assign dictionaries containing the clustering results from the class instance `cluster_embeddings`.
        """
        self.emb_sess_test_dict = cluster_embeddings.emb_sess_test_dict
        self.clus_test_label_dict = cluster_embeddings.clus_test_label_dict
        self.emb_seq_test = cluster_embeddings.emb_seq_test

    def prepare_splits(self):
        """Prepare training and validation set and save the split timestamp dictionaries.
        """
        self.train_multiscale_timestamp_dict = self.prepare_split_data(
            self.cfg_msdd_model.train_ds.manifest_filepath, self.cfg_msdd_model.train_ds.emb_dir,
        )
        self.validation_multiscale_timestamp_dict = self.prepare_split_data(
            self.cfg_msdd_model.validation_ds.manifest_filepath, self.cfg_msdd_model.validation_ds.emb_dir,
        )

    def prepare_split_data(self, manifest_filepath, _out_dir):
        """
        Prepare multiscale timestamp data for training. Oracle VAD timestamps from RTTM files are used as VAD timestamps.
        In this function, timestamps for embedding extraction are extracted without extracting the embedding vectors.

        Args:
            manifest_filepath (str):
                Input manifest file for creating audio-to-RTTM mapping.
            _out_dir (str):
                Output directory where timestamp json files are saved.

        Returns:
            multiscale_args_dict (dict):
                - Dictionary containing two types of arguments: multi-scale weights and subsegment timestamps for each data sample.
                - Each data sample has two keys: `multiscale_weights` and `scale_dict`.
                    - `multiscale_weights` key contains a list containing multiscale weights.
                    - `scale_dict` is indexed by integer keys which are scale index.
                - Each data sample is indexed by using the following naming convention: `<uniq_id>_<start time in ms>_<end time in ms>`
                    Example: `fe_03_00106_mixed_626310_642300`
        """
        self._speaker_dir = os.path.join(_out_dir, 'speaker_outputs')
        # Only if this is for the first run of modelPT instance, remove temp folders.
        if self.trainer.global_rank == 0:
            if os.path.exists(self._speaker_dir):
                shutil.rmtree(self._speaker_dir)
            os.makedirs(self._speaker_dir)
        split_audio_rttm_map = audio_rttm_map(manifest_filepath, attatch_dur=True)
        out_rttm_dir = os.path.join(_out_dir, 'pred_rttms')
        os.makedirs(out_rttm_dir, exist_ok=True)

        # Speech Activity Detection part
        _speaker_manifest_path = os.path.join(self._speaker_dir, f'oracle_vad_manifest.json')
        logging.info(f"Extracting oracle VAD timestamps and saving at {self._speaker_dir}")
        if not os.path.exists(_speaker_manifest_path):
            write_rttm2manifest(split_audio_rttm_map, _speaker_manifest_path, include_uniq_id=True)

        multiscale_and_timestamps = {}

        # Segmentation
        for scale_idx, (window, shift) in self.multiscale_args_dict['scale_dict'].items():
            subsegments_manifest_path = os.path.join(self._speaker_dir, f'subsegments_scale{scale_idx}.json')
            if not os.path.exists(subsegments_manifest_path):
                # Segmentation for the current scale (scale_idx)
                self.run_segmentation(
                    window, shift, self._speaker_dir, _speaker_manifest_path, scale_tag=f'_scale{scale_idx}'
                )
            multiscale_timestamps = self._extract_timestamps(subsegments_manifest_path)
            multiscale_and_timestamps[scale_idx] = multiscale_timestamps

        multiscale_timestamps_dict = self.get_timestamps(multiscale_and_timestamps, self.multiscale_args_dict)
        return multiscale_timestamps_dict

    def _extract_timestamps(self, manifest_file: str):
        """
        This method extracts speaker embeddings from segments passed through manifest_file. Optionally you may save the
        intermediate speaker embeddings for debugging or any use.

        Args:
            manifest_file (str):
                Manifest file containing segmentation information.
        Returns:
            time_stamps (dict):
                Dictionary containing lists of timestamps.
        """
        logging.info(f"Extracting timestamps from {manifest_file} for multiscale subsegmentation.")
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
        The embeddings and timestamps in multiscale_embeddings_and_timestamps dictionary are indexed by scale index.
        This function rearranges the extracted speaker embedding and timestamps by unique ID to make the further processing more convenient.

        Args:
            multiscale_embeddings_and_timestamps (dict):
                Dictionary of timestamps for each scale.
            multiscale_args_dict (dict):
                Dictionary of scale information: window, shift and multiscale weights.

        Returns:
            timestamps_dict (dict)
                A dictionary containing embeddings and timestamps of each scale, indexed by unique ID.
        """
        timestamps_dict = {uniq_id: {'scale_dict': {}} for uniq_id in multiscale_and_timestamps[0].keys()}
        for scale_idx in sorted(multiscale_args_dict['scale_dict'].keys()):
            time_stamps = multiscale_and_timestamps[scale_idx]
            for uniq_id in time_stamps.keys():
                timestamps_dict[uniq_id]['scale_dict'][scale_idx] = {
                    'time_stamps': time_stamps[uniq_id],
                }

        return timestamps_dict

    def __setup_dataloader_from_config(self, config: DictConfig, multiscale_timestamp_dict):
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
        self, config: DictConfig, emb_dict: dict, emb_seq: dict, clus_label_dict: dict, pairwise_infer=False
    ):
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
            use_single_scale_clus=False,
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
        """
        Perform segmenation without extracting embedding vectors.

        Args:
            window (float):
                Window length (scale length) for segmentation.
            shift (float):
                Shift (hop length) for segmentation.
            _speaker_dir (str):
                Path to the directory that contains segmentation and embedding extraction results.
            _speaker_manifest_path (str):
                Path to the input manifest file.
            scale_tag (str):
                String variable for indicating the scale index.

        Returns:
            subsegments_manifest_path (str):
                Path to the output subsegment _manifest file.
        """
        subsegments_manifest_path = os.path.join(_speaker_dir, f'subsegments{scale_tag}.json')
        logging.info(
            f"Subsegmentation for timestamp extraction:{scale_tag.replace('_',' ')}, {subsegments_manifest_path}"
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

    def get_ms_emb_seq(
        self, embs: torch.Tensor, scale_mapping: torch.Tensor, ms_seg_counts: torch.Tensor
    ) -> torch.Tensor:
        """
        Reshape the given tensor and organize the embedding sequence based on the original sequence counts.
        Repeat the embeddings according to the scale_mapping information so that the final embedding sequence has
        the identical length for all scales.

        Args:
            embs (Tensor):
                Merged embeddings without zero-padding in the batch. See `ms_seg_counts` for details.
                Shape: (Total number of segments in the batch, emb_dim)
            scale_mapping (Tensor):
		The element at the m-th row and the n-th column of the scale mapping matrix indicates the (m+1)-th scale
		segment index which has the closest center distance with (n+1)-th segment in the base scale.
		Example:
		    scale_mapping_argmat[2][101] = 85
		In the above example, it means that 86-th segment in the 3rd scale (python index is 2) is mapped with
		102-th segment in the base scale. Thus, the longer segments bound to have more repeating numbers since
		multiple base scale segments (since the base scale has the shortest length) fall into the range of the
		longer segments. At the same time, each row contains N numbers of indices where N is number of
		segments in the base-scale (i.e., the finest scale).
                Shape: (batch_size, scale_n, self.diar_window_length)
            ms_seg_counts (Tensor)
                Cumulative sum of the number of segments in each scale. This information is needed to reconstruct
                the multi-scale input matrix during forward propagating.

		Example: `batch_size=3, scale_n=6, emb_dim=192`
                    ms_seg_counts =
                     [[8,  9, 12, 16, 25, 51],
                      [11, 13, 14, 17, 25, 51],
                      [ 9,  9, 11, 16, 23, 50]]

		In this function, `ms_seg_counts` is used to get the actual length of each embedding sequence without
		zero-padding.

        Returns:
            ms_emb_seq (Tensor):
	        Multi-scale embedding sequence that is mapped, matched and repeated. The longer scales are less repeated,
                while shorter scales are more frequently repeated following the scale mapping tensor.
        """
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
    def get_cluster_avg_embs_model(
        self, embs: torch.Tensor, clus_label_index: torch.Tensor, ms_seg_counts: torch.Tensor, scale_mapping
    ):
        """
        Calculate the cluster-average speaker embedding based on the ground-truth speaker labels (i.e., cluster labels).

        Args:
            embs (Tensor):
                Merged embeddings without zero-padding in the batch. See `ms_seg_counts` for details.
                Shape: (Total number of segments in the batch, emb_dim)
            clus_label_index (Tensor):
                Merged ground-truth cluster labels from all scales with zero-padding. Each scale's index can be
                retrieved by using segment index in `ms_seg_counts`.
                Shape: (batch_size, maximum total segment count among the samples in the batch)
            ms_seg_counts
                Cumulative sum of the number of segments in each scale. This information is needed to reconstruct
                multi-scale input tensors during forward propagating.

                Example: `batch_size=3, scale_n=6, emb_dim=192`
                    ms_seg_counts =
                     [[8,  9, 12, 16, 25, 51],
                      [11, 13, 14, 17, 25, 51],
                      [ 9,  9, 11, 16, 23, 50]]
                    Counts of merged segments: (121, 131, 118)
                    embs has shape of (370, 192)
                    clus_label_index has shape of (3, 131)

                Shape: (batch_size, scale_n)

        Returns:
            ms_avg_embs (Tensor):
                Multi-scale cluster-average speaker embedding vectors. These embedding vectors are used as reference for
                each speaker to predict the speaker label for the given multi-scale embedding sequences.
                Shape: (batch_size, scale_n, emb_dim, self.num_spks_per_model)
        """
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
        assert (
            ms_avg_embs.requires_grad == False
        ), "ms_avg_embs.requires_grad = True. ms_avg_embs should be detached from the torch graph."
        return ms_avg_embs

    @torch.no_grad()
    def get_ms_mel_feat(
        self,
        processed_signal: torch.Tensor,
        processed_signal_len: torch.Tensor,
        ms_seg_timestamps: torch.Tensor,
        ms_seg_counts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load acoustic feature from audio segments for each scale and save it into a torch.tensor matrix.
        In addition, create variables containing the information of the multiscale subsegmentation information.

        Args:
            processed_signal: (Tensor)
                Zero-padded Feature input.
                Shape: (batch_size, feat_dim, the longest feature sequence length)
            processed_signal_len: (Tensor)
                The actual legnth of feature input without zero-padding.
                Shape: (batch_size,)
            ms_seg_timestamps: (Tensor)
                Timestamps of the base-scale segments.
                Shape: (batch_size, scale_n, number of base-scale segments, self.num_spks_per_model)
            ms_seg_counts: (Tensor)
                Cumulative sum of the number of segments in each scale. This information is needed to reconstruct
                the multi-scale input matrix during forward propagating.
                Shape: (batch_size, scale_n)

        Returns:
            ms_mel_feat: (Tensor)
                Feature input stream split into the same length.
                Shape: (total number of segments, feat_dim, self.frame_per_sec * the-longest-scale-length)
            ms_mel_feat_len: (Tensor)
                The actual length of feature without zero-padding.
                Shape: (total number of segments,)
            seq_len: (Tensor)
                The length of the input embedding sequences.
                Shape: (total number of segments,)
            detach_ids: (tuple)
                Tuple containing both detached embeding indices and attached embedding indices
        """
        device = torch.cuda.current_device()
        _emb_batch_size = min(self.emb_batch_size, ms_seg_counts.sum().item() - self.min_detached_embs)
        feat_dim = self.preprocessor._cfg.features
        max_sample_count = int(self.multiscale_args_dict["scale_dict"][0][0] * self.frame_per_sec)
        ms_mel_feat_len_list, sequence_lengths_list, ms_mel_feat_list = [], [], []
        total_seg_count = torch.sum(ms_seg_counts)

        batch_size = processed_signal.shape[0]
        for batch_idx in range(batch_size):
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

        # For detached embeddings
        with torch.no_grad():
            self.msdd._speaker_model.eval()
            logits, embs_d = self.msdd._speaker_model.forward_for_export(
                processed_signal=audio_signal[detach_ids[1]], processed_signal_len=audio_signal_len[detach_ids[1]]
            )
            embs = torch.zeros(audio_signal.shape[0], embs_d.shape[1]).cuda()
            embs[detach_ids[1], :] = embs_d.detach()

        # For attached  embeddings
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
        return preds, scale_weights

    def training_step(self, batch: list, batch_idx: int):
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

    def validation_step(self, batch: list, batch_idx: int, dataloader_idx: int = 0):
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

    def multi_validation_epoch_end(self, outputs: list, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        f1_acc = self._accuracy_valid.compute()
        self._accuracy_valid.reset()

        self.log('val_loss', val_loss_mean)
        self.log('val_f1_acc', f1_acc)
        return {
            'val_loss': val_loss_mean,
            'val_f1_acc': f1_acc,
        }

    def multi_test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]], dataloader_idx: int = 0):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        f1_acc = self._accuracy_test.compute()
        self._accuracy_test.reset()
        self.log('val_f1_acc', f1_acc)
        return {
            'test_loss': test_loss_mean,
            'test_f1_acc': f1_acc,
        }

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


class OverlapAwareDiarizer:
    """
    Class for inference based on multiscale diarization decoder (MSDD). MSDD requires initializing clustering results from
    clustering diarizer. Overlap-aware diarizer requires separate RTTM generation and evaluation modules to check the effect of
    overlap detection in speaker diarization.
    """

    def __init__(self, cfg: DictConfig):
        """ """
        self._cfg = cfg

        self._init_msdd_model(cfg)
        self.diar_window_length = cfg.diarizer.msdd_model.parameters.diar_window_length
        self.msdd_model.cfg = self.transfer_diar_params_to_model_params(self.msdd_model, cfg)
        self.manifest_filepath = self.msdd_model.cfg.test_ds.manifest_filepath
        self.AUDIO_RTTM_MAP = audio_rttm_map(self.manifest_filepath)

        # Initialize clustering and embedding preparation instance (as a diarization encoder).
        self.clustering_embedding = ClusterEmbedding(cfg_base=cfg, cfg_msdd_model=self.msdd_model.cfg)
        self.clustering_embedding.spk_emb_state_dict = self.spk_emb_state_dict

        # Parameters for creating diarization results from MSDD outputs.
        self.clustering_max_spks = cfg.diarizer.clustering.parameters.max_num_speakers
        self.overlap_infer_spk_limit = cfg.diarizer.msdd_model.parameters.get(
            'overlap_infer_spk_limit', self.clustering_max_spks
        )
        self.use_clus_as_main = cfg.diarizer.msdd_model.parameters.get('use_clus_as_main', False)
        self.max_overlap_spks = cfg.diarizer.msdd_model.parameters.get('max_overlap_spks', 2)
        self.num_spks_per_model = cfg.diarizer.msdd_model.parameters.get('num_spks_per_model', 2)
        self.use_adaptive_thres = cfg.diarizer.msdd_model.parameters.get('use_adaptive_thres', True)
        self.max_pred_length = cfg.diarizer.msdd_model.parameters.get('max_pred_length', 0)

    def transfer_diar_params_to_model_params(self, msdd_model, cfg):
        """
        Transfer the parameters that are needed for MSDD inference from the diarizer config files to `msdd_model.cfg`.
        """
        msdd_model.cfg.base.diarizer.out_dir = cfg.diarizer.out_dir
        msdd_model.cfg.test_ds.manifest_filepath = cfg.diarizer.manifest_filepath
        msdd_model.cfg.test_ds.emb_dir = cfg.diarizer.out_dir
        msdd_model.cfg.test_ds.batch_size = cfg.diarizer.msdd_model.parameters.infer_batch_size
        msdd_model.cfg_base = cfg
        msdd_model._cfg.base.diarizer.clustering.parameters.max_num_speakers = (
            cfg.diarizer.clustering.parameters.max_num_speakers
        )
        return msdd_model.cfg

    def extract_standalone_model(self, ext: str = '.ckpt') -> str:
        """
        MSDD model file contains speaker embedding model and MSDD model. This function extracts standalone MSDD model and save it to
        disc to separateley load standalone MSDD model. Speaker embedding model weights are saved to `self.spk_emb_state_dict` to be
        loaded separately for clustering diarizer.

        Args:
            ext (str):
                File-name extension of the provided model path.
        Returns:
            standalone_model_path (str):
                Path to the extracted standalone model without speaker embedding extractor model.
        """
        loaded_model = torch.load(self._cfg.diarizer.msdd_model.model_path)
        self.spk_emb_state_dict = {}
        spk_emb_module_names = []
        for name in loaded_model['state_dict'].keys():
            if 'msdd._speaker_model' in name:
                spk_emb_module_names.append(name)

        for name in spk_emb_module_names:
            org_name = name.replace('msdd._speaker_model.', '')
            self.spk_emb_state_dict[org_name] = loaded_model['state_dict'][name]
            del loaded_model['state_dict'][name]

        standalone_model_path = self._cfg.diarizer.msdd_model.model_path.replace(ext, f'.{ext}')
        torch.save(loaded_model, standalone_model_path)
        return standalone_model_path

    def _init_msdd_model(self, cfg: DictConfig):
        """
        Initialized MSDD model with the provided config. Load either from `.nemo` file or `.ckpt` checkpoint files.
        """
        self.device = 'cuda'
        if not torch.cuda.is_available():
            self.device = 'cpu'
            logging.warning(
                "Running model on CPU, for faster performance it is adviced to use atleast one NVIDIA GPUs"
            )

        if cfg.diarizer.msdd_model.model_path.endswith('.nemo'):
            logging.info(f"Using local nemo file from {cfg.diarizer.msdd_model.model_path}")
            msdd_only_model_path = self.extract_standalone_model('.nemo')
            self.msdd_model = EncDecDiarLabelModel.restore_from(restore_path=msdd_only_model_path)
        elif cfg.diarizer.msdd_model.model_path.endswith('.ckpt'):
            logging.info(f"Using local checkpoint from {cfg.diarizer.msdd_model.model_path}")
            msdd_only_model_path = self.extract_standalone_model('.ckpt')
            self.msdd_model = EncDecDiarLabelModel.load_from_checkpoint(checkpoint_path=msdd_only_model_path)

    def get_pred_mat(self, data_list: List[Union[Tuple[int], List[torch.Tensor]]]) -> torch.Tensor:
        """
        This module puts together the pairwise, two-speaker, predicted results to form a finalized matrix that has dimension of
        `(total_len, n_est_spks)`. The pairwise results are evenutally averaged. For example, in 4 speaker case (speaker 1, 2, 3, 4),
        the sum of the pairwise results (1, 2), (1, 3), (1, 4) are then divided by 3 to take average of the sigmoid values.

        Args:
            data_list (list):
                List containing data points from `test_data_collection` variable. `data_list` has sublists `data` as follows:
                data[0]: `target_spks` tuple
                    Examples: (0, 1, 2)
                data[1]: Tensor containing estimaged sigmoid values.
                   [[0.0264, 0.9995],
		    [0.0112, 1.0000],
		    ...,
		    [1.0000, 0.0512]]

        Returns:
            sum_pred (Tensor):
                Tensor containing the averaged sigmoid values for each speaker.
        """
        all_tups = tuple()
        for data in data_list:
            all_tups += data[0]
        n_est_spks = len(set(all_tups))
        digit_map = dict(zip(sorted(set(all_tups)), range(n_est_spks)))
        total_len = max([sess[1].shape[1] for sess in data_list])
        sum_pred = torch.zeros(total_len, n_est_spks)
        for (_dim_tup, pred_mat) in data_list:
            dim_tup = [digit_map[x] for x in _dim_tup]
            if len(pred_mat.shape) == 3:
                pred_mat = pred_mat.squeeze(0)
            if n_est_spks <= self.num_spks_per_model:
                sum_pred = pred_mat
            else:
                _end = pred_mat.shape[0]
                sum_pred[:_end, dim_tup] += pred_mat.cpu().float()
        sum_pred = sum_pred / (n_est_spks - 1)
        return sum_pred

    def get_integrated_preds_list(
        self, uniq_id_list: List[str], test_data_collection: List[Any], preds_list: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Merge multiple sequence inference outputs into a session level result.

        Args:
            uniq_id_list (list):
                List containing `uniq_id` values.
            test_data_collection (collections.DiarizationLabelEntity):
                Class instance that is containing session information such as targeted speaker indices, audio filepaths and RTTM filepaths.
            preds_list (list):
                List containing tensors filled with sigmoid values.

        Returns:
            output_list (list):
                List containing session-level estimated prediction matrix.
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
        Launch diarization pipeline which starts from VAD (or a oracle VAD stamp generation), initialization clustering and multiscale diarization decoder (MSDD).
        Note that the result of MSDD can include multiple speakers at the same time. Therefore, RTTM output of MSDD needs to be based on `make_rttm_with_overlap()`
        function that can generate overlapping timestamps. `self.run_overlap_aware_eval()` function performs DER evaluation.
        """
        torch.set_grad_enabled(False)
        self.clustering_embedding.prepare_cluster_embs_infer()
        self.msdd_model.pairwise_infer = True
        self.msdd_model.get_emb_clus_infer(self.clustering_embedding)
        preds_list, targets_list, signal_lengths_list = self.run_pairwise_diarization()
        for threshold in list(self._cfg.diarizer.msdd_model.parameters.sigmoid_threshold):
            self.run_overlap_aware_eval(preds_list, threshold)

    def get_range_average(
        self, signals: torch.Tensor, emb_vectors: torch.Tensor, diar_window_index: int, test_data_collection: List[Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        This function is only used when `split_infer=True`. This module calculates cluster-average embeddings for the given short range.
        The range length is set by `self.diar_window_length`, and each cluster-average is only calculated for the specified range.
        Note that if the specified range does not contain some speakers (e.g. the range contains speaker 1, 3) compared to the global speaker sets
        (e.g. speaker 1, 2, 3, 4) then the missing speakers (e.g. speakers 2, 4) are assigned with zero-filled cluster-average speaker embedding.

        Args:
            signals (Tensor):
                Zero-padded Input multi-scale embedding sequences.
                Shape: (length, scale_n, emb_vectors, emb_dim)
            emb_vectors (Tensor):
                Cluster-average multi-scale embedding vectors.
                Shape: (length, scale_n, emb_vectors, emb_dim)
            diar_window_index (int):
                Index of split diarization wondows.
            test_data_collection (collections.DiarizationLabelEntity)
                Class instance that is containing session information such as targeted speaker indices, audio filepath and RTTM filepath.

        Returns:
            return emb_vectors_split (Tensor):
                Cluster-average speaker embedding vectors for each scale.
            emb_seq (Tensor):
                Zero-padded multi-scale embedding sequences.
            seq_len (int):
                Length of the sequence determined by `self.diar_window_length` variable.
        """
        emb_vectors_split = torch.zeros_like(emb_vectors)
        uniq_id = os.path.splitext(os.path.basename(test_data_collection.audio_file))[0]
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

    def get_range_clus_avg_emb(
        self, test_batch: List[torch.Tensor], _test_data_collection: List[Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This function is only used when `get_range_average` function is called. This module calculates cluster-average embeddings for
        the given short range. The range length is set by `self.diar_window_length`, and each cluster-average is only calculated for the specified range.

        Args:
            test_batch: (list)
                List containing embedding sequences, length of embedding sequences, ground truth labels (if exists) and initializing embedding vectors.
            test_data_collection: (list)
                List containing test-set dataloader contents. test_data_collection includes wav file path, RTTM file path, clustered speaker indices.

        Returns:
            sess_emb_vectors (Tensor):
                Tensor of cluster-average speaker embedding vectors.
                Shape: (batch_size, scale_n, emb_dim, 2*num_of_spks)
            sess_emb_seq (Tensor):
                Tensor of input multi-scale embedding sequences.
                Shape: (batch_size, length, scale_n, emb_dim)
            sess_sig_lengths (Tensor):
                Tensor of the actucal sequence length without zero-padding.
                Shape: (batch_size)
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

    def diar_infer(
        self, test_batch: List[torch.Tensor], test_data_collection: List[Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
                Tensor containing predicted values which are generated from MSDD model.
            targets (Tensor):
                Tensor containing binary ground-truth values.
            signal_lengths (Tensor):
                The actual Session length (number of steps = number of base-scale segments) without zero padding.
        """
        signals, signal_lengths, _targets, emb_vectors = test_batch
        if self._cfg.diarizer.msdd_model.parameters.split_infer:
            split_count = torch.ceil(torch.tensor(signals.shape[1] / self.diar_window_length)).int()
            sess_emb_vectors, sess_emb_seq, sess_sig_lengths = self.get_range_clus_avg_emb(
                test_batch, test_data_collection,
            )
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
            with autocast():
                _preds, scale_weights = self.msdd_model.forward_infer(
                    input_signal=signals, input_signal_length=signal_lengths, emb_vectors=emb_vectors, targets=_targets
                )
        self.max_pred_length = max(_preds.shape[1], self.max_pred_length)
        preds = torch.zeros(_preds.shape[0], self.max_pred_length, _preds.shape[2])
        targets = torch.zeros(_preds.shape[0], self.max_pred_length, _preds.shape[2])
        preds[:, : _preds.shape[1], :] = _preds
        return preds, targets, signal_lengths

    def run_pairwise_diarization(
        self, embedding_dir: str = './', device: str = 'cuda'
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Setup the parameters needed for batch inference and run batch inference. Note that each sample is pairwise speaker input.
        The pairwise inference results are reconstructed to make session-wise prediction results.

        Args:
            embedding_dir: (str)
                Path to the embedding directory in string format.
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
            f1_score, simple_acc = self.msdd_model.compute_accuracies()
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
        logging.info(
            f"     [Threshold: {threshold:.4f}] [use_clus_as_main={self.use_clus_as_main}] [diar_window={self.diar_window_length}]"
        )
        for k, (collar, ignore_overlap) in enumerate([(0.25, True), (0.25, False), (0.0, False)]):
            all_reference, all_hypothesis = make_rttm_with_overlap(
                self.manifest_filepath,
                self.msdd_model.clus_test_label_dict,
                preds_list,
                threshold=threshold,
                infer_overlap=(not ignore_overlap),
                use_clus_as_main=self.use_clus_as_main,
                overlap_infer_spk_limit=self.overlap_infer_spk_limit,
                use_adaptive_thres=self.use_adaptive_thres,
                max_overlap_spks=self.max_overlap_spks,
            )
            score_labels(
                self.AUDIO_RTTM_MAP, all_reference, all_hypothesis, collar=collar, ignore_overlap=ignore_overlap,
            )
        logging.info(f"  \n")

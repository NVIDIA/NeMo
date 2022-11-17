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

import os
import time
from copy import deepcopy

import numpy as np
import torch
from omegaconf import DictConfig

from nemo.collections.asr.models import ClusteringDiarizer
from nemo.collections.asr.parts.utils.nmesc_clustering import (
    OnlineSpeakerClustering,
    getTempInterpolMultiScaleCosAffinityMatrix,
    split_input_data,
)
from nemo.collections.asr.parts.utils.speaker_utils import (
    OnlineSegmentor,
    audio_rttm_map,
    generate_cluster_labels,
    get_embs_and_timestamps,
)
from nemo.utils import logging, model_utils


__all__ = ['OnlineDiarizer']

def timeit(method):
    """
    Monitor elapsed time of the corresponding function displaying the method name.
    """
    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()
        if 'log_time' in kwargs:
            name = kwargs.get('log_name', method.__name__.upper())
            kwargs['log_time'][name] = int((te - ts) * 1000)
        else:
            logging.info('%2.2fms %r' % ((te - ts) * 1000, method.__name__))
        return result

    return timed

class OnlineDiarizer(ClusteringDiarizer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.cfg = model_utils.convert_model_config_to_dict_config(cfg)
        self._diarizer_params = self.cfg.diarizer
        self.base_scale_index = max(self.multiscale_args_dict['scale_dict'].keys())

        self.uniq_id = None
        self.decimals = 2
        self.AUDIO_RTTM_MAP = audio_rttm_map(self.cfg.diarizer.manifest_filepath)
        self.sample_rate = self.cfg.sample_rate
        self._cfg_diarizer = self.cfg.diarizer
        torch.manual_seed(0)

        self._out_dir = self._cfg_diarizer.out_dir
        if not os.path.exists(self._out_dir):
            os.mkdir(self._out_dir)

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.reset()

        # Initialize an online segmentor module
        self.online_segmentor = OnlineSegmentor(self.cfg.sample_rate)
        # Set speaker embedding model in eval mode
        self._speaker_model.eval()

    def _init_online_clustering_module(self, clustering_params):
        """Initialize online speaker clustering module
        """
        self.online_clus = OnlineSpeakerClustering(
            max_num_speakers=clustering_params.max_num_speakers,
            max_rp_threshold=clustering_params.max_rp_threshold,
            sparse_search_volume=clustering_params.sparse_search_volume,
            history_buffer_size=clustering_params.history_buffer_size,
            current_buffer_size=clustering_params.current_buffer_size,
        )
        self.history_n = clustering_params.history_buffer_size
        self.current_n = clustering_params.current_buffer_size

        self.max_num_speakers = self.online_clus.max_num_speakers

    def _init_online_segmentor_module(self, sample_rate):
        """Initialize an online segmentor module
        """
        self.online_segmentor = OnlineSegmentor(sample_rate)

    def _init_memory_buffer(self):
        """These variables are kept for the future updates
        """
        self.memory_margin = 0
        self.memory_segment_ranges = {key: [] for key in self.multiscale_args_dict['scale_dict'].keys()}
        self.memory_segment_indexes = {key: [] for key in self.multiscale_args_dict['scale_dict'].keys()}
        self.memory_cluster_labels = np.array([])
        self.cumulative_speech_labels = []

    def _init_temporal_major_voting_module(self):
        """Variables needed for taking majority votes for speaker labels
        """
        self.use_temporal_label_major_vote = False
        self.temporal_label_major_vote_buffer_size = 11
        self.base_scale_label_dict = {}

    def _init_segment_variables(self):
        self.embs_array = {self.uniq_id: {}}
        self.time_stamps = {self.uniq_id: {}}
        self.segment_range_ts = {self.uniq_id: {}}
        self.segment_raw_audio = {self.uniq_id: {}}
        self.segment_indexes = {self.uniq_id: {}}

        for scale_idx, (window, shift) in self.multiscale_args_dict['scale_dict'].items():
            self.multiscale_embeddings_and_timestamps[scale_idx] = [None, None]
            self.embs_array[scale_idx] = None
            self.time_stamps[scale_idx] = []
            self.segment_range_ts[scale_idx] = []
            self.segment_raw_audio[scale_idx] = []
            self.segment_indexes[scale_idx] = []

    def _init_buffer_frame_timestamps(self):
        """Timing variables trasferred from OnlineDiarWithASR class
        """
        self.frame_index = 0
        self.frame_start = 0.0
        self.buffer_start = 0.0
        self.buffer_end = 0.0

    def _transfer_timestamps_to_segmentor(self):
        """Pass the timing information from streaming ASR buffers
        """
        self.online_segmentor.frame_start = self.frame_start
        self.online_segmentor.buffer_start = self.buffer_start
        self.online_segmentor.buffer_end = self.buffer_end

    def reset(self):
        """Reset all the necessary variables and initialize classes.
        """
        self.n_embed_seg_len = int(
            self.sample_rate * self.multiscale_args_dict['scale_dict'][self.base_scale_index][0]
        )
        self._init_segment_variables()
        self._init_online_clustering_module(self._cfg_diarizer.clustering.parameters)
        self._init_online_segmentor_module(self.cfg.sample_rate)
        self._init_memory_buffer()
        self._init_temporal_major_voting_module()
        self._init_buffer_frame_timestamps()

    def _clear_memory(self, scale_idx):
        """
        Calculate how many segments should be removed from memory.
        """
        base_scale_shift = self.multiscale_args_dict['scale_dict'][self.base_scale_index][1]
        self.memory_margin = int((self.buffer_end - self.buffer_start) / base_scale_shift)

        scale_buffer_size = int(
            len(set(self.scale_mapping_dict[scale_idx].tolist()))
            / len(set(self.scale_mapping_dict[self.base_scale_index].tolist()))
            * (self.history_n + self.current_n)
        )
        keep_range = scale_buffer_size + self.memory_margin
        self.embs_array[scale_idx] = self.embs_array[scale_idx][-keep_range:]
        self.segment_raw_audio[scale_idx] = self.segment_raw_audio[scale_idx][-keep_range:]
        self.segment_range_ts[scale_idx] = self.segment_range_ts[scale_idx][-keep_range:]
        self.segment_indexes[scale_idx] = self.segment_indexes[scale_idx][-keep_range:]

    @timeit
    def _temporal_label_major_vote(self):
        """
        Take a majority voting for every segment on temporal steps. This feature significantly reduces the error coming
        from unstable speaker counting in the beginning of sessions.

        Args:

        Returns:
            maj_vote_labels (list):
                List containing the major-voted speaker labels on temporal domain
        """
        maj_vote_labels = []
        for seg_idx in self.memory_segment_indexes[self.base_scale_index]:
            if seg_idx not in self.base_scale_label_dict:
                self.base_scale_label_dict[seg_idx] = [self.memory_cluster_labels[seg_idx]]
            else:
                while len(self.base_scale_label_dict[seg_idx]) > self.temporal_label_major_vote_buffer_size:
                    self.base_scale_label_dict[seg_idx].pop(0)
                self.base_scale_label_dict[seg_idx].append(self.memory_cluster_labels[seg_idx])

            maj_vote_labels.append(torch.mode(torch.tensor(self.base_scale_label_dict[seg_idx]))[0].item())
        return maj_vote_labels

    def save_history_data(
        self, 
        scale_idx: int, 
        total_cluster_labels: torch.Tensor, 
        isOnline: bool,
    ):
        """
        Save the temporary input to the class memory buffer.
        - Clustering is done for (hist_N + curr_N) number of embeddings.
        - Thus, we need to remove the clustering results on the embedding memory.
        - If self.diar.history_buffer_seg_end is not None, that indicates streaming diarization system
          is starting to save embeddings to its memory. Thus, the new incoming clustering label should be separated.
        - If `isOnline = True`, old embeddings outside the window are removed to save GPU memory.

        Args:

        Returns:
        """
        total_cluster_labels = total_cluster_labels.tolist()

        if not isOnline:
            self.memory_segment_ranges[scale_idx] = deepcopy(self.segment_range_ts[scale_idx])
            self.memory_segment_indexes[scale_idx] = deepcopy(self.segment_indexes[scale_idx])
            if scale_idx == self.base_scale_index:
                self.memory_cluster_labels = deepcopy(total_cluster_labels)

        # Only if there are newly obtained embeddings, update ranges and embeddings.
        elif self.segment_indexes[scale_idx][-1] > self.memory_segment_indexes[scale_idx][-1]:
            global_idx = max(self.memory_segment_indexes[scale_idx]) - self.memory_margin

            # convert global index global_idx to buffer index buffer_idx
            segment_indexes_mat = torch.tensor(self.segment_indexes[scale_idx])
            buffer_idx = torch.where(segment_indexes_mat == global_idx)[0][0]

            self.memory_segment_ranges[scale_idx][global_idx:] = deepcopy(
                self.segment_range_ts[scale_idx][buffer_idx:]
            )
            self.memory_segment_indexes[scale_idx][global_idx:] = deepcopy(
                self.segment_indexes[scale_idx][buffer_idx:]
            )
            if scale_idx == self.base_scale_index:
                self.memory_cluster_labels[global_idx:] = deepcopy(total_cluster_labels[global_idx:])
                assert len(self.memory_cluster_labels) == len(self.memory_segment_ranges[scale_idx])

            # Remove unnecessary old values
            self._clear_memory(scale_idx)

        assert (
            len(self.embs_array[scale_idx])
            == len(self.segment_raw_audio[scale_idx])
            == len(self.segment_indexes[scale_idx])
            == len(self.segment_range_ts[scale_idx])
        )

        if self.use_temporal_label_major_vote:
            cluster_label_hyp = self._temporal_label_major_vote()
        else:
            cluster_label_hyp = self.memory_cluster_labels
        return cluster_label_hyp

    @timeit
    @torch.no_grad()
    def _run_embedding_extractor(self, audio_signal):
        """
        Call `forward` function of the speaker embedding model.

        Args:

        Returns:
        """
        audio_signal = torch.stack(audio_signal).float().to(self.device)
        audio_signal_lens = torch.tensor([self.n_embed_seg_len for k in range(audio_signal.shape[0])]).to(self.device)
        _, torch_embs = self._speaker_model.forward(input_signal=audio_signal, input_signal_length=audio_signal_lens)
        return torch_embs

    @timeit
    def _extract_embeddings(self, audio_signal, segment_ranges, indexes, embeddings):
        """
        Extract speaker embeddings based on audio_signal and segment_ranges varialbes. Unlike offline speaker diarization,
        speaker embedding and subsegment ranges are not saved on the disk.

        Args:
            embeddings (Tensor):
            audio_signal (Tensor):
            segment_ranges(Tensor):
        Returns:
            embeddings (Tensor):
        """
        stt_idx = 0 if embeddings is None else embeddings.shape[0]
        end_idx = len(segment_ranges)

        if end_idx > stt_idx:
            torch_embs = self._run_embedding_extractor(audio_signal[stt_idx:end_idx])
            if embeddings is None:
                embeddings = torch_embs
            else:
                embeddings = torch.vstack((embeddings[:stt_idx, :], torch_embs))
        elif end_idx < stt_idx:
            embeddings = embeddings[: len(segment_ranges)]

        if len(segment_ranges) != embeddings.shape[0]:
            raise ValueError("Segment ranges and embeddings shapes do not match.")
        return embeddings

    @timeit
    def _perform_online_clustering(
        self, uniq_embs_and_timestamps, oracle_num_speakers=None, cuda=False,
    ):
        device = torch.device("cuda") if cuda else torch.device("cpu")

        # Get base-scale (the highest index) information from uniq_embs_and_timestamps.
        embeddings_in_scales, timestamps_in_scales = split_input_data(
            embeddings_in_scales=uniq_embs_and_timestamps['embeddings'],
            timestamps_in_scales=uniq_embs_and_timestamps['timestamps'],
            multiscale_segment_counts=uniq_embs_and_timestamps['multiscale_segment_counts'],
        )

        curr_emb, self.scale_mapping_dict = getTempInterpolMultiScaleCosAffinityMatrix(
            multiscale_weights=uniq_embs_and_timestamps['multiscale_weights'],
            embeddings_in_scales=embeddings_in_scales,
            timestamps_in_scales=timestamps_in_scales,
            device=device,
        )

        concat_emb, add_new = self.online_clus.get_reduced_mat(
            emb=curr_emb, base_segment_indexes=self.segment_indexes[self.base_scale_index]
        )

        Y_concat = self.online_clus.forward(emb=concat_emb, frame_index=self.frame_index, cuda=True, device=device,)

        merged_clus_labels = self.online_clus.macth_labels(Y_concat, add_new, self.online_clus.isOnline)

        for scale_idx, (window, shift) in self.multiscale_args_dict['scale_dict'].items():
            cluster_label_hyp = self.save_history_data(scale_idx, merged_clus_labels, self.online_clus.isOnline)

        return cluster_label_hyp

    def _get_interim_output(self):
        """
        In case buffer is not filled or there is no speech activity in the input, generate temporary output.

        Returns:
            (Tensor): Speaker labels based on the previously saved segments and speaker labels 
        """
        if len(self.memory_cluster_labels) == 0 or self.buffer_start < 0:
            return generate_cluster_labels([[0.0, self.total_buffer_in_secs]], [0])
        else:
            return generate_cluster_labels(
                self.memory_segment_ranges[self.base_scale_index], self.memory_cluster_labels
            )

    @timeit
    def diarize_step(
        self, 
        audio_buffer: torch.Tensor, 
        vad_timestamps: torch.Tensor):
        """
        A function for a unit diarization step. A diarization step goes through the following steps:
        
        1. Segmentation:
            Using `OnlineSegmentor` class, call `run_online_segmentation` methoed to get the segmentations.
        2. Embedding Extraction:
            Extract multiscale embeddings from the extracted speech segments.
        3. Online Clustering & Counting
            Perform online speaker clustering by using `OnlineSpeakerClustering` class.
        4. Generate speaker labels:
            Generate start and end timestamps of speaker labels based on the diarization results.

        c.f.) Also see method `diarize` in `ClusteringDiarizer` class.

        Args:
            audio_buffer (Tensor):
                Tensor variable containing the time series signal at the current frame
                Dimensions: (Number of audio time-series samples) x 1
            vad_timestamps (Tensor):
                List containing VAD timestamps.
                Dimensions: (Number of segments) x 2
                Example:
                    >>> vad_timestamps = torch.Tensor([[0.05, 2.52], [3.12, 6.85]])

        Returns:
            diar_hyp (Tensor):
                Speaker label hypothesis from the start of the session to the current position
        """
        self._transfer_timestamps_to_segmentor()
        # In case buffer is not filled or there is no speech activity in the input
        if self.buffer_start < 0 or len(vad_timestamps) == 0:
            return self._get_interim_output()

        # Segmentation: (c.f. see `diarize` function in ClusteringDiarizer class)
        for scale_idx, (window, shift) in self.multiscale_args_dict['scale_dict'].items():

            # Step 1: Get subsegments for embedding extraction.
            audio_sigs, segment_ranges, range_inds = self.online_segmentor.run_online_segmentation(
                audio_buffer=audio_buffer,
                vad_timestamps=vad_timestamps,
                segment_raw_audio=self.segment_raw_audio[scale_idx],
                segment_range_ts=self.segment_range_ts[scale_idx],
                segment_indexes=self.segment_indexes[scale_idx],
                window=window,
                shift=shift,
            )
            self.segment_raw_audio[scale_idx] = audio_sigs
            self.segment_range_ts[scale_idx] = segment_ranges
            self.segment_indexes[scale_idx] = range_inds

            # Step 2-1: Extract speaker embeddings from the extracted subsegment timestamps.
            embeddings = self._extract_embeddings(
                audio_signal=self.segment_raw_audio[scale_idx],
                segment_ranges=self.segment_range_ts[scale_idx],
                indexes=self.segment_indexes[scale_idx],
                embeddings=self.embs_array[scale_idx],
            )

            # Step 2-2:Save the embeddings and segmentation timestamps in memory
            self.embs_array[scale_idx] = embeddings

            self.multiscale_embeddings_and_timestamps[scale_idx] = [
                {self.uniq_id: embeddings},
                {self.uniq_id: segment_ranges},
            ]

        embs_and_timestamps = get_embs_and_timestamps(
            self.multiscale_embeddings_and_timestamps, self.multiscale_args_dict
        )

        # Step 3: Clustering: Perform an online version of clustering algorithm
        cluster_label_hyp = self._perform_online_clustering(embs_and_timestamps[self.uniq_id], cuda=True,)

        # Step 4: Generate RTTM style diarization labels from segment ranges and cluster labels
        diar_hyp = generate_cluster_labels(self.memory_segment_ranges[self.base_scale_index], cluster_label_hyp)
        return diar_hyp

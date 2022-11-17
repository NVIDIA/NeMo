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

import numpy as np
import pytest
import torch

from nemo.collections.asr.parts.utils.nmesc_clustering import (
    SpeakerClustering,
    get_closest_embeddings,
    get_minimal_indices,
    getCosAffinityMatrix,
    merge_emb,
    run_reducer,
    split_input_data,
    stitch_cluster_labels,
)
from nemo.collections.asr.parts.utils.speaker_utils import (
    combine_float_overlaps,
    combine_int_overlaps,
    get_subsegments,
)


def check_range_values(target, source):
    bool_list = []
    for tgt, src in zip(target, source):
        for x, y in zip(src, tgt):
            bool_list.append(abs(x - y) < 1e-6)
    return all(bool_list)


def check_labels(target, source):
    bool_list = []
    for x, y in zip(target, source):
        bool_list.append(abs(x - y) < 1e-6)
    return all(bool_list)


def matrix(mat, use_tensor=True, dtype=torch.long):
    if use_tensor:
        mat = torch.Tensor(mat).to(dtype)
    else:
        mat = np.array(mat)
    return mat


def generate_mock_emb(n_emb_per_spk, perturb_sigma, emb_dim):
    """Generate a set of artificial embedding vectors from random numbers
    """
    return torch.rand(1, emb_dim).repeat(n_emb_per_spk, 1) + perturb_sigma * torch.rand(n_emb_per_spk, emb_dim)


def generate_mock_data(
    n_spks=2,
    spk_dur=3,
    emb_dim=192,
    perturb_sigma=0.01,
    ms_window=[1.5, 1.0, 0.5],
    ms_shift=[0.75, 0.5, 0.25],
    torch_seed=0,
):

    torch.manual_seed(torch_seed)
    spk_timestamps = [(spk_dur * k, spk_dur) for k in range(n_spks)]
    emb_list, seg_list = [], []
    multiscale_segment_counts = [0 for _ in range(len(ms_window))]
    ground_truth = []
    for scale_idx, (window, shift) in enumerate(zip(ms_window, ms_shift)):
        for spk_idx, (offset, dur) in enumerate(spk_timestamps):
            segments = get_subsegments(offset=offset, window=window, shift=shift, duration=dur)
            emb = generate_mock_emb(n_emb_per_spk=len(segments), perturb_sigma=perturb_sigma, emb_dim=emb_dim,)
            seg_list.extend(segments)
            emb_list.append(emb)
            multiscale_segment_counts[scale_idx] += emb.shape[0]

            if scale_idx == len(multiscale_segment_counts) - 1:
                ground_truth.extend([spk_idx] * emb.shape[0])

    emb_tensor = torch.concat(emb_list)
    multiscale_segment_counts = torch.tensor(multiscale_segment_counts)
    segm_tensor = torch.tensor(seg_list)
    multiscale_weights = torch.ones(len(ms_window)).unsqueeze(0)
    ground_truth = torch.tensor(ground_truth)
    return emb_tensor, segm_tensor, multiscale_segment_counts, multiscale_weights, spk_timestamps, ground_truth


@pytest.mark.run_only_on('GPU')
def test_speaker_counting(n_spks=3, total_dur_sec=30, num_speakers=-1, max_num_speakers=5, cuda=True):
    speaker_clustering_python = SpeakerClustering(maj_vote_spk_count=False, cuda=cuda)
    assert isinstance(speaker_clustering_python, SpeakerClustering)
    each_spk_dur = float(total_dur_sec / n_spks)
    em, ts, mc, mw, _, _ = generate_mock_data(n_spks=n_spks, spk_dur=each_spk_dur)
    Y = speaker_clustering_python.forward_infer(
        embeddings_in_scales=em,
        timestamps_in_scales=ts,
        multiscale_segment_counts=mc,
        multiscale_weights=mw,
        oracle_num_speakers=num_speakers,
        max_num_speakers=max_num_speakers,
    )
    return len(set(Y.tolist()))


class TestDiarizationUtilFunctions:
    """
    Tests diarization and speaker-task related utils.
    Test functions include:
        - Segment interval merging function
        - Embedding merging
    """

    @pytest.mark.unit
    def test_combine_float_overlaps(self):
        intervals = [[0.25, 1.7], [1.5, 3.0], [2.8, 5.0], [5.5, 10.0]]
        target = [[0.25, 5.0], [5.5, 10.0]]
        merged = combine_float_overlaps(intervals)
        assert check_range_values(target, merged)

    @pytest.mark.unit
    def test_combine_int_overlaps(self):
        intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]
        target = [[1, 6], [8, 10], [15, 18]]
        merged = combine_int_overlaps(intervals)
        assert check_range_values(target, merged)

    @pytest.mark.unit
    def test_combine_int_overlaps_edge(self):
        intervals = [[1, 4], [4, 5]]
        target = [[1, 5]]
        merged = combine_int_overlaps(intervals)
        assert check_range_values(target, merged)

    @pytest.mark.unit
    def test_minimal_index(self):
        Y = matrix([3, 3, 3, 4, 4, 5])
        min_Y = get_minimal_indices(Y)
        target = matrix([0, 0, 0, 1, 1, 2])
        assert check_labels(target, min_Y)

    @pytest.mark.unit
    def test_stitch_cluster_labels_label_switch(self):
        Y_old = matrix([0, 0, 0, 0, 0, 0])
        Y_new = matrix([0, 0, 0, 0, 0, 0]) + 1
        target = matrix([0, 0, 0, 0, 0, 0])
        result = stitch_cluster_labels(Y_old, Y_new)
        assert check_labels(target, result)

    @pytest.mark.unit
    def test_stitch_cluster_labels_label_many_to_one(self):
        Y_old = matrix([0, 1, 2, 3, 4, 5])
        Y_new = matrix([0, 0, 0, 0, 0, 0])
        target = matrix([0, 0, 0, 0, 0, 0])
        with_history = False
        result = stitch_cluster_labels(Y_old, Y_new)
        assert check_labels(target, result)

    @pytest.mark.unit
    def test_stitch_cluster_labels_label_one_to_many(self):
        Y_old = matrix([0, 0, 0, 0, 0, 0])
        Y_new = matrix([0, 1, 2, 3, 4, 5])
        target = matrix([0, 1, 2, 3, 4, 5])
        with_history = False
        result = stitch_cluster_labels(Y_old, Y_new)
        assert check_labels(target, result)

    @pytest.mark.unit
    def test_stitch_cluster_labels_one_label_replaced(self, N=3):
        Y_old = matrix([0] * N + [1] * N + [2] * N)
        Y_new = matrix([1] * N + [2] * N + [3] * N)
        target = matrix([0, 0, 0, 1, 1, 1, 2, 2, 2])
        result = stitch_cluster_labels(Y_old, Y_new)
        assert check_labels(target, result)

    @pytest.mark.unit
    def test_stitch_cluster_labels_confusion_error(self, N=3):
        Y_old = matrix([0] * N + [1] * (N - 1) + [2] * (N + 1))
        Y_new = matrix([1] * N + [2] * N + [3] * N)
        target = matrix([0, 0, 0, 1, 1, 1, 2, 2, 2])
        result = stitch_cluster_labels(Y_old, Y_new)
        assert check_labels(target, result)

    @pytest.mark.unit
    def test_stitch_cluster_labels_speaker_more_speakers(self, N=3):
        Y_old = matrix([0] * N + [1] * (N - 1) + [2] * (N + 1) + [0, 0, 0])
        Y_new = matrix([1] * N + [0] * N + [2] * N + [4, 5, 6])
        target = matrix([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 4, 5])
        result = stitch_cluster_labels(Y_old, Y_new)
        assert check_labels(target, result)

    @pytest.mark.unit
    def test_stitch_cluster_labels_speaker_longer_sequence(self, N=3):
        Y_old = matrix([0] * N + [1] * N + [2] * N + [0, 0, 0])
        Y_new = matrix([1] * N + [2] * N + [0] * N + [1, 2, 3, 1, 2, 3])
        target = matrix([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 3, 0, 1, 3])
        result = stitch_cluster_labels(Y_old, Y_new)
        assert check_labels(target, result)

    @pytest.mark.unit
    def test_embedding_merger(self):
        em, ts, mc, mw, spk_ts, gt = generate_mock_data(n_spks=2, spk_dur=2, perturb_sigma=0.1)
        em_s, ts_s = split_input_data(em, ts, mc)
        base_scale_idx = len(em_s) - 1
        target_spk = 0
        target_num = 3
        pre_clus_labels = gt
        ndx = torch.where(pre_clus_labels == target_spk)[0]
        pre_embs = em_s[base_scale_idx]
        affinity_mat = getCosAffinityMatrix(pre_embs)
        cmat = torch.tril(affinity_mat[:, ndx][ndx, :])
        # Check the dimension of the selected affinity values
        assert cmat.shape[0] == cmat.shape[1] == torch.sum(pre_clus_labels == target_spk).item()
        index_2d = get_closest_embeddings(cmat, ndx, target_num)
        # Check the most closest affinity value
        assert torch.max(cmat.flatten()) == cmat[index_2d[0, 0], index_2d[1, 0]]
        spk_cluster_labels, emb_ndx = pre_clus_labels[ndx], pre_embs[ndx]
        merged_embs, merged_clus_labels = merge_emb(index_2d, emb_ndx, spk_cluster_labels)
        # Check the number of merged embeddings and labels
        assert (torch.sum(gt == target_spk).item() - target_num) == merged_clus_labels.shape[0]

    @pytest.mark.unit
    def test_embedding_reducer(self):
        em, ts, mc, mw, spk_ts, gt = generate_mock_data(n_spks=2, spk_dur=10)
        em_s, ts_s = split_input_data(em, ts, mc)
        base_scale_idx = len(em_s) - 1
        target_spk = 0
        target_num = 10
        merged_embs, merged_clus_labels = run_reducer(
            pre_embs=em_s[base_scale_idx], target_spk_idx=target_spk, target_num=target_num, pre_clus_labels=gt
        )
        assert (torch.sum(gt == target_spk).item() - target_num) == merged_clus_labels.shape[0]


class TestSpeakerClustering:
    """
    Test speaker clustering module
    Test functions include:
        - script module export
        - speaker counting feature
    """

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_clus_script_export(self):
        exported_filename = 'speaker_clustering_script.pt'
        speaker_clustering_python = SpeakerClustering(maj_vote_spk_count=False, cuda=True)
        speaker_clustering_scripted_source = torch.jit.script(speaker_clustering_python)
        torch.jit.save(speaker_clustering_scripted_source, exported_filename)
        speaker_clustering_scripted = torch.jit.load(exported_filename)
        assert os.path.exists(exported_filename)
        os.remove(exported_filename)
        assert not os.path.exists(exported_filename)
        total_dur_sec = 30
        n_spks = 3
        each_spk_dur = float(total_dur_sec / n_spks)
        em, ts, mc, mw, spk_ts, gt = generate_mock_data(n_spks=n_spks, spk_dur=each_spk_dur)
        num_speakers = -1
        max_num_speakers = 8
        enhanced_count_thres = 80
        sparse_search_volume = 10
        max_rp_threshold = 0.15
        fixed_thres = -1.0

        # Function call for NeMo python pipeline (unexported) in python
        Y_py = speaker_clustering_python.forward_infer(
            embeddings_in_scales=em,
            timestamps_in_scales=ts,
            multiscale_segment_counts=mc,
            multiscale_weights=mw,
            oracle_num_speakers=num_speakers,
            max_num_speakers=max_num_speakers,
            enhanced_count_thres=enhanced_count_thres,
            sparse_search_volume=sparse_search_volume,
            max_rp_threshold=max_rp_threshold,
            fixed_thres=fixed_thres,
        )

        # Function call for exported module but in python
        Y_tjs = speaker_clustering_scripted.forward_infer(
            embeddings_in_scales=em,
            timestamps_in_scales=ts,
            multiscale_segment_counts=mc,
            multiscale_weights=mw,
            oracle_num_speakers=num_speakers,
            max_num_speakers=max_num_speakers,
            enhanced_count_thres=enhanced_count_thres,
            sparse_search_volume=sparse_search_volume,
            max_rp_threshold=max_rp_threshold,
            fixed_thres=fixed_thres,
        )

        clustering_param_dict = {
            'embeddings': em,
            'timestamps': ts,
            'multiscale_segment_counts': mc,
            'multiscale_weights': mw,
            'oracle_num_speakers': torch.LongTensor([num_speakers]),
            'max_num_speakers': torch.LongTensor([max_num_speakers]),
            'enhanced_count_thres': torch.LongTensor([enhanced_count_thres]),
            'sparse_search_volume': torch.LongTensor([sparse_search_volume]),
            'max_rp_threshold': torch.tensor([max_rp_threshold]),
            'fixed_thres': torch.tensor([fixed_thres]),
        }

        # Function call for an exported module in Triton server environment
        Y_prd = speaker_clustering_scripted.forward(clustering_param_dict)

        # All three types of function call should generate exactly the same output.
        assert len(set(Y_tjs.tolist())) == len(set(Y_py.tolist())) == len(set(Y_prd.tolist())) == n_spks
        assert (
            all(Y_tjs == Y_py) == all(Y_py == Y_prd) == True
        ), f"Script module and python module are showing different clustering results"

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_speaker_counting_1_cpu(self, n_spks=1):
        est_n_spk = test_speaker_counting(n_spks=n_spks, total_dur_sec=10, cuda=False)
        assert est_n_spk == n_spks, f"Clustering test failed at n_spks={n_spks} speaker test"
        est_n_spk = test_speaker_counting(n_spks=n_spks, total_dur_sec=20, cuda=False)
        assert est_n_spk == n_spks, f"Clustering test failed at n_spks={n_spks} speaker test"

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_speaker_counting_1spk_gpu(self, n_spks=1):
        est_n_spk = test_speaker_counting(n_spks=n_spks, total_dur_sec=10)
        assert est_n_spk == n_spks, f"Clustering test failed at n_spks={n_spks} speaker test"
        est_n_spk = test_speaker_counting(n_spks=n_spks, total_dur_sec=20)
        assert est_n_spk == n_spks, f"Clustering test failed at n_spks={n_spks} speaker test"

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_speaker_counting_2spk_gpu(self, n_spks=2):
        est_n_spk = test_speaker_counting(n_spks=n_spks)
        assert est_n_spk == n_spks, f"Clustering test failed at n_spks={n_spks} speaker test"

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_speaker_counting_3spk_gpu(self, n_spks=3):
        est_n_spk = test_speaker_counting(n_spks=n_spks)
        assert est_n_spk == n_spks, f"Clustering test failed at n_spks={n_spks} speaker test"

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_speaker_counting_4spk_gpu(self, n_spks=4):
        est_n_spk = test_speaker_counting(n_spks=n_spks)
        assert est_n_spk == n_spks, f"Clustering test failed at n_spks={n_spks} speaker test"

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_speaker_counting_5spk_gpu(self, n_spks=5):
        est_n_spk = test_speaker_counting(n_spks=n_spks)
        assert est_n_spk == n_spks, f"Clustering test failed at n_spks={n_spks} speaker test"

    def test_offline_clustering(self):
        # TODO
        pass

    def test_online_clustering(self):
        # TODO
        pass

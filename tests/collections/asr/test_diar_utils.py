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

from itertools import permutations

import pytest
import torch

from nemo.collections.asr.parts.utils.speaker_utils import combine_int_overlaps, combine_float_overlaps
from nemo.collections.asr.models.online_diarizer import stitch_cluster_labels, get_minimal_indices
import numpy as np

def check_range_values(target, source):
    bool_list = []
    for tgt, src in zip(target, source):
        for x, y in zip(src, tgt):
            bool_list.append(abs(x-y) < 1e-6)
    return all(bool_list)

def check_labels(target, source):
    bool_list = []
    for x, y in zip(target, source):
        bool_list.append(abs(x-y) < 1e-6)
    return all(bool_list)

def matrix(mat, torch=False):
    if torch:
        return torch.tensor(mat)
    else:
        return np.array(mat)

class TestDiarizationUtilFunctions:
    """
    Tests for cpWER calculation.
    """

    @pytest.mark.unit
    def test_combine_float_overlaps(self):
        intervals = [[0.25, 1.7], [1.5, 3.0], [2.8, 5.0], [5.5, 10.0]]
        target = [[0.25, 5.0], [5.5, 10.0]]
        merged = combine_float_overlaps(intervals)
        assert check_range_values(target, merged)

    @pytest.mark.unit
    def test_combine_int_overlaps(self):
        intervals = [[1,3],[2,6],[8,10],[15,18]]
        target = [[1,6],[8,10],[15,18]]
        merged = combine_int_overlaps(intervals)
        assert check_range_values(target, merged)

    @pytest.mark.unit
    def test_combine_int_overlaps_edge(self):
        intervals = [[1,4],[4,5]]
        target = [[1,5]]
        merged = combine_int_overlaps(intervals)
        assert check_range_values(target, merged)
    
    @pytest.mark.unit
    def test_minimal_index(self):
        Y = [3, 3, 3, 4, 4, 5]
        min_Y = get_minimal_indices(Y)
        target = matrix([0, 0, 0, 1, 1, 2])
        assert check_labels(target, min_Y)

    @pytest.mark.unit
    def test_stitch_cluster_labels(self):
        N = 3
        Y_old = np.zeros(2*N,).astype(int)
        Y_new = np.zeros(2*N,).astype(int) + 1
        target = matrix( [0,0,0,0,0,0] )
        result = stitch_cluster_labels(Y_old, Y_new)
        assert check_labels(target, result)
        
        Y_old = matrix([0,1,2,3,4,5])
        Y_new = matrix([0,0,0,0,0,0])
        target = matrix([0,0,0,0,0,0])
        with_history=False
        result = stitch_cluster_labels(Y_old, Y_new)
        assert check_labels(target, result)
        
        Y_old = matrix([0,0,0,0,0,0])
        Y_new = matrix([0,1,2,3,4,5])
        target = matrix([0,1,2,3,4,5])
        with_history=False
        result = stitch_cluster_labels(Y_old, Y_new)
        assert check_labels(target, result)
        
        Y_old = matrix( [0] * N + [1] * N + [2] * N )
        Y_new = matrix( [1] * N + [2] * N + [3] * N )
        target= matrix( [0, 0, 0, 1, 1, 1, 2, 2, 2] )
        result = stitch_cluster_labels(Y_old, Y_new)
        assert check_labels(target, result)
        
        Y_old = matrix( [0] * N + [1] * (N-1)+ [2] * (N+1))
        Y_new = matrix( [1] * N + [2] * N +    [3] * N )
        target= matrix( [0, 0, 0, 1, 1, 1, 2, 2, 2] )
        result = stitch_cluster_labels(Y_old, Y_new)
        assert check_labels(target, result)
        
        Y_old = matrix( [0] * N + [1] * (N-1)+ [2] * (N+1) + [0, 0, 0])
        Y_new = matrix( [1] * N + [0] * N +    [2] * N     + [4, 5, 6])
        target= matrix( [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 4, 5] )
        result = stitch_cluster_labels(Y_old, Y_new)
        assert check_labels(target, result)

        Y_old = matrix( [0] * N + [1] * N + [2] * N + [0, 0, 0])
        Y_new = matrix( [1] * N + [2] * N + [0] * N + [1, 2, 3, 1, 2, 3])
        target= matrix( [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 3, 0, 1, 3] )
        result = stitch_cluster_labels(Y_old, Y_new)
        assert check_labels(target, result)
    
    def test_embedding_merge(self):
        # TODO
        pass
    
    def test_offline_clustering(self):
        # TODO
        pass

    def test_online_clustering(self):
        # TODO
        pass

    def test_clustering_export(self):
        # TODO
        pass


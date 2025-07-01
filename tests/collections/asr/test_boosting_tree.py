# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.parts.context_biasing.boosting_graph_batched import GPUBoostingTreeModel
from nemo.collections.asr.parts.context_biasing.context_graph_universal import ContextGraph
from torch.nn.utils.rnn import pad_sequence

DEVICES = [torch.device("cpu")]

if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))


@pytest.fixture(scope="module")
def test_context_graph():
    phrases = ["abc", "abd", "c"]
    phrases_ids = [[1, 2, 3], [1, 2, 4], [3]]
    scores = [0.0, 0.0, 0.0]
    context_graph = ContextGraph(context_score=1.0, depth_scaling=1.0)
    context_graph.build(token_ids=phrases_ids, phrases=phrases, scores=scores, uniform_weights=False)
    return context_graph


class TestGPUBoostingTreeModel:
    
    @pytest.mark.unit
    def test_bulding_context_graph(self, test_context_graph):
        context_graph = test_context_graph
        assert context_graph.num_nodes == 5
        # end nodes
        assert context_graph.root.next[1].next[2].next[3].is_end
        assert context_graph.root.next[1].next[2].next[4].is_end
        assert context_graph.root.next[3].is_end
        # words in the end nodes
        assert context_graph.root.next[1].next[2].next[3].phrase == "abc"
        assert context_graph.root.next[1].next[2].next[4].phrase == "abd"
        assert context_graph.root.next[3].phrase == "c"
        # fail links
        assert context_graph.root.next[1].next[2].next[3].fail.token == 3
        assert context_graph.root.next[1].next[2].next[4].fail.token == -1 # root
        assert context_graph.root.next[3].fail.token == -1 # root
        # node scores
        assert round(context_graph.root.next[1].next[2].next[3].node_score, 2) == 4.79
        assert round(context_graph.root.next[1].next[2].next[4].node_score, 2) == 4.79
        assert round(context_graph.root.next[3].node_score, 2) == 1.0


    @pytest.mark.unit
    def test_building_boosting_tree(self, test_context_graph):
        context_graph = test_context_graph
        boosting_tree = GPUBoostingTreeModel.from_cb_tree(
            cb_tree=context_graph,
            vocab_size=5,
            unk_score=0.0,
            final_eos_score=0.0,
            use_triton=True,
            uniform_weights=False
        )

        sentences_ids = [[1, 2, 3, 2, 1], [2, 2, 1, 2, 4], [3, 1, 2, 1]] # [abcba, bbabd, caba]
        device = torch.device("cuda")
        boosting_tree = boosting_tree.cuda()

        boosting_scores = boosting_tree(
            labels=pad_sequence([torch.LongTensor(sentence) for sentence in sentences_ids], batch_first=True).to(device),
            labels_lengths=torch.LongTensor([len(sentence) for sentence in sentences_ids]).to(device),
            bos=False,
            eos=False,
        )

        correct_answer  = torch.tensor([[ 1.0000,  1.6931,  2.0986,  0.0000,  1.0000],
                                        [ 0.0000,  0.0000,  1.0000,  1.6931,  2.0986],
                                        [ 1.0000,  1.0000,  1.6931, -1.6931,  0.0000]], device=device)
        assert torch.allclose(boosting_scores, correct_answer, atol=1e-4)


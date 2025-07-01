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
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("batch_size", [1, 3, 8])
    def test_advance_method(self, test_context_graph, device, batch_size):
        """Test advance method with different batch sizes"""
        boosting_tree = GPUBoostingTreeModel.from_cb_tree(
            cb_tree=test_context_graph, vocab_size=5
        ).to(device)
        
        # Test with initial states
        init_states = boosting_tree.get_init_states(batch_size=batch_size, bos=True)
        scores, next_states = boosting_tree.advance(init_states)
        
        assert scores.shape == (batch_size, 5)  # vocab_size=5
        assert next_states.shape == (batch_size, 5)


    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_get_final_method(self, test_context_graph, device):
        """Test get_final method for EOS scoring"""
        boosting_tree = GPUBoostingTreeModel.from_cb_tree(
            cb_tree=test_context_graph, vocab_size=5
        ).to(device)
        
        # Test with various states
        states = torch.tensor([0, 1, 2], dtype=torch.long, device=device)
        final_scores = boosting_tree.get_final(states)
        
        assert final_scores.shape == (3,)


    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_building_boosting_tree(self, test_context_graph, device):
        context_graph = test_context_graph
        boosting_tree = GPUBoostingTreeModel.from_cb_tree(
            cb_tree=context_graph,
            vocab_size=5,
            unk_score=0.0,
            final_eos_score=0.0,
            use_triton=True,
            uniform_weights=False
        )

        sentences_ids = [[1, 2, 3, 2, 1], [2, 2, 1, 2, 4], [3, 1, 2, 1], []] # ['abcba', 'bbabd', 'caba', '']
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
                                        [ 1.0000,  1.0000,  1.6931, -1.6931,  0.0000],
                                        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]], device=device)
        assert torch.allclose(boosting_scores, correct_answer, atol=1e-4)


    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_triton_vs_pytorch_consistency(self, test_context_graph):
        """Compare Triton vs PyTorch implementations"""
        device = torch.device("cuda")
        
        # Create two identical models with different implementations
        boosting_tree_triton = GPUBoostingTreeModel.from_cb_tree(
            cb_tree=test_context_graph, vocab_size=5, use_triton=True
        ).to(device)
        
        boosting_tree_pytorch = GPUBoostingTreeModel.from_cb_tree(
            cb_tree=test_context_graph, vocab_size=5, use_triton=False
        ).to(device)
        
        # Test with same input
        sentences_ids = [[1, 2, 3, 2, 1], [2, 2, 1, 2, 4]]
        labels = pad_sequence([torch.LongTensor(s) for s in sentences_ids], batch_first=True).to(device)
        lengths = torch.LongTensor([len(s) for s in sentences_ids]).to(device)
        
        scores_triton = boosting_tree_triton(labels=labels, labels_lengths=lengths, bos=False, eos=False)
        scores_pytorch = boosting_tree_pytorch(labels=labels, labels_lengths=lengths, bos=False, eos=False)
        
        assert torch.allclose(scores_triton, scores_pytorch, atol=1e-5)


    @pytest.mark.unit
    def test_eos_handling(self, test_context_graph):
        """Test EOS token handling"""
        boosting_tree = GPUBoostingTreeModel.from_cb_tree(
            cb_tree=test_context_graph, vocab_size=5, unk_score=0.0, final_eos_score=1.0
        )
        
        # Test advance with EOS
        init_states = torch.tensor([1, 2], dtype=torch.long)
        scores, next_states = boosting_tree.advance(init_states, eos_id=0)
        
        # state 2 in the 1st batch should have final_eos_score value
        assert round(scores[0, 0].item(), 2) == 1.69 # (1.69+0): 1.69 as max score for state 1 and 0 because it is not final state
        assert scores[1, 0] == 2.0 # (1+1): 1 as max score for state 2 and 1 because it is final state
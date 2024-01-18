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

from typing import List

import pytest
import torch

from nemo.collections.asr.modules.rnnt import RNNTDecoder, RNNTJoint
from nemo.collections.common.parts.rnn import label_collate

DEVICES: List[torch.device] = [torch.device("cpu")]

if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))

if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICES.append(torch.device("mps"))


class TestJointTorchScript:
    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_compilable_simple(self, device):
        joint = RNNTJoint(
            jointnet={
                "encoder_hidden": 16,
                "pred_hidden": 8,
                "joint_hidden": 32,
                "activation": "relu",
                "dropout": 0.2,
            },
            num_classes=10,
        )
        joint_jit = torch.jit.script(joint)
        joint = joint.to(device).eval()
        joint_jit = joint_jit.to(device).eval()

        sample_encoder = torch.rand([2, 16, 4], device=device)
        sample_prednet = torch.rand([2, 8, 3], device=device)
        joint_forward = joint(encoder_outputs=sample_encoder, decoder_outputs=sample_prednet)
        joint_jit_forward = joint_jit(encoder_outputs=sample_encoder, decoder_outputs=sample_prednet)
        assert torch.allclose(joint_forward, joint_jit_forward)

        sample_f = torch.rand([2, 1, 16], device=device)
        sample_g = torch.rand([2, 1, 8], device=device)
        joint_joint = joint.joint(f=sample_f, g=sample_g)
        joint_jit_joint = joint_jit.joint(f=sample_f, g=sample_g)
        assert torch.allclose(joint_joint, joint_jit_joint)


# class RNNTDecoderDummy(AbstractRNNTDecoder):
#     def predict(
#         self,
#         y: Optional[torch.Tensor] = None,
#         state: Optional[torch.Tensor] = None,
#         add_sos: bool = False,
#         batch_size: Optional[int] = None,
#     ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
#         return torch.rand([1, 2]), []
#
#     def initialize_state(self, y: torch.Tensor) -> List[torch.Tensor]:
#         return []
#
#     def score_hypothesis(
#             self, hypothesis, cache: Dict[Tuple[int], Any]
#     ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
#         return torch.rand([1, 2]), [], torch.rand([1, 2])


class TestPredictionNetworkTorchScript:
    # @pytest.mark.unit
    # @pytest.mark.parametrize("device", DEVICES)
    # def test_compilable_dummy(self, device):
    #     prednet = RNNTDecoderDummy(blank_idx=32, blank_as_pad=True, vocab_size=32)
    #     prednet_jit = torch.jit.script(prednet)

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_compilable_simple(self, device):
        prednet = RNNTDecoder(
            prednet={"pred_hidden": 16, "pred_rnn_layers": 1, "dropout": 0.2, "t_max": None}, vocab_size=32
        )
        prednet = prednet.to(device).eval()
        prednet_jit = torch.jit.script(prednet)
        prednet_jit = prednet_jit.to(device).eval()

        targets = torch.tensor([[0, 1], [3, 5]], device=device)
        targets_lengths = torch.tensor([2, 1], device=device)

        etalon_output, _, etalon_state = prednet(targets=targets, target_length=targets_lengths, states=None)
        jit_output, _, jit_state = prednet_jit(targets=targets, target_length=targets_lengths, states=None)
        assert torch.allclose(etalon_output, jit_output)
        assert torch.allclose(etalon_state[0], jit_state[0])
        assert torch.allclose(etalon_state[1], jit_state[1])

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_label_collate(self, device):
        @torch.jit.script
        def label_collate_wrapper(device: torch.device):
            labels = torch.tensor([[2, 7]], device=device)
            y = label_collate(labels)
            return y

        print(label_collate_wrapper(device))

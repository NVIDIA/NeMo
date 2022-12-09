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

import pytest
import torch

from nemo.collections.asr.models import ASRModel


class TestASRLocalAttention:
    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_forward(self):
        asr_model = ASRModel.from_pretrained("stt_en_conformer_ctc_small")
        asr_model = asr_model.eval()

        len = 16000 * 60 * 30  # 30 minutes, OOM without local attention
        input_signal = torch.randn(size=(1, len), device=asr_model.device)
        length = torch.tensor([len], device=asr_model.device)

        # switch to local attn
        asr_model.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=(64, 64))
        with torch.no_grad():
            asr_model.forward(input_signal=input_signal, input_signal_length=length)

        # switch context size only (keep local)
        asr_model.change_attention_model(att_context_size=(192, 192))
        with torch.no_grad():
            asr_model.forward(input_signal=input_signal, input_signal_length=length)

        len = 16000 * 60 * 4
        input_signal = torch.randn(size=(1, len), device=asr_model.device)
        length = torch.tensor([len], device=asr_model.device)

        # test switching back
        asr_model.change_attention_model(self_attention_model="rel_pos", att_context_size=(-1, -1))
        with torch.no_grad():
            asr_model.forward(input_signal=input_signal, input_signal_length=length)

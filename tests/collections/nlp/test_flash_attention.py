# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import random

import pytest
import torch
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.modules.common.megatron.attention import CoreAttention
from nemo.collections.nlp.modules.common.megatron.megatron_init import initialize_model_parallel_for_nemo
from nemo.collections.nlp.modules.common.megatron.utils import build_attention_mask_3d
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy

try:
    from apex.transformer.enums import AttnMaskType

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


@pytest.mark.run_only_on('GPU')
@pytest.mark.skipif(not HAVE_APEX, reason="apex is not installed")
class TestFlashAttention:
    @classmethod
    def setup_class(cls):
        if not torch.cuda.is_available():
            return

        GPUS = 1
        TP_SIZE = GPUS
        PP_SIZE = 1
        MB_SIZE = 4
        GB_SIZE = 8
        SEED = 1234
        trainer = Trainer(strategy=NLPDDPStrategy(), devices=GPUS, accelerator='gpu', num_nodes=1, logger=None,)

        initialize_model_parallel_for_nemo(
            world_size=trainer.world_size,
            global_rank=trainer.global_rank,
            local_rank=trainer.local_rank,
            tensor_model_parallel_size=TP_SIZE,
            pipeline_model_parallel_size=PP_SIZE,
            micro_batch_size=MB_SIZE,
            global_batch_size=GB_SIZE,
            seed=SEED,
            apex_transformer_log_level=30,
        )

        def dummy():
            return

        if trainer.strategy.launcher is not None:
            trainer.strategy.launcher.launch(dummy, trainer=trainer)
        trainer.strategy.setup_environment()
        torch.distributed.barrier()

    @pytest.fixture()
    def cfg(self):
        cfg = {
            'bz': random.randint(1, 7),
            'sl': random.randint(1, 7),
            'head': random.randint(1, 7),
            'device': torch.cuda.current_device(),
        }
        cfg['hidden'] = cfg['head'] * 8
        return cfg

    @pytest.mark.unit
    def test_flash_attention(self, cfg):
        device = cfg['device']
        bz, sl, np, h = cfg['bz'], cfg['sl'], cfg['head'], cfg['hidden']
        hn = h // np

        q = torch.rand(sl, bz, np, hn, device=device).half()
        k = torch.rand(sl, bz, np, hn, device=device).half()
        v = torch.rand(sl, bz, np, hn, device=device).half()

        q_m = torch.tril(torch.ones(bz, sl, device=device))
        k_m = torch.tril(torch.ones(bz, sl, device=device))
        attention_mask_padding = build_attention_mask_3d(
            source_mask=q_m, target_mask=k_m, attn_mask_type=AttnMaskType.padding
        ).unsqueeze(1)

        attention_mask_causal = build_attention_mask_3d(
            source_mask=q_m, target_mask=k_m, attn_mask_type=AttnMaskType.causal
        ).unsqueeze(1)

        # Non-causal
        attention = CoreAttention(
            layer_number=1,
            num_attention_heads=np,
            hidden_size=h,
            attn_mask_type=AttnMaskType.padding,
            attention_dropout=0.0,
        )

        attention_fa = CoreAttention(
            layer_number=1,
            num_attention_heads=np,
            hidden_size=h,
            attn_mask_type=AttnMaskType.padding,
            attention_dropout=0.0,
            use_flash_attention=True,
        )

        out = attention(q, k, v, attention_mask_padding)
        out_fa = attention_fa(q, k, v, attention_mask_padding)
        assert torch.allclose(out, out_fa, rtol=1e-3, atol=1e-3)

        # Causal
        attention = CoreAttention(
            layer_number=1,
            num_attention_heads=np,
            hidden_size=h,
            attn_mask_type=AttnMaskType.causal,
            attention_dropout=0.0,
        )

        attention_fa = CoreAttention(
            layer_number=1,
            num_attention_heads=np,
            hidden_size=h,
            attn_mask_type=AttnMaskType.causal,
            attention_dropout=0.0,
            use_flash_attention=True,
        )

        out = attention(q, k, v, attention_mask_causal)
        out_fa = attention_fa(q, k, v, attention_mask_causal)
        assert torch.allclose(out, out_fa, rtol=1e-3, atol=1e-3)

    @pytest.mark.unit
    def test_flash_attention_triton(self, cfg):
        device = cfg['device']
        bz, sl, np, h = cfg['bz'], cfg['sl'], cfg['head'], cfg['hidden']
        hn = h // np

        q = torch.rand(sl, bz, np, hn, device=device).half()
        k = torch.rand(sl, bz, np, hn, device=device).half()
        v = torch.rand(sl, bz, np, hn, device=device).half()

        q_m = torch.tril(torch.ones(bz, sl, device=device))
        k_m = torch.tril(torch.ones(bz, sl, device=device))
        attention_mask_padding = build_attention_mask_3d(
            source_mask=q_m, target_mask=k_m, attn_mask_type=AttnMaskType.padding
        ).unsqueeze(1)

        attention_mask_causal = build_attention_mask_3d(
            source_mask=q_m, target_mask=k_m, attn_mask_type=AttnMaskType.causal
        ).unsqueeze(1)

        attention_bias = torch.rand(bz, np, sl, sl, device=device)

        # Non-causal
        attention = CoreAttention(
            layer_number=1,
            num_attention_heads=np,
            hidden_size=h,
            attn_mask_type=AttnMaskType.padding,
            attention_dropout=0.0,
        )

        attention_fa = CoreAttention(
            layer_number=1,
            num_attention_heads=np,
            hidden_size=h,
            attn_mask_type=AttnMaskType.padding,
            attention_dropout=0.0,
            use_flash_attention=True,
        )

        out = attention(q, k, v, attention_mask_padding, relative_position_bias=attention_bias)
        out_fa = attention_fa(q, k, v, attention_mask_padding, relative_position_bias=attention_bias)
        assert torch.allclose(out, out_fa, rtol=1e-3, atol=1e-3)

        # Causal
        attention = CoreAttention(
            layer_number=1,
            num_attention_heads=np,
            hidden_size=h,
            attn_mask_type=AttnMaskType.causal,
            attention_dropout=0.0,
        )

        attention_fa = CoreAttention(
            layer_number=1,
            num_attention_heads=np,
            hidden_size=h,
            attn_mask_type=AttnMaskType.causal,
            attention_dropout=0.0,
            use_flash_attention=True,
        )

        out = attention(q, k, v, attention_mask_causal, relative_position_bias=attention_bias)
        out_fa = attention_fa(q, k, v, attention_mask_causal, relative_position_bias=attention_bias)
        assert torch.allclose(out, out_fa, rtol=1e-3, atol=1e-3)

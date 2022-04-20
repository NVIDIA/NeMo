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

import pytest
import torch
from pytorch_lightning.trainer.trainer import Trainer
from nemo.collections.nlp.modules.common.megatron.layer_type import LayerType

from nemo.collections.nlp.modules.common.megatron.megatron_init import initialize_model_parallel_for_nemo
from nemo.collections.nlp.modules.common.megatron.rotary_pos_embedding import RotaryEmbedding
from nemo.collections.nlp.modules.common.megatron.transformer import ParallelChunkedCrossAttention
from nemo.collections.nlp.modules.common.megatron.retrieval_transformer import MegatronRetrievalTransformerEncoderModule
from nemo.collections.nlp.modules.common.megatron.utils import init_method_normal, scaled_init_method_normal
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin


class TestCrossAttn:

    @classmethod
    @pytest.mark.run_only_on('GPU')
    def setup_class(cls):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
        plugin = NLPDDPPlugin()

        TP_SIZE = 2
        PP_SIZE = 1
        MB_SIZE = 4
        GB_SIZE = 8
        SEED = 1234

        trainer = Trainer(plugins=plugin, devices=2, accelerator='gpu', num_nodes=1, logger=None, log_gpu_memory=None)

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

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_cross_attn(self):
        init_method_std = 0.02
        num_layers = 1

        # rotary pos emb dim
        dim = 128
        num_attention_heads = 8
        rot_dim = dim // num_attention_heads
        rotary_pos_emb = RotaryEmbedding(rot_dim).cuda().half()
        hidden = torch.rand(2048, 2, dim).cuda().half()  # (seq, batch, dim)
        pad_id = 19999

        text_chunk_size = 64
        context_chunk_size = 128
        retrieved = torch.randint(0, 20000, (32, 2, 128, 2)).cuda().half()
        # retrieved tokens - (num chunks, num retrieved neighbors, retrieved chunk with continuation, batch)

        # context attention mask [b, np, sq, sk]
        pad_id = 19999
        context_mask = retrieved != pad_id
        retrieved_emb = torch.rand(32, 2, 128, 2, dim).cuda().half()
        # retrieved tokens - (num chunks, num retrieved neighbors, retrieved chunk with continuation, batch, hidden)

        device = retrieved.device
        # need to add extra chunk size, since it will be shifted
        cross_attn_q_pos_emb = rotary_pos_emb(text_chunk_size + text_chunk_size - 1, device=device, offset=0)
        cross_attn_k_pos_emb = rotary_pos_emb(context_chunk_size, device=device)
        cross_attn_pos_emb = (cross_attn_q_pos_emb, cross_attn_k_pos_emb)

        init_method = init_method_normal(init_method_std)

        scaled_init_method = scaled_init_method_normal(init_method_std, num_layers)
        cross_attn = (
            ParallelChunkedCrossAttention(
                init_method=init_method,
                output_layer_init_method=scaled_init_method,
                layer_number=0,
                num_attention_heads=num_attention_heads,
                hidden_size=dim,
                precision=16,
                chunk_size=text_chunk_size,
            )
            .cuda()
            .half()
        )
        out, bias = cross_attn(hidden, context_mask, encoder_output=retrieved_emb, pos_emb=cross_attn_pos_emb)

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_retrival_encoder(self):

        init_method_std = 0.02

        # rotary pos emb dim
        dim = 128
        pad_id = 19999
        num_attention_heads = 8
        # rot_dim = dim // num_attention_heads
        # rotary_pos_emb = RotaryEmbedding(rot_dim).cuda().half()
        hidden = torch.rand(2, 2048).cuda().half()  # (batch, seq, dim)
        hidden_mask = (hidden == pad_id).type(torch.int32).cuda()

        hidden_emb = torch.rand(2, 2048, dim).cuda().half()  # (batch, seq, dim)

        text_chunk_size = 64
        # context_chunk_size = 128
        retrieved = torch.randint(0, 20000, (2, 32, 2, 128)).cuda().half()
        # retrieved tokens - (batch, num chunks, num retrieved neighbors, retrieved chunk with continuation)

        # context attention mask [b, np, sq, sk]
        pad_id = 19999
        context_mask = (retrieved == pad_id).type(torch.int32).cuda()
        retrieved_emb = torch.rand(2, 32, 2, 128, dim).cuda().half()
        # retrieved tokens - (batch, num chunks, num retrieved neighbors, retrieved chunk with continuation, hidden)

        # device = retrieved.device
        # need to add extra chunk size, since it will be shifted
        # cross_attn_q_pos_emb = rotary_pos_emb(text_chunk_size + text_chunk_size - 1, device=device, offset=0)
        # cross_attn_k_pos_emb = rotary_pos_emb(context_chunk_size, device=device)
        # cross_attn_pos_emb = (cross_attn_q_pos_emb, cross_attn_k_pos_emb)
        layer_type = [LayerType.encoder, LayerType.encoder, LayerType.retrieval_encoder]
        num_layers = len(layer_type)

        init_method = init_method_normal(init_method_std)
        scaled_init_method = scaled_init_method_normal(init_method_std, num_layers)
        encoder = MegatronRetrievalTransformerEncoderModule(
                init_method=init_method,
                output_layer_init_method=scaled_init_method,
                hidden_size=dim,
                ffn_hidden_size=dim*4,
                num_layers=num_layers,
                num_attention_heads=num_attention_heads,
                precision=16,
                chunk_size=text_chunk_size,
                layer_type=layer_type,
        ).cuda().half()
        out = encoder(retrieved_emb, context_mask, context_attn_mask=hidden_mask, encoder_output=hidden_emb)

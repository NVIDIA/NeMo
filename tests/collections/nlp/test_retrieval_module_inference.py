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
from einops import rearrange
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.modules.common.megatron.layer_type import LayerType
from nemo.collections.nlp.modules.common.megatron.megatron_init import initialize_model_parallel_for_nemo
from nemo.collections.nlp.modules.common.megatron.retrieval_token_level_encoder_decoder import (
    MegatronRetrievalTokenLevelEncoderDecoderModule,
)
from nemo.collections.nlp.modules.common.megatron.retrieval_transformer import (
    MegatronRetrievalTransformerDecoderModule,
    MegatronRetrievalTransformerEncoderModule,
)
from nemo.collections.nlp.modules.common.megatron.rotary_pos_embedding import RotaryEmbedding
from nemo.collections.nlp.modules.common.megatron.transformer import ParallelChunkedCrossAttention
from nemo.collections.nlp.modules.common.megatron.utils import (
    build_attention_mask_3d,
    init_method_normal,
    scaled_init_method_normal,
)
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin

try:
    from apex.transformer.enums import AttnMaskType

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


@pytest.mark.run_only_on('GPU')
@pytest.mark.skipif(not HAVE_APEX, reason="apex is not installed")
class TestRetrievalModuleInference:
    @classmethod
    def setup_class(cls):
        if not torch.cuda.is_available():
            return
        GPUS = 1
        plugins = [NLPDDPPlugin()]
        TP_SIZE = GPUS
        PP_SIZE = 1
        MB_SIZE = 4
        GB_SIZE = 8
        SEED = 1234
        trainer = Trainer(
            plugins=plugins, devices=GPUS, accelerator='gpu', num_nodes=1, logger=None, log_gpu_memory=None
        )

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

    @pytest.mark.unit
    def test_retrieval_encoder(self):

        init_method_std = 0.02

        batch = 2
        neighbors = 2
        # rotary pos emb dim
        dim = 128
        pad_id = 19999
        num_attention_heads = 8
        chunks = 32
        text_chunk_size = 64
        input_length = chunks * text_chunk_size
        vocab_size = 20000

        hidden = torch.randint(0, vocab_size, (batch, input_length)).cuda()  # (seq, batch, dim)
        hidden_mask = (hidden != pad_id).cuda()

        hidden_emb = torch.rand(batch, input_length, dim).cuda().half()  # (batch, seq, dim)
        retrieved = torch.randint(0, vocab_size, (batch, chunks, neighbors, 2 * text_chunk_size)).cuda()
        pad_id = vocab_size - 1
        context_mask = (retrieved != pad_id).cuda()
        retrieved_emb = torch.rand(batch, chunks, neighbors, 2 * text_chunk_size, dim).cuda().half()

        layer_type = [LayerType.encoder, LayerType.retrieval_encoder, LayerType.encoder, LayerType.retrieval_encoder]
        num_layers = len(layer_type)

        init_method = init_method_normal(init_method_std)
        scaled_init_method = scaled_init_method_normal(init_method_std, num_layers)
        encoder = (
            MegatronRetrievalTransformerEncoderModule(
                init_method=init_method,
                output_layer_init_method=scaled_init_method,
                hidden_size=dim,
                ffn_hidden_size=dim * 4,
                num_layers=num_layers,
                num_attention_heads=num_attention_heads,
                precision=16,
                chunk_size=text_chunk_size,
                layer_type=layer_type,
                hidden_dropout=0.0,
                attention_dropout=0.0,
            )
            .cuda()
            .half()
        )
        out_gt = encoder(retrieved_emb, context_mask, context_attn_mask=hidden_mask, encoder_output=hidden_emb)
        assert out_gt.shape == torch.Size([batch, chunks, neighbors, 2 * text_chunk_size, dim])

        out_1 = encoder(
            None,
            None,
            context_attn_mask=hidden_mask[:, :63],
            encoder_output=hidden_emb[:, :63, :],
            set_inference_key_value_memory=True,
            inference_max_sequence_len=input_length,
            neighbors=neighbors,
        )

        out_2 = encoder(
            retrieved_emb[:, :1],
            context_mask[:, :1],
            context_attn_mask=hidden_mask[:, 63:64],
            encoder_output=hidden_emb[:, 63:64, :],
            set_inference_key_value_memory=False,
            inference_max_sequence_len=input_length,
            neighbors=neighbors,
        )
        # assert (out_gt[:, 0,] - out_2[:, 0]).abs().max().item() < 1e-5
        print((out_gt[:, 0,] - out_2[:, 0]).abs().max().item())

        for i in range(64, 127):
            out_3 = encoder(
                retrieved_emb[:, :1],
                context_mask[:, :1],
                context_attn_mask=hidden_mask[:, i:i+1],
                encoder_output=hidden_emb[:, i:i+1, :],
                set_inference_key_value_memory=False,
                inference_max_sequence_len=input_length,
                neighbors=neighbors,
            )
        print((encoder.encoder_output - hidden_emb[:, :64, :]).abs().max())
        out_test = encoder(retrieved_emb[:, :1], context_mask[:, :1], context_attn_mask=hidden_mask[:, :64], encoder_output=hidden_emb[:, :64])
        print((out_gt[:, 0,] - out_test[:, 0]).abs().max().item())

        print((out_gt[:, 0,] - out_3[:, 0]).abs().max().item())
        # test inference
        for i in range(127, 191):
            out_4 = encoder(
                retrieved_emb[:, :2],
                context_mask[:, :2],
                context_attn_mask=hidden_mask[:, i:i+1],
                encoder_output=hidden_emb[:, i:i+1, :],
                set_inference_key_value_memory=False,
                inference_max_sequence_len=input_length,
                neighbors=neighbors,
            )

        print((out_gt[:, 1,] - out_4[:, 1]).abs().max().item())

        print((encoder.encoder_output - hidden_emb[:, 64:128, :]).abs().max())
        out_test = encoder(retrieved_emb[:, :1], context_mask[:, :1], context_attn_mask=hidden_mask[:, :64], encoder_output=hidden_emb[:, :64])

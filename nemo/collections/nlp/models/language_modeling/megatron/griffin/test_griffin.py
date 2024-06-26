# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import os

import megatron.core.parallel_state as ps
import pytest
import torch
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer

from nemo.collections.nlp.models.language_modeling.megatron.griffin.griffin_layer_spec import (
    griffin_mqa_layer_with_transformer_engine_spec,
    griffin_recurrent_layer_with_transformer_engine_spec,
)
from nemo.collections.nlp.models.language_modeling.megatron.griffin.griffin_model import GriffinModel
from nemo.collections.nlp.models.language_modeling.megatron.griffin.recurrent_layer import (
    RecurrentBlock,
    RecurrentBlockSubmodules,
)


class Utils:

    world_size = 1  # torch.cuda.device_count()
    rank = 0  # int(os.environ['LOCAL_RANK'])

    @staticmethod
    def initialize_distributed():
        if not torch.distributed.is_initialized():
            print(f'Initializing torch.distributed with rank: {Utils.rank}, world_size: {Utils.world_size}')
            torch.cuda.set_device(Utils.rank % torch.cuda.device_count())
            init_method = 'tcp://'
            master_ip = os.getenv('MASTER_ADDR', 'localhost')
            master_port = os.getenv('MASTER_PORT', '6000')
            init_method += master_ip + ':' + master_port
            torch.distributed.init_process_group(
                backend='nccl', world_size=Utils.world_size, rank=Utils.rank, init_method=init_method
            )

    @staticmethod
    def destroy_model_parallel():
        ps.destroy_model_parallel()
        torch.distributed.barrier()

    @staticmethod
    def initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        pipeline_model_parallel_split_rank=None,
        **kwargs,
    ):
        ps.destroy_model_parallel()
        Utils.initialize_distributed()
        ps.initialize_model_parallel(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank,
            **kwargs,
        )


class TestParallelAttention:

    def setup_method(self, method=None):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = TransformerConfig(
            bias_activation_fusion=True,
            gated_linear_unit=True,
            hidden_size=2560,
            ffn_hidden_size=7680,
            num_attention_heads=10,
            num_layers=1,
            window_size=[2, 0],
            num_query_groups=1,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            rotary_interleaved=False,
        )
        from megatron.core.transformer.enums import AttnMaskType

        self.parallel_attention = SelfAttention(
            self.transformer_config,
            griffin_mqa_layer_with_transformer_engine_spec.submodules.self_attention.submodules,
            attn_mask_type=AttnMaskType.causal,
            layer_number=1,
        )

        new_state_dict = {}
        dm_model_weight = torch.load('/home/ataghibakhsh/deepmind/space_gemma_model/2b.pt')

        new_state_dict['linear_proj.weight'] = dm_model_weight['blocks.2.attention_block.proj_final.weight']
        new_state_dict['linear_proj.bias'] = dm_model_weight['blocks.2.attention_block.proj_final.bias']
        new_state_dict['linear_qkv.weight'] = torch.cat(
            [
                dm_model_weight['blocks.2.attention_block.proj_q.weight'],
                dm_model_weight['blocks.2.attention_block.proj_k.weight'],
                dm_model_weight['blocks.2.attention_block.proj_v.weight'],
            ]
        )
        new_state_dict['linear_qkv.bias'] = torch.zeros(new_state_dict['linear_qkv.weight'].shape[0])
        new_state_dict['linear_proj._extra_state'] = self.parallel_attention.state_dict()['linear_proj._extra_state']
        new_state_dict['linear_qkv._extra_state'] = self.parallel_attention.state_dict()['linear_qkv._extra_state']

        self.parallel_attention.load_state_dict(new_state_dict, strict=True)
        self.parallel_attention = self.parallel_attention.half()

    def teardown_method(self, method=None):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.parallel_attention, SelfAttention)
        assert self.parallel_attention.layer_number == 1

        num_weights = sum([p.numel() for p in self.parallel_attention.parameters()])
        assert num_weights == 648

    def test_cpu_forward(self):
        # we can't currently do this because the global memory buffer is on GPU
        pass

    def test_gpu_forward(self):

        config = self.parallel_attention.config
        sequence_length = 2048
        micro_batch_size = 2

        rotary_pos_emb = RotaryEmbedding(
            kv_channels=config.kv_channels,
            rotary_percent=1.0,
            rotary_interleaved=config.rotary_interleaved,
            seq_len_interpolation_factor=None,
            rotary_base=10000,
        )

        self.parallel_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.rand(sequence_length, micro_batch_size, 2560) * 0 + (
            1
            + torch.cat(
                [
                    torch.arange(sequence_length).unsqueeze(dim=1)
                    for i in range(self.parallel_attention.config.hidden_size)
                ],
                dim=1,
            )
        ).unsqueeze(1)
        hidden_states = hidden_states / sequence_length * 0.0001
        hidden_states = hidden_states.half().cuda()

        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()  # .tril()

        self.parallel_attention.input_tensor = None
        rotary_seq_len = rotary_pos_emb.get_rotary_seq_len(None, self.parallel_attention, hidden_states, config)
        rotary_pos_emb = rotary_pos_emb(rotary_seq_len)

        output, bias = self.parallel_attention(hidden_states, attention_mask, rotary_pos_emb=rotary_pos_emb)

        assert config.recompute_granularity is None
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size


class TestTransfomer:

    def setup_method(self, method=None):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = TransformerConfig(
            bias_activation_fusion=True,
            gated_linear_unit=True,
            hidden_size=2560,
            ffn_hidden_size=7680,
            num_attention_heads=10,
            num_layers=1,
            window_size=[1024, 0],
            num_query_groups=1,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            rotary_interleaved=False,
            normalization="RMSNorm",
            layernorm_epsilon=1e-6,
        )

        from megatron.core.transformer.transformer_layer import TransformerLayer

        # from megatron.core.transformer.transfomer_layer2 import TransformerLayer

        self.transfomer_layer = TransformerLayer(
            self.transformer_config,
            griffin_mqa_layer_with_transformer_engine_spec.submodules,
        )

        new_state_dict = {}
        dm_model_weight = torch.load('/home/ataghibakhsh/deepmind/space_gemma_model/2b.pt')

        new_state_dict['self_attention.linear_qkv.layer_norm_weight'] = (
            dm_model_weight['blocks.2.temporal_pre_norm.scale'] + 1
        )
        # new_state_dict['input_layernorm.weight'] = dm_model_weight['blocks.2.temporal_pre_norm.scale']

        new_state_dict['self_attention.linear_proj.weight'] = dm_model_weight[
            'blocks.2.attention_block.proj_final.weight'
        ]
        new_state_dict['self_attention.linear_proj.bias'] = dm_model_weight['blocks.2.attention_block.proj_final.bias']
        new_state_dict['self_attention.linear_qkv.weight'] = torch.cat(
            [
                dm_model_weight['blocks.2.attention_block.proj_q.weight'],
                dm_model_weight['blocks.2.attention_block.proj_k.weight'],
                dm_model_weight['blocks.2.attention_block.proj_v.weight'],
            ]
        )
        new_state_dict['self_attention.linear_qkv.bias'] = torch.zeros(
            new_state_dict['self_attention.linear_qkv.weight'].shape[0]
        )
        new_state_dict['self_attention.linear_proj._extra_state'] = self.transfomer_layer.state_dict()[
            'self_attention.linear_proj._extra_state'
        ]
        new_state_dict['self_attention.linear_qkv._extra_state'] = self.transfomer_layer.state_dict()[
            'self_attention.linear_qkv._extra_state'
        ]

        new_state_dict['mlp.linear_fc1.layer_norm_weight'] = dm_model_weight['blocks.2.channel_pre_norm.scale'] + 1
        # new_state_dict['pre_mlp_layernorm.weight'] = dm_model_weight['blocks.2.channel_pre_norm.scale']

        new_state_dict['mlp.linear_fc1.weight'] = torch.cat(
            [
                dm_model_weight['blocks.2.mlp_block.ffw_up.w'].permute(0, 2, 1)[0],
                dm_model_weight['blocks.2.mlp_block.ffw_up.w'].permute(0, 2, 1)[1],
            ]
        )
        new_state_dict['mlp.linear_fc1.bias'] = dm_model_weight['blocks.2.mlp_block.ffw_up.b'].flatten()
        new_state_dict['mlp.linear_fc2.weight'] = dm_model_weight['blocks.2.mlp_block.ffw_down.weight']
        new_state_dict['mlp.linear_fc2.bias'] = dm_model_weight['blocks.2.mlp_block.ffw_down.bias']
        new_state_dict['mlp.linear_fc1._extra_state'] = self.transfomer_layer.state_dict()[
            'mlp.linear_fc1._extra_state'
        ]
        new_state_dict['mlp.linear_fc2._extra_state'] = self.transfomer_layer.state_dict()[
            'mlp.linear_fc2._extra_state'
        ]

        self.transfomer_layer.load_state_dict(new_state_dict, strict=True)
        self.transfomer_layer = self.transfomer_layer.half()

    def teardown_method(self, method=None):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.transfomer_layer, TransformerLayer)
        assert self.transfomer_layer.layer_number == 1

        num_weights = sum([p.numel() for p in self.transfomer_layer.parameters()])
        assert num_weights == 648

    def test_cpu_forward(self):
        # we can't currently do this because the global memory buffer is on GPU
        pass

    def test_gpu_forward(self):

        config = self.transfomer_layer.config
        sequence_length = 4
        micro_batch_size = 2

        rotary_pos_emb = RotaryEmbedding(
            kv_channels=config.kv_channels,
            rotary_percent=1.0,
            rotary_interleaved=config.rotary_interleaved,
            seq_len_interpolation_factor=None,
            rotary_base=10000,
        )

        self.transfomer_layer.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.rand(sequence_length, micro_batch_size, 2560) * 0 + (
            1 + torch.cat([torch.arange(sequence_length).unsqueeze(dim=1) for i in range(config.hidden_size)], dim=1)
        ).unsqueeze(1)
        hidden_states = hidden_states / sequence_length * 0.0001
        hidden_states[:, :, 1280:] = hidden_states[:, :, 1280:] * 3.141592
        hidden_states = hidden_states.half().cuda()

        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

        self.transfomer_layer.input_tensor = None
        rotary_seq_len = rotary_pos_emb.get_rotary_seq_len(None, self.transfomer_layer, hidden_states, config)
        rotary_pos_emb = rotary_pos_emb(rotary_seq_len).half()
        print(rotary_pos_emb.shape)
        print(hidden_states.shape)
        output = self.transfomer_layer(hidden_states, attention_mask, rotary_pos_emb=rotary_pos_emb)[0].permute(
            1, 0, 2
        )
        print(output.shape)
        print(output)


class TestRecurrent:

    def setup_method(self, method=None):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = TransformerConfig(
            bias_activation_fusion=True,
            gated_linear_unit=True,
            hidden_size=2560,
            ffn_hidden_size=7680,
            num_attention_heads=10,
            num_layers=1,
            window_size=[1024, 0],
            num_query_groups=1,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            rotary_interleaved=False,
            normalization="RMSNorm",
            layernorm_epsilon=1e-6,
        )

        self.recurrent_layer = RecurrentBlock(
            self.transformer_config,
            griffin_recurrent_layer_with_transformer_engine_spec.submodules,
        )

        new_state_dict = {}
        dm_model_weight = torch.load('/home/ataghibakhsh/deepmind/space_gemma_model/2b.pt')

        # new_state_dict['input_layernorm.weight'] = dm_model_weight['blocks.0.temporal_pre_norm.scale']

        new_state_dict['recurrent_layer.linear_y.layer_norm_weight'] = (
            dm_model_weight['blocks.0.temporal_pre_norm.scale'] + 1
        )
        new_state_dict['recurrent_layer.linear_x.layer_norm_weight'] = (
            dm_model_weight['blocks.0.temporal_pre_norm.scale'] + 1
        )
        new_state_dict['recurrent_layer.linear_y.weight'] = dm_model_weight['blocks.0.recurrent_block.linear_y.weight']
        new_state_dict['recurrent_layer.linear_y.bias'] = dm_model_weight['blocks.0.recurrent_block.linear_y.bias']
        new_state_dict['recurrent_layer.linear_x.weight'] = dm_model_weight['blocks.0.recurrent_block.linear_x.weight']
        new_state_dict['recurrent_layer.linear_x.bias'] = dm_model_weight['blocks.0.recurrent_block.linear_x.bias']
        new_state_dict['recurrent_layer.linear_out.weight'] = dm_model_weight[
            'blocks.0.recurrent_block.linear_out.weight'
        ]
        new_state_dict['recurrent_layer.linear_out.bias'] = dm_model_weight['blocks.0.recurrent_block.linear_out.bias']

        new_state_dict['recurrent_layer.conv_1d.conv_1d.weight'] = (
            dm_model_weight['blocks.0.recurrent_block.conv_1d.w'].unsqueeze(0).permute(2, 0, 1)
        )
        new_state_dict['recurrent_layer.conv_1d.conv_1d.bias'] = dm_model_weight['blocks.0.recurrent_block.conv_1d.b']

        new_state_dict['recurrent_layer.rg_lru.a_param'] = dm_model_weight['blocks.0.recurrent_block.rg_lru.a_param']
        new_state_dict['recurrent_layer.rg_lru.input_gate.w'] = dm_model_weight[
            'blocks.0.recurrent_block.rg_lru.input_gate.w'
        ]
        new_state_dict['recurrent_layer.rg_lru.input_gate.b'] = dm_model_weight[
            'blocks.0.recurrent_block.rg_lru.input_gate.b'
        ]
        new_state_dict['recurrent_layer.rg_lru.a_gate.w'] = dm_model_weight['blocks.0.recurrent_block.rg_lru.a_gate.w']
        new_state_dict['recurrent_layer.rg_lru.a_gate.b'] = dm_model_weight['blocks.0.recurrent_block.rg_lru.a_gate.b']

        # new_state_dict['pre_mlp_layernorm.weight'] = dm_model_weight['blocks.0.channel_pre_norm.scale']
        new_state_dict['mlp.linear_fc1.layer_norm_weight'] = dm_model_weight['blocks.0.channel_pre_norm.scale'] + 1
        new_state_dict['mlp.linear_fc1.weight'] = torch.cat(
            [
                dm_model_weight['blocks.0.mlp_block.ffw_up.w'].permute(0, 2, 1)[0],
                dm_model_weight['blocks.0.mlp_block.ffw_up.w'].permute(0, 2, 1)[1],
            ]
        )
        new_state_dict['mlp.linear_fc1.bias'] = dm_model_weight['blocks.0.mlp_block.ffw_up.b'].flatten()
        new_state_dict['mlp.linear_fc2.weight'] = dm_model_weight['blocks.0.mlp_block.ffw_down.weight']
        new_state_dict['mlp.linear_fc2.bias'] = dm_model_weight['blocks.0.mlp_block.ffw_down.bias']
        new_state_dict['mlp.linear_fc1._extra_state'] = self.recurrent_layer.state_dict()[
            'mlp.linear_fc1._extra_state'
        ]
        new_state_dict['mlp.linear_fc2._extra_state'] = self.recurrent_layer.state_dict()[
            'mlp.linear_fc2._extra_state'
        ]

        new_state_dict['recurrent_layer.linear_y._extra_state'] = self.recurrent_layer.state_dict()[
            'recurrent_layer.linear_y._extra_state'
        ]
        new_state_dict['recurrent_layer.linear_x._extra_state'] = self.recurrent_layer.state_dict()[
            'recurrent_layer.linear_x._extra_state'
        ]
        new_state_dict['recurrent_layer.linear_out._extra_state'] = self.recurrent_layer.state_dict()[
            'recurrent_layer.linear_out._extra_state'
        ]

        self.recurrent_layer.load_state_dict(new_state_dict, strict=True)
        self.recurrent_layer = self.recurrent_layer.half()

    def teardown_method(self, method=None):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.recurrent_layer, TransformerLayer)
        assert self.recurrent_layer.layer_number == 1

        num_weights = sum([p.numel() for p in self.recurrent_layer.parameters()])
        assert num_weights == 648

    def test_cpu_forward(self):
        # we can't currently do this because the global memory buffer is on GPU
        pass

    def test_gpu_forward(self):

        config = self.recurrent_layer.config
        sequence_length = 128
        micro_batch_size = 2

        rotary_pos_emb = RotaryEmbedding(
            kv_channels=config.kv_channels,
            rotary_percent=1.0,
            rotary_interleaved=config.rotary_interleaved,
            seq_len_interpolation_factor=None,
            rotary_base=10000,
        )

        self.recurrent_layer.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.rand(sequence_length, micro_batch_size, 2560) * 0 + (
            1 + torch.cat([torch.arange(sequence_length).unsqueeze(dim=1) for i in range(config.hidden_size)], dim=1)
        ).unsqueeze(1)
        hidden_states = hidden_states / sequence_length * 0.0001
        hidden_states[:, :, 1280:] = hidden_states[:, :, 1280:] * 3.141592
        hidden_states = hidden_states.half().cuda()

        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

        self.recurrent_layer.input_tensor = None
        rotary_seq_len = rotary_pos_emb.get_rotary_seq_len(None, self.recurrent_layer, hidden_states, config)
        rotary_pos_emb = rotary_pos_emb(rotary_seq_len)

        output = self.recurrent_layer(hidden_states, attention_mask)
        print(output.shape)
        print(output.sum())
        print(output)


class TestFullPipe:

    def setup_method(self, method=None):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = TransformerConfig(
            bias_activation_fusion=True,
            gated_linear_unit=True,
            hidden_size=2560,
            ffn_hidden_size=7680,
            num_attention_heads=10,
            num_layers=26,
            window_size=[1024, 0],
            num_query_groups=1,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            rotary_interleaved=False,
            normalization="RMSNorm",
            layernorm_epsilon=1e-6,
            layernorm_zero_centered_gamma=True,
        )
        self.transformer_config.activations_checkpoint_recurrent = False
        self.griffin = GriffinModel(self.transformer_config, vocab_size=256128).cuda()
        # print(self.griffin.state_dict().keys())
        # print(self.griffin.state_dict()['decoder.layers.25.recurrent_layer.linear_in.bias'].shape)
        # print(self.griffin.state_dict()['decoder.layers.25.recurrent_layer.linear_in.weight'].shape)
        # print(self.griffin.state_dict()['decoder.layers.25.recurrent_layer.linear_in.layer_norm_weight'].shape)

        new_state_dict = {}
        dm_model_weight = torch.load('/home/ataghibakhsh/deepmind/space_gemma_model/2b.pt')

        new_state_dict['embedding.word_embeddings.weight'] = dm_model_weight['embedder.input_embedding']
        new_state_dict['decoder.final_layernorm.weight'] = dm_model_weight['final_norm.scale']

        for l in range(26):
            # print(f"Converting Layer {l}")
            # print("********************")

            new_state_dict[f'decoder.layers.{l}.mlp.linear_fc1.weight'] = torch.cat(
                [
                    dm_model_weight[f'blocks.{l}.mlp_block.ffw_up.w'].permute(0, 2, 1)[0],
                    dm_model_weight[f'blocks.{l}.mlp_block.ffw_up.w'].permute(0, 2, 1)[1],
                ]
            )
            new_state_dict[f'decoder.layers.{l}.mlp.linear_fc1.bias'] = dm_model_weight[
                f'blocks.{l}.mlp_block.ffw_up.b'
            ].flatten()
            new_state_dict[f'decoder.layers.{l}.mlp.linear_fc2.weight'] = dm_model_weight[
                f'blocks.{l}.mlp_block.ffw_down.weight'
            ]
            new_state_dict[f'decoder.layers.{l}.mlp.linear_fc2.bias'] = dm_model_weight[
                f'blocks.{l}.mlp_block.ffw_down.bias'
            ]
            new_state_dict[f'decoder.layers.{l}.mlp.linear_fc1._extra_state'] = self.griffin.state_dict()[
                f'decoder.layers.{l}.mlp.linear_fc1._extra_state'
            ]
            new_state_dict[f'decoder.layers.{l}.mlp.linear_fc2._extra_state'] = self.griffin.state_dict()[
                f'decoder.layers.{l}.mlp.linear_fc2._extra_state'
            ]

            new_state_dict[f'decoder.layers.{l}.mlp.linear_fc1.layer_norm_weight'] = dm_model_weight[
                f'blocks.{l}.channel_pre_norm.scale'
            ]

            if l % 3 == 2:
                new_state_dict[f'decoder.layers.{l}.self_attention.linear_proj.weight'] = dm_model_weight[
                    f'blocks.{l}.attention_block.proj_final.weight'
                ]
                new_state_dict[f'decoder.layers.{l}.self_attention.linear_proj.bias'] = dm_model_weight[
                    f'blocks.{l}.attention_block.proj_final.bias'
                ]
                new_state_dict[f'decoder.layers.{l}.self_attention.linear_qkv.layer_norm_weight'] = dm_model_weight[
                    f'blocks.{l}.temporal_pre_norm.scale'
                ]
                new_state_dict[f'decoder.layers.{l}.self_attention.linear_qkv.weight'] = torch.cat(
                    [
                        dm_model_weight[f'blocks.{l}.attention_block.proj_q.weight'],
                        dm_model_weight[f'blocks.{l}.attention_block.proj_k.weight'],
                        dm_model_weight[f'blocks.{l}.attention_block.proj_v.weight'],
                    ]
                )
                new_state_dict[f'decoder.layers.{l}.self_attention.linear_qkv.bias'] = torch.zeros(
                    new_state_dict[f'decoder.layers.{l}.self_attention.linear_qkv.weight'].shape[0]
                )
                new_state_dict[f'decoder.layers.{l}.self_attention.linear_proj._extra_state'] = (
                    self.griffin.state_dict()[f'decoder.layers.{l}.self_attention.linear_proj._extra_state']
                )
                new_state_dict[f'decoder.layers.{l}.self_attention.linear_qkv._extra_state'] = (
                    self.griffin.state_dict()[f'decoder.layers.{l}.self_attention.linear_qkv._extra_state']
                )

            else:

                new_state_dict[f'decoder.layers.{l}.recurrent_layer.linear_in.layer_norm_weight'] = dm_model_weight[
                    f'blocks.{l}.temporal_pre_norm.scale'
                ]
                new_state_dict[f'decoder.layers.{l}.recurrent_layer.linear_in.weight'] = torch.cat(
                    [
                        dm_model_weight[f'blocks.{l}.recurrent_block.linear_x.weight'],
                        dm_model_weight[f'blocks.{l}.recurrent_block.linear_y.weight'],
                    ]
                )
                new_state_dict[f'decoder.layers.{l}.recurrent_layer.linear_in.bias'] = torch.cat(
                    [
                        dm_model_weight[f'blocks.{l}.recurrent_block.linear_x.bias'],
                        dm_model_weight[f'blocks.{l}.recurrent_block.linear_y.bias'],
                    ]
                )
                new_state_dict[f'decoder.layers.{l}.recurrent_layer.linear_out.weight'] = dm_model_weight[
                    f'blocks.{l}.recurrent_block.linear_out.weight'
                ]
                new_state_dict[f'decoder.layers.{l}.recurrent_layer.linear_out.bias'] = dm_model_weight[
                    f'blocks.{l}.recurrent_block.linear_out.bias'
                ]

                new_state_dict[f'decoder.layers.{l}.recurrent_layer.conv_1d.conv_1d.weight'] = (
                    dm_model_weight[f'blocks.{l}.recurrent_block.conv_1d.w'].unsqueeze(0).permute(2, 0, 1)
                )
                new_state_dict[f'decoder.layers.{l}.recurrent_layer.conv_1d.conv_1d.bias'] = dm_model_weight[
                    f'blocks.{l}.recurrent_block.conv_1d.b'
                ]

                new_state_dict[f'decoder.layers.{l}.recurrent_layer.rg_lru.a_param'] = dm_model_weight[
                    f'blocks.{l}.recurrent_block.rg_lru.a_param'
                ]
                new_state_dict[f'decoder.layers.{l}.recurrent_layer.rg_lru.input_gate.w'] = dm_model_weight[
                    f'blocks.{l}.recurrent_block.rg_lru.input_gate.w'
                ]
                new_state_dict[f'decoder.layers.{l}.recurrent_layer.rg_lru.input_gate.b'] = dm_model_weight[
                    f'blocks.{l}.recurrent_block.rg_lru.input_gate.b'
                ]
                new_state_dict[f'decoder.layers.{l}.recurrent_layer.rg_lru.a_gate.w'] = dm_model_weight[
                    f'blocks.{l}.recurrent_block.rg_lru.a_gate.w'
                ]
                new_state_dict[f'decoder.layers.{l}.recurrent_layer.rg_lru.a_gate.b'] = dm_model_weight[
                    f'blocks.{l}.recurrent_block.rg_lru.a_gate.b'
                ]

                new_state_dict[f'decoder.layers.{l}.recurrent_layer.linear_in._extra_state'] = (
                    self.griffin.state_dict()[f'decoder.layers.{l}.recurrent_layer.linear_in._extra_state']
                )
                new_state_dict[f'decoder.layers.{l}.recurrent_layer.linear_out._extra_state'] = (
                    self.griffin.state_dict()[f'decoder.layers.{l}.recurrent_layer.linear_out._extra_state']
                )

        self.griffin.load_state_dict(new_state_dict, strict=True)
        self.griffin = self.griffin.half()

    def teardown_method(self, method=None):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.recurrent_layer, TransformerLayer)
        assert self.recurrent_layer.layer_number == 1

        num_weights = sum([p.numel() for p in self.recurrent_layer.parameters()])
        assert num_weights == 648

    def test_cpu_forward(self):
        # we can't currently do this because the global memory buffer is on GPU
        pass

    def test_gpu_forward(self):

        # BATCH_SIZE = 2
        # SEQ_LEN = 64

        # prompt = torch.arange(BATCH_SIZE*SEQ_LEN).view(BATCH_SIZE,SEQ_LEN).cuda()
        # attention_mask = torch.ones((1, 1, SEQ_LEN, SEQ_LEN), dtype=bool).cuda()

        from transformers import AutoModelForCausalLM, AutoTokenizer

        input_texts = [
            'query: how much protein should a female eat',
            'query: summit define',
            "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
            "passage: Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
        ]
        # Tokenize the input texts
        hf_tokenizer = AutoTokenizer.from_pretrained("google/recurrentgemma-2b")
        batch_dict = hf_tokenizer(
            input_texts, max_length=128, pad_to_max_length=True, truncation=True, return_tensors='pt'
        )
        batch_dict_cuda = {k: v.cpu() for k, v in batch_dict.items()}
        prompt = batch_dict['input_ids'].cuda()
        self.griffin = self.griffin.eval()
        logits = self.griffin.forward(prompt, attention_mask=None)
        # print("nemo_start = ", logits[:,:5,256000-5:256000].detach().cpu())
        # print("prompt shape = ", prompt.shape)

        # print("nemo_end = ", logits[:,-10:,256000-10:256000].detach().cpu())
        # print("nemo_start = ", logits[:,:10, :30].detach().cpu().float().sum(dim=-2))
        # print("nemo_end = ", logits[:,-10:, :30].detach().cpu().float().sum(dim=-2))
        # print(logits.shape)

        # logits = self.griffin.forward(prompt, attention_mask)
        # logits = logits.permute(1,0,2)
        print(logits)
        print(logits.shape)
        # print(cache['blocks.0'])
        # print(cache['blocks.2'])

        # sampled_tokens = []
        # sampled_logits = []
        # # sampled_tokens.extend(prompt.flatten().tolist())
        # NUM_SAMPLING_STEPS = 64
        # for i in range(7,NUM_SAMPLING_STEPS):
        #     next_token = torch.argmax(logits[:, i], axis=-1)
        #     logit_maxes = torch.max(logits[:, i].detach().cpu(), axis=-1).values
        #     prompt[:,i] = next_token

        #     logits = self.griffin.forward(prompt, attention_mask=None)

        #     sampled_tokens.append(next_token[0].item())
        #     sampled_logits.append(logit_maxes[0].item())

        # print(sampled_tokens)
        # print(sampled_logits)


# test = TestParallelAttention()
# test = TestTransfomer()
# test = TestRecurrent()
test = TestFullPipe()

test.setup_method()
test.test_gpu_forward()

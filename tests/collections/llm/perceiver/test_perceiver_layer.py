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
from typing import Optional

from bionemo.testing import megatron_parallel_state_utils
from perceiver import PerceiverLayer, PerceiverConfig, perceiver_layer_spec

@pytest.fixture
def perceiver_config():
    config = PerceiverConfig(
        num_layers=1,
        hidden_size=512,
        num_attention_heads=16,
        num_latents=2,
        num_self_attention_per_cross_attention=5,
        bias_dropout_fusion=True,
        hidden_dropout=0.1,
        layernorm_epsilon=1e-5,
        attention_dropout=0.1,
    )
    return config

@pytest.fixture
def perceiver_layer(perceiver_config):
    """Initialize perceiver layer using the spec from perceiver.py"""
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        perceiver = PerceiverLayer(
            config=perceiver_config,
            submodules=perceiver_layer_spec.submodules,
            layer_number=1
        )

        return perceiver

class TestPerceiverLayer:
    
    def test_initialization(self, perceiver_layer, perceiver_config):
        """Test that the layer initializes correctly"""
        with megatron_parallel_state_utils.distributed_model_parallel_state():
            assert hasattr(perceiver_layer, 'input_cross_attention')
            assert hasattr(perceiver_layer, 'self_attention_layers')
            assert hasattr(perceiver_layer, 'final_layernorm')
            assert len(perceiver_layer.self_attention_layers) == perceiver_config.num_self_attention_per_cross_attention

    def test_forward_shape(self, perceiver_layer, perceiver_config):
        """Test that the forward pass maintains correct shapes"""
        with megatron_parallel_state_utils.distributed_model_parallel_state():
            batch_size = 4
            seq_length = 128
            latent_dim = perceiver_config.num_latents
            hidden_size = perceiver_config.hidden_size

            # Create input tensors
            latent_states = torch.randn(latent_dim, batch_size, hidden_size).cuda()
            input_sequence = torch.randn(seq_length, batch_size, hidden_size).cuda()
            
            # Create attention masks
            cross_attention_mask_q = torch.zeros(batch_size, 1, 1, latent_dim).cuda()
            cross_attention_mask_kv = torch.zeros(batch_size, 1, 1, seq_length).cuda()
            self_attention_mask = torch.zeros(batch_size, 1, 1, latent_dim).cuda()

            # Convert masks to boolean
            cross_attention_masks = (cross_attention_mask_q > 0, cross_attention_mask_kv > 0)
            self_attention_mask = self_attention_mask > 0
            
            # Run forward pass
            output = perceiver_layer(
                latent_states=latent_states,
                input_sequence=input_sequence,
                cross_attention_masks=cross_attention_masks,
                self_attention_mask=self_attention_mask,
            )

            # Check output shape
            assert output.shape == (latent_dim, batch_size, hidden_size)

    def test_attention_mask(self, perceiver_layer, perceiver_config):
        """Test that the layer handles attention masks correctly"""
        with megatron_parallel_state_utils.distributed_model_parallel_state():
            batch_size = 4
            seq_length = 128
            latent_dim = perceiver_config.num_latents
            hidden_size = perceiver_config.hidden_size

            latent_states = torch.randn(latent_dim, batch_size, hidden_size).cuda()
            input_sequence = torch.randn(seq_length, batch_size, hidden_size).cuda()
            
            # Create attention masks with some positions masked
            cross_attention_mask_q = torch.zeros(batch_size, 1, 1, latent_dim).cuda()
            cross_attention_mask_kv = torch.zeros(batch_size, 1, 1, seq_length).cuda()
            cross_attention_mask_kv[:, :, :, seq_length//2:] = 0  # Mask half the sequence
            
            self_attention_mask = torch.zeros(batch_size, 1, 1, latent_dim).cuda()
            self_attention_mask[:, :, :, latent_dim//2:] = 0  # Mask half the latents

            # Convert masks to boolean
            cross_attention_masks = (cross_attention_mask_q > 0, cross_attention_mask_kv > 0)
            self_attention_mask = self_attention_mask > 0
            
            output = perceiver_layer(
                latent_states=latent_states,
                input_sequence=input_sequence,
                cross_attention_masks=cross_attention_masks,
                self_attention_mask=self_attention_mask,
            )

            assert output.shape == (latent_dim, batch_size, hidden_size)

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    @pytest.mark.parametrize("seq_length", [64, 128, 256])
    def test_various_batch_and_sequence_sizes(
        self, perceiver_layer, batch_size, seq_length
    ):
        """Test that the layer handles different batch and sequence sizes"""
        with megatron_parallel_state_utils.distributed_model_parallel_state():
            latent_dim = 2
            hidden_size = 512

            latent_states = torch.randn(latent_dim, batch_size, hidden_size).cuda()
            input_sequence = torch.randn(seq_length, batch_size, hidden_size).cuda()

            # Create attention masks
            cross_attention_mask_q = torch.zeros(batch_size, 1, 1, latent_dim).cuda()
            cross_attention_mask_kv = torch.zeros(batch_size, 1, 1, seq_length).cuda()
            self_attention_mask = torch.zeros(batch_size, 1, 1, latent_dim).cuda()

            # Convert masks to boolean
            cross_attention_masks = (cross_attention_mask_q > 0, cross_attention_mask_kv > 0)
            self_attention_mask = self_attention_mask > 0

            output = perceiver_layer(
                latent_states=latent_states,
                input_sequence=input_sequence,
                cross_attention_masks=cross_attention_masks,
                self_attention_mask=self_attention_mask,
            )

            assert output.shape == (latent_dim, batch_size, hidden_size)

    def test_gradient_flow(self, perceiver_layer):
        """Test that gradients flow through the layer properly"""
        with megatron_parallel_state_utils.distributed_model_parallel_state():
            batch_size = 4
            seq_length = 128
            latent_dim = 2
            hidden_size = 512

            # Create input tensors and ensure they're leaf tensors
            latent_states = torch.randn(latent_dim, batch_size, hidden_size, requires_grad=True).cuda()
            input_sequence = torch.randn(seq_length, batch_size, hidden_size, requires_grad=True).cuda()
            
            # Create optimizer for both tensors
            optimizer = torch.optim.Adam(perceiver_layer.parameters(), lr=0.01)

            # Create attention masks
            cross_attention_mask_q = torch.zeros(batch_size, 1, 1, latent_dim).cuda()
            cross_attention_mask_kv = torch.zeros(batch_size, 1, 1, seq_length).cuda()
            self_attention_mask = torch.zeros(batch_size, 1, 1, latent_dim).cuda()

            # Convert masks to boolean
            cross_attention_masks = (cross_attention_mask_q > 0, cross_attention_mask_kv > 0)
            self_attention_mask = self_attention_mask > 0

            # Forward pass
            output = perceiver_layer(
                latent_states=latent_states,
                input_sequence=input_sequence,
                cross_attention_masks=cross_attention_masks,
                self_attention_mask=self_attention_mask,
            )

            # Store initial parameter values
            initial_params = {name: param.clone() for name, param in perceiver_layer.named_parameters()}
            
            # Compute loss and backward pass
            loss = output.max()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            failing_params = []
            
            # Verify that parameters changed after optimizer step
            for name, param in perceiver_layer.named_parameters():
                if not name.endswith('linear_proj.bias'):
                    if torch.allclose(param, initial_params[name], rtol=1e-4, atol=1e-4):
                        failing_params.append(name)

            # Assert that no parameters failed to update
            assert len(failing_params) == 0, f"The following parameters did not update during optimization: {failing_params}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_placement(self, perceiver_layer):
        """Test that the layer can be moved to GPU"""
        with megatron_parallel_state_utils.distributed_model_parallel_state():
            batch_size = 4
            seq_length = 128
            latent_dim = 2
            hidden_size = 512

            latent_states = torch.randn(
                latent_dim, batch_size, hidden_size, device='cuda'
            ).cuda()
            input_sequence = torch.randn(
                seq_length, batch_size, hidden_size, device='cuda'
            ).cuda()

            # Create attention masks
            cross_attention_mask_q = torch.zeros(batch_size, 1, 1, latent_dim, device='cuda').cuda()
            cross_attention_mask_kv = torch.zeros(batch_size, 1, 1, seq_length, device='cuda').cuda()
            self_attention_mask = torch.zeros(batch_size, 1, 1, latent_dim, device='cuda').cuda()

            # Convert masks to boolean
            cross_attention_masks = (cross_attention_mask_q > 0, cross_attention_mask_kv > 0)
            self_attention_mask = self_attention_mask > 0

            output = perceiver_layer(
                latent_states=latent_states,
                input_sequence=input_sequence,
                cross_attention_masks=cross_attention_masks,
                self_attention_mask=self_attention_mask,
            )

            assert output.device.type == 'cuda'

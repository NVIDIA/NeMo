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
import torch
import torch.nn.functional as F
import pytest
from nemo.automodel.loss.linear_ce import fused_linear_cross_entropy, HAVE_LINEAR_LOSS_CE


@pytest.mark.skipif(not HAVE_LINEAR_LOSS_CE, reason="Linear loss CE is not installed")
def test_fused_cross_entropy():
    """
    Tests fused_linear_cross_entropy against PyTorch's cross_entropy implementation, fused_linear_cross_entropy should:
        * has close output with PyTorch's cross_entropy
        * uses less memory than PyTorch's cross_entropy
    """
    if not torch.cuda.is_available():
        pytest.skip("This test requires a GPU")

    device = torch.device('cuda')
    batch_size = 32
    hidden_dim = 4096
    vocab_size = 128256
    dtype = torch.bfloat16

    # Create inputs on GPU
    hidden_states = torch.randn(batch_size, hidden_dim, dtype=dtype, device=device)
    weight = torch.randn(vocab_size, hidden_dim, dtype=dtype, device=device)  # Note: transposed shape
    targets = torch.randint(0, vocab_size, (batch_size,), device=device)

    # Measure memory for PyTorch implementation
    torch.cuda.reset_peak_memory_stats()
    with torch.amp.autocast(device_type='cuda', dtype=dtype):
        logits = torch.matmul(hidden_states, weight.t())  # Use transpose for matmul
        pytorch_loss = F.cross_entropy(logits, targets, reduction="mean")
    pytorch_memory = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    # Measure memory for fused implementation
    torch.cuda.reset_peak_memory_stats()
    with torch.amp.autocast(device_type='cuda', dtype=dtype):
        fused_loss = fused_linear_cross_entropy(hidden_states, weight, targets)
    fused_memory = torch.cuda.max_memory_allocated()

    # Compare results and memory usage
    print("\nMemory usage comparison:")
    print(f"PyTorch implementation: {pytorch_memory / 1024**2:.2f} MB")
    print(f"Fused implementation: {fused_memory / 1024**2:.2f} MB")
    print(f"Memory savings: {(pytorch_memory - fused_memory) / 1024**2:.2f} MB")

    # Convert both losses to float32 for comparison
    pytorch_loss = pytorch_loss.float()
    fused_loss = fused_loss.float()

    # Check if the losses are close
    assert torch.allclose(fused_loss, pytorch_loss, rtol=1e-2, atol=1e-2), \
        f"Loss mismatch: PyTorch={pytorch_loss.item()}, Fused={fused_loss.item()}"
    # Check if the fused implementation uses less memory
    assert fused_memory < pytorch_memory, \
        "Fused implementation should use less memory than PyTorch implementation"

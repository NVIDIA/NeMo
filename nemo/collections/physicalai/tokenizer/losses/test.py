# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
from config import ColorConfig, FlowConfig, PerceptualConfig, VideoLoss
from loss import ColorLoss, FlowLoss, PerceptualLoss, TokenizerLoss

VIDEO_KEY = "video"
RECON_KEY = "reconstructions"
LATENT_KEY = "latent"
INPUT_KEY = "INPUT"
MASK_KEY = "loss_mask"
RECON_CONSISTENCY_KEY = f"{RECON_KEY}_consistency"

# Setup input
device = torch.device("cuda")  # Move to GPU

input_t = torch.randn([2, 3, 10, 256, 256], dtype=torch.bfloat16, device=device)
mask_t = torch.ones_like(input_t, requires_grad=False, dtype=torch.bfloat16, device=device)
inputs = {INPUT_KEY: input_t, MASK_KEY: mask_t}

reconstructions = torch.randn([2, 3, 10, 256, 256], dtype=torch.bfloat16, device=device)
output_batch = {RECON_KEY: reconstructions}

# Create loss (assuming these loss functions support bf16)
colorLoss = ColorLoss(config=ColorConfig()).to(device).to(torch.bfloat16)
perceptualLoss = PerceptualLoss(config=PerceptualConfig()).to(device).to(torch.bfloat16)
flowLoss = FlowLoss(config=FlowConfig()).to(device).to(torch.bfloat16)
videoLoss = TokenizerLoss(config=VideoLoss()).to(device).to(torch.bfloat16)

print("#" * 20)

# Test color loss
cLoss = colorLoss(inputs, output_batch, 0)
print("color loss shape:", cLoss["color"].shape)

# Test LPIPS loss
pLoss = perceptualLoss(inputs, output_batch, 0)
print("LPIPS loss:", pLoss.keys(), pLoss['lpips'].shape)

# Test flow loss
fLoss = flowLoss(inputs, output_batch, 250001)
print("Flow loss:", fLoss.keys(), fLoss['flow'].shape)

# Test video loss
vLoss, total_vLoss = videoLoss(inputs, output_batch, 250001)
print("Video loss:", vLoss.keys(), vLoss, total_vLoss)

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import gc
from typing import List

import torch

from cosmos1.models.autoregressive.configs.inference.inference_config import DiffusionDecoderSamplingConfig
from cosmos1.models.autoregressive.diffusion_decoder.model import LatentDiffusionDecoderModel
from cosmos1.models.autoregressive.diffusion_decoder.utils import linear_blend_video_list, split_with_overlap
from cosmos1.utils import log


def diffusion_decoder_process_tokens(
    model: LatentDiffusionDecoderModel,
    indices_tensor: List[torch.Tensor],
    dd_sampling_config: DiffusionDecoderSamplingConfig = None,
    original_video_example: torch.Tensor = None,
    t5_emb_batch: List[torch.Tensor] = None,
):
    _, T, H, W = original_video_example.shape
    if dd_sampling_config is None:
        dd_sampling_config = DiffusionDecoderSamplingConfig()
    # indices_tensor is assumed to be a list of tensors with shape 1LHW
    data_batch_list = []
    for sample_num, token_CTHW in enumerate(indices_tensor):
        token_BCTHW = token_CTHW.unsqueeze(0).unsqueeze(1)
        token_BCTHW = split_with_overlap(
            token_BCTHW,
            (dd_sampling_config.dd_train_num_video_frames - 1) // 8 + 1,
            overlap=dd_sampling_config.overlap,
            tobf16=False,
        )
        data_batch_list.append(
            {
                "token_chunks": token_BCTHW,
                "t5_text_embeddings": t5_emb_batch[sample_num].to(torch.bfloat16),
                "t5_text_mask": torch.ones(1, 512, dtype=torch.bfloat16).cuda(),
                # other conditions
                "image_size": torch.tensor([[H, W, H, W]] * 1, dtype=torch.bfloat16).cuda(),
                "fps": torch.tensor([dd_sampling_config.fps] * 1, dtype=torch.bfloat16).cuda(),
                "num_frames": torch.tensor(
                    [dd_sampling_config.dd_train_num_video_frames] * 1, dtype=torch.bfloat16
                ).cuda(),
                "padding_mask": torch.zeros((1, 1, H, W), dtype=torch.bfloat16).cuda(),
            }
        )

    out_videos_batch = []

    for idx, data_batch_template in enumerate(data_batch_list):
        full_length_sample = []
        iterations = min(len(data_batch_template["token_chunks"]), dd_sampling_config.max_iter)
        for iter in range(iterations):
            gc.collect()
            torch.cuda.empty_cache()

            data_batch = copy.deepcopy(data_batch_template)
            data_batch["video"] = data_batch_template["token_chunks"][iter].cuda().to("cuda")

            log.debug(f"Run iter {iter} for video # {idx} at length {data_batch['video'].shape[2]}")
            # org_video,
            with torch.no_grad():
                samples_latent = model.generate_samples_from_batch(
                    data_batch,
                    guidance=dd_sampling_config.guidance,
                    sigma_min=dd_sampling_config.sigma_min,
                    state_shape=[
                        dd_sampling_config.continuous_tokenizer_channel,
                        dd_sampling_config.continuous_tokenizer_spatial_compression_ratio,
                        H // 8,
                        W // 8,
                    ],
                    apply_corruptor=False,
                    return_recon_x=False,
                    # corrupt_sigma=dd_sampling_config.sigma,
                    preencode_condition=True,  # We are using discrete model, so the input is already pre-encoded
                    num_steps=dd_sampling_config.num_steps,
                )
                log.debug(f"Current sample shape {samples_latent.shape} for video # {idx} ")
            full_length_sample.append(samples_latent.detach())

            # Turn off because we remove CP
            # distributed.barrier()
            del data_batch

            torch.cuda.empty_cache()

        gc.collect()
        torch.cuda.empty_cache()

        # Decode full-length samples and free GPU memory
        full_length_sample_pixs = [model.decode(item).clamp(-1, 1).cpu() for item in full_length_sample]
        torch.cuda.empty_cache()

        # Blend pixel samples
        if len(full_length_sample_pixs) > 1:
            full_length_sample_pixel_blend = linear_blend_video_list(
                full_length_sample_pixs, dd_sampling_config.overlap
            )[:, :, :T]
        else:
            full_length_sample_pixel_blend = full_length_sample_pixs[0][:, :, :T]

        # Batch size of full_length_sample_pixel_blend is always 1
        out_videos_batch.append((1 + full_length_sample_pixel_blend[0].cpu()) / 2)
    return out_videos_batch

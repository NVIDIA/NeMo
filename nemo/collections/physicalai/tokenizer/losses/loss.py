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

# pylint: disable=C0115,C0116,C0301

from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torchvision.models.optical_flow as optical_flow
from einops import rearrange

from nemo.collections.physicalai.tokenizer.losses.lpips import LPIPS

IMAGE_KEY = "images"
VIDEO_KEY = "video"
RECON_KEY = "reconstructions"
LATENT_KEY = "latent"
INPUT_KEY = "INPUT"
MASK_KEY = "loss_mask"
RECON_CONSISTENCY_KEY = f"{RECON_KEY}_consistency"


def time2batch(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    batch_size = x.shape[0]
    return rearrange(x, "b c t h w -> (b t) c h w"), batch_size


def batch2time(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    return rearrange(x, "(b t) c h w -> b c t h w", b=batch_size)


class WeightScheduler(torch.nn.Module):
    def __init__(self, boundaries, values):
        super().__init__()
        self.boundaries = list(boundaries)
        self.values = list(values)

    def forward(self, iteration):
        for boundary, value in zip(self.boundaries, self.values):
            if iteration < boundary:
                return value
        return self.values[-1]


class ColorLoss(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.schedule = WeightScheduler(boundaries=config.boundaries, values=config.values)

    def forward(self, inputs, output_batch, iteration) -> dict[str, torch.Tensor]:
        reconstructions = output_batch[RECON_KEY]
        weights = inputs[MASK_KEY]
        recon = weights * torch.abs(inputs[INPUT_KEY].contiguous() - reconstructions.contiguous())
        color_weighted = self.schedule(iteration) * recon
        if torch.isnan(color_weighted).any():
            raise ValueError("[COLOR] NaN detected in loss")
        return dict(color=color_weighted)


class PerceptualLoss(LPIPS):
    """Relevant changes that're internal to us:

    - Remove linear projection layers, simply use the raw pre-normalized features.
    - Use pyramid-layer weights: [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10.0 / 1.5].
    - Accepts pixel-wise masks and modulates the features before norm calculation.
    - Implements gram-matrix and correlation losses.
    """

    def __init__(self, config):
        super(PerceptualLoss, self).__init__(config.checkpoint_activations)
        self.net = self.net.eval()
        self.gram_loss = config.gram_loss
        self.corr_loss = config.corr_loss
        self.layer_weights = list(config.layer_weights)
        self.lpips_schedule = WeightScheduler(config.lpips_boundaries, config.lpips_values)
        self.gram_schedule = WeightScheduler(config.gram_boundaries, config.gram_values)
        self.corr_schedule = WeightScheduler(config.corr_boundaries, config.corr_values)
        self.checkpoint_activations = config.checkpoint_activations

    def _temporal_gram_matrix(self, x, batch_size=None):
        x = batch2time(x, batch_size)
        c, t, h, w = x.shape[-4], x.shape[-3], x.shape[-2], x.shape[-1]
        reshaped_x = torch.reshape(x, [-1, c, t * h * w])
        return torch.matmul(reshaped_x, reshaped_x.transpose(1, 2)) / float(t * h * w)

    def _gram_matrix(self, x, batch_size=None):
        if batch_size is not None and x.shape[0] != batch_size:
            return self._temporal_gram_matrix(x, batch_size)
        c, h, w = x.shape[-3], x.shape[-2], x.shape[-1]
        reshaped_x = torch.reshape(x, [-1, c, h * w])
        return torch.matmul(reshaped_x, reshaped_x.transpose(1, 2)) / float(h * w)

    def forward(self, inputs, output_batch, iteration):
        output_dict = dict()
        reconstructions = output_batch[RECON_KEY]
        weights = inputs[MASK_KEY]
        input_images = inputs[INPUT_KEY]

        if input_images.ndim == 5:
            input_images, batch_size = time2batch(input_images)
            reconstructions, _ = time2batch(reconstructions)
            weights, _ = time2batch(weights)
        else:
            batch_size = input_images.shape[0]

        in0_input, in1_input = (self.scaling_layer(input_images), self.scaling_layer(reconstructions))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)

        _layer_weights = self.layer_weights
        weights_map, res, diffs = {}, {}, {}
        for kk in range(len(self.chns)):
            weights_map[kk] = torch.nn.functional.interpolate(weights[:, :1, ...], outs0[kk].shape[-2:])
            diffs[kk] = weights_map[kk] * torch.abs(outs0[kk] - outs1[kk])
            res[kk] = _layer_weights[kk] * diffs[kk].mean([1, 2, 3], keepdim=True)

        val = res[0]
        for ll in range(1, len(self.chns)):
            val += res[ll]
        # Scale by number of pixels to match pixel-wise losses.
        val = val.expand(-1, input_images.shape[-3], input_images.shape[-2], input_images.shape[-1])
        if batch_size != input_images.shape[0]:
            val = batch2time(val, batch_size)
        if torch.isnan(val).any():
            raise ValueError("[LPIPS] NaN detected in loss")
        output_dict["lpips"] = self.lpips_schedule(iteration) * val

        if self.gram_loss and self.gram_schedule(iteration) > 0.0:
            num_chans = len(self.chns)
            grams0 = [self._gram_matrix(weights_map[kk] * outs0[kk], batch_size) for kk in range(num_chans)]
            grams1 = [self._gram_matrix(weights_map[kk] * outs1[kk], batch_size) for kk in range(num_chans)]
            gram_diffs = [(grams0[kk] - grams1[kk]) ** 2 for kk in range(num_chans)]
            grams_res = [_layer_weights[kk] * gram_diffs[kk].mean([1, 2], keepdim=True) for kk in range(num_chans)]
            gram_val = grams_res[0]
            for ll in range(1, len(self.chns)):
                gram_val += grams_res[ll]

            # Scale by number of total pixels to match pixel-wise losses.
            gram_val = gram_val.unsqueeze(1).expand(
                -1, input_images.shape[-3], input_images.shape[-2], input_images.shape[-1]
            )
            if batch_size != input_images.shape[0]:
                gram_val = batch2time(gram_val, batch_size)
            if torch.isnan(gram_val).any():
                raise ValueError("[GRAM] NaN detected in loss")
            output_dict["gram"] = self.gram_schedule(iteration) * gram_val
        return output_dict

    def torch_compile(self):
        """
        This method invokes torch.compile() on this loss
        """
        # cuda-graphs crash after 1k iterations
        self.net = torch.compile(self.net, dynamic=False)


class FlowLoss(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.schedule = WeightScheduler(config.boundaries, config.values)
        self.scale = config.scale
        self.dtype = getattr(torch, config.dtype)
        self.checkpoint_activations = config.checkpoint_activations

        current_device = torch.device(torch.cuda.current_device())

        # In order to be able to run model in bf16 we need to change make_coords_grid()
        # to allow it to return arbitrary type provided by us in argument
        # the line from orginal implementation that caused results to be only fp32 is commented
        # Additionally I've changed that function to run on GPU instead of CPU, which results in
        # less graph breaks when torch.compile() is used
        # This function is copied from
        # https://github.com/pytorch/vision/blob/main/torchvision/models/optical_flow/_utils.py#L22
        # commit: b06ea39d5f0adbe949d08257837bda912339e415
        def make_coords_grid(
            batch_size: int, h: int, w: int, device: torch.device = current_device, dtype: torch.dtype = self.dtype
        ):
            # Original: def make_coords_grid(batch_size: int, h: int, w: int, device: str = "cpu"):
            device = torch.device(device)
            coords = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
            coords = torch.stack(coords[::-1], dim=0).to(dtype)
            # Original: coords = torch.stack(coords[::-1], dim=0).float()
            return coords[None].repeat(batch_size, 1, 1, 1)

        # We also need to specify output dtype of torch.linspace() in index_pyramid()
        # method of CorrBlock, otherwise it uses default fp32 dtype as output.
        # Additionally I've changed that function to run on GPU instead of CPU, which results in
        # less graph breaks when torch.compile() is used
        # This function is copied from
        # https://github.com/pytorch/vision/blob/main/torchvision/models/optical_flow/raft.py#L394
        # commit: b06ea39d5f0adbe949d08257837bda912339e415
        def index_pyramid(
            self, centroids_coords, dtype: torch.dtype = self.dtype, device: torch.device = current_device
        ):
            # Original: def index_pyramid(self, centroids_coords):
            """Return correlation features by indexing from the pyramid."""
            neighborhood_side_len = 2 * self.radius + 1  # see note in __init__ about out_channels
            di = torch.linspace(-self.radius, self.radius, neighborhood_side_len, dtype=dtype, device=device)
            dj = torch.linspace(-self.radius, self.radius, neighborhood_side_len, dtype=dtype, device=device)
            # Original: di = torch.linspace(-self.radius, self.radius, neighborhood_side_len)
            # Original: dj = torch.linspace(-self.radius, self.radius, neighborhood_side_len)
            delta = torch.stack(torch.meshgrid(di, dj, indexing="ij"), dim=-1).to(centroids_coords.device)
            delta = delta.view(1, neighborhood_side_len, neighborhood_side_len, 2)

            batch_size, _, h, w = centroids_coords.shape  # _ = 2
            centroids_coords = centroids_coords.permute(0, 2, 3, 1).reshape(batch_size * h * w, 1, 1, 2)

            indexed_pyramid = []
            for corr_volume in self.corr_pyramid:
                sampling_coords = centroids_coords + delta  # end shape is (batch_size * h * w, side_len, side_len, 2)
                indexed_corr_volume = optical_flow.raft.grid_sample(
                    corr_volume, sampling_coords, align_corners=True, mode="bilinear"
                ).view(batch_size, h, w, -1)
                indexed_pyramid.append(indexed_corr_volume)
                centroids_coords = centroids_coords / 2

            corr_features = torch.cat(indexed_pyramid, dim=-1).permute(0, 3, 1, 2).contiguous()

            expected_output_shape = (batch_size, self.out_channels, h, w)
            if corr_features.shape != expected_output_shape:
                raise ValueError(
                    f"Output shape of index pyramid is incorrect. Should be {expected_output_shape}, got {corr_features.shape}"
                )

            return corr_features

        optical_flow.raft.make_coords_grid = make_coords_grid
        optical_flow.raft.CorrBlock.index_pyramid = index_pyramid

        flow_model = optical_flow.raft_large(pretrained=True, progress=False)
        flow_model.requires_grad_(False)
        flow_model.eval()
        flow_model = flow_model.to(self.dtype)

        self.flow_model = flow_model

    def _run_model(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """Runs flow_model in the forward mode on explicit dtype=float32.

        Args:
            input1: First video frames batch, layout (T, C, H, W), bfloat16.
            input2: Next video frames batch, layout (T, C, H, W), bfloat16.

        Returns:
            Forward optical flow, (T, 2, H, W), bfloat16.
        """
        input_dtype = input1.dtype
        flow_output = self.flow_model.to(self.dtype)(input1.to(self.dtype), input2.to(self.dtype))[-1]
        return flow_output.to(input_dtype)

    def _run_model_fwd(self, input_video: torch.Tensor) -> torch.Tensor:
        """Runs foward flow on a batch of videos, one batch at a time.
        Args:
            input_video: The input batch of videos, layout (B, T, C, H, W).

        Returns:
            Forward optical flow, layout (B, 2, T-1, H, W).
        """
        output_list = list()
        for fwd_input_frames in input_video:
            fwd_input_frames = fwd_input_frames.transpose(1, 0)
            fwd_flow_output = self._run_model(fwd_input_frames[:-1], fwd_input_frames[1:])
            output_list.append(fwd_flow_output.transpose(1, 0))
        return torch.stack(output_list, dim=0)

    def _bidirectional_flow(self, input_video: torch.Tensor) -> torch.Tensor:
        """The bidirectional optical flow on a batch of videos.

        The forward and backward flows are averaged to get the bidirectional flow.
        To reduce memory pressure, the input video is scaled down by a factor of `self.scale`,
        and rescaled back to match other pixel-wise losses.

        Args:
            input_video: The input batch of videos, layout (B, T, C, H, W).

        Returns:
            Biderectinoal flow, layout (B, 2, T-1, H, W).
        """
        # scale down the input video to reduce memory pressure.
        t, h, w = input_video.shape[-3:]
        input_video_scaled = F.interpolate(input_video, (t, h // self.scale, w // self.scale), mode="trilinear")

        # forward flow.
        if self.checkpoint_activations:
            fwd_flow_output = checkpoint.checkpoint(self._run_model_fwd, input_video_scaled, use_reentrant=False)
        else:
            fwd_flow_output = self._run_model_fwd(input_video_scaled)

        # backward flow.
        input_video_scaled = input_video_scaled.flip([2])
        if self.checkpoint_activations:
            bwd_flow_output = checkpoint.checkpoint(self._run_model_fwd, input_video_scaled, use_reentrant=False)
        else:
            bwd_flow_output = self._run_model_fwd(input_video_scaled)
        bwd_flow_output = bwd_flow_output.flip([2])

        # bidirectional flow, concat fwd and bwd along temporal axis.
        flow_input = torch.cat([fwd_flow_output, bwd_flow_output], dim=2)
        return self.scale * F.interpolate(flow_input, (2 * (t - 1), h, w), mode="trilinear")

    def forward(
        self, inputs: dict[str, torch.Tensor], output_batch: dict[str, torch.Tensor], iteration: int
    ) -> dict[str, torch.Tensor]:
        input_images = inputs[INPUT_KEY]
        if input_images.ndim == 4 or input_images.shape[2] == 1:
            return dict()
        if self.schedule(iteration) == 0.0:
            return dict()

        # Biderectional flow (B, 2, 2*(T-1), H, W)
        flow_input = self._bidirectional_flow(input_images)
        flow_recon = self._bidirectional_flow(output_batch[RECON_KEY])

        # L1 loss on the flow. (B, 1, 2*(T-1), H, W)
        flow_loss = torch.abs(flow_input - flow_recon).mean(dim=1, keepdim=True)

        flow_loss_weighted = self.schedule(iteration) * flow_loss
        if torch.isnan(flow_loss_weighted).any():
            raise ValueError("[FLOW] NaN detected in loss")
        return dict(flow=flow_loss_weighted)

    def torch_compile(self):
        """
        This method invokes torch.compile() on this loss
        """
        self.flow_model = torch.compile(self.flow_model, dynamic=False)


######################################
_VALID_LOSS_NAMES = ["color", "perceptual", "flow", "kl", "video_consistency"]


def _mean(recon: torch.Tensor) -> torch.Tensor:
    return torch.mean(recon)


def _sum_per_frame(recon: torch.Tensor) -> torch.Tensor:
    batch_size = recon.shape[0] * recon.shape[2] if recon.ndim == 5 else recon.shape[0]
    return torch.sum(recon) / batch_size


def _sum(recon: torch.Tensor) -> torch.Tensor:
    return torch.sum(recon) / recon.shape[0]


class ReduceMode(Enum):
    MEAN = "MEAN"
    SUM_PER_FRAME = "SUM_PER_FRAME"
    SUM = "SUM"

    @property
    def function(self):
        if self == ReduceMode.MEAN:
            return _mean
        elif self == ReduceMode.SUM_PER_FRAME:
            return _sum_per_frame
        elif self == ReduceMode.SUM:
            return _sum
        else:
            raise ValueError("Invalid ReduceMode")


class TokenizerLoss(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        _reduce = ReduceMode(config.reduce.upper()) if hasattr(config, "reduce") else None
        self.reduce = _reduce.function

        # manually create these losses
        self.loss_modules = nn.ModuleDict()
        self.loss_modules["color"] = ColorLoss(config=config.color)
        self.loss_modules["perceptual"] = PerceptualLoss(config=config.perceptual)
        self.loss_modules["flow"] = FlowLoss(config=config.flow)

    def forward(self, inputs, output_batch, iteration) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        loss = dict()
        total_loss = 0.0

        # human loss_mask is only enabled after iter=start_human_mask
        if iteration < self.config.start_human_mask:
            inputs[MASK_KEY] = torch.ones_like(inputs[INPUT_KEY])

        # Calculates reconstruction losses (`total_loss`).
        for key, module in self.loss_modules.items():
            curr_loss = module(inputs, output_batch, iteration)
            loss.update({k: torch.mean(v) for k, v in curr_loss.items()})
            total_loss += sum([self.reduce(v) if (v.dim() > 0) else v for v in curr_loss.values()])

        return dict(loss=loss), total_loss

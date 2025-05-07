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

"""LPIPS loss.

Adapted from: github.com/CompVis/stable-diffusion/ldm/modules/losses/contperceptual.py.
"""

import hashlib
import os
from collections import namedtuple
from typing import Optional

import requests
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from loguru import logger as logging
from torchvision import models
from tqdm import tqdm


def get_rank(group: Optional[dist.ProcessGroup] = None) -> int:
    """Get the rank (GPU device) of the worker.

    Returns:
        rank (int): The rank of the worker.
    """
    rank = 0
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank(group)
    return rank


def is_rank0() -> bool:
    """Check if current process is the master GPU.

    Returns:
        (bool): True if this function is called from the master GPU, else False.
    """
    return get_rank() == 0


_TORCH_HOME = os.getenv("TORCH_HOME", "~/.cache/my_model")
# TODO(freda): Update the download link to a PBSS location, safer.
_URL_MAP = {"vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"}
_CKPT_MAP = {"vgg_lpips": "vgg.pth"}
_MD5_MAP = {"vgg_lpips": "d507d7349b931f0638a25a48a722f98a"}


def _download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def _md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def _get_ckpt_path(name, root, check=False):
    assert name in _URL_MAP
    path = os.path.join(root, _CKPT_MAP[name])
    if not os.path.exists(path) or (check and not _md5_hash(path) == _MD5_MAP[name]):
        logging.info("Downloading {} model from {} to {}".format(name, _URL_MAP[name], path))
        _download(_URL_MAP[name], path)
        md5 = _md5_hash(path)
        assert md5 == _MD5_MAP[name], md5
    return path


class LPIPS(nn.Module):
    def __init__(self, checkpoint_activations: bool = False):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=True, requires_grad=False, checkpoint_activations=checkpoint_activations)

        if dist.is_initialized() and not is_rank0():
            dist.barrier()
        self.load_from_pretrained()
        if dist.is_initialized() and is_rank0():
            dist.barrier()

        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = _get_ckpt_path(name, f"{_TORCH_HOME}/hub/checkpoints")
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        logging.info("Loaded pretrained LPIPS loss from {}".format(ckpt))

    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        if name != "vgg_lpips":
            raise NotImplementedError
        model = cls()
        ckpt = _get_ckpt_path(name)
        model.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        return model

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [diffs[kk].mean([1, 2, 3], keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for i in range(1, len(self.chns)):
            val += res[i]
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer("shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None], persistent=False)
        self.register_buffer("scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None], persistent=False)

    def forward(self, inp):
        return (inp - self.shift) / self.scale


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, checkpoint_activations: bool = False):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.checkpoint_activations = checkpoint_activations
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        if self.checkpoint_activations:
            h = checkpoint.checkpoint(self.slice1, X, use_reentrant=False)
        else:
            h = self.slice1(X)
        h_relu1_2 = h

        if self.checkpoint_activations:
            h = checkpoint.checkpoint(self.slice2, h, use_reentrant=False)
        else:
            h = self.slice2(h)
        h_relu2_2 = h

        if self.checkpoint_activations:
            h = checkpoint.checkpoint(self.slice3, h, use_reentrant=False)
        else:
            h = self.slice3(h)
        h_relu3_3 = h

        if self.checkpoint_activations:
            h = checkpoint.checkpoint(self.slice4, h, use_reentrant=False)
        else:
            h = self.slice4(h)
        h_relu4_3 = h

        if self.checkpoint_activations:
            h = checkpoint.checkpoint(self.slice5, h, use_reentrant=False)
        else:
            h = self.slice5(h)
        h_relu5_3 = h

        vgg_outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out

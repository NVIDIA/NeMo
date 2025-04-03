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

import numpy as np
import torch
from pytorch_retinaface.utils.nms.py_cpu_nms import py_cpu_nms

from cosmos1.utils import log


# Adapted from https://github.com/biubug6/Pytorch_Retinaface/blob/master/detect.py
def filter_detected_boxes(boxes, scores, confidence_threshold, nms_threshold, top_k, keep_top_k):
    """Filter boxes based on confidence score and remove overlapping boxes using NMS."""
    # Keep detections with confidence above threshold
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # Sort by confidence and keep top K detections
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    scores = scores[order]

    # Run non-maximum-suppression (NMS) to remove overlapping boxes
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    dets = dets[:keep_top_k, :]
    boxes = dets[:, :-1]
    return boxes


# Adapted from https://github.com/biubug6/Pytorch_Retinaface/blob/master/utils/box_utils.py to handle batched inputs
def decode_batch(loc, priors, variances):
    """Decode batched locations from predictions using priors and variances.

    Args:
        loc (tensor): Batched location predictions for loc layers.
            Shape: [batch_size, num_priors, 4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors, 4]
        variances: (list[float]): Variances of prior boxes.

    Return:
        Decoded batched bounding box predictions
            Shape: [batch_size, num_priors, 4]
    """
    batch_size = loc.size(0)
    priors = priors.unsqueeze(0).expand(batch_size, -1, -1)

    boxes = torch.cat(
        (
            priors[:, :, :2] + loc[:, :, :2] * variances[0] * priors[:, :, 2:],
            priors[:, :, 2:] * torch.exp(loc[:, :, 2:] * variances[1]),
        ),
        dim=2,
    )

    boxes[:, :, :2] -= boxes[:, :, 2:] / 2
    boxes[:, :, 2:] += boxes[:, :, :2]
    return boxes


# Adapted from https://github.com/biubug6/Pytorch_Retinaface/blob/master/detect.py
def _check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    log.debug("Missing keys:{}".format(len(missing_keys)))
    log.debug("Unused checkpoint keys:{}".format(len(unused_pretrained_keys)))
    log.debug("Used keys:{}".format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, "load NONE from pretrained checkpoint"
    return True


# Adapted from https://github.com/biubug6/Pytorch_Retinaface/blob/master/detect.py
def _remove_prefix(state_dict, prefix):
    """Old version of the model is stored with all names of parameters sharing common prefix 'module.'"""
    log.debug("Removing prefix '{}'".format(prefix))

    def f(x):
        return x.split(prefix, 1)[-1] if x.startswith(prefix) else x

    return {f(key): value for key, value in state_dict.items()}


# Adapted from https://github.com/biubug6/Pytorch_Retinaface/blob/master/detect.py
def load_model(model, pretrained_path, load_to_cpu):
    log.debug("Loading pretrained model from {}".format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage, weights_only=True)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(device), weights_only=True
        )
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = _remove_prefix(pretrained_dict["state_dict"], "module.")
    else:
        pretrained_dict = _remove_prefix(pretrained_dict, "module.")
    _check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

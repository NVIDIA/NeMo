# =============================================================================
# Copyright (c) 2020 NVIDIA. All Rights Reserved.
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
# =============================================================================

import itertools
import math
import sys

import torch

import nemo
from nemo.collections.cv.modules.faster_rcnn import FasterRCNN
from nemo.collections.cv.modules.nll_loss import NLLLoss
from nemo.collections.cv.modules.penn_fudan_person_detection_datalayer import (
    PennFudanDataLayer,
    PennFudanPedestrianDataset,
)
from nemo.core import DeviceType, NeuralType

# 0. Instantiate Neural Factory with supported backend
nf = nemo.core.NeuralModuleFactory(placement=DeviceType.GPU)

# 1. Instantiate necessary neural modules
PennDL = PennFudanDataLayer(batch_size=4, shuffle=True, data_folder="~/data/PennFudanPed")

# Question: how to pass 2 from DL to model?
model = FasterRCNN(2)

# 2. Describe activation's flow
ids, imgs, boxes, targets, masks, areas, iscrowds, num_objs = PennDL()
losses = model(images=imgs, bounding_boxes=boxes, targets=targets, num_objects=num_objs)


# Invoke "train" action
nf.train([losses], callbacks=[], optimization_params={"num_epochs": 10, "lr": 0.001}, optimizer="adam")


# Test our model.

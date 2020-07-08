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

import pytorch_lightning as ptl
from torch.utils.data import DataLoader

from nemo.collections.cv.models import MNISTLeNet5

from nemo.utils import logging

if __name__ == "__main__":

    # The "model" - with dataset.
    lenet5 = MNISTLeNet5([])

    # Create trainer.
    trainer = ptl.Trainer()

    # Train.
    trainer.fit(model=lenet5)

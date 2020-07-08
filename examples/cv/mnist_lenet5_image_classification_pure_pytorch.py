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

from torch import optim
from torch.utils.data import DataLoader

from nemo.core.config import set_config, NeMoConfig, DataLoaderConfig
from dataclasses import dataclass, asdict
from omegaconf import DictConfig

from nemo.collections.cv.datasets import MNISTDataset, MNISTDatasetConfig
from nemo.collections.cv.losses import NLLLoss
from nemo.collections.cv.modules import LeNet5
from nemo.utils import logging


@dataclass
class AppConfig(NeMoConfig):
    """
    This is structured config for this application.
    As in the example we hardcode the optimizer, so will just enable the user to play with learning rate (lr).

    Args:
        name: Description of the application.
        dataset: contains configuration of dataset.
        dataloader: contains configuration of dataloader.
        lr: learning rate passed to the optimizer.
        freq: display frequency.
    """
    name: str="Training of a LeNet-5 Neural Module using a custom training loop written in pure PyTorch."
    dataset: MNISTDatasetConfig=MNISTDatasetConfig(width=32, height=32)
    dataloader: DataLoaderConfig=DataLoaderConfig(batch_size=128, shuffle=True)
    lr: float=0.001
    freq: int=10


@set_config(config=AppConfig)
def main(cfg: DictConfig):

    # Show configuration - user can influence every parameter from command line!
    print("="*80 + " Hydra says hello! " + "="*80)
    print(cfg.pretty())

    # Dataset.
    mnist_ds = MNISTDataset(cfg.dataset)
    # The "model".
    lenet5 = LeNet5()
    # Loss.
    nll_loss = NLLLoss()

    # Create optimizer.
    opt = optim.Adam(lenet5.parameters(), lr=cfg.lr)

    # Configure data loader.
    train_loader = DataLoader(dataset=mnist_ds, **(cfg.dataloader))

    # Iterate over the whole dataset - in batches.
    for step, (_, images, targets, _) in enumerate(train_loader):

        # Reset the gradients.
        opt.zero_grad()

        # Forward pass.
        predictions = lenet5(images=images)

        # Calculate loss.
        loss = nll_loss(predictions=predictions, targets=targets)

        # Print loss.
        if step % cfg.freq == 0:
            logging.info("Step: {} Training Loss: {}".format(step, loss))

        # Backpropagate the gradients.
        loss.backward()

        # Update the parameters.
        opt.step()
    # Epoch ended.

if __name__ == "__main__":
    main()
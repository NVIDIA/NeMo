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

from nemo.collections.diffusion.data.diffusion_energon_datamodule import DiffusionDataModule
from nemo.collections.physicalai.tokenizer.train_tokenizer import ImageTaskEncoder
    
    
datamodule = DiffusionDataModule(
    path='/lustre/fsw/portfolios/coreai/users/yihuih/data/cosmos_tokenizer/videoVAE',
    task_encoder=ImageTaskEncoder(),
    global_batch_size=2,
    micro_batch_size=1,
    num_workers=1,
)

train = datamodule.train_dataloader()
a = next(iter(train))
from IPython import embed; embed()
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

FROM nvcr.io/nvidia/nemo:25.02.rc3

RUN pip install --no-cache-dir imageio[ffmpeg] pyav iopath better_profanity peft lru-dict pyquaternion git+https://github.com/NVlabs/Pytorch_Retinaface.git@b843f45 git+https://github.com/NVIDIA/NeMo-Run.git@f07877f80dbfc91c4d37683864cf2552b96d62f1
RUN pip install moviepy==1.0.3

RUN rm -rf /opt/megatron-lm
RUN git clone --single-branch --branch cosmos1.0.1 https://github.com/NVIDIA/Megatron-LM.git /opt/megatron-lm && cd /opt/megatron-lm && git checkout 86dad2c39a52c241e372b774f07041201f8a57f4 && pip install -e . && cd megatron/core/datasets && make

RUN rm -rf /opt/NeMo
COPY . /opt/NeMo
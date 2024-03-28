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

echo "unset all SLURM_, PMI_, PMIX_ Variables"
set -x
for i in $(env | grep ^SLURM_ | cut -d"=" -f 1); do unset -v $i; done
for i in $(env | grep ^PMI_ | cut -d"=" -f 1); do unset -v $i; done
for i in $(env | grep ^PMIX_ | cut -d"=" -f 1); do unset -v $i; done
set +x

py.test -s /opt/NeMo/tests/deploy/test_nemo_deploy.py::test_LLAMA2_70B_base_2gpu
py.test -s /opt/NeMo/tests/deploy/test_nemo_deploy.py::test_NV_GPT_8B_Base_4k_1gpu
py.test -s /opt/NeMo/tests/deploy/test_nemo_deploy.py::test_NV_GPT_8B_QA_4k_1gpu
py.test -s /opt/NeMo/tests/deploy/test_nemo_deploy.py::test_NV_GPT_8B_Chat_4k_SFT_1gpu
py.test -s /opt/NeMo/tests/deploy/test_nemo_deploy.py::test_NV_GPT_8B_Chat_4k_RLHF_1gpu
py.test -s /opt/NeMo/tests/deploy/test_nemo_deploy.py::test_LLAMA2_7B_base_1gpu
py.test -s /opt/NeMo/tests/deploy/test_nemo_deploy.py::test_LLAMA2_13B_base_1gpu

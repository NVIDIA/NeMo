# +
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

git lfs install
git clone https://huggingface.co/meta-llama/Llama-2-7b-hf

docker run -it -p 8080:8080 -p 8088:8088 --rm --gpus all --ipc=host --network host -v $(pwd):/workspace nvcr.io/nvidia/nemo:25.02

cd Llama-2-7b-hf

# convert from .hf to nemo format to be saved at default path /root/.cache/nemo/models/
python3 ../convert.py

# delete the original download to save disk
cd ..
rm -rf Llama-2-7b-hf/

# call run_sft.py script
cd code/

python3 run_sft.py
python3 run_inference.py

# conduct evaluation on sft model
python3 /opt/NeMo/scripts/metric_calculation/compute_rouge.py --ground-truth /workspace/data/verilog/test.jsonl --preds /workspace/inference/base_llama_prediction.jsonl --answer-field "output" 
python3 /opt/NeMo/scripts/metric_calculation/compute_rouge.py --ground-truth /workspace/data/verilog/test.jsonl --preds /workspace/inference/sft_prediction--val_loss=1.3609.jsonl --answer-field "output"
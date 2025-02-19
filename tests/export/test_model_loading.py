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

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

dummy_module = MagicMock()
dummy_module.torch_to_numpy = lambda torch_tensor: torch_tensor.detach().cpu().numpy()


def test_model_loading(nemo_ckpt_path: str, trt_llm_export_path: str) -> None:
    """
    Test if model loading works without tensorrt_llm.

    Args:
        nemo_ckpt_path (str): Path to the nemo checkpoint.
        trt_llm_export_path (str): Export path.
    Returns:
        None
    """
    export_path = Path(trt_llm_export_path)
    export_path.mkdir(parents=True, exist_ok=True)
    export_path_mcore = export_path / 'mcore_export'
    export_path_local = export_path / 'local_export'

    nemo_path = Path(nemo_ckpt_path)

    with patch.dict(
        'sys.modules',
        {
            'tensorrt_llm': dummy_module,
            'tensorrt_llm._utils': dummy_module,
        },
    ):
        from nemo.export.trt_llm.nemo_ckpt_loader.nemo_file import load_nemo_model

        load_nemo_model(nemo_path, export_path_local, False)
        load_nemo_model(nemo_path, export_path_mcore, True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nemo_ckpt_path', type=str, required=True)
    parser.add_argument('--trt_llm_export_path', type=str, required=True)
    args = parser.parse_args()
    test_model_loading(args.nemo_ckpt_path, args.trt_llm_export_path)

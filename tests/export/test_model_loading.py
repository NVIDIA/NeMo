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

import builtins
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nemo.collections import llm

HF_PATH = "/home/TestData/nlp/megatron_llama/llama-ci-hf"
OUTPUT_PATH = '/tmp/imported_nemo2'

dummy_module = MagicMock()
dummy_module.torch_to_numpy = lambda torch_tensor: torch_tensor.detach().cpu().numpy()


class DummyMCore:
    """
    Allow only for MCoreSavePlan import from megatron.core.dist_checkpointing.strategies.torch
    """

    def __getattr__(self, name):
        if name.startswith('__'):
            return MagicMock()
        if name == 'MCoreSavePlan':

            class Dummy:
                pass

            return Dummy
        raise ModuleNotFoundError(f"Module {name} is not available")


dummy_module_mcore = DummyMCore()


@pytest.mark.run_only_on('GPU')
@pytest.mark.unit
def test_model_loading() -> None:
    """
    Test if model loading works for tensorrt_llm export.
    """

    model = llm.LlamaModel(config=llm.Llama2Config7B)
    nemo_path = llm.import_ckpt(model, 'hf://' + HF_PATH, output_path=Path(OUTPUT_PATH))

    assert nemo_path.exists()
    assert (nemo_path / 'weights').exists()
    assert (nemo_path / 'context').exists()

    export_path = Path('/tmp/trtllm_exported_model')
    export_path.mkdir(parents=True, exist_ok=True)
    export_path_mcore = export_path / 'mcore_export'
    export_path_local = export_path / 'local_export'

    original_import = builtins.__import__

    def custom_import(name, globals=None, locals=None, fromlist=(), level=0):
        """Disables megatron imports."""
        if name == "megatron.core.dist_checkpointing.strategies.torch":
            return original_import(
                'megatron.core.dist_checkpointing.strategies.torch', globals, locals, fromlist, level
            )
        if name == "megatron" or name.startswith("megatron."):
            raise ModuleNotFoundError(f"Module {name} is not available!")
        return original_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=custom_import):
        with patch.dict(
            'sys.modules',
            {
                'tensorrt_llm': dummy_module,
                'tensorrt_llm._utils': dummy_module,
                'megatron': dummy_module,
                'megatron.core': dummy_module,
                'megatron.core.dist_checkpointing': dummy_module,
                'megatron.core.dist_checkpointing.strategies': dummy_module,
                'megatron.core.dist_checkpointing.strategies.torch': dummy_module_mcore,
            },
        ):
            from nemo.export.trt_llm.nemo_ckpt_loader.nemo_file import load_nemo_model

            load_nemo_model(nemo_path, export_path_local, False)
            load_nemo_model(nemo_path, export_path_mcore, True)

    shutil.rmtree(OUTPUT_PATH, ignore_errors=True)
    shutil.rmtree(export_path, ignore_errors=True)

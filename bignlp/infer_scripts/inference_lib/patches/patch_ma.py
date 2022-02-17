#!/usr/bin/env python3
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
import logging
import pathlib
import site
import subprocess

LOGGER = logging.getLogger(__name__)

BIGNLP_SCRIPTS_PATH = pathlib.Path(__file__).parent.parent.parent.parent


def main():
    logging.basicConfig(level=logging.INFO)
    model_analyzer_dir = [
        list(pathlib.Path(sp).rglob("model_analyzer"))[0]
        for sp in site.getsitepackages()
        if list(pathlib.Path(sp).rglob("model_analyzer"))
    ][0]
    patched_flag_path = model_analyzer_dir.parent / "model_analyzer.patched"
    if not patched_flag_path.exists():
        patch_file_path = BIGNLP_SCRIPTS_PATH / "infer_scripts/inference_lib/patches/model_analyzer.patch"
        cmd = [
            "bash",
            "-c",
            f"cd {model_analyzer_dir} && patch -p2 < {patch_file_path}",
        ]
        LOGGER.info(f"Patching Triton Model Analyzer: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True)
        LOGGER.info(f"returncode: {result.returncode}")
        LOGGER.info(f"stdout: \n{result.stdout.decode('utf-8')}")
        LOGGER.info(f"stderr: \n{result.stderr.decode('utf-8')}")
        patched_flag_path.touch(exist_ok=True)
    else:
        LOGGER.info("Triton Model Analyzer already patched")


if __name__ == "__main__":
    main()

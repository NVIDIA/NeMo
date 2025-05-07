# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=C0115,C0116,C0301

import os

from huggingface_hub import snapshot_download


def download_diffusion_nemo():
    """
    Downloads all Cosmos Diffusion NeMo assets to HF_HOME directory.
    Make sure to set HF_HOME to your desired path before running this function.
    """
    snapshot_download("nvidia/Cosmos-1.0-Guardrail")
    snapshot_download("nvidia/Cosmos-1.0-Tokenizer-CV8x8x8")
    snapshot_download("nvidia/Cosmos-1.0-Diffusion-7B-Text2World", allow_patterns=["nemo/*"])
    snapshot_download("nvidia/Cosmos-1.0-Diffusion-14B-Text2World", allow_patterns=["nemo/*"])
    snapshot_download("nvidia/Cosmos-1.0-Diffusion-7B-Video2World", allow_patterns=["nemo/*"])
    snapshot_download("nvidia/Cosmos-1.0-Diffusion-14B-Video2World", allow_patterns=["nemo/*"])
    snapshot_download("nvidia/Cosmos-1.0-Prompt-Upsampler-12B-Text2World")
    snapshot_download("google-t5/t5-11b", ignore_patterns=["*.h5"])


def main():
    # Check if HF_HOME is set
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        raise EnvironmentError(
            "The HF_HOME environment variable is not set. "
            "Please set it to your desired path before running this script."
        )

    # Download Cosmos Diffusion NeMo checkpoints
    download_diffusion_nemo()


if __name__ == "__main__":
    main()

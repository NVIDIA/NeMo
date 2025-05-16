# **NeMo Export and Deploy**

## Introduction

NVIDIA NeMo Export and Deploy library ...

For technical documentation, please see the [NeMo Framework User
Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html).

### Get Started with Export and Deploy Library

- Refer to the [Quickstart](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html) for examples of using NeMo-Run to launch NeMo 2.0 experiments locally and on a slurm cluster.
- For more information about NeMo 2.0, see the [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html).
- [NeMo 2.0 Recipes](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes) contains additional examples of launching large-scale runs using NeMo 2.0 and NeMo-Run.
- For an in-depth exploration of the main features of NeMo 2.0, see the [Feature Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/features/index.html#feature-guide).
- To transition from NeMo 1.0 to 2.0, see the [Migration Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/migration/index.html#migration-guide) for step-by-step instructions.

## Get Started with NeMo Framework

Getting started with NeMo ...

## Key Features

- Exporting ...

## Requirements

- Python 3.10 or above (Recommended: Python 3.12)
- Pytorch 2.5 or above (Recommended: Pytorch 2.6)
- NVIDIA GPU (if you intend to do model training)

## Install NeMo Export and Deploy

Start an interactive terminal inside a `nvidia/cuda` container:

```bash
docker run --rm -it --entrypoint bash --runtime nvidia --gpus all nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
```

Install NeMo Export-Deploy:

```bash
apt update
apt install -y python3-pip python3-venv git libopenmpi-dev vim
git clone https://github.com/nvidia/nemo
cd nemo
git switch ko3n1g/onur/nemo-export-deploy-repo
python3 -m venv venv
source venv/bin/activate

# Required for TransformerEngine 2.2
pip install "torch==2.6.0" wheel_stub pybind11 "Cython>=3.0.0"
export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"

# Install NeMo Export-Deploy
pip install --no-build-isolation --no-cache-dir '.[all]'

# Use devel version of NeMo-Run and Mcore
pip install git+https://github.com/NVIDIA/Megatron-LM.git 
pip install git+https://github.com/NVIDIA/NeMo-Run.git 
```

### Support matrix

NeMo-Framework provides tiers of support based on OS / Platform and mode of installation. Please refer the following overview of support levels:

- Fully supported: Max performance and feature-completeness.
- Limited supported: Used to explore NeMo.
- No support yet: In development.
- Deprecated: Support has reached end of life.

Please refer to the following table for current support levels:

| OS / Platform              | Install from PyPi | Source into NGC container |
|----------------------------|-------------------|---------------------------|
| `linux` - `amd64/x84_64`   | Limited support   | Full support              |
| `linux` - `arm64`          | Limited support   | Limited support           |
| `darwin` - `amd64/x64_64`  | Deprecated        | Deprecated                |
| `darwin` - `arm64`         | Limited support   | Limited support           |
| `windows` - `amd64/x64_64` | No support yet    | No support yet            |
| `windows` - `arm64`        | No support yet    | No support yet            |

## Licenses

- [NeMo GitHub Apache 2.0
  license](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file#readme)
- NeMo is licensed under the [NVIDIA AI PRODUCT
  AGREEMENT](https://www.nvidia.com/en-us/data-center/products/nvidia-ai-enterprise/eula/).
  By pulling and using the container, you accept the terms and
  conditions of this license.

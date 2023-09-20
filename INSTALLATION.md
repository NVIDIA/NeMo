# Installing NeMo

## Conda

We recommend installing NeMo in a fresh Conda environment.

```bash
conda create --name nemo python==3.10.12
conda activate nemo
```

Install PyTorch using their [configurator](https://pytorch.org/get-started/locally/).

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

The command used to install PyTorch may depend on your system. Please use the configurator linked above to find the right command for your system.

## Pip
Use this installation mode if you want the latest released version.

```bash
apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython
pip install nemo_toolkit['all']
```

Depending on the shell used, you may need to use `"nemo_toolkit[all]"` instead in the above command.

## Pip from source

Use this installation mode if you want the version from a particular GitHub branch (e.g main).

```bash
apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython
python -m pip install git+https://github.com/NVIDIA/NeMo.git@{BRANCH}#egg=nemo_toolkit[all]
```

## From source
Use this installation mode if you are contributing to NeMo.

```bash
apt-get update && apt-get install -y libsndfile1 ffmpeg
git clone https://github.com/NVIDIA/NeMo
cd NeMo
./reinstall.sh
```

If you only want the toolkit without additional conda-based dependencies, you may replace `reinstall.sh`
with `pip install -e .` when your PWD is the root of the NeMo repository.

## RNNT

Note that RNNT requires numba to be installed from conda.

```bash
conda remove numba
pip uninstall numba
conda install -c conda-forge numba
```

## NeMo Megatron

NeMo Megatron training requires NVIDIA Apex to be installed.
Install it manually if not using the NVIDIA PyTorch container.

To install Apex, run

```bash
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout 52e18c894223800cb611682dce27d88050edf1de
pip install -v --no-build-isolation --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_layer_norm" --global-option="--distributed_adam" --global-option="--deprecated_fused_adam" ./
```

It is highly recommended to use the NVIDIA PyTorch or NeMo container if having issues installing Apex or any other dependencies.

While installing Apex, it may raise an error if the CUDA version on your system does not match the CUDA version torch was compiled with.
This raise can be avoided by commenting it [here](https://github.com/NVIDIA/apex/blob/master/setup.py#L32).

`cuda-nvprof` is needed to install Apex. The version should match the CUDA version that you are using:

```bash
conda install -c nvidia cuda-nvprof=11.8
```

`packaging` is also needed:

```bash
pip install packaging
```

With the latest versions of Apex, the `pyproject.toml` file in Apex may need to be deleted in order to install locally.


## Transformer Engine

NeMo Megatron GPT has been integrated with [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine). Transformer Engine enables FP8 training on NVIDIA Hopper GPUs.
[Install](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html) it manually if not using the NVIDIA PyTorch container.

```bash
pip install --upgrade git+https://github.com/NVIDIA/TransformerEngine.git@stable
```

It is highly recommended to use the NVIDIA PyTorch or NeMo container if having issues installing Transformer Engine or any other dependencies.

Transformer Engine requires PyTorch to be built with CUDA 11.8.


## Flash Attention
Transformer Engine already supports Flash Attention for GPT models. If you want to use Flash Attention for non-causal models or use with attention bias (introduced from position encoding, e.g. Alibi), please install [flash-attn](https://github.com/HazyResearch/flash-attention).

```bash
pip install flash-attn
pip install triton==2.0.0.dev20221202
```

## NLP inference UI
To launch the inference web UI server, please install [gradio](https://gradio.app/).

```bash
pip install gradio==3.34.0
```

## NeMo Text Processing
NeMo Text Processing, which provides functionality for Text Normalization & Inverse Text Normalization, is now a separate [repository](https://github.com/NVIDIA/NeMo-text-processing).

## Docker containers:
We release NeMo containers alongside NeMo releases. For example, NeMo `r1.20.0` comes with container `nemo:23.06`, you may find more details about released containers in [releases page](https://github.com/NVIDIA/NeMo/releases).

To use a built container, please run

```bash
docker pull nvcr.io/nvidia/nemo:23.06
```

To build a nemo container with a Dockerfile from a branch, please run

```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile -t nemo:latest .
```

If you chose to work with the main branch, we recommend using NVIDIA's PyTorch container version `23.06-py3` and then installing from GitHub.

```bash
docker run --gpus all -it --rm -v <nemo_github_folder>:/NeMo --shm-size=8g \
-p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit \
stack=67108864 --device=/dev/snd nvcr.io/nvidia/pytorch:23.06-py3
```

FROM nvcr.io/nvidia/pytorch:20.09-py3

COPY requirements /tmp/requirements
RUN for f in $(ls /tmp/requirements/*.txt); do pip install --disable-pip-version-check --no-cache-dir -r $f; done; rm -rf /tmp/requirements

RUN pip install --upgrade torch torchvision torchaudio torchtext
RUN pip install pytorch_lightning==1.1.5

COPY . /tmp/nemo
RUN cd /tmp/nemo && pip install ".[all]"

RUN LC_ALL=C.UTF-8
RUN LANG=C.UTF-8
RUN wandb login 21cdf23a1ec59443afa54c8b44b5accca37f0238

ARG CACHE_VER=xxx
COPY . /workspace/nemo
WORKDIR /workspace/nemo

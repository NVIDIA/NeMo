FROM nvcr.io/nvidia/pytorch:20.09-py3

COPY requirements /tmp/requirements
RUN for f in $(ls /tmp/requirements/*.txt); do pip install --disable-pip-version-check --no-cache-dir -r $f; done; rm -rf /tmp/requirements

ARG CACHE_VER=xxx
COPY . /workspace/nemo
WORKDIR /workspace/nemo

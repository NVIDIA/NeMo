ARG FROM_IMAGE_NAME=gitlab-master.nvidia.com:5005/dl/dgx/bignlp/train:22.04-py3-base
FROM ${FROM_IMAGE_NAME}

WORKDIR /lustre/fsw/joc/yuya/bignlp/bignlp-scripts_ci
COPY . .


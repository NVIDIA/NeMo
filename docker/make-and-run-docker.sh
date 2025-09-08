# Usage (at the root of the repository):
#  $ export MODEL_DIR=/path/to/my/models
export DATASET_DIR=/data/cjss7894/load/Cambrian737k
#  $ ./docker/make-and-run-docker.sh


BASE_IMAGE_WITH_TAG=nvcr.io/nvidia/nemo:25.07

# Build the base image
# docker build \
#   -t ${BASE_IMAGE_WITH_TAG} \
#   -f docker/Dockerfile .

# Build the user image
docker build \
  --build-arg BASE_IMAGE_WITH_TAG=${BASE_IMAGE_WITH_TAG} \
  --build-arg USER_ID=$(id -u) \
  --build-arg USER_NAME=$(id -un) \
  --build-arg GROUP_ID=$(id -g) \
  --build-arg GROUP_NAME=$(id -gn) \
  -t ${BASE_IMAGE_WITH_TAG}-$(whoami) \
  -f docker/Dockerfile.user .

# Run the container
docker run --rm -it --gpus all \
  --name nemo-$(whoami) --net=host \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_DATASETS_CACHE=/datasets/.cache/hf \
  -v "$(pwd)":/workspace \
  -v "${MODEL_DIR}":/models \
  -v "${DATASET_DIR}":/datasets \
  ${BASE_IMAGE_WITH_TAG}-$(whoami) \
  bash

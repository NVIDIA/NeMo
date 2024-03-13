
docker run --gpus all -it --rm -v /home/heh:/home/heh -v /media:/media  --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 gitlab-master.nvidia.com/heh/nemo_containers:nemo-main-23.01-dev bash

# nvcr.io/nvidia/nemo:23.04 bash




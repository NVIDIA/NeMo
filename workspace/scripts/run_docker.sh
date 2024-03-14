
docker run --gpus all -it --rm -v /home/heh:/home/heh -v /media:/media  --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 nemo-24.01 bash

# nvcr.io/nvidia/nemo:23.04 bash




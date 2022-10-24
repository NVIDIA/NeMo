echo $PWD
docker run --gpus all -it --rm --shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864 --net=host --ipc=host \
-v $PWD:/NeMo \
-v $PWD/project/synth_audio_train:$PWD/project/synth_audio_train \
-v $PWD/project/synth_audio_val:$PWD/project/synth_audio_val \
gitlab-master.nvidia.com/heh/nemo_containers:nemo-main-22.09 /bin/bash
# cmd="cd /NeMo/project && PL_DISABLE_FORK=1 python run_debug_train.py"
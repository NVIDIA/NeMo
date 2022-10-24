$CURR=pwd
sudo docker run --gpus all -it --rm --shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864 --device=/dev/snd -it \
-v /media/data/projects/NeMo-vad:/NeMo \
-v /media/data/datasets/vad_sd:/media/data/datasets/vad_sd \
-v /media/data/projects/NeMo-vad/project/synth_audio_train:/media/data/projects/NeMo-vad/project/synth_audio_train \
-v /media/data/projects/NeMo-vad/project/synth_audio_val:/media/data/projects/NeMo-vad/project/synth_audio_val \
gitlab-master.nvidia.com/heh/nemo_containers:nemo-main-22.09 /bin/bash

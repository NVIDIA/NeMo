## Server
```
docker run --gpus '"device=0"' -it -v $PWD:/ws --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 --device=/dev/snd --net=host nvcr.io/nvidia/tritonserver:24.03-py3

apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython
pip install nemo_toolkit['all']

```

## client
```
pip install tritonclient[all]

cd client/
python client.py --audio_file=xxx.wav
```
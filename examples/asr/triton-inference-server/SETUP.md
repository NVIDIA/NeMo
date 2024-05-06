## Prepare perf_analyzer's input
```
cd client/
pip install fire datasets
python3 create_perf_analyzer_json.py --dataset_name=dummy --tag=dummy --output_json=perf_analyzer_librispeech_clean.json
```

## Server
```
# Use --cap-add=SYS_ADMIN to help nsight systems: https://docs.nvidia.com/nsight-systems/UserGuide/index.html#enable-docker-collection
docker run --cap-add=SYS_ADMIN --gpus '"device=0"' -it -w /ws -v /home/dgalvez/scratch/data:/home/dgalvez/scratch/data -v $PWD:/ws -v $PWD/../../..:/NeMo/ --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 --device=/dev/snd --net=host nvcr.io/nvidia/tritonserver:24.03-py3

apt-get update && apt-get install -y libsndfile1 ffmpeg && pip install Cython cuda-python tritonclient[all] && pip install -e /NeMo/['all']

# You can save this container via "docker commit" to avoid having to reinstall the dependencies every time you run it.

# Use perf-analyzer to send data
nsys launch --session-new=galvez1 tritonserver --model-control-mode poll --repository-poll-secs 5 --model-repository asr_ctc/model_repo 1> tritonserver_stdout.log 2> tritonserver_stderr.log &
nsys start --session=galvez1
perf_analyzer -m asr_ctc -b 1 -i gRPC -u localhost:8001 --max-thread=16 --input-data=client/perf_analyzer_librispeech_clean.json -s 1000 -a --max-thread=16 --concurrency-range=16
nsys stop --session=galvez1
nsys shutdown
```

## client
```
pip install tritonclient[all]

cd client/
python client.py --audio_file=xxx.wav

# verify manifest accuracy
pip install nemo_toolkit['asr']
python client.py --manifest=/home/dgalvez/scratch/data/test_other_sorted_downward.json  --do_wer_cer=1  # 1 for wer, 2 for cer

```
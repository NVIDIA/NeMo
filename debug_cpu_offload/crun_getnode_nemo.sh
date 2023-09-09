crun -t 04:00:00 -q "chip=ga100 and memory_capacity_gb=80" -i -img nvcr.io/ea-bignlp/nemofw-training:23.07-py3 -a "--ipc=host --ulimit memlock=-1 --ulimit stack=67108864"

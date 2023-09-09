NEMO=/home/scratch.guyueh_sw/2023su/cpu_offload/NeMo

export PYTHONPATH=${NEMO}:$PYTHONPATH

# python toy_transformer.py

/home/scratch.svc_compute_arch/release/nsightSystems/x86_64/rel/2023.2.1.122/bin/nsys \
profile -s none -t nvtx,cuda -o ./computelab_a100_toy_transformer_group --force-overwrite true \
--capture-range=cudaProfilerApi --capture-range-end=stop \
python toy_transformer_group.py
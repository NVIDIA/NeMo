NEMO=/home/scratch.guyueh_sw/2023su/cpu_offload/NeMo

export PYTHONPATH=${NEMO}:$PYTHONPATH

/home/scratch.svc_compute_arch/release/nsightSystems/x86_64/rel/2023.2.1.122/bin/nsys \
profile -s none -t nvtx,cuda -o ./mlp --force-overwrite true \
--capture-range=cudaProfilerApi --capture-range-end=stop \
python mlp.py

/home/scratch.svc_compute_arch/release/nsightSystems/x86_64/rel/2023.2.1.122/bin/nsys \
profile -s none -t nvtx,cuda -o ./mlp_offload_jit --force-overwrite true \
--capture-range=cudaProfilerApi --capture-range-end=stop \
python mlp.py

/home/scratch.svc_compute_arch/release/nsightSystems/x86_64/rel/2023.2.1.122/bin/nsys \
profile -s none -t nvtx,cuda -o ./mlp_offload_prefetch --force-overwrite true \
--capture-range=cudaProfilerApi --capture-range-end=stop \
python mlp.py
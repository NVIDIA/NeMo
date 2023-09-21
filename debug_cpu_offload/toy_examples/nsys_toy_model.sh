NEMO=/home/scratch.guyueh_sw/2023su/cpu_offload/NeMo

export PYTHONPATH=${NEMO}:$PYTHONPATH

nsys profile -s none -t nvtx,cuda -o ./toy_model_offload --force-overwrite true \
--capture-range=cudaProfilerApi --capture-range-end=stop \
python toy_model_offload.py
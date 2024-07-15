from typing import List
from nemo.utils import logging
import glob
import os
from tensorboard.backend.event_processing import event_accumulator

# From GPU datasheets (numbers with Tensor Core and dense computation)
# H200: https://nvdam.widen.net/s/nb5zzzsjdf/hpc-datasheet-sc23-h200-datasheet-3002446
# H100: https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet?ncid=no-ncid
# A100: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf

GPU_HW_FLOPS_MAP = {
    "h100": {
        "int8": 1979,
        "fp8": 1979,
        "fp16": 1979/2,
        "bf16": 1979/2,
        "tf32": 989/2,
        "fp64": 67,
    },
    "h200": {
        "int8": 1979,
        "fp8": 1979,
        "fp16": 1979/2,
        "bf16": 1979/2,
        "tf32": 989/2,
        "fp64": 67,
    },
    "a100": {
        "int8": 624,
        "fp16": 624/2,
        "bf16": 624/2,
        "tf32": 156,
        "fp64": 19.5,
    }
}

LLM_VOCAB_SIZE_MAP = {
    "gpt3": 51200,
    "llama2": 32000,
    "llama3": 128256,
    "nemotron": 256000,
    "bert": 29000,
    "mixtral": 32000,
    }

def read_tb_log(path, summary_name: str) -> List:
    """
    Reads a TensorBoard Events file from the input path, and returns the
    summary specified.

    Args:
        path: str, path to the dir where the events file is located.
        summary_name: str, name of the summary to read from the TB logs.
    Returns:
        summary_list: list, the values in the read summary list, formatted as a list.
    """

    files = glob.glob(f"{path}/events*tfevents*")
    files.sort(key=lambda x: os.path.getmtime(os.path.join(path, x)))
    if len(files) == 0 or not os.path.isfile(files[0]):
        raise FileNotFoundError(f"Missing TensorBoard log file.")

    events_file = files[0]
    ea = event_accumulator.EventAccumulator(events_file)
    ea.Reload()
    try:
        summary = ea.Scalars(summary_name)
        summary_list = [round(x.value, 2) for x in summary]
    except KeyError:
        logging.error(f"{summary_name} not found in {events_file}")

    return summary_list
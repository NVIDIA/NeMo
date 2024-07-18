import os
import sys

import pytest
import yaml

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(os.path.dirname(current)))
print(parent)
sys.path.append(parent)

from nemo.collections.common.metrics.perf_metrics import FLOPsMeasurementCallback
from tests.collections.common.test_perf_metrics_data import LLAMA2_CFG_STR, NEMOTRON_CFG_STR, UNSUPPORTED_MODEL_CFG_STR


@pytest.mark.unit
def test_flops_measurement():
    llama2_cfg = yaml.safe_load(LLAMA2_CFG_STR)

    # extract model name from cfg
    flops_callback = FLOPsMeasurementCallback(llama2_cfg, model_name=None)
    tflops_per_sec_per_gpu = flops_callback.eval_tflops_per_sec_per_gpu(train_step_time=8)
    assert tflops_per_sec_per_gpu == pytest.approx(377.53, rel=1e-5)

    # override model name from args
    flops_callback = FLOPsMeasurementCallback(llama2_cfg, model_name="llama2")
    tflops_per_sec_per_gpu = flops_callback.eval_tflops_per_sec_per_gpu(train_step_time=8)
    assert tflops_per_sec_per_gpu == pytest.approx(377.53, rel=1e-5)

    # list of train step times
    tflops_per_sec_per_gpu = flops_callback.eval_tflops_per_sec_per_gpu(train_step_time=[8, 8, 8, 8])
    assert tflops_per_sec_per_gpu == pytest.approx(377.53, rel=1e-5)

    nemotron_cfg = yaml.safe_load(NEMOTRON_CFG_STR)

    flops_callback = FLOPsMeasurementCallback(nemotron_cfg, model_name=None)
    tflops_per_sec_per_gpu = flops_callback.eval_tflops_per_sec_per_gpu(train_step_time=1.31)
    assert tflops_per_sec_per_gpu == pytest.approx(642.71, rel=1e-5)

    # extract valid model name with delimiter='-'
    nemotron_cfg["run"]["name"] = nemotron_cfg["run"]["name"].replace("-", ".")
    flops_callback = FLOPsMeasurementCallback(nemotron_cfg, model_name=None)
    tflops_per_sec_per_gpu = flops_callback.eval_tflops_per_sec_per_gpu(train_step_time=1.31)
    assert tflops_per_sec_per_gpu == pytest.approx(642.71, rel=1e-5)

    # extract valid model name from a string
    nemotron_cfg["run"]["name"] = nemotron_cfg["run"]["name"].replace(".", "")
    flops_callback = FLOPsMeasurementCallback(nemotron_cfg, model_name=None)
    tflops_per_sec_per_gpu = flops_callback.eval_tflops_per_sec_per_gpu(train_step_time=1.31)
    assert tflops_per_sec_per_gpu == pytest.approx(642.71, rel=1e-5)

    # model_name=None, both as a param and in config
    llama2_cfg["run"]["name"] = None
    flops_callback = FLOPsMeasurementCallback(llama2_cfg, model_name=None)
    with pytest.raises(
        KeyError, match="Failed to extract valid model name from or missing FLOPs calculations for None"
    ):
        tflops_per_sec_per_gpu = flops_callback.eval_tflops_per_sec_per_gpu(train_step_time=1)

    unsupported_model_cfg = yaml.safe_load(UNSUPPORTED_MODEL_CFG_STR)
    flops_callback = FLOPsMeasurementCallback(unsupported_model_cfg, model_name=None)
    with pytest.raises(
        KeyError, match="Failed to extract valid model name from or missing FLOPs calculations for unsupported_model"
    ):
        tflops_per_sec_per_gpu = flops_callback.eval_tflops_per_sec_per_gpu(train_step_time=1)


@pytest.mark.unit
def test_eval_mfu():
    llama2_cfg = yaml.safe_load(LLAMA2_CFG_STR)

    # precision = BF16
    flops_callback = FLOPsMeasurementCallback(llama2_cfg, gpu_name="H100")
    mfu = flops_callback.eval_mfu(tflops_per_sec_per_gpu=542)
    assert mfu == pytest.approx(54.8, rel=1e-3)

    nemotron_cfg = yaml.safe_load(NEMOTRON_CFG_STR)

    # precision = FP8
    flops_callback = FLOPsMeasurementCallback(nemotron_cfg, gpu_name="H100")
    mfu = flops_callback.eval_mfu(tflops_per_sec_per_gpu=643)
    assert mfu == pytest.approx(32.5, rel=1e-3)

    # unsupported GPU
    flops_callback = FLOPsMeasurementCallback(nemotron_cfg, gpu_name="")
    with pytest.raises(KeyError, match="Missing hardware FLOPs for self.gpu_name=''"):
        mfu = flops_callback.eval_mfu(tflops_per_sec_per_gpu=643)

    # precision not supported
    unssuported_cfg = yaml.safe_load(UNSUPPORTED_MODEL_CFG_STR)
    flops_callback = FLOPsMeasurementCallback(unssuported_cfg, gpu_name="a100")
    with pytest.raises(KeyError, match="Missing hardware FLOPs for precision='bf64'"):
        mfu = flops_callback.eval_mfu(tflops_per_sec_per_gpu=643)

    # 'precision' key missing
    unssuported_cfg["trainer"].pop("precision")
    with pytest.raises(KeyError, match="Missing hardware FLOPs for precision=''"):
        mfu = flops_callback.eval_mfu(tflops_per_sec_per_gpu=643)

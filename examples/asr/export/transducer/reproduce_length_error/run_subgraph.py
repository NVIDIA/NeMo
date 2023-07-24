from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx
from polygraphy.backend.trt import EngineFromBytes, EngineFromNetwork, NetworkFromOnnxPath, save_engine, TrtRunner
from polygraphy.backend.trt.config import CreateConfig
from polygraphy.backend.trt.profile import Profile
from polygraphy.comparator import Comparator, CompareFunc

import numpy as np

import os
import subprocess

from nemo.collections.asr.models import ASRModel


def main():
    if not os.path.exists("encoder-temp_rnnt.onnx"):
        nemo_model = ASRModel.from_pretrained("stt_en_conformer_transducer_large", map_location='cuda')
        nemo_model.export("temp_rnnt.onnx", onnx_opset_version=18)

    subprocess.check_call("polygraphy surgeon extract encoder-temp_rnnt.onnx --inputs length:auto:auto --outputs encoded_lengths:auto -o just_length_computation.onnx", shell=True)

    build_onnxrt_session = SessionFromOnnx("just_length_computation.onnx")
    build_onnxrt_cuda_session = SessionFromOnnx("just_length_computation.onnx", ["cuda"])

    config = CreateConfig(fp16=True, profiles=[
        (Profile()
         .add("length", min=(1, ), opt=(16, ), max=(32, ))
        )
    ]
    )

    build_engine = EngineFromNetwork(
        NetworkFromOnnxPath("just_length_computation.onnx"),
        config)

    engine = build_engine()

    runners = [
        TrtRunner(engine),
        OnnxrtRunner(build_onnxrt_session),
        OnnxrtRunner(build_onnxrt_cuda_session)
    ]

    run_results = Comparator.run(runners, data_loader=[{"length": np.array([2135], dtype=np.int64)}])

    trt_runner_name, onnxrt_runner_name, onnxrt_cuda_runner_name = list(run_results.keys())

    print(run_results[trt_runner_name][0]["encoded_lengths"])
    print(run_results[onnxrt_runner_name][0]["encoded_lengths"])
    print(run_results[onnxrt_cuda_runner_name][0]["encoded_lengths"])

    assert run_results[trt_runner_name][0]["encoded_lengths"] == run_results[onnxrt_runner_name][0]["encoded_lengths"], f'{run_results[trt_runner_name][0]["encoded_lengths"]} vs. {run_results[onnxrt_runner_name][0]["encoded_lengths"]}'

if __name__ == "__main__":
    main()

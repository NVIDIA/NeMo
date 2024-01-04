import os

import numpy as np
from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx
from polygraphy.backend.trt import EngineFromBytes, EngineFromNetwork, NetworkFromOnnxPath, TrtRunner, save_engine
from polygraphy.backend.trt.config import CreateConfig
from polygraphy.backend.trt.profile import Profile
from polygraphy.comparator import Comparator, CompareFunc

from nemo.collections.asr.models import ASRModel


def main():
    if not os.path.exists("encoder-temp_rnnt.onnx"):
        nemo_model = ASRModel.from_pretrained("stt_en_conformer_transducer_large", map_location='cuda')
        nemo_model.export("temp_rnnt.onnx", onnx_opset_version=18)

    build_onnxrt_session = SessionFromOnnx("encoder-temp_rnnt.onnx")

    if not os.path.exists("encoder-temp_rnnt.engine"):
        timing_cache = "trt.cache"

        config = CreateConfig(
            fp16=True,
            profiles=[
                (
                    Profile()
                    .add("audio_signal", min=(1, 80, 25), opt=(16, 80, 1024), max=(32, 80, 4096))
                    .add("length", min=(1,), opt=(16,), max=(32,))
                )
            ],
        )

        build_engine = EngineFromNetwork(NetworkFromOnnxPath("encoder-temp_rnnt.onnx"), config, timing_cache)

        engine = build_engine()
        save_engine(engine, "encoder-temp_rnnt.engine")
    else:
        engine = EngineFromBytes(open("encoder-temp_rnnt.engine", "rb").read())

    runners = [
        TrtRunner(engine),
        OnnxrtRunner(build_onnxrt_session),
    ]

    run_results = Comparator.run(
        runners,
        data_loader=[
            {"audio_signal": np.zeros((1, 80, 2135), dtype=np.float32), "length": np.array([2135], dtype=np.int64)}
        ],
    )

    trt_runner_name, onnxrt_runner_name = list(run_results.keys())

    assert (
        run_results[trt_runner_name][0]["encoded_lengths"] == run_results[onnxrt_runner_name][0]["encoded_lengths"]
    ), f'{run_results[trt_runner_name][0]["encoded_lengths"]} vs. {run_results[onnxrt_runner_name][0]["encoded_lengths"]}'


if __name__ == "__main__":
    main()

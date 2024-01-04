#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
This script runs an identity model with ONNX-Runtime and TensorRT,
then compares outputs.
"""
import numpy as np
from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx
from polygraphy.backend.trt import EngineFromNetwork, NetworkFromOnnxPath, TrtRunner
from polygraphy.backend.trt.config import CreateConfig
from polygraphy.backend.trt.profile import Profile
from polygraphy.comparator import Comparator, CompareFunc


def main():
    build_onnxrt_session = SessionFromOnnx("just_length_computation2.onnx")

    config = CreateConfig(fp16=True, profiles=[(Profile().add("length", min=(1,), opt=(16,), max=(32,)))])
    build_engine = EngineFromNetwork(NetworkFromOnnxPath("just_length_computation2.onnx"), config)

    runners = [
        TrtRunner(build_engine),
        OnnxrtRunner(build_onnxrt_session),
    ]

    run_results = Comparator.run(runners, data_loader=[{"length": np.array([2135], dtype=np.int64)}])

    assert bool(Comparator.compare_accuracy(run_results, compare_func=CompareFunc.simple(atol=1e-8)))
    run_results.save("inference_results.json")


if __name__ == "__main__":
    main()

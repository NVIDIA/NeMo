# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from time import time

import api.nmt_pb2 as nmt
import api.nmt_pb2_grpc as nmtsrv
import grpc


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="es", type=str)
    parser.add_argument("--source", default="en", type=str)
    parser.add_argument("--text", default="", type=str)
    parser.add_argument("--port", default=50052, type=int, required=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    with grpc.insecure_channel(f'localhost:{args.port}') as channel:
        stub = nmtsrv.JarvisTranslateStub(channel)

        iterations = 1
        start_time = time()
        for _ in range(iterations):
            req = nmt.TranslateTextRequest(texts=[args.text], source_language=args.source, target_language=args.target)
            result = stub.TranslateText(req)
        end_time = time()
        print(f"Time to complete {iterations} synchronous requests: {end_time-start_time}")
        print(result)
        print(result.translations[0].translation)

        # iterations = 1
        # start_time = time()
        # futures = []
        # for _ in range(iterations):
        #     req = nmt.TranslateTextRequest(texts=["Hello, can you hear me?"])
        #     futures.append(stub.TranslateText.future(req))
        # for f in futures:
        #     f.result()
        # end_time = time()
        # print(f"Time to complete {iterations} asynchronous requests: {end_time-start_time}")
        # print(futures[0].result())

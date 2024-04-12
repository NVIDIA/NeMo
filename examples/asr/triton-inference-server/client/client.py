# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


import numpy as np
import soundfile as sf
import argparse
import os
import json
import time

import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import np_to_triton_dtype

import asyncio
from asyncio_pool import AioPool
import hashlib


class OfflineSpeechClient(object):

    def __init__(self, triton_client, model_name, protocol_client):
        self.triton_client = triton_client
        self.protocol_client = protocol_client
        self.model_name = model_name

    async def recognize(self, wav_file):
        waveform, sample_rate = sf.read(wav_file, dtype=np.float32)
        samples = np.array([waveform], dtype=np.float32)
        lengths = np.array([[len(waveform)]], dtype=np.int32)
        sequence_id = int(hashlib.sha1(wav_file.encode("utf-8")).hexdigest(), 16) % (10 ** 5)
        result = ""
        inputs = [
            self.protocol_client.InferInput("WAV", samples.shape,
                                            np_to_triton_dtype(samples.dtype)),
            self.protocol_client.InferInput("WAV_LENS", lengths.shape,
                                            np_to_triton_dtype(lengths.dtype)),
        ]
        inputs[0].set_data_from_numpy(samples)
        inputs[1].set_data_from_numpy(lengths)
        outputs = [self.protocol_client.InferRequestedOutput("TRANSCRIPTS")]
        start = time.time()
        response = await self.triton_client.infer(
            self.model_name,
            inputs,
            request_id=str(sequence_id),
            outputs=outputs,
        )
        latency = time.time() - start
        decoding_results = response.as_numpy("TRANSCRIPTS")[0]
        if type(decoding_results) == np.ndarray:
            result = b" ".join(decoding_results).decode("utf-8")
        else:
            result = decoding_results.decode("utf-8")
        print("Recognized: ", wav_file, result)
        return (wav_file, result, latency)


async def main(args):
    filepaths = []
    transcripts = []
    if args.audio_file is not None:
        path = args.audio_file
        if os.path.exists(path):
            filepaths = [path]
    elif args.manifest is not None:
        with open(args.manifest, "r") as f:
            for line in f:
                item = json.loads(line)
                filepaths.append(item["audio_filepath"])
                transcripts.append(item["text"])
    
    
    triton_client = grpcclient.InferenceServerClient(
        url=args.url, verbose=args.verbose)
    
    speech_client = OfflineSpeechClient(triton_client, args.model_name, grpcclient)
    concurrency = args.concurrency
    pool = AioPool(size=concurrency)
    print(f'=== Parallel Generation Requests Start - Total number of Tasks:{len(filepaths)} - Concurrency Rate:{pool.size} ===')
    start = time.time()
    responses = await pool.map(speech_client.recognize, filepaths)
    end = time.time()
    print(f"Time taken is {(end - start):.2f} seconds")
    print(f"Avg latency per audio is {(end-start)*1000/len(responses):.2f} ms")
    
    predictions = []
    server_latency = []
    for response in responses:
        predictions.append((response[0], response[1]))
        server_latency.append(response[2])
    print(f"Avg latency per request: \
        {np.mean(server_latency)*1000:.2f} ms")
    
    predictions = [(id, predict_text) for (id, predict_text) in predictions]
    # dump prediction
    if args.output_file is not None:
        with open(args.output_file, "w") as f:
            for item in predictions:
                filepath = item[0]
                predict_text = item[1]
                f.write(filepath + "\t" + predict_text + "\n")
    
    # calculate WER
    if args.do_wer_cer > 0 and args.manifest is not None:
        # you need to install nemo asr
        try:
            from nemo.collections.asr.metrics.wer import word_error_rate
        except:
            raise ImportError("Please install nemo asr to calculate WER/CER")
        
        use_cer = args.do_wer_cer == 2
        predict_text = [li[1] for li in predictions]
        wer = word_error_rate(hypotheses=predict_text, references=transcripts, use_cer=use_cer)
        print(f'WER/CER: {wer * 100:.2f}%')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8001",
        help="Inference server URL. Default is "
        "localhost:8001.",
    )
    parser.add_argument(
        "--model_name",
        required=False,
        default="asr_ctc",
        choices=["asr_ctc", "asr_ctc_ensemble", "asr_ctc_chunked_offline"],
        help="the model to send request to",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=False,  
    )
    parser.add_argument(
        "--audio_file",
        type=str,
        required=False,
        default=None,
        help="single wav file path",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        required=False,
        default=1,
        help="number of tasks to be processed together",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=False,
        default=None,
        help="output file to save the recognition result",
    )
    parser.add_argument(
        "--do_wer_cer",
        type=int,
        default=0,
        choices=[0, 1, 2],
        required=False,
        help="0 for no wer/cer calculation, 1 for wer, 2 for cer",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
    
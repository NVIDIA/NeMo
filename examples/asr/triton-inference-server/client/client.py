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


import numpy as np
import soundfile as sf
import argparse
import os
import json
from multiprocessing import Pool

import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype



class OfflineSpeechClient(object):

    def __init__(self, triton_client, model_name, protocol_client):
        self.triton_client = triton_client
        self.protocol_client = protocol_client
        self.model_name = model_name

    def recognize(self, wav_file, idx=0):
        waveform, sample_rate = sf.read(wav_file)
        samples = np.array([waveform], dtype=np.float32)
        lengths = np.array([[len(waveform)]], dtype=np.int32)
        sequence_id = 10086 + idx
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
        response = self.triton_client.infer(
            self.model_name,
            inputs,
            request_id=str(sequence_id),
            outputs=outputs,
        )
        decoding_results = response.as_numpy("TRANSCRIPTS")[0]
        if type(decoding_results) == np.ndarray:
            result = b" ".join(decoding_results).decode("utf-8")
        else:
            result = decoding_results.decode("utf-8")
        return [result]


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
        "--output_file",
        type=str,
        required=False,
        default=None,
        help="the file to dump predicted text",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=1,
        help="number of workers to send requests together",
    )

    args = parser.parse_args()
    filepaths = []
    if args.audio_file is not None:
        path = args.audio_file
        if os.path.exists(path):
            filepaths = [path]
    elif args.manifest is not None:
        with open(args.manifest, "r") as f:
            for line in f:
                item = json.loads(line)
                filepaths.append(item["audio_filepath"])
    
    speech_client_cls = OfflineSpeechClient
    def single_job(client_files):
        with grpcclient.InferenceServerClient(
                url=args.url, verbose=args.verbose) as triton_client:
            protocol_client = grpcclient
            speech_client = speech_client_cls(triton_client, args.model_name,
                                              protocol_client)
            idx, audio_files = client_files
            predictions = []
            for li in audio_files:
                result = speech_client.recognize(li, idx)
                print("Recognized {}:{}".format(li, result[0]))
                predictions += result
        return predictions
    
    # start to do inference
    # Group requests in batches
    predictions = []
    tasks = []
    num_workers = args.num_workers
    splits = np.array_split(filepaths, num_workers)

    for idx, per_split in enumerate(splits):
        cur_files = per_split.tolist()
        tasks.append((idx, cur_files))

    #with Pool(processes=num_workers) as pool:
    #    predictions = pool.map(single_job, tasks)
    # remove pool to better debug 
    predictions = single_job(tasks[0])
    
    predictions = [item for sublist in predictions for item in sublist]
    # dump prediction
    if args.output_file is not None:
        with open(args.output_file, "w") as f:
            for item in predictions:
                f.write(item + "\n")
    
#!/usr/bin/python

# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import sys, os
from pathlib import Path

from nemo.deploy import DeployPyTriton, NemoQuery
from nemo.export import TensorRTLLM
from nemo.utils import logging

from builtins import range
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import statistics

try:
    from contextlib import nullcontext
except ImportError:
    # handle python < 3.7
    from contextlib import suppress as nullcontext


def get_args(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"Deploy nemo models to Triton and benchmark the models",
    )

    parser.add_argument(
        "-nc", 
        "--nemo_checkpoint", 
        type=str, 
        help="Source .nemo file"
    )

    parser.add_argument(
        "-mt",
        "--model_type",
        type=str, default="gptnext",
        choices=["gptnext", "llama"],
        help="Type of the model. gpt or llama are only supported."
    )

    parser.add_argument(
        "-tmn", 
        "--triton_model_name", 
        default="LLM_Model", 
        type=str, 
        help="Name for the service"
    )

    parser.add_argument(
        "-tmv", 
        "--triton_model_version", 
        default=1, 
        type=int, 
        help="Version for the service"
    )

    parser.add_argument(
        "-tv", 
        "--triton_port", 
        default=8000, 
        type=int, 
        help="Port for the Triton server to listen for requests"
    )

    parser.add_argument(
        "-tha", 
        "--triton_http_address", 
        default="0.0.0.0", 
        type=str, 
        help="HTTP address for the Triton server"
    )

    parser.add_argument(
        "-tlf", 
        "--trt_llm_folder", 
        default=None, 
        type=str, 
        help="Folder for the trt-llm conversion"
    )

    parser.add_argument(
        "-ng", 
        "--num_gpus", 
        default=1, 
        type=int, 
        help="Number of GPUs for the deployment"
    )

    parser.add_argument(
        "-d", 
        "--dtype",
        choices=["bf16", "fp16", "fp8", "int8"],
        default="bf16",
        type=str,
        help="dtype of the model on TensorRT-LLM",
    )

    parser.add_argument(
        "-mil", 
        "--max_input_len", 
        default=512, 
        type=int, 
        help="Max input length of the model"
    )

    parser.add_argument(
        "-mol",
        "--max_output_len",
        default=512, 
        type=int, 
        help="Max output length of the model"
    )

    parser.add_argument(
        "-mbs", 
        "--max_batch_size", 
        default=512, 
        type=int, 
        help="Max batch size of the model"
    )

    parser.add_argument(
        '-w',
        '--warm_up',
        action="store_true",
        required=False,
        default=False,
        help='Enable warm_up before benchmark'
    )

    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        default=8,
        required=False,
        help='Specify batch size'
    )

    parser.add_argument(
        '-top_k',
        '--top_k',
        type=int,
        default=1,
        required=False,
        help='top k for sampling'
    )

    parser.add_argument(
        '-top_p',
        '--top_p',
        type=float,
        default=0.0,
        required=False,
        help='top p for sampling'
    )

    parser.add_argument(
        '-temperature',
        '--temperature',
        type=float,
        default=0.0,
        required=False,
        help='top p for sampling'
    )

    parser.add_argument(
        '-sl',
        '--start_len',
        type=int,
        default=8,
        required=False,
        help='Specify input length'
    )

    parser.add_argument(
        '-nr',
        '--num_runs',
        type=int,
        default=8,
        required=False,
        help='Specify input length'
    )

    args = parser.parse_args(argv)
    return args


def nemo_deploy(args):

    if args.dtype != "bf16":
        logging.error("Only bf16 is currently supported for the optimized deployment with TensorRT-LLM. "
                      "Support for the other precisions will be added in the coming releases.")
        return

    if args.trt_llm_folder is None:
        trt_llm_path = "/tmp/trt_llm_model_dir/"
        logging.info(
            "/tmp/trt_llm_model_dir/ path will be used as the TensorRT LLM folder. "
            "Please set this parameter if you'd like to use a path that has already "
            "included the TensorRT LLM model files."
        )
        Path(trt_llm_path).mkdir(parents=True, exist_ok=True)
    else:
        trt_llm_path = args.trt_llm_folder

    if args.nemo_checkpoint is None and not os.path.isdir(args.trt_llm_folder):
        logging.error(
            "The provided trt_llm_folder is not a valid TensorRT-LLM model "
            "directory. Please provide a --nemo_checkpoint."
        )

    trt_llm_exporter = TensorRTLLM(model_dir=trt_llm_path)

    if args.nemo_checkpoint is not None:
        trt_llm_exporter.export(
            nemo_checkpoint_path=args.nemo_checkpoint,
            model_type=args.model_type,
            n_gpus=args.num_gpus,
            max_input_token=args.max_input_len,
            max_output_token=args.max_output_len,
            max_batch_size=args.max_batch_size,
        )

        run_forward(trt_llm_exporter, args)

    nm = DeployPyTriton(
        model=trt_llm_exporter,
        triton_model_name=args.triton_model_name,
        triton_model_version=args.triton_model_version,
        max_batch_size=args.max_batch_size,
        port=args.triton_port,
        http_address=args.triton_http_address,
    )
    
    nm.deploy()

    try:
        logging.info("Triton deploy function is called.")
        nm.run()
    except:
        logging.info("An error has occurred and will stop serving the model.")
        return None

    return nm


def get_inputs():
    test_input_128 = ["Who designed the Gold State Coach? Adjacent to the palace is the Royal Mews, also designed by Nash, where the royal carriages, including the Gold State Coach, are housed. This rococo gilt coach, designed by Sir William Chambers in 1760, has painted panels by G. B. Cipriani. It was first used for the State Opening of Parliament by George III in 1762 and has been used by the monarch for every coronation since George IV. It was last used for the Golden Jubilee of Elizabeth II. Also housed in the mews are the coach horses used at royal ceremonial processions."]
    test_input_200 = ["The Princess Theatre, Regent Theatre, and Forum Theatre are members of which of Melbourne's theater districts? Melbourne's live performance institutions date from the foundation of the city, with the first theatre, the Pavilion, opening in 1841. The city's East End Theatre District includes theatres that similarly date from 1850s to the 1920s, including the Princess Theatre, Regent Theatre, Her Majesty's Theatre, Forum Theatre, Comedy Theatre, and the Athenaeum Theatre. The Melbourne Arts Precinct in Southbank is home to Arts Centre Melbourne, which includes the State Theatre, Hamer Hall, the Playhouse and the Fairfax Studio. The Melbourne Recital Centre and Southbank Theatre (principal home of the MTC, which includes the Sumner and Lawler performance spaces) are also located in Southbank. The Sidney Myer Music Bowl, which dates from 1955, is located in the gardens of Kings Domain; and the Palais Theatre is"]
 
    return {"input_128": {"output_len": 20, "input": test_input_128}, "input_200": {"output_len": 200, "input": test_input_200}}


def run_forward(trt_llm_exporter, args):
    if True:
        input_info = get_inputs()

        for inpt, ol in input_info.items():
            for batch_size in [1, 2, 4, 8]:
                inputs = ol["input"] * batch_size
                # print(inputs)
            
                # warm up
                if args.warm_up:
                    #print("[INFO] sending requests to warm up")
                    output = trt_llm_exporter.forward(input_texts=inputs, max_output_token=ol["output_len"])
                    #print("----------output-----------")
                    #print(output)
            
                
                latencies = []
                for i in range(args.num_runs):
                    start_time = datetime.now()
            
                    output = trt_llm_exporter.forward(input_texts=inputs, max_output_token=ol["output_len"])
            
                    stop_time = datetime.now()
                    latencies.append((stop_time - start_time).total_seconds() * 1000.0)
            
                
                if args.num_runs > 1:
                    latency = statistics.mean(latencies)
                else:
                    latency = latencies[0]

                latency = round(latency, 3)
                throughput = round(1000 / latency * batch_size, 3)
                print(
                    f"[INFO] ** TRT-LLM Dataset: {inpt} Batch size: {batch_size}, Output len: {ol['output_len']}"
                )
                print(f"[INFO] ** TRT-LLM Latency: {latency} ms")
                print(f"[INFO] ** TRT-LLM Throughput: {throughput} sentences / sec")

def send_queries(args):
    nq = NemoQuery(url="localhost:8000", model_name=args.triton_model_name)
 
    input_info = get_inputs()
    
    for inpt, ol in input_info.items():
        for batch_size in [1, 2, 4, 8]:
            inputs = ol["input"] * batch_size
            # print(inputs)
        
            # warm up
            if args.warm_up:
                #print("[INFO] sending requests to warm up")
                output = nq.query_llm(prompts=inputs, max_output_token=ol["output_len"])
                #print("----------output-----------")
                #print(output)
        
            
            latencies = []
            for i in range(args.num_runs):
                start_time = datetime.now()
        
                output = nq.query_llm(prompts=inputs, max_output_token=ol["output_len"])
        
                stop_time = datetime.now()
                latencies.append((stop_time - start_time).total_seconds() * 1000.0)
        
            
            if args.num_runs > 1:
                latency = statistics.mean(latencies)
            else:
                latency = latencies[0]

            latency = round(latency, 3)
            throughput = round(1000 / latency * batch_size, 3)
            print(
                f"[INFO] Dataset: {inpt} Batch size: {batch_size}, Output len: {ol['output_len']}"
            )
            print(f"[INFO] Latency: {latency} ms")
            print(f"[INFO] Throughput: {throughput} sentences / sec")


if __name__ == '__main__':
    args = get_args(sys.argv[1:])

    loglevel = logging.INFO
    logging.setLevel(loglevel)
    logging.info("Logging level set to {}".format(loglevel))
    logging.info(args)

    nm = nemo_deploy(args)

    if nm is None:
        logging.info("Model serving will be stopped.")
    else:
        try:
            send_queries(args)
        except:
            logging.info("There are issues with sending queries.")

        try:
            logging.info("Model serving will be stopped.")
            nm.stop()
        except:
            logging.info("Model could not be stopped properly.")

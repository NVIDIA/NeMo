#!/usr/bin/python
 
import os
import sys
 
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import argparse
import statistics as s
from builtins import range
from datetime import datetime
 
import numpy as np
#from utils import utils
import utils
from nemo.deploy import NemoQuery
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v',
        '--verbose',
        action="store_true",
        required=False,
        default=False,
        help='Enable verbose output'
    )

    parser.add_argument(
        '-u',
        '--url',
        type=str,
        required=False,
        help='Inference server URL.'
    
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
        '-beam',
        '--beam_width',
        type=int,
        default=1,
        required=False,
        help='Specify beam width'
    )

    parser.add_argument(
        '-topk',
        '--topk',
        type=int,
        default=1,
        required=False,
        help='topk for sampling'
    )

    parser.add_argument(
        '-topp',
        '--topp',
        type=float,
        default=0.0,
        required=False,
        help='topp for sampling'
    )

    parser.add_argument(
        '-s',
        '--start_len',
        type=int,
        default=8,
        required=False,
        help='Specify input length'
    )

    parser.add_argument(
        '-o',
        '--output_len',
        type=int,
        default=10,
        required=False,
        help='Specify output length'
    )

    parser.add_argument(
        '-n',
        '--num_runs',
        type=int,
        default=1,
        required=False,
        help="Spedifty number of runs to get the average latency"
    )
 
    FLAGS = parser.parse_args()
 
    if FLAGS.url is None:
        FLAGS.url = "localhost:8000" if FLAGS.protocol == "http" else "localhost:8001"
     
    nq = NemoQuery(url="localhost:8000", model_name="GPT-2B")
 
    test_input_128 = ["Who designed the Gold State Coach? Adjacent to the palace is the Royal Mews, also designed by Nash, where the royal carriages, including the Gold State Coach, are housed. This rococo gilt coach, designed by Sir William Chambers in 1760, has painted panels by G. B. Cipriani. It was first used for the State Opening of Parliament by George III in 1762 and has been used by the monarch for every coronation since George IV. It was last used for the Golden Jubilee of Elizabeth II. Also housed in the mews are the coach horses used at royal ceremonial processions."]
    
    test_input_200 = ["The Princess Theatre, Regent Theatre, and Forum Theatre are members of which of Melbourne's theater districts? Melbourne's live performance institutions date from the foundation of the city, with the first theatre, the Pavilion, opening in 1841. The city's East End Theatre District includes theatres that similarly date from 1850s to the 1920s, including the Princess Theatre, Regent Theatre, Her Majesty's Theatre, Forum Theatre, Comedy Theatre, and the Athenaeum Theatre. The Melbourne Arts Precinct in Southbank is home to Arts Centre Melbourne, which includes the State Theatre, Hamer Hall, the Playhouse and the Fairfax Studio. The Melbourne Recital Centre and Southbank Theatre (principal home of the MTC, which includes the Sumner and Lawler performance spaces) are also located in Southbank. The Sidney Myer Music Bowl, which dates from 1955, is located in the gardens of Kings Domain; and the Palais Theatre is"]
 
    input_data= ""
    if FLAGS.start_len==128:
        input_data = test_input_128
    if FLAGS.start_len==200:
        input_data = test_input_200
 
    inputs = input_data * FLAGS.batch_size
    print(inputs)
 
    # warm up
    if FLAGS.warm_up:
        print("[INFO] sending requests to warm up")
        output = nq.query_llm(prompts=inputs, max_output_len=FLAGS.output_len)
        print("----------output-----------")
        print(output)
 
    latencies = []
    for i in range(FLAGS.num_runs):
        start_time = datetime.now()
 
        output = nq.query_llm(prompts=test_input_128, max_output_len=FLAGS.output_len)
 
 
        stop_time = datetime.now()
        latencies.append((stop_time - start_time).total_seconds() * 1000.0)
 
     
    if FLAGS.num_runs > 1:
        latency = s.mean(latencies)
    else:
        latency = latencies[0]
    latency = round(latency, 3)
    throughput = round(1000 / latency * FLAGS.batch_size, 3)
    print(
        f"[INFO] Batch size: {FLAGS.batch_size}, Start len: {FLAGS.start_len}, Output len: {FLAGS.output_len}"
    )
    print(f"[INFO] Latency: {latency} ms")
    print(f"[INFO] Throughput: {throughput} sentences / sec")
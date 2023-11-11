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
import json
import sys, os
from pathlib import Path

from nemo.deploy import DeployPyTriton, NemoQuery
from nemo.export import TensorRTLLM
from nemo.utils import logging

from builtins import range
from datetime import datetime
import numpy as np
import torch
import traceback

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
        default=250, 
        type=int, 
        help="Max input length of the model"
    )

    parser.add_argument(
        "-mol",
        "--max_output_len",
        default=200, 
        type=int, 
        help="Max output length of the model"
    )

    parser.add_argument(
        "-mbs", 
        "--max_batch_size", 
        default=8, 
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
        nargs='+',
        default=["1", "2", "4", "8"],
        required=False,
        help='Specify batch size'
    )
    
    parser.add_argument(
        '-l',
        '--out_lens',
        nargs='+',
        default=[20, 100, 200, 300],
        type=int, 
        required=False,
        help='Lengths of outputs'
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
        '-nr',
        '--num_runs',
        type=int,
        default=8,
        required=False,
        help='Specify input length'
    )

    parser.add_argument(
        '-rt',
        '--run_trt_llm',
        choices=[0, 1],
        type=int,
        default=0,
        required=False,
        help='Run TRT-LLM without PyTriton'
    )

    parser.add_argument(
        '--out_jsonl', 
        type=argparse.FileType('w'), 
        required=False
    )

    parser.add_argument(
        '-ptl',
        '--ptuning_table_len',
        type=int,
        default=0,
        required=False,
        help='Prompt embedding table len'
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
        if args.ptuning_table_len > 0:
            hs = trt_llm_exporter.get_hidden_size()
            prompt_embedding_table = np.random.rand(args.ptuning_table_len, hs)

        trt_llm_exporter.export(
            nemo_checkpoint_path=args.nemo_checkpoint,
            model_type=args.model_type,
            n_gpus=args.num_gpus,
            max_input_token=args.max_input_len,
            max_output_token=args.max_output_len,
            max_batch_size=args.max_batch_size,
            prompt_embeddings_table=prompt_embedding_table if args.ptuning_table_len > 0 else None
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
    except Exception as e:
        logging.error("An error has occurred and will stop serving the model.")
        logging.error(repr(e))
        logging.error(traceback.format_exc())
        return None

    return nm

def get_inputs(args):
    out_lens = [int(l) for l in args.out_lens]
    if args.model_type == 'llama':
        return get_inputs_llama(out_lens)
    else:
        return get_inputs_gptnext(out_lens)

# dmitrym for LLama using LLM Service to count tokens
def get_inputs_llama(out_lens):
    test_input_128 = ["Who designed the Gold State Coach? Adjacent to the palace is the Royal Mews, also designed by Nash, where the royal carriages, including the Gold State Coach, are housed. This rococo gilt coach, designed by Sir William Chambers in 1760, has painted panels by G. B. Cipriani. It was first used for the State Opening of Parliament by George III in 1762 and has been used by the monarch for every coronation since George IV. It was last used for the Golden Jubilee of Elizabeth II"]
    test_input_256 = ["Paragliding is the recreational and competitive adventure sport of flying paragliders: lightweight, free-flying, foot-launched glider aircraft with no rigid primary structure. The pilot sits in a harness or in a cocoon-like 'pod' suspended below a fabric wing. Wing shape is maintained by the suspension lines, the pressure of air entering vents in the front of the wing, and the aerodynamic forces of the air flowing over the outside. Despite not using an engine, paraglider flights can last many hours and cover many hundreds of kilometres, though flights of one to five hours and covering some tens of kilometres are more the norm. By skillful exploitation of sources of lift, the pilot may gain height, often climbing to altitudes of a few thousand metres. History In 1966, Canadian Domina Jalbert was granted a patent for a multi-cell wing type aerial device—a wing having a flexible canopy constituting an upper skin and with a plurality of longitudinally extending ribs forming in effect a wing corresponding to an airplane wing airfoil ... More particularly the invention contemplates"]
    test_input_512 = ["Paragliding is the recreational and competitive adventure sport of flying paragliders: lightweight, free-flying, foot-launched glider aircraft with no rigid primary structure. The pilot sits in a harness or in a cocoon-like 'pod' suspended below a fabric wing. Wing shape is maintained by the suspension lines, the pressure of air entering vents in the front of the wing, and the aerodynamic forces of the air flowing over the outside. Despite not using an engine, paraglider flights can last many hours and cover many hundreds of kilometres, though flights of one to five hours and covering some tens of kilometres are more the norm. By skillful exploitation of sources of lift, the pilot may gain height, often climbing to altitudes of a few thousand metres. History In 1966, Canadian Domina Jalbert was granted a patent for a multi-cell wing type aerial device—a wing having a flexible canopy constituting an upper skin and with a plurality of longitudinally extending ribs forming in effect a wing corresponding to an airplane wing airfoil ... More particularly the invention contemplates the provision of a wing of rectangular or other shape having a canopy or top skin and a lower spaced apart bottom skin, a governable gliding parachute with multi-cells and controls for glide. Governador Valadares, Brazil is known internationally for the World Paragliding Championships that has been held at Ibituruna Peak (1,123 m (3,684 ft)) In 1954, Walter Neumark predicted (in an article in Flight magazine) a time when a glider pilot would be able to launch himself by running over the edge of a cliff or down a slope ... whether on a rock-climbing holiday in Skye or skiing in the Alps. In 1961, the French engineer Pierre Lemongine produced improved parachute designs that led to the Para-Commander (PC). The Para-Commander had cutouts at the rear and sides that enabled it to be towed into the air and steered, leading to parasailing/parascending. Domina Jalbert invented the parafoil, which had sectioned cells in an aerofoil shape"]
    test_input_2048 = ["Paragliding, like hang gliding, developed partially out of designs created for the NASA space program. Other designs, along with test flights completed independently on the other side of the world, also helped provide the foundation of paragliding and contributed to its development. David Barish 1965American pilot David Barish created one of the first airfoils that helped jump-start the evolution of modern paragliding. After the end of WWII, Barish left the Air Force to study aerodynamics at the California Institute of Technology, then became a consultant for NASA. In 1955, he designed the Vortex Ring, a lighter, more stable parachute with improved gliding capabilities. Then, in the early 1960s, he built on his previous work to design a parachute, called the Sailwing, to aid in the return of NASA space capsules to Earth. Barish first flew his Sailwing—a single-surface, rectangular parachute—in 1965 from a ski resort in New York. He called the activity “slope soaring,” and in the summer of 1966 he toured ski resorts all the way to California to try to popularize the ground-skimming hobby. After NASA decided on other methods to recover the space capsule, however, Barish largely shifted his focus to other projects. Around the same time, others were also furthering parachute designs. In 1964, American Domina Jalbert patented the Parafoil, a multi-celled, double-surface, ram-air type parachute. The design used the motion of air blowing through the cells to inflate the parachute, giving it an airfoil shape that allowed it to glide. Jean-Claude BétempsThe sport of paragliding finally took off in 1978. On June 25, skydivers Jean-Claude Bétemps and André Bohn decided to try to get aloft by launching from the steep slope of Mont Pertuiset in Mieussy, France. Bétemps took off first, and both glided to the valley below. Their flights gained attention from the media, attracting others to the sport, and Bétemps became known by many as the inventor of paragliding. After that, the sport grew rapidly. The first paragliding school was founded in 1979, with Bétemps serving as an instructor. In 1985, Laurent de Kalbermatten began manufacturing and selling the first wing intended specifically for paragliding, and other companies soon followed. Paragliding began spreading to the U.S. in the mid-to-late 1980s and continued to grow during the 1990s. New paraglider pilots quickly started competing. The first Paragliding World Championships were held in Austria in 1989. The same year, Hans Jörg Bachmair set the first straight distance world record of 69.15 km that was recorded by the World Air Sports Federation (FAI). It was broken by two other pilots by the end of that year, then jumped to nearly 150 km by December 1990. Records for straight distance flown on a paraglider continued to increase, breaking 400 km in 2007. The current straight distance record of 564.3 km (350 miles) was set on October 13, 2016 by Donizete Baldessar Lemos, Rafael Monteiro Saladini, and Samuel Nascimento. Now, it’s easier, safer, and more exciting than ever to learn to paraglide. Glider designs are continuously improving, making paragliders lighter, more stable, and easier to fly, as well as giving them increasingly better performance. And with thousands of participants worldwide, paragliding has something for everyone: hiking (and even camping) with your wing, soaring the coastline, flying cross-country to beat your personal—or world—records, performing aerobatics, and participating in competitions, to list a few options. The sport is also continuing to evolve, with new designs shrinking paragliders into speedwings and mini-wings that allow pilots to fly low and fast down mountainsides. If you’ve ever dreamed of flying, paragliding offers the most accessible way to soar like a bird. With over 400 paragliding instructors and 40 schools certified by the Professional Air Sports Association (PASA), you won’t have to go far to get into the air. There’s no better time to get started, so find a school or instructor and join us in the sky! Site flying is the most classic form of paragliding and is suitable for beginners. This paragliding activity can be done in several ways. You have the easy launch site flight. This involves easy take-offs and flights in calm, stable conditions. You can practice it on low hills or gentle slopes. We make this type of flying accessible to beginners with an EN-A or EN-B glider. If you are a beginner, we recommend an introductory course at a specialised school to master the various manoeuvres. This has an impact on the level of the paraglider and allows you to know the code that governs this practice. The other technique for flying on site is the difficult take-off. This type of paragliding is usually done on higher mountain sites. The difficulty here is with changing weather conditions and stronger winds. We recommend difficult launch flying to intermediate pilots. They can do it with an EN-B or EN-C glider. In France, advanced training in a paragliding school is required to master this paragliding practice. Cross-country paragliding is the practice of flying long distances between several take-off sites. It is carried out with the ascending air currents in order to fly as long as possible. This technique is not accessible to everyone. You must be an experienced pilot. You need to be trained and have advanced experience in paragliding and navigation. You also need to be able to read the weather and air currents to plan your flight. Our blog contains a more detailed article that explains how does a paraglider fly. For cross-country paragliding, it is essential to have a glider suitable for the activity. The equipment should be EN-C or EN-D certified for good performance and stability at altitude. Semi-light gliders are also popular for cross country flying. They combine performance and lightness. Acrobatic paragliding is an advanced activity that involves performing aerobatics. You can perform rolls, loops, wing-overs and spirals during your flight. The aim is to master the techniques of piloting and controlling the glider in difficult situations. To practice aerobatics, you need a glider with an EN-D certification or higher. This type of equipment also makes it easier for the pilot to manoeuvre. Stunt gliders are generally smaller and lighter than standard paragliders. This makes it easier to move and manoeuvre in flight. Paragliding is a very demanding sport and requires specialised training in aerobatic flying. Advanced paragliding experience is highly recommended. We would like to remind you that acrobatic flight presents important risks. These risks include speed loss and loss of control of the wings. You must therefore respect the safety rules. These are strict in terms of preparation, checking of equipment and speed limits. Soaring is a paragliding practice that consists of playing with the wind as it rises along a relief. The latter can be a slope, a cliff, a dune, etc. It provokes ascents allowing you to fly without descending. You play close to the ground or to the earth. Very popular with paragliders, this game is addictive and provides a lot of adrenaline. We recommend rando flying to nature and paragliding lovers. Very sportive, this activity consists, for the paraglider, to reach the summit on foot. This type of paragliding is done with light material (the wing) with a maximum weight of 5 kg. Some newer models weigh less than 1.5 kg for the glider and harness! Paragliding offers a wide range of activities, from site flying to soaring to hiking. Would you like to make one of these paragliding flights? Adrenaline Parapente is a team of professionals specialised in these different practices. Whether you are a beginner or an experienced amateur, one of our experienced instructors will provide you with personalized assistance. The harness is one of the most important parts of a paragliding equipment. It provides protection for the pilot and transmits information about the air mass. It is an indispensable piloting tool. The design and adjustment of the harness has a significant impact on the behaviour of the glider. A good harness must be able to reduce the imbalance caused by turbulence and to limit movements, while favouring precise piloting. The choice "]
    inputs_avail = {
        128: test_input_128,
        256: test_input_256,
        512: test_input_512,
        2048: test_input_2048
    }
    inputs = {}
    for olen in out_lens:
        for inp, txt in inputs_avail.items():
            inputs[f"{inp}_{olen}"] = {"output_len": olen, "input": txt}

    return inputs

def get_inputs_gptnext(out_lens):
    test_input_128 = ["Who designed the Gold State Coach? Adjacent to the palace is the Royal Mews, also designed by Nash, where the royal carriages, including the Gold State Coach, are housed. This rococo gilt coach, designed by Sir William Chambers in 1760, has painted panels by G. B. Cipriani. It was first used for the State Opening of Parliament by George III in 1762 and has been used by the monarch for every coronation since George IV. It was last used for the Golden Jubilee of Elizabeth II. Also housed in the mews are the coach horses used at royal ceremonial processions."]
    base_text="Paragliding is the recreational and competitive adventure sport of flying paragliders: lightweight, free-flying, foot-launched glider aircraft with no rigid primary structure. The pilot sits in a harness or in a cocoon-like 'pod' suspended below a fabric wing. Wing shape is maintained by the suspension lines, the pressure of air entering vents in the front of the wing, and the aerodynamic forces of the air flowing over the outside. Despite not using an engine, paraglider flights can last many hours and cover many hundreds of kilometres, though flights of one to five hours and covering some tens of kilometres are more the norm. By skillful exploitation of sources of lift, the pilot may gain height, often climbing to altitudes of a few thousand metres. History In 1966, Canadian Domina Jalbert was granted a patent for a multi-cell wing type aerial device—a wing having a flexible canopy constituting an upper skin and with a plurality of longitudinally extending ribs forming in effect a wing corresponding to an airplane wing airfoil."
    test_input_256 = " ".join([base_text]*1 + [base_text[:224]])
    test_input_512 = " ".join([base_text]*2 + [base_text[:455]])
    test_input_2048 = " ".join([base_text]*9 + [base_text[:777]])
    inputs_avail = {
       128: test_input_128,
       256: test_input_256,
       512: test_input_512,
       2048: test_input_2048
    }
    inputs = {}
    for olen in out_lens:
        for inp, txt in inputs_avail.items():
            inputs[f"{inp}_{olen}"] = {"output_len": olen, "input": txt}

    return inputs

class FakeNeMoQuery:
    def __init__(self, trt_llm_exporter) -> None:
        self.trt_llm_exporter = trt_llm_exporter
        pass

    def query_llm(self, prompts, max_output_token):
        return self.trt_llm_exporter.forward(input_texts=prompts, max_output_token=max_output_token)


def run_forward(trt_llm_exporter, args):
    if args.run_trt_llm == 1:
        fake_nq = FakeNeMoQuery(trt_llm_exporter)
        perform_benchmark(args, fake_nq)



def perform_benchmark(args, nemo_query):
    nq = NemoQuery(url="localhost:8000", model_name=args.triton_model_name)
 
    input_info = get_inputs(args)
    
    for inpt, ol in input_info.items():
        for batch_size in args.batch_size:
            batch_size = int(batch_size)
            inputs = ol["input"] * batch_size
            # print(inputs)

            failed = False
        
            # warm up
            if args.warm_up:
                #print("[INFO] sending requests to warm up")
                output = nq.query_llm(prompts=inputs, max_output_token=ol["output_len"])
                # print("----------output-----------")
                # print(output)
                if output[0][0].startswith("An error occurred"):
                    failed = True
                # The full error response is 
                # [['An error occurred: CUDA out of memory. Tried to allocate 1.95 GiB. GPU 0 has a total capacty of 79.14 GiB of which 1.49 GiB is free. Process 110652 has 77.08 GiB memory in use. Process 111275 has 586.00 MiB memory in use. Of the allocated memory 37.13 GiB is allocated by PyTorch, and 292.06 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF']]
        
            
            latencies = []
            for i in range(args.num_runs):
                start_time = datetime.now()
        
                output = nq.query_llm(prompts=inputs, max_output_token=ol["output_len"])
                if output[0][0].startswith("An error occurred"):
                    failed = True
                    break

                stop_time = datetime.now()
                latencies.append((stop_time - start_time).total_seconds() * 1000.0)
        
            if not failed:
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
                print(f"[INFO] Throughput: {throughput} prompts / sec")
            else:
                latency = None

            if args.out_jsonl:
                measurement = {
                    "input_output": inpt,
                    "failed":failed,
                    "batch_size": batch_size,
                    "input_len": int(inpt.split("_")[0]),
                    "output_len": ol['output_len'],
                    "latency": latency,
                    "all_latencies": latencies,
                    "nemo_checkpoint_path": args.nemo_checkpoint,
                    "model_type": args.model_type,
                    "n_gpus": args.num_gpus,
                    "device": torch.cuda.get_device_name(),
                    "device_properties": str(torch.cuda.get_device_properties(0)),
                    "full_args":vars(args),
                }
                def custom_serializer(obj):
                    try:
                        return json.JSONEncoder().default(obj)
                    except TypeError:
                        return f"Unserializable: {type(obj).__name__}"
                args.out_jsonl.write(json.dumps(measurement, default=custom_serializer) + "\n")
                args.out_jsonl.flush()

def send_queries(args):
    nq = NemoQuery(url="localhost:8000", model_name=args.triton_model_name)
    perform_benchmark(args, nq)

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
        except Exception as e:
            logging.error("An error has occurred while sending queries.")
            logging.error(repr(e))
            logging.error(traceback.format_exc())
        try:
            logging.info("Model serving will be stopped.")
            nm.stop()
        except Exception as e:
            logging.error("Model could not be stopped properly.")
            logging.error(repr(e))
            logging.error(traceback.format_exc()) 


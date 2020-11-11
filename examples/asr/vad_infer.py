# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""
During inference, we perform frame-level prediction by two approaches: 
    1) shift the window of length time_length (e.g. 0.63s) by shift_length (e.g. 10ms) to generate the frame and use the prediction of the window to represent the label for the frame;
       [this script demonstrate how to do this approach]
    2) generate predictions with overlapping input segments. Then a smoothing filter is applied to decide the label for a frame spanned by multiple segments. 
       [get frame level prediction by this script and use vad_overlap_posterior.py in NeMo/scripts/
       One can also find posterior about converting frame level prediction 
       to speech/no-speech segment in start and end times format in that script.]
   
Usage:
python vad_infer.py  --vad_model="MatchboxNet-VAD-3x2" --dataset=<FULL PATH OF MANIFEST TO BE PERFORMED INFERENCE ON> --out_dir='frame/demo' --time_length=0.63

"""


import json
import os
from argparse import ArgumentParser

import torch

from nemo.collections.asr.models import EncDecClassificationModel
from nemo.utils import logging

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--vad_model", type=str, default="MatchboxNet-VAD-3x2", required=False, help="Pass: 'MatchboxNet-VAD-3x2'"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path of json file of evaluation data. Audio files should have unique names.",
    )
    parser.add_argument("--out_dir", type=str, default="vad_frame", help="Dir of your vad outputs")
    parser.add_argument("--time_length", type=float, default=0.63)
    parser.add_argument("--shift_length", type=float, default=0.01)
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    if args.vad_model.endswith('.nemo'):
        logging.info(f"Using local VAD model from {args.vad_model}")
        vad_model = EncDecClassificationModel.restore_from(restore_path=args.vad_model)
    else:
        logging.info(f"Using NGC cloud VAD model {args.vad_model}")
        vad_model = EncDecClassificationModel.from_pretrained(model_name=args.vad_model)

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    # setup_test_data
    vad_model.setup_test_data(
        test_data_config={
            'vad_stream': True,
            'sample_rate': 16000,
            'manifest_filepath': args.dataset,
            'labels': ['infer',],
            'num_workers': 20,
            'shuffle': False,
            'time_length': args.time_length,
            'shift_length': args.shift_length,
            'trim_silence': False,
        }
    )

    vad_model = vad_model.to(device)
    vad_model.eval()

    data = []
    for line in open(args.dataset, 'r'):
        file = json.loads(line)['audio_filepath'].split("/")[-1]
        data.append(file.split(".wav")[0])
    print(f"Inference on {len(data)} audio files/json lines!")

    time_unit = int(args.time_length / args.shift_length)
    trunc = int(time_unit / 2)
    trunc_l = time_unit - trunc
    all_len = 0

    for i, test_batch in enumerate(vad_model.test_dataloader()):
        if i == 0:
            status = 'start' if data[i] == data[i + 1] else 'single'
        elif i == len(data) - 1:
            status = 'end' if data[i] == data[i - 1] else 'single'
        else:
            if data[i] != data[i - 1] and data[i] == data[i + 1]:
                status = 'start'
            elif data[i] == data[i - 1] and data[i] == data[i + 1]:
                status = 'next'
            elif data[i] == data[i - 1] and data[i] != data[i + 1]:
                status = 'end'
            else:
                status = 'single'
        print(data[i], status)

        test_batch = [x.to(device) for x in test_batch]
        with autocast():
            log_probs = vad_model(input_signal=test_batch[0], input_signal_length=test_batch[1])
            probs = torch.softmax(log_probs, dim=-1)
            pred = probs[:, 1]

            if status == 'start':
                to_save = pred[:-trunc]
            elif status == 'next':
                to_save = pred[trunc:-trunc_l]
            elif status == 'end':
                to_save = pred[trunc_l:]
            else:
                to_save = pred
            all_len += len(to_save)

            outpath = os.path.join(args.out_dir, data[i] + ".frame")
            with open(outpath, "a") as fout:
                for f in range(len(to_save)):
                    fout.write('{0:0.4f}\n'.format(to_save[f]))

        del test_batch
        if status == 'end' or status == 'single':
            print(f"Overall length of prediction of {data[i]} is {all_len}!")
            all_len = 0


if __name__ == '__main__':
    main()

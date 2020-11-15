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
import glob
import os
import time
from argparse import ArgumentParser
from itertools import repeat
from multiprocessing import Pool

import numpy as np
import pandas as pd


"""
This script serves two purposes:
 
    1) gen_overlap_seq: 
        Generate predictions with overlapping input segments by using the frame level prediction from NeMo/examples/asr/vad_infer.py. 
        Then a smoothing filter is applied to decide the label for a frame spanned by multiple segments. 
       
    2）gen_seg_table: 
        Converting frame level prediction to speech/no-speech segment in start and end times format.
   
Usage:

python vad_overlap_posterior.py --gen_overlap_seq --gen_seg_table --frame_folder=<FULL PATH OF YOU STORED FRAME LEVEL PREDICTION> --method='median' --overlap=0.875 --num_workers=20 --threshold=0.8 
 
"""


def gen_overlap_seq(frame_filepath, per_args):
    """
    Given a frame level prediction, generate predictions with overlapping input segments by using it
    Args:
        frame_filepath : frame prediction file to be processed.
        per_args:
            method : Median or mean smoothing filter.
            overlap : Amounts of overlap.
            seg_len : Length of window for generating the frame.
            shift_len : Amount of shift of window for generating the frame.
            out_dir : Output dir of generated prediction.
    """

    try:
        method = per_args['method']
        overlap = per_args['overlap']
        seg_len = per_args['seg_len']
        shift_len = per_args['shift_len']
        out_dir = per_args['out_dir']

        frame = np.loadtxt(frame_filepath)
        name = os.path.basename(frame_filepath).split(".frame")[0] + "." + method
        overlap_filepath = os.path.join(out_dir, name)

        shift = int(shift_len / 0.01)  # number of units of shift
        seg = int((seg_len / 0.01 + 1))  # number of units of each window/segment

        jump_on_target = int(seg * (1 - overlap))  # jump on target generated sequence
        jump_on_frame = int(jump_on_target / shift)  # jump on input frame sequence

        if jump_on_frame < 1:
            raise ValueError(
                f"Note we jump over frame sequence to generate overlapping input segments. \n \
            Your input makes jump_on_fram={jump_on_frame} < 1 which is invalid because it cannot jump and will stuck.\n \
            Please try different seg_len, shift_len and overlap choices. \n \
            jump_on_target = int(seg * (1 - overlap)) \n \
            jump_on_frame  = int(jump_on_frame/shift) "
            )

        target_len = int(len(frame) * shift)

        if method == 'mean':
            preds = np.zeros(target_len)
            pred_count = np.zeros(target_len)

            for i, og_pred in enumerate(frame):
                if i % jump_on_frame != 0:
                    continue
                start = i * shift
                end = start + seg
                preds[start:end] = preds[start:end] + og_pred
                pred_count[start:end] = pred_count[start:end] + 1

            preds = preds / pred_count
            last_non_zero_pred = preds[pred_count != 0][-1]
            preds[pred_count == 0] = last_non_zero_pred

        elif method == 'median':
            preds = [[] for _ in range(target_len)]
            for i, og_pred in enumerate(frame):
                if i % jump_on_frame != 0:
                    continue

                start = i * shift
                end = start + seg
                for j in range(start, end):
                    if j <= target_len - 1:
                        preds[j].append(og_pred)

            preds = np.array([np.median(l) for l in preds])
            nan_idx = np.isnan(preds)
            last_non_nan_pred = preds[~nan_idx][-1]
            preds[nan_idx] = last_non_nan_pred

        else:
            raise ValueError("method should be either mean or median")

        round_final = np.round(preds, 4)
        np.savetxt(overlap_filepath, round_final, delimiter='\n')
        print(f"Finished! {overlap_filepath}!")

    except Exception as e:
        raise (e)


def gen_seg_table(frame_filepath, per_args):

    """
    Convert frame level prediction to speech/no-speech segment in start and end times format.
    And save to csv file  in rttm-like format
            0, 10, speech
            10,12, no-speech
    Args:
        frame_filepath : frame prediction file to be processed.
        per_args : 
            threshold : threshold for prediction score (from 0 to 1).
            shift_len : Amount of shift of window for generating the frame. 
            out_dir : Output dir of generated table/csv file.                   
    """
    threshold = per_args['threshold']
    shift_len = per_args['shift_len']
    out_dir = per_args['out_dir']

    print(f"process {frame_filepath}")
    name = frame_filepath.split("/")[-1].rsplit(".", 1)[0]

    sequence = np.loadtxt(frame_filepath)
    start = 0
    end = 0
    start_list = [0]
    end_list = []
    state_list = []

    for i in range(len(sequence) - 1):
        current_sate = "non-speech" if sequence[i] <= threshold else "speech"
        next_state = "non-speech" if sequence[i + 1] <= threshold else "speech"
        if next_state != current_sate:
            end = i * shift_len + shift_len  # shift_len for handling joint
            state_list.append(current_sate)
            end_list.append(end)

            start = (i + 1) * shift_len
            start_list.append(start)

    end_list.append((i + 1) * shift_len + shift_len)
    state_list.append(current_sate)

    seg_table = pd.DataFrame({'start': start_list, 'end': end_list, 'vad': state_list})

    save_name = name + ".txt"
    save_path = os.path.join(out_dir, save_name)
    seg_table.to_csv(save_path, sep='\t', index=False, header=False)


if __name__ == '__main__':
    start = time.time()
    parser = ArgumentParser()
    parser.add_argument("--gen_overlap_seq", default=False, action='store_true')
    parser.add_argument("--gen_seg_table", default=False, action='store_true')
    parser.add_argument("--frame_folder", type=str, required=True)
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        help="Use mean/median for overlapped prediction. Use frame for gen_seg_table of frame prediction",
    )
    parser.add_argument("--overlap_out_dir", type=str)
    parser.add_argument("--table_out_dir", type=str)
    parser.add_argument("--overlap", type=float, default=0.875, help="Overlap percentatge. Default is 0.875")
    parser.add_argument("--seg_len", type=float, default=0.63)
    parser.add_argument("--shift_len", type=float, default=0.01)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    if args.gen_overlap_seq:
        p = Pool(processes=args.num_workers)
        print("Generating predictions with overlapping input segments")
        frame_filepathlist = glob.glob(args.frame_folder + "/*.frame")

        if not args.overlap_out_dir:
            overlap_out_dir = "./overlap_smoothing_output" + "_" + args.method + "_" + str(args.overlap)
        else:
            overlap_out_dir = args.overlap_out_dir
        if not os.path.exists(overlap_out_dir):
            print(f"Creating output dir {overlap_out_dir}")
            os.mkdir(overlap_out_dir)

        per_args = {
            "method": args.method,
            "overlap": args.overlap,
            "seg_len": args.seg_len,
            "shift_len": args.shift_len,
            "out_dir": overlap_out_dir,
        }

        p.starmap(gen_overlap_seq, zip(frame_filepathlist, repeat(per_args)))
        p.close()
        p.join()

        end = time.time()
        print(f"Generate overlapped prediction takes {end-start} seconds!\n Save to {overlap_out_dir}")

    if args.gen_seg_table:
        p = Pool(processes=args.num_workers)
        print("Converting frame level prediction to speech/no-speech segment in start and end times format.")

        if args.gen_overlap_seq:
            print("Use overlap prediction. Change if you want to use basic frame level prediction")
            frame_filepath = overlap_out_dir
            shift_len = 0.01
        else:
            print("Use basic frame level prediction")
            frame_filepath = args.frame_folder
            shift_len = args.shift_len

        frame_filepathlist = glob.glob(frame_filepath + "/*." + args.method)

        if not args.table_out_dir:
            table_out_dir = "table_output_" + str(args.threshold)
        else:
            table_out_dir = args.table_out_dir
        if not os.path.exists(table_out_dir):
            print(f"Creating rttm-like table output dir {table_out_dir}")
            os.mkdir(table_out_dir)

        per_args = {
            "threshold": args.threshold,
            "shift_len": shift_len,
            "out_dir": table_out_dir,
        }

        p.starmap(gen_seg_table, zip(frame_filepathlist, repeat(per_args)))
        p.close()
        p.join()

        end = time.time()
        print(f"Generate rttm-like table for {frame_filepath} takes {end-start} seconds!\n Save to {table_out_dir}")

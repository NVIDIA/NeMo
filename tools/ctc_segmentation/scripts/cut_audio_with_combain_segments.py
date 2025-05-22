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

import argparse
import json
import os
from glob import glob

import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
from dataclasses import dataclass
from random import choices


parser = argparse.ArgumentParser(description="Cut audio on the segments based on segments")
parser.add_argument("--output_dir", type=str, help="Path to output directory", required=True)
parser.add_argument(
    "--alignment",
    type=str,
    required=True,
    help="Path to a data directory with alignments or a single .txt file with timestamps - result of the ctc-segmentation",
)
parser.add_argument("--threshold", type=float, default=-5, help="Minimum score value accepted")
parser.add_argument("--offset", type=float, default=0, help="Offset, s")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
parser.add_argument(
    "--edge_duration",
    type=float,
    help="Duration of audio for mean absolute value calculation at the edges, s",
    default=0.05,
)
parser.add_argument("--sample_rate", type=int, help="Sample rate, Hz", default=16000)
parser.add_argument(
    "--max_duration",
    type=int,
    help="Maximum audio duration (seconds). Samples that are longer will be dropped",
    default=40,
)
parser.add_argument(
    "--max_silence",
    type=float,
    help="Maximum silence duration to combain segments, s",
    default=1.5,
)

@dataclass
class SpeechSegment:
    start_time: float = None
    end_time: float = None
    text_processed: str = ""
    text_no_preprocessing: str = ""
    text_normalized: str = ""



def process_alignment(alignment_file: str, manifest: str, clips_dir: str, args):
    """ Cut original audio file into audio segments based on alignment_file

    Args:
        alignment_file: path to the file with segmented text and corresponding time stamps.
            The first line of the file contains the path to the original audio file
        manifest: path to .json manifest to save segments metadata
        clips_dir: path to a directory to save audio clips
        args: main script args
    """
    if not os.path.exists(alignment_file):
        raise ValueError(f"{alignment_file} not found")

    base_name = os.path.basename(alignment_file).replace("_segments.txt", "")

    # read the segments, note the first line contains the path to the original audio
    segments = []
    ref_text_processed = []
    ref_text_no_preprocessing = []
    ref_text_normalized = []
    with open(alignment_file, "r") as f:
        for line in f:
            line = line.split("|")
            # read audio file name from the first line
            if len(line) == 1:
                audio_file = line[0].strip()
                continue
            ref_text_processed.append(line[1].strip())
            ref_text_no_preprocessing.append(line[2].strip())
            ref_text_normalized.append(line[3].strip())
            line = line[0].split()
            start = float(line[0])
            if start < 1:
                start = 0.0
            # segments.append((float(line[0]) + args.offset / 1000, float(line[1]) + args.offset / 1000, float(line[2])))
            segments.append((start, float(line[1]) + args.offset / 1000, float(line[2])))

    # cut the audio into segments and save the final manifests at output_dir
    sampling_rate, signal = wavfile.read(audio_file)
    original_duration = len(signal) / sampling_rate

    num_samples = int(args.edge_duration * args.sample_rate)
    low_score_dur = 0
    high_score_dur = 0
    
    # set distribution
    population = [1, 2]
    weights = [0.80, 0.20]

    with open(manifest, "a", encoding="utf8") as f:
        new_segment = None
        for i, (st, end, score) in enumerate(segments):
            current_duration = end - st
            # prune low score segments
            if score < args.threshold or current_duration > args.max_duration:
                low_score_dur += current_duration
                continue
            
            # create or exted new segment
            if not new_segment:
                new_segment = SpeechSegment(start_time=st,
                                            end_time=end,
                                            text_processed=ref_text_processed[i].strip(),
                                            text_no_preprocessing=ref_text_no_preprocessing[i].strip(),
                                            text_normalized=ref_text_normalized[i].strip(),)
            else:
                do_merge = choices(population, weights)[0] == 1
                if do_merge and st - new_segment.end_time < args.max_silence and end - new_segment.start_time <= args.max_duration:
                    new_segment.end_time = end
                    new_segment.text_processed += f" {ref_text_processed[i].strip()}"
                    new_segment.text_no_preprocessing += f" {ref_text_no_preprocessing[i].strip()}"
                    new_segment.text_normalized += f" {ref_text_normalized[i].strip()}"
                else:
                    # save previous accomulated segment
                    segment_samples = signal[round(new_segment.start_time * sampling_rate) : round(new_segment.end_time * sampling_rate)]
                    duration = len(segment_samples) / sampling_rate
                    if duration <= args.max_duration: 
                        audio_filepath = os.path.join(clips_dir, f"{base_name}_{i:04}.wav")
                        wavfile.write(audio_filepath, sampling_rate, segment_samples)
                        assert len(signal.shape) == 1 and sampling_rate == args.sample_rate, "check sampling rate"

                        info = {
                            "audio_filepath": audio_filepath,
                            "duration": duration,
                            "text": new_segment.text_processed,
                            "text_pc": new_segment.text_normalized,
                            "text_origin": new_segment.text_no_preprocessing,
                            "start_abs": float(np.mean(np.abs(segment_samples[:num_samples]))),
                            "end_abs": float(np.mean(np.abs(segment_samples[-num_samples:]))),
                        }
                        json.dump(info, f, ensure_ascii=False)
                        f.write("\n")
                        high_score_dur += duration
                    else:
                        low_score_dur += duration

                    # make new segment
                    new_segment = SpeechSegment(start_time=st,
                                                end_time=end,
                                                text_processed=ref_text_processed[i].strip(),
                                                text_no_preprocessing=ref_text_no_preprocessing[i].strip(),
                                                text_normalized=ref_text_normalized[i].strip(),)

        # import ipdb; ipdb.set_trace()
        
        if new_segment:
            # save last segment
            # if len(segments) == 1:
            #     segment_samples = signal
            #     duration = len(segment_samples) / sampling_rate
            # else:
            #     segment_samples = signal[round(new_segment.start_time * sampling_rate) : round(new_segment.end_time * sampling_rate)]
            #     duration = len(segment_samples) / sampling_rate
            segment_samples = signal[round(new_segment.start_time * sampling_rate) : round(new_segment.end_time * sampling_rate)]
            duration = len(segment_samples) / sampling_rate
            if duration <= args.max_duration: 
                audio_filepath = os.path.join(clips_dir, f"{base_name}_{i+1:04}.wav")
                wavfile.write(audio_filepath, sampling_rate, segment_samples)
                assert len(signal.shape) == 1 and sampling_rate == args.sample_rate, "check sampling rate"

                info = {
                    "audio_filepath": audio_filepath,
                    "duration": duration,
                    "text": new_segment.text_processed,
                    "text_pc": new_segment.text_normalized,
                    "text_origin": new_segment.text_no_preprocessing,
                    "start_abs": float(np.mean(np.abs(segment_samples[:num_samples]))),
                    "end_abs": float(np.mean(np.abs(segment_samples[-num_samples:]))),
                }
                json.dump(info, f, ensure_ascii=False)
                f.write("\n")
                high_score_dur += duration
            else:
                low_score_dur += duration

        # import ipdb; ipdb.set_trace()


    # keep track of duration of the deleted segments
    del_duration = 0
    begin = 0

    for i, (st, end, _) in enumerate(segments):
        if st - begin > 0.01:
            segment = signal[int(begin * sampling_rate) : int(st * sampling_rate)]
            duration = len(segment) / sampling_rate
            del_duration += duration
        begin = end

    segment = signal[int(begin * sampling_rate) :]
    duration = len(segment) / sampling_rate
    del_duration += duration

    stats = (
        args.output_dir,
        base_name,
        round(original_duration),
        round(high_score_dur),
        round(low_score_dur),
        round(del_duration),
    )
    return stats


if __name__ == "__main__":
    args = parser.parse_args()
    print("Splitting audio files into segments...")

    if os.path.isdir(args.alignment):
        alignment_files = glob(f"{args.alignment}/*_segments.txt")
    else:
        alignment_files = [args.alignment]

    # create a directory to store segments with alignement confindence score avove the threshold
    args.output_dir = os.path.abspath(args.output_dir)
    clips_dir = os.path.join(args.output_dir, "clips")
    manifest_dir = os.path.join(args.output_dir, "manifests")
    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)

    manifest = os.path.join(manifest_dir, "manifest.json")
    if os.path.exists(manifest):
        os.remove(manifest)

    stats_file = os.path.join(args.output_dir, "stats.tsv")
    with open(stats_file, "w") as f:
        f.write("Folder\tSegment\tOriginal dur (s)\tHigh quality dur (s)\tLow quality dur (s)\tDeleted dur (s)\n")

        high_score_dur = 0
        low_score_dur = 0
        del_duration = 0
        original_dur = 0

        for alignment_file in tqdm(alignment_files):
            stats = process_alignment(alignment_file, manifest, clips_dir, args)
            original_dur += stats[-4]
            high_score_dur += stats[-3]
            low_score_dur += stats[-2]
            del_duration += stats[-1]
            stats = "\t".join([str(t) for t in stats]) + "\n"
            f.write(stats)

        f.write(f"Total\t\t{round(high_score_dur)}\t{round(low_score_dur)}\t{del_duration}")

    print(f"Original duration  : {round(original_dur / 60)}min")
    print(f"High score segments: {round(high_score_dur / 60)}min ({round(high_score_dur/original_dur*100)}%)")
    print(f"Low score segments : {round(low_score_dur / 60)}min ({round(low_score_dur/original_dur*100)}%)")
    print(f"Deleted segments   : {round(del_duration / 60)}min ({round(del_duration/original_dur*100)}%)")
    print(f"Stats saved at {stats_file}")
    print(f"Manifest saved at {manifest}")

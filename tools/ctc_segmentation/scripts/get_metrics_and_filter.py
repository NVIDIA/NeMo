import argparse
import json
from glob import glob

import editdistance
from joblib import Parallel, delayed
from tqdm import tqdm

from nemo.collections.asr.parts.preprocessing.segment import AudioSegment

parser = argparse.ArgumentParser("Calculate metrics and filters out samples based on thresholds")
parser.add_argument(
    '--manifest', required=True, help='Path .json manifest file with ASR predictions saved' 'at `pred_text` field.',
)
parser.add_argument(
    '--tail_len', type=int, help='Number of characters to use for CER calculation at the edges', default=5
)
parser.add_argument('--sample_rate', type=int, help='Audio sample rate', default=16000)
parser.add_argument('--audio_dir', type=str, help='Path to original .wav files', default=None)
parser.add_argument('--max_cer', type=int, help='Threshold CER value', default=30)
parser.add_argument('--max_wer', type=int, help='Threshold WER value', default=75)
parser.add_argument('--max_len_diff', type=float, help='Threshold for len diff', default=0.3)
parser.add_argument('--max_edge_cer', type=int, help='Threshold edge CER value', default=60)
parser.add_argument('--max_duration', type=int, help='Max duration of a segment', default=-1)
parser.add_argument(
    "--num_jobs",
    default=-2,
    type=int,
    help="The maximum number of concurrently running jobs, `-2` - all CPUs but one are used",
)
parser.add_argument(
    '--only_filter',
    action='store_true',
    help='Set to True to perform only filtering (when transcripts' 'are already available)',
)


def _calculate(line, tail_len):
    eps = 1e-9

    text = line['text'].split()
    pred_text = line['pred_text'].split()

    num_words = max(len(text), eps)
    word_dist = editdistance.eval(text, pred_text)
    line['WER'] = word_dist / num_words * 100.0
    num_chars = max(len(line['text']), eps)
    char_dist = editdistance.eval(line['text'], line['pred_text'])
    line['CER'] = char_dist / num_chars * 100.0

    line['start_CER'] = editdistance.eval(line['text'][:tail_len], line['pred_text'][:tail_len]) / tail_len * 100
    line['end_CER'] = editdistance.eval(line['text'][-tail_len:], line['pred_text'][-tail_len:]) / tail_len * 100
    line['len_diff'] = 1.0 * abs(len(text) - len(pred_text)) / max(len(pred_text), eps)
    return line


def get_metrics():
    manifest_out = args.manifest.replace(".json", "_metrics.json")

    with open(args.manifest, "r") as f:
        lines = f.readlines()

    lines = Parallel(n_jobs=args.num_jobs)(
        delayed(_calculate)(json.loads(line), tail_len=args.tail_len) for line in tqdm(lines)
    )
    with open(manifest_out, "w") as f_out:
        for line in lines:
            f_out.write(json.dumps(line) + '\n')
    print(f"Metrics save at {manifest_out}")
    return manifest_out


def _apply_filters(manifest, max_cer, max_wer, max_edge_cer, max_len_diff, max_dur=-1, original_duration=0):
    manifest_out = manifest.replace(".json", "_thresholded.json")

    remaining_duration = 0
    segmented_duration = 0
    with open(manifest, 'r') as f, open(manifest_out, "w") as f_out:
        for line in f:
            item = json.loads(line)
            cer = item["CER"]
            wer = item["WER"]
            len_diff = item["len_diff"]
            duration = item['duration']
            segmented_duration += duration
            if (
                cer <= max_cer
                and wer <= max_wer
                and len_diff <= max_len_diff
                and item["end_CER"] <= max_edge_cer
                and item["start_CER"] <= max_edge_cer
                and (max_dur == -1 or (max_dur > -1 and duration < max_dur))
            ):
                remaining_duration += duration
                f_out.write(json.dumps(item) + '\n')

    print("-" * 50)
    print("Threshold values:")
    print(f"max WER: {max_wer}")
    print(f"max CER: {max_cer}")
    print(f"max edge CER: {max_edge_cer}")
    print(f"max Word len diff: {max_len_diff}")
    print(f"max Duration: {max_dur}")
    print("-" * 50)

    remaining_duration = remaining_duration / 60
    original_duration = original_duration / 60
    segmented_duration = segmented_duration / 60
    print(f"Original audio dur: {round(original_duration, 2)} min")
    print(
        f"Segmented duration: {round(segmented_duration, 2)} min ({round(100 * segmented_duration / original_duration, 2)}% of original audio)"
    )
    print(
        f'Retained {round(remaining_duration, 2)} min ({round(100*remaining_duration/original_duration, 2)}% of original or {round(100 * remaining_duration / segmented_duration, 2)}% of segmented audio).'
    )
    print(f'Retained data saved to {manifest_out}')


def filter(manifest):
    original_duration = 0
    if args.audio_dir:
        audio_files = glob(f"{args.audio_dir}*")
        for audio in audio_files:
            try:
                audio_data = AudioSegment.from_file(audio)
                duration = len(audio_data._samples) / audio_data._sample_rate
                original_duration += duration
            except:
                print(f'Skipping {audio}')

    _apply_filters(
        manifest=manifest,
        max_cer=args.max_cer,
        max_wer=args.max_wer,
        max_edge_cer=args.max_edge_cer,
        max_len_diff=args.max_len_diff,
        max_dur=args.max_duration,
        original_duration=original_duration,
    )


if __name__ == '__main__':
    args = parser.parse_args()

    if not args.only_filter:
        manifest_with_metrics = get_metrics()
    else:
        manifest_with_metrics = args.manifest
    filter(manifest_with_metrics)

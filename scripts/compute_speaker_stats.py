import argparse
import json
import os
from multiprocessing import Pool

import librosa
import torch

from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer

speaker_wise_data = {}
num_wavs_per_speaker = 50
wav_featurizer = WaveformFeaturizer(sample_rate=16000, int_values=False, augmentor=None)


def get_pitch_contour(wav, pitch_mean=None, pitch_std=None):
    ssl_hop_length = int(0.01 * 16000)

    f0, _, _ = librosa.pyin(
        wav.numpy(),
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        frame_length=ssl_hop_length * 16,
        hop_length=ssl_hop_length * 4,
        sr=16000,
        center=True,
        fill_na=0.0,
    )
    pitch_contour = torch.tensor(f0, dtype=torch.float32)
    if (pitch_mean is not None) and (pitch_std is not None):
        pitch_contour = pitch_contour - pitch_mean
        pitch_contour[pitch_contour == -pitch_mean] = 0.0
        pitch_contour = pitch_contour / pitch_std

    return pitch_contour


def _is_valid_pitch(pitch_mean, pitch_std):
    c1 = pitch_mean > 0 and pitch_mean < 1000
    c2 = pitch_std > 0 and pitch_std < 1000
    return c1 and c2


def compute_speaker_stats(speaker_id):
    print("computing for speaker: {}".format(speaker_id))
    wav_paths = speaker_wise_data[speaker_id][:num_wavs_per_speaker]
    non_zero_pc = []
    for wav_path in wav_paths:
        wav = wav_featurizer.process(wav_path)
        pitch_contour = get_pitch_contour(wav)
        pitch_contour_nonzero = pitch_contour[pitch_contour != 0]
        if len(pitch_contour_nonzero) > 0:
            non_zero_pc.append(pitch_contour_nonzero)

    if len(non_zero_pc) > 0:
        non_zero_pc = torch.cat(non_zero_pc)
        valid = "True"
        pitch_mean = non_zero_pc.mean().item()
        pitch_std = non_zero_pc.std().item()
        if not _is_valid_pitch(pitch_mean, pitch_std):
            print("invalid pitch: {}".format(speaker_id))
            pitch_mean = 212.0
            pitch_std = 70.0
            valid = "False"
    else:
        print("could not find pitch contour for speaker {}".format(speaker_id))
        valid = "False"
        pitch_mean = 212.0
        pitch_std = 70.0

    return {"pitch_mean": pitch_mean, "pitch_std": pitch_std, "valid": valid}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path", type=str, default="/home/pneekhara/NeMo2022/libri_train_formatted.json")
    parser.add_argument("--output_dir", type=str, default="/home/pneekhara/NeMo2022/")
    args = parser.parse_args()

    global speaker_wise_data
    with open(args.manifest_path, "r") as f:
        all_lines = f.readlines()
        for line in all_lines:
            record = json.loads(line)
            if 'speaker' in record:
                speaker = record['speaker']
            else:
                speaker = 0

            if speaker not in speaker_wise_data:
                speaker_wise_data[speaker] = []
            speaker_wise_data[speaker].append(record['audio_filepath'])

    speakers = list(speaker_wise_data.keys())
    with Pool(40) as p:
        results = p.map(compute_speaker_stats, speakers)

    speaker_wise_stats = {}
    for sidx, speaker in enumerate(speakers):
        speaker_wise_stats[speaker] = results[sidx]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, "speaker_wise_stats.json"), "w") as f:
        json.dump(speaker_wise_stats, f)


if __name__ == "__main__":
    main()

import argparse
import copy
import csv
import glob
import json
import os
import random
import shutil
import wave
from io import BytesIO
from pathlib import Path

### from nemo.collections.tts.models import AudioCodecModel
import librosa
import numpy as np
import soundfile as sf
import torch
from lhotse import AudioSource, CutSet, MonoCut, Recording, SupervisionSegment
from lhotse.array import Array, TemporalArray
from lhotse.audio import RecordingSet, save_audio
from lhotse.cut.base import Cut
from lhotse.features.base import Features, FeatureSet
from lhotse.shar.readers.lazy import LazySharIterator
from lhotse.shar.writers import AudioTarWriter
from matplotlib import pyplot as plt
from tqdm import tqdm

from nemo.utils import logging

#  python -m pdb -c continue /lustre/fsw/portfolios/llmservice/users/zhehuaic/works/mod_speech_llm/code/NeMo_s2s_duplex2/scripts/speech_data_generation/create_shars_duplex_multi_from_single.py --manifest /lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/tmp/msmarco_train_normalized.conversation_style_manifest_normalized_with_correctpath_with_evaluations.json.200 --out_shar_dir /lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/tmp/msmarco_train_normalized.b.duplex.200/shars --num_shard 1


def json_reader(filename):
    with open(filename) as f:
        for line in f:
            yield json.loads(line)


def create_shar_from_manifest(audio_path, manifest, out_shar_dir, num_shard=10, segment_size=120):
    # manifest = "/lustre/csfs12/portfolios/adlr/projects/adlr_audio_speech/datasets/duplex_speech/transcripts/manifests/callhome_eng_LDC97S42.ndjson"
    # audio_path = "/lustre/csfs12/portfolios/adlr/projects/adlr_audio_speech/datasets/duplex_speech/speakersep/callhome_eng_LDC97S42/8khz/"
    new_cuts = []
    in_manifest = {}
    for line in json_reader(manifest):
        in_manifest[os.path.basename(line['audio_filepath'])] = line
    audio1_names = glob.glob(audio_path + '/*s1.wav')
    for audio1_name in audio1_names:
        audio1_manifest = in_manifest[os.path.basename(audio1_name)]
        audio2_name = audio1_name.replace("s1.wav", "s0.wav")
        audio2_manifest = in_manifest[os.path.basename(audio2_name)]

        def get_wav(audio_name):
            with wave.open(audio_name, 'rb') as wav:
                ts = np.frombuffer(wav.readframes(-1), dtype=np.int16)
                params = wav.getparams()
                return ts, params.framerate

        audio1, sample_rate1 = get_wav(audio1_name)
        audio2, sample_rate2 = get_wav(audio2_name)
        assert len(audio1) == len(audio2)
        assert sample_rate1 == sample_rate2

        def filter_empty_segment(segments):
            return [s for s in segments if s['text'] != '']

        audio1_manifest['segments'] = filter_empty_segment(audio1_manifest['segments'])
        audio2_manifest['segments'] = filter_empty_segment(audio2_manifest['segments'])
        # decide which one is user and which one is agent
        if audio1_manifest['segments'][0]['start'] < audio2_manifest['segments'][0]['start']:
            user_audio = audio1
            user_manifest = audio1_manifest
            agent_audio = audio2
            agent_manifest = audio2_manifest
        else:
            user_audio = audio2
            user_manifest = audio2_manifest
            agent_audio = audio1
            agent_manifest = audio1_manifest
        segment_i_start = 0
        while segment_i_start < user_manifest['segments'][-1]['start']:

            def get_first_segment(segments, start_time):
                for segment in segments:
                    if segment['start'] >= start_time:
                        return segment
                return None

            segment_i_start = get_first_segment(user_manifest['segments'], segment_i_start)['start']
            assert segment_i_start is not None
            segment_i_end = segment_i_start + segment_size

            def get_segements_between(segments, start_time, end_time):
                result = []
                for segment in segments:
                    if segment['start'] >= start_time and segment['end'] <= end_time:
                        result.append(segment)
                return result

            user_segments = get_segements_between(user_manifest['segments'], segment_i_start, segment_i_end)
            agent_segments = get_segements_between(agent_manifest['segments'], segment_i_start, segment_i_end)
            if len(user_segments) <= 0 or len(agent_segments) <= 0:
                print(f"skip {segment_i_start} {segment_i_end}")
                segment_i_start += segment_size // 2
                continue

            def get_step_size(dur):
                return int(dur * sample_rate1)

            segment_i_end = max(user_segments[-1]['end'], agent_segments[-1]['end'])
            assert segment_i_end - segment_i_start <= segment_size

            if len(user_audio) < get_step_size(segment_i_end):
                user_audio = np.pad(user_audio, (0, get_step_size(segment_i_end) - len(user_audio)))
                agent_audio = np.pad(agent_audio, (0, get_step_size(segment_i_end) - len(agent_audio)))

            # TODO: produce an example for every 60 sec chunk
            new_cut = MonoCut(
                id=f"{os.path.basename(audio1_name)}_{segment_i_start}",
                start=0,
                duration=segment_i_end - segment_i_start,
                channel=0,
                supervisions=[],
            )

            def offset_segments(segments, offset):
                segments = copy.deepcopy(segments)
                for segment in segments:
                    segment['start'] -= offset
                    segment['end'] -= offset
                return segments

            new_cut.user_segments = offset_segments(user_segments, segment_i_start)
            new_cut.agent_segments = offset_segments(agent_segments, segment_i_start)
            if len(new_cut.user_segments) < 2 or len(new_cut.agent_segments) < 2:  # too short
                print(f"skip {segment_i_start} {segment_i_end}")
                segment_i_start += segment_size // 2
                continue
            user_stream = BytesIO()
            agent_stream = BytesIO()

            save_audio(
                dest=user_stream,
                src=user_audio[get_step_size(segment_i_start) : (get_step_size(segment_i_end) + 1)],
                sampling_rate=sample_rate1,
                format="wav",
            )
            save_audio(
                dest=agent_stream,
                src=agent_audio[get_step_size(segment_i_start) : (get_step_size(segment_i_end) + 1)],
                sampling_rate=sample_rate1,
                format="wav",
            )
            user_stream.seek(0)
            agent_stream.seek(0)
            new_cut.recording = Recording.from_bytes(user_stream.getvalue(), f"{new_cut.id}_user")
            new_cut.target_audio = Recording.from_bytes(agent_stream.getvalue(), f"{new_cut.id}_agent")
            segment_i_start += segment_size // 2
            new_cuts.append(new_cut)
        print(audio1_name)

    cuts = CutSet(cuts=new_cuts)
    shard_size = int(len(new_cuts) / num_shard)
    if len(in_manifest) % shard_size != 0:
        shard_size += 1
    print(f"shard_size {shard_size} num_shards {num_shard}")

    print("...Making Shars")
    out_shar_dir = Path(out_shar_dir)
    out_shar_dir.mkdir(parents=True, exist_ok=True)
    # assert len(user_recordings) % shard_size != 0, "Lhotse breaks if feat_list is a multiple of shard_size"
    exported = cuts.to_shar(
        out_shar_dir, fields={"recording": "flac", "target_audio": "flac"}, num_jobs=1, shard_size=shard_size
    )
    print(f"...share created")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--manifest',
        type=str,
        default="/lustre/fsw/portfolios/edgeai/projects/edgeai_riva_rivamlops/data/ALM/SpeechQA/Mixtral8x22b_MMLPC_en/original_manifests/manifest_0.json",
    )
    parser.add_argument(
        '--audio_path',
        type=str,
        default="/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/s2s_synthetic_data/Mixtral8x22b_MMLPC_en/question_shars/manifest_0_answer_small/",
    )
    parser.add_argument(
        '--out_shar_dir',
        type=str,
        default="/lustre/fs7/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/s2s_synthetic_data/s2s_lhotse_with_wavs/squadv2/",
    )
    parser.add_argument(
        '--num_shard',
        type=int,
        default=10,
    )

    args = parser.parse_args()
    print(f"out_shar_dir {args.out_shar_dir}")
    print(f"num_shard {args.num_shard}")

    create_shar_from_manifest(
        audio_path=args.audio_path,
        manifest=args.manifest,
        out_shar_dir=args.out_shar_dir,
        num_shard=args.num_shard,
    )


if __name__ == "__main__":
    main()

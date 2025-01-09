import argparse
import csv
import json
import os
import shutil
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
from lhotse.shar.writers import AudioTarWriter
from matplotlib import pyplot as plt
from tqdm import tqdm

from nemo.utils import logging

#  python -m pdb -c continue /lustre/fsw/portfolios/llmservice/users/zhehuaic/works/mod_speech_llm/code/NeMo_s2s_duplex2/scripts/speech_data_generation/create_shars_duplex_multi_from_single.py --manifest /lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/tmp/msmarco_train_normalized.conversation_style_manifest_normalized_with_correctpath_with_evaluations.json.200 --out_shar_dir /lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/tmp/msmarco_train_normalized.b.duplex.200/shars --num_shard 1


def json_reader(filename):
    with open(filename) as f:
        for line in f:
            yield json.loads(line)


def create_shar_from_manifest(
    manifest,
    out_shar_dir,
    prompt_wav,
    num_shard=10,
    dataset_name='squadv2',
    turn_silence_sec=0.32,
    newpath='/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/RIVA-TTS/en/',
    sample_rate=22050,
):
    in_manifest = list(json_reader(manifest))
    print(f"...loaded {manifest} # of datapoints {len(in_manifest)}")

    prev_user_audio = None
    prev_agent_audio = None
    prev_text = None
    new_cuts = []
    for i, line in tqdm(enumerate(in_manifest)):
        # For single turn convs is a list of 2 elements
        # First element is user speech and second is agent speech
        audiopath = line["audio_filepath"].replace("/home/jasoli/data_prime/RIVA-TTS/en/", newpath)
        audio = Recording.from_file(audiopath).resample(sample_rate).load_audio()
        prompt_audio = Recording.from_file(prompt_wav).resample(sample_rate).load_audio()
        text = line["text"]

        silence_padding = np.zeros((1, int(turn_silence_sec * sample_rate)))
        cur_agent_audio = audio
        id = f"{os.path.basename(audiopath)}"
        user_audio_list = []
        agent_audio_list = []
        cur_user = [
            prompt_audio,
            silence_padding,
            audio,
            silence_padding,
            np.zeros_like(cur_agent_audio),
            silence_padding,
        ]
        cur_user_dur = (
            prompt_audio.shape[1] + silence_padding.shape[1] + audio.shape[1] + silence_padding.shape[1]
        ) / sample_rate
        cur_user_audio = np.concatenate(cur_user, axis=1)
        cur_agent = [
            np.zeros_like(prompt_audio),
            silence_padding,
            np.zeros_like(audio),
            silence_padding,
            cur_agent_audio,
            silence_padding,
        ]
        cur_agent_audio = np.concatenate(cur_agent, axis=1)
        supervisions = []
        if prev_user_audio is not None and prev_agent_audio is not None:
            user_audio = np.concatenate([prev_user_audio, cur_user_audio], axis=1)
            agent_audio = np.concatenate([prev_agent_audio, cur_agent_audio], axis=1)
            supervisions.append(
                SupervisionSegment(
                    id=prev_id,
                    recording_id=prev_id,
                    start=0,
                    duration=prev_user_dur,
                    text="Can you repeat after me? " + prev_text,
                    speaker="user",
                    language="EN",
                ),
            )
            supervisions.append(
                SupervisionSegment(
                    id=prev_id,
                    recording_id=prev_id,
                    start=prev_user_dur,
                    duration=prev_agent_audio.shape[1] / sample_rate - prev_user_dur,
                    text=prev_text,
                    speaker="agent",
                    language="EN",
                ),
            )
            supervisions.append(
                SupervisionSegment(
                    id=id,
                    recording_id=id,
                    start=prev_agent_audio.shape[1] / sample_rate,
                    duration=cur_user_dur,
                    text="Can you repeat after me? " + text,
                    speaker="user",
                    language="EN",
                ),
            )
            supervisions.append(
                SupervisionSegment(
                    id=id,
                    recording_id=id,
                    start=prev_agent_audio.shape[1] / sample_rate + cur_user_dur,
                    duration=cur_agent_audio.shape[1] / sample_rate - cur_user_dur,
                    text=text,
                    speaker="agent",
                    language="EN",
                ),
            )
            total_dur = cur_agent_audio.shape[1] / sample_rate + prev_agent_audio.shape[1] / sample_rate
            cut = MonoCut(
                id=id,
                start=0,
                duration=total_dur,
                channel=0,
                supervisions=supervisions,
            )
            user_stream = BytesIO()
            agent_stream = BytesIO()
            save_audio(dest=user_stream, src=user_audio, sampling_rate=sample_rate, format="wav")
            save_audio(dest=agent_stream, src=agent_audio, sampling_rate=sample_rate, format="wav")
            user_stream.seek(0)
            agent_stream.seek(0)
            cut.recording = Recording.from_bytes(user_stream.getvalue(), f"{cut.id}_user")
            cut.target_audio = Recording.from_bytes(agent_stream.getvalue(), f"{cut.id}_agent")
            new_cuts.append(cut)

        prev_agent_audio = cur_agent_audio
        prev_user_audio = cur_user_audio
        prev_text = text
        prev_user_dur = cur_user_dur
        prev_id = id

    cuts = CutSet(cuts=new_cuts)
    shard_size = int(len(new_cuts) / num_shard)
    if len(in_manifest) % shard_size != 0:
        shard_size += 1
    print(f"shard_size {shard_size} num_shards {num_shard}")

    print("...Making Shars")
    out_shar_dir = Path(out_shar_dir)
    out_shar_dir.mkdir(parents=True, exist_ok=True)
    shard_size = shard_size
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
        default="/lustre/fsw/portfolios/convai/users/subhankarg/manifests/s2s/squadv2/conversation_style_manifest_normalized_with_correctpath_with_evaluations.json",
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
    parser.add_argument(
        '--prompt_wav',
        type=str,
        default="",
    )

    args = parser.parse_args()
    print(f"manifest {args.manifest}")
    print(f"out_shar_dir {args.out_shar_dir}")
    print(f"num_shard {args.num_shard}")

    create_shar_from_manifest(
        manifest=args.manifest,
        out_shar_dir=args.out_shar_dir,
        num_shard=args.num_shard,
        prompt_wav=args.prompt_wav,
    )


if __name__ == "__main__":
    main()

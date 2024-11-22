import argparse
import csv
import json
import os
import shutil
from pathlib import Path

### from nemo.collections.tts.models import AudioCodecModel
import librosa
import numpy as np
import soundfile as sf
import torch
from lhotse import AudioSource, CutSet, Recording, SupervisionSegment
from lhotse.array import Array, TemporalArray
from lhotse.audio import RecordingSet, save_audio
from lhotse.cut.base import Cut
from lhotse.features.base import Features, FeatureSet
from lhotse.shar.writers import AudioTarWriter
from matplotlib import pyplot as plt
from tqdm import tqdm

#  python -m pdb -c continue /lustre/fsw/portfolios/llmservice/users/zhehuaic/works/mod_speech_llm/code/NeMo_s2s_duplex/scripts/speech_data_generation/create_shars_duplex_from_multi.py --manifest /lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/tmp/msmarco_train_normalized.b.multiturn/conversation_style_manifest_normalized_with_correctpath_with_evaluations.multi2.json --out_shar_dir /lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/tmp/msmarco_train_normalized.b.duplex/shars --num_shard 1


def json_reader(filename):
    with open(filename) as f:
        for line in f:
            yield json.loads(line)


def create_shar_from_manifest(manifest, out_shar_dir, num_shard=10):
    in_manifest = list(json_reader(manifest))
    print(f"...loaded {manifest} # of datapoints {len(in_manifest)}")
    shard_size = int(len(in_manifest) / num_shard)
    if len(in_manifest) % shard_size != 0:
        shard_size += 1
    print(f"shard_size {shard_size} num_shards {num_shard}")

    user_recordings = []
    answer_list = []
    instructions = []
    source_language = []
    target_language = []
    target_recordings = []
    for i, line in tqdm(enumerate(in_manifest)):
        # For single turn convs is a list of 2 elements
        # First element is user speech and second is agent speech
        convs = line["conversations"]
        for conv in convs:
            conv["value"] = conv["value"].replace("fs7", "fsw")

        # User_Speech
        user_recording = Recording.from_file(convs[0]['value'])
        user_recordings.append(user_recording)

        # Instructions from the user. In case the question is part of the source audio this is a static text "Transcribe and answer",
        # If not then this is the actual question from the user but in text.
        # For direct_s2s instructions are always empty (else part)
        if "instruction" in convs[0]:
            instructions.append(convs[0]["instruction"])
        else:
            instructions.append("")

        # Language source
        if "lang" in convs[0]:
            source_language.append(convs[0]["lang"])
        else:
            source_language.append("EN")

        # Loading agent audio and using only the extracted features as nd.array
        target_recordings.append(Recording.from_file(convs[1]['value']))
        # Agent answer transcript
        answer_list.append(convs[1]["transcript"])
        # Language target
        if "lang" in convs[1]:
            target_language.append(convs[1]["lang"])
        else:
            target_language.append("EN")

    print("Done extracting data from manifest")
    print(len(user_recordings))
    cuts = CutSet.from_manifests(recordings=RecordingSet.from_recordings(user_recordings))

    # Attach text
    for j, cut in tqdm(enumerate(cuts)):
        user_audio = np.array([[]])
        agent_audio = np.array([[]])
        total_dur = 0
        convs = in_manifest[j]["conversations"]
        for i in range(0, len(convs), 2):

            user_duration = Recording.from_file(convs[i]['value']).duration
            sample_rate = Recording.from_file(convs[i + 1]['value']).sampling_rate
            agent_duration = Recording.from_file(convs[i + 1]['value']).duration
            cur_user_audio = Recording.from_file(convs[i]['value']).resample(sample_rate).load_audio()
            cur_agent_audio = Recording.from_file(convs[i + 1]['value']).load_audio()
            user_audio = np.concatenate([user_audio, cur_user_audio, 0 * cur_agent_audio], axis=1)
            agent_audio = np.concatenate([agent_audio, 0 * cur_user_audio, cur_agent_audio], axis=1)
            cut.supervisions.append(
                SupervisionSegment(
                    id=cut.id,
                    recording_id=cut.id,
                    start=total_dur,
                    duration=user_duration,
                    text=convs[i]["instruction"],
                    speaker=convs[i]["from"],
                    language="EN",
                ),
            )
            cut.supervisions.append(
                SupervisionSegment(
                    id=cut.id,
                    recording_id=cut.id,
                    start=total_dur + user_duration,
                    duration=agent_duration,
                    text=convs[i + 1]["transcript"],
                    speaker=convs[i + 1]["from"],
                    language="EN",
                ),
            )
            total_dur += user_duration + agent_duration
        save_audio(f"/tmp/u{j}1.wav", user_audio, sample_rate)
        cut.recording = Recording.from_file(f"/tmp/u{j}1.wav")
        save_audio(f"/tmp/u{j}2.wav", agent_audio, sample_rate)
        cut.target_audio = Recording.from_file(f"/tmp/u{j}2.wav")

    print("...Making Shars")
    out_shar_dir = Path(out_shar_dir)
    out_shar_dir.mkdir(parents=True, exist_ok=True)
    shard_size = shard_size
    # assert len(user_recordings) % shard_size != 0, "Lhotse breaks if feat_list is a multiple of shard_size"
    exported = cuts.to_shar(out_shar_dir, fields={}, num_jobs=1, shard_size=shard_size)
    print(f"...share created")
    for i, path in tqdm(enumerate(exported["cuts"])):
        out_path = path.replace("cuts", "target_audio").replace(".jsonl.gz", ".tar")
        with AudioTarWriter(out_path, shard_size=None, format="wav") as writer:
            for cut in CutSet.from_file(path):
                writer.write(
                    cut.id, cut.target_audio.load_audio(), manifest=cut.target_audio, sampling_rate=sample_rate
                )
        out_path = path.replace("cuts", "recording").replace(".jsonl.gz", ".tar")
        with AudioTarWriter(out_path, shard_size=None, format="wav") as writer:
            for cut in CutSet.from_file(path):
                writer.write(cut.id, cut.recording.load_audio(), manifest=cut.recording, sampling_rate=sample_rate)


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

    args = parser.parse_args()
    print(f"manifest {args.manifest}")
    print(f"out_shar_dir {args.out_shar_dir}")
    print(f"num_shard {args.num_shard}")

    create_shar_from_manifest(
        manifest=args.manifest,
        out_shar_dir=args.out_shar_dir,
        num_shard=args.num_shard,
    )


if __name__ == "__main__":
    main()

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
from lhotse import AudioSource, CutSet, Recording, SupervisionSegment
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


def create_shar_from_manifest(manifest, out_shar_dir, num_shard=10, dataset_name='squadv2', turn_silence_sec=0.32):
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
    cleaned_manifest = []
    for i, line in tqdm(enumerate(in_manifest)):
        # For single turn convs is a list of 2 elements
        # First element is user speech and second is agent speech
        convs = line["conversations"]
        for conv in convs:
            conv["value"] = conv["value"].replace("fs7", "fsw")

        if convs[1]["transcript"] != 'I could not find the answer in the audio.':
            try:
                if 'value' not in convs[0] or not os.path.exists(convs[0]['value']):
                    raise FileNotFoundError(f"File not found for convs[0]['value']: {convs[0]['value']}")

                if 'value' not in convs[1] or not os.path.exists(convs[1]['value']):
                    raise FileNotFoundError(f"File not found for convs[1]['value']: {convs[1]['value']}")

                cleaned_manifest.append(line)
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
            except:
                logging.info(f'Skipping {i}th json record.')
    in_manifest = cleaned_manifest

    print("Done extracting data from manifest")
    print(len(user_recordings))
    cuts = CutSet.from_manifests(recordings=RecordingSet.from_recordings(user_recordings))

    # Attach text
    num_cuts = len(cuts)
    for j, cut in tqdm(enumerate(cuts)):
        user_audio_list = []
        agent_audio_list = []
        total_dur = 0

        convs = in_manifest[j]["conversations"] + in_manifest[num_cuts - j - 1]["conversations"]
        for i in range(0, len(convs), 2):

            user_recording = Recording.from_file(convs[i]['value'])
            agent_recording = Recording.from_file(convs[i + 1]['value'])

            sample_rate = agent_recording.sampling_rate
            user_duration = user_recording.duration + turn_silence_sec
            agent_duration = agent_recording.duration
            cur_user_audio = user_recording.resample(sample_rate).load_audio()
            cur_agent_audio = agent_recording.load_audio()

            silence_padding = np.zeros((1, int(turn_silence_sec * sample_rate)))
            user_audio_list.extend([cur_user_audio, silence_padding, np.zeros_like(cur_agent_audio)])
            agent_audio_list.extend([np.zeros_like(cur_user_audio), silence_padding, cur_agent_audio])

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

        user_audio = np.concatenate(user_audio_list, axis=1)
        agent_audio = np.concatenate(agent_audio_list, axis=1)
        # append trailing silence to help agent learn to stop
        user_audio_list.append(silence_padding)
        agent_audio_list.append(silence_padding)

        cut.duration = total_dur + turn_silence_sec
        cut.duration_no_sil = total_dur

        cut.start = 0.0

        user_stream = BytesIO()
        agent_stream = BytesIO()
        save_audio(dest=user_stream, src=user_audio, sampling_rate=sample_rate, format="wav")
        save_audio(dest=agent_stream, src=agent_audio, sampling_rate=sample_rate, format="wav")
        user_stream.seek(0)
        agent_stream.seek(0)
        cut.recording = Recording.from_bytes(user_stream.getvalue(), f"{cut.id}_user")
        cut.target_audio = Recording.from_bytes(agent_stream.getvalue(), f"{cut.id}_agent")

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
        '--dataset_name',
        type=str,
        default="squadv2",
    )

    args = parser.parse_args()
    print(f"manifest {args.manifest}")
    print(f"out_shar_dir {args.out_shar_dir}")
    print(f"num_shard {args.num_shard}")

    create_shar_from_manifest(
        manifest=args.manifest,
        out_shar_dir=args.out_shar_dir,
        num_shard=args.num_shard,
        dataset_name=args.dataset_name,
    )


if __name__ == "__main__":
    main()

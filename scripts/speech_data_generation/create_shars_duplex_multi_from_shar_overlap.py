import argparse
import copy
import csv
import glob
import json
import os
import random
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


def create_shar_from_manifest(in_dir, in_dir_question, manifest, out_shar_dir, num_shard=10, turn_silence_sec=0.32):
    in_manifest = list(json_reader(manifest))
    source_text_dict = {}
    for line in in_manifest:
        source_text_dict[line['audio_filepath'].replace(".wav", "")] = line['text']
    cuts_name = glob.glob(in_dir + '/cuts.*')

    def replace_cuts_name(cuts_name, new_name=None, new_root=None):
        cuts_name = [name.replace("jsonl.gz", "tar") for name in cuts_name]
        if new_root is not None:
            cuts_name = [os.path.join(new_root, os.path.basename(name)) for name in cuts_name]
        if new_name is not None:
            cuts_name = [name.replace("cuts", new_name) for name in cuts_name]
        return cuts_name

    cuts = LazySharIterator(
        {
            "cuts": cuts_name,
            "recording": replace_cuts_name(cuts_name, new_name="recording"),
            "target_audio": replace_cuts_name(cuts_name, new_name="target_audio"),
            "question_recording": replace_cuts_name(
                cuts_name, new_name="question_recording", new_root=in_dir_question
            ),
        }
    )
    new_cuts = []
    prev_cut2 = None
    prev_cut = None

    def create_cut_from_single_cut(cut, new_cut, total_dur, overlap_sec=0, limit_agent_length=0):
        agent_recording = cut.target_audio
        cur_user_context_audio = cut.recording
        cur_user_question_audio = cut.question_recording
        sample_rate = agent_recording.sampling_rate
        user_duration = cur_user_context_audio.duration + cur_user_question_audio.duration + turn_silence_sec
        agent_duration = agent_recording.duration

        def get_audio(recording, sample_rate):
            return recording.resample(sample_rate).load_audio()

        user_context = source_text_dict[cut.recording.id]
        if random.random() < 0.5:
            cur_user_audio = np.concatenate(
                [get_audio(cur_user_context_audio, sample_rate), get_audio(cur_user_question_audio, sample_rate)],
                axis=1,
            )
            cur_user_text = user_context + " ; " + cut.supervisions[0].text
        else:
            cur_user_audio = np.concatenate(
                [get_audio(cur_user_question_audio, sample_rate), get_audio(cur_user_context_audio, sample_rate)],
                axis=1,
            )
            cur_user_text = cut.supervisions[0].text + " ; " + user_context
        cur_agent_audio = agent_recording.load_audio()
        if limit_agent_length > 0 and limit_agent_length < agent_duration:
            cur_agent_audio = cur_agent_audio[:, : int(limit_agent_length * sample_rate)]
            agent_duration = limit_agent_length

        if overlap_sec > 0:
            overlap_sec = min(overlap_sec, user_audio_list[-1].shape[1] / sample_rate)
            overlap_samples = int(overlap_sec * sample_rate)
            user_audio_list[-1] = user_audio_list[-1][:, :-overlap_samples]
            agent_audio_list.extend(
                [np.zeros_like(cur_user_audio)[:, :-overlap_samples], silence_padding, cur_agent_audio]
            )
            total_dur -= overlap_sec
        else:
            agent_audio_list.extend([np.zeros_like(cur_user_audio), silence_padding, cur_agent_audio])

        user_audio_list.extend([cur_user_audio, silence_padding, np.zeros_like(cur_agent_audio)])

        new_cut.supervisions.append(
            SupervisionSegment(
                id=cut.id,
                recording_id=cut.id,
                start=total_dur,
                duration=user_duration,
                text=cur_user_text,
                speaker="user",
                language="EN",
            ),
        )
        new_cut.supervisions.append(
            SupervisionSegment(
                id=cut.id,
                recording_id=cut.id,
                start=total_dur + user_duration,
                duration=agent_duration,
                text=cut.supervisions[1].text,
                speaker="agent",
                language="EN",
            ),
        )
        total_dur += user_duration + agent_duration
        return total_dur

    for j, cut in tqdm(enumerate(cuts)):
        assert len(cut.supervisions) == 2

        user_audio_list = []
        agent_audio_list = []
        total_dur = 0
        sample_rate = cut.target_audio.sampling_rate
        if (
            cut.supervisions[0].text
            == "Transcribe the spoken content to written German text, with punctuations and capitalizations."
        ):
            continue  # ASR data is not useful
        if prev_cut2 is not None:
            silence_padding = np.zeros((1, int(turn_silence_sec * sample_rate)))
            new_cut = copy.deepcopy(cut)
            del new_cut.question_recording
            new_cut.supervisions = []
            total_dur = create_cut_from_single_cut(prev_cut2, new_cut, total_dur, limit_agent_length=2.5)
            total_dur = create_cut_from_single_cut(
                prev_cut, new_cut, total_dur, limit_agent_length=2.0, overlap_sec=1.28
            )
            total_dur = create_cut_from_single_cut(cut, new_cut, total_dur, overlap_sec=1.28)
            # append trailing silence to help agent learn to stop
            user_audio_list.append(silence_padding)
            agent_audio_list.append(silence_padding)

            user_audio = np.concatenate(user_audio_list, axis=1)
            agent_audio = np.concatenate(agent_audio_list, axis=1)
            assert user_audio.shape == agent_audio.shape

            new_cut.duration = user_audio.shape[1] / sample_rate
            new_cut.duration_no_sil = total_dur - turn_silence_sec
            new_cut.start = 0.0

            user_stream = BytesIO()
            agent_stream = BytesIO()
            save_audio(dest=user_stream, src=user_audio, sampling_rate=sample_rate, format="wav")
            save_audio(dest=agent_stream, src=agent_audio, sampling_rate=sample_rate, format="wav")
            user_stream.seek(0)
            agent_stream.seek(0)
            new_cut.recording = Recording.from_bytes(user_stream.getvalue(), f"{cut.id}_user")
            new_cut.target_audio = Recording.from_bytes(agent_stream.getvalue(), f"{cut.id}_agent")
            new_cuts.append(new_cut)
        prev_cut2 = copy.deepcopy(prev_cut)
        prev_cut = copy.deepcopy(cut)
    cuts = CutSet(cuts=new_cuts)
    shard_size = int(j / num_shard)
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
        '--in_dir',
        type=str,
        default="/lustre/fsw/portfolios/edgeai/projects/edgeai_riva_rivamlops/data/ALM/SpeechQA/Mixtral8x22b_MMLPC_en/onfly_timestamp_s2s_shars/manifest_0_answer_small/",
    )
    parser.add_argument(
        '--manifest',
        type=str,
        default="/lustre/fsw/portfolios/edgeai/projects/edgeai_riva_rivamlops/data/ALM/SpeechQA/Mixtral8x22b_MMLPC_en/original_manifests/manifest_0.json",
    )
    parser.add_argument(
        '--in_dir_question',
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
    parser.add_argument(
        '--turn_silence_sec',
        type=float,
        default=0.32,
    )

    args = parser.parse_args()
    print(f"out_shar_dir {args.out_shar_dir}")
    print(f"num_shard {args.num_shard}")

    create_shar_from_manifest(
        in_dir=args.in_dir,
        manifest=args.manifest,
        in_dir_question=args.in_dir_question,
        out_shar_dir=args.out_shar_dir,
        num_shard=args.num_shard,
        turn_silence_sec=args.turn_silence_sec,
    )


if __name__ == "__main__":
    main()

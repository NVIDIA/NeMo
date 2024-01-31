import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from omegaconf import OmegaConf
from tqdm import tqdm

from nemo.collections.asr.parts.preprocessing.perturb import NoisePerturbation, WhiteNoisePerturbation
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.tts.models import AudioCodecModel
from nemo.collections.tts.modules.transformer import mask_from_lens
from nemo.collections.tts.parts.utils.tts_dataset_utils import get_base_dir
from nemo.core.classes import Dataset
from nemo.utils import logging


class AudioDataset(Dataset):
    def __init__(
        self,
        manifest_paths,
        min_duration=1.0,
        max_duration=22.0,
        sample_rate=24000,
        pad_multiple=320,
    ):
        self.data = []
        for manifest_path in manifest_paths:
            with open(manifest_path, "r") as f:
                for line in f:
                    record = json.loads(line)
                    if 'answer_duration' not in record:
                        record['answer_duration'] = record['duration']

                    if record['answer_duration'] < min_duration or record['answer_duration'] > max_duration:
                        continue

                    self.data.append(record)

        self.sample_rate = sample_rate
        self.pad_multiple = pad_multiple
        self.base_data_dir = get_base_dir([item["audio_filepath"] for item in self.data])

    def __len__(self):
        return len(self.data)

    def _get_wav_from_filepath(self, audio_filepath, perturb=False):
        features = AudioSegment.segment_from_file(
            audio_filepath, target_sr=self.sample_rate, n_segments=-1, trim=False,
        )
        audio_samples = features.samples
        audio = torch.tensor(audio_samples)
        audio = torch.nn.functional.pad(audio, (0, self.pad_multiple - audio.size(0) % self.pad_multiple), value=0)
        audio_length = torch.tensor(audio.size(0)).long()
        return audio, audio_length

    def pad_collate_fn(self, batch):
        final_batch = {}
        for row in batch:
            for key in row:
                if key not in final_batch:
                    final_batch[key] = []
                final_batch[key].append(row[key])

        max_audio_len = max([_audio_len.item() for _audio_len in final_batch["audio_len"]])

        audios_padded = []
        for audio in final_batch["audio"]:
            audio_padded = torch.nn.functional.pad(audio, (0, max_audio_len - audio.size(0)), value=0)
            audios_padded.append(audio_padded)

        final_batch["audio"] = audios_padded

        non_tensor_keys = [
            "audio_filepath",
            "duration",
            "rel_audio_path_as_text_id",
        ]

        for key in final_batch:
            if key not in non_tensor_keys:
                final_batch[key] = torch.stack(final_batch[key])

        return final_batch

    def __getitem__(self, index):
        sample = self.data[index]
        rel_audio_path = Path(sample["audio_filepath"]).relative_to(self.base_data_dir).with_suffix("")
        rel_audio_path_as_text_id = str(rel_audio_path).replace("/", "_")

        audio, audio_length = self._get_wav_from_filepath(
            sample["audio_filepath"], perturb=False
        )

        return {
            "audio": audio,
            "audio_len": audio_length,
            "rel_audio_path_as_text_id": rel_audio_path_as_text_id,
            "audio_filepath": sample["audio_filepath"],
            "duration": sample["answer_duration"],
        }

def save_batch_audios(batch, bidx, temp_dir, codec_model, codec_model_type='encodec', codec_model_sample_rate=24000):
    for sidx in range(batch["audio"].shape[0]):
        sample_audio = batch["audio"][sidx]
        sample_audio_len = batch["audio_len"][sidx].item()
        sample_audio = sample_audio[:sample_audio_len]

        # Save sample_audio
        sample_audio_path = os.path.join(temp_dir, f"{bidx}_{sidx}_sample.wav")
        torchaudio.save(sample_audio_path, sample_audio[None].cpu(), codec_model_sample_rate)

        with torch.no_grad():
            for key in batch:
                if "CODEC" in key:
                    codec = batch[key][sidx]  # (8, T)
                    codec_len = torch.Tensor([codec.shape[1]]).long().cuda()
                    codec_decoded_audio, _ = codec_model.decode(tokens=codec.unsqueeze(0), tokens_len=codec_len)
                    codec_decoded_audio = codec_decoded_audio[0]

                    codec_decoded_audio_path = os.path.join(temp_dir, f"{bidx}_{sidx}_{key}_decoded.wav")
                    torchaudio.save(codec_decoded_audio_path, codec_decoded_audio[None].cpu(), codec_model_sample_rate)

def main():
    parser = argparse.ArgumentParser(description='Create multiple tasks')
    parser.add_argument(
        '--manifest_paths',
        type=str,
        default=None,
    )
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--out_dir', type=str, default='/Data/CodecDatasets/speechllm_codecdatasets/')
    parser.add_argument('--dataset_name', type=str, default='LibriTTSCorrectContext_train')
    parser.add_argument('--codec_model_path', type=str, default='/Data/Checkpoints/rlang_codec/SpeechCodec.nemo')
    parser.add_argument('--codec_model', type=str, default='nemo_codec')  # encodec, uniaudio_codec, dac or nemo_codec
    parser.add_argument('--use_context_as_same_speaker_audio', action='store_true')
    parser.add_argument('--save_only_tts_records', action='store_true')
    parser.add_argument('--shuffle', action='store_true')
    args = parser.parse_args()

    model_path = args.codec_model_path
    codec_model = AudioCodecModel.restore_from(model_path)
    codec_model.to('cuda')
    codec_model.eval()
    codec_model_sample_rate = 22050
    codec_model_downsampling_factor = 256.0

    dataset = AudioDataset(
        manifest_paths=[args.manifest_paths],
        sample_rate=codec_model_sample_rate,
        pad_multiple=int(codec_model_downsampling_factor),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=args.batch_size, collate_fn=dataset.pad_collate_fn, shuffle=False, num_workers=8,
    )

    _exp_name = "{}_{}_".format(args.dataset_name, args.codec_model)
    temp_dir = os.path.join(args.out_dir, "temp_{}".format(_exp_name))
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    codec_base_dir = os.path.join(args.out_dir, "codecs")
    audiocodec_out_dir = os.path.join(codec_base_dir, _exp_name)

    if not os.path.exists(audiocodec_out_dir):
        os.makedirs(audiocodec_out_dir)

    for bidx, batch in enumerate(tqdm(dataloader)):
        # print("bidx", bidx+1, "of", len(dataloader))
        cuda_keys = ['audio', 'audio_len']
        for key in cuda_keys:
            batch[key] = batch[key].cuda()
        with torch.no_grad():
            original_codec_codes, _ = codec_model.encode(audio=batch["audio"], audio_len=batch["audio_len"])

        target_codecs = []
        for sidx in range(batch['audio'].shape[0]):

            codec_len = math.ceil(batch['audio_len'][sidx].item() / codec_model_downsampling_factor)
            sample_codec_codes = original_codec_codes[sidx][:, :codec_len]
            target_codecs.append(sample_codec_codes)

            example_name = batch['rel_audio_path_as_text_id'][sidx]
            target_codec_filepath = os.path.join(audiocodec_out_dir, "target_codes_{}.pt".format(example_name))
            torch.save(sample_codec_codes.cpu().type(torch.int16), target_codec_filepath)

        batch['target_CODEC'] = target_codecs

        if bidx == 0:
            save_batch_audios(batch, bidx, temp_dir, codec_model, args.codec_model, codec_model_sample_rate)

if __name__ == '__main__':
    main()
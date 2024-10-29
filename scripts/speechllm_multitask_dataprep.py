# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from encodec import EncodecModel
from omegaconf import OmegaConf
from tqdm import tqdm

from nemo.collections.asr.parts.preprocessing.perturb import NoisePerturbation, WhiteNoisePerturbation
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.tts.models import AudioCodecModel
from nemo.collections.tts.modules.transformer import mask_from_lens
from nemo.collections.tts.parts.utils.tts_dataset_utils import get_base_dir
from nemo.core.classes import Dataset
from nemo.utils import logging

try:
    from models.soundstream import SoundStream
except:
    logging.warning("SoundStream not found, uniaudio cannot be used")

try:
    import dac
except:
    logging.warning("DAC not found")


class AudioDataset(Dataset):
    def __init__(
        self,
        manifest_paths,
        min_duration=0.0,
        max_duration=22.0,
        sample_rate=24000,
        noise_manifest_path=None,
        min_snr_db=0,
        max_snr_db=5,
        max_same_speaker_audios=1,
        use_context_as_same_speaker_audio=False,
        pad_multiple=320,
        audio_type="actual", # actual or noise or silence
    ):
        self.data = []
        speakerwise_records = {}
        for manifest_path in manifest_paths:
            with open(manifest_path, "r") as f:
                for line in f:
                    record = json.loads(line)
                    if 'answer_duration' not in record:
                        record['answer_duration'] = record['duration']
                    
                    if isinstance(record['speaker'], str) and 'mls_english_' in record['speaker']:
                        record['speaker'] = record['speaker'].replace('mls_english_', '')
                        record['speaker'] = int(record['speaker'])

                    if record['answer_duration'] < min_duration or record['answer_duration'] > max_duration:
                        continue

                    if ('context_duration' in record) and (
                        record['context_duration'] < min_duration or record['context_duration'] > max_duration
                    ):
                        continue
                    
                    if self._is_record_valid(record):
                        self.data.append(record)
                        if record['speaker'] not in speakerwise_records:
                            speakerwise_records[record['speaker']] = []
                        speakerwise_records[record['speaker']].append(record)

        self.speakerwise_records = speakerwise_records
        self.speaker_list = list(self.speakerwise_records.keys())

        self.sample_rate = sample_rate
        self.audio_type = audio_type

        # TODO: Using White Noise Perturbation right now (dont have noise manifest)

        # self.noise_perturber = NoisePerturbation(
        #     manifest_path=noise_manifest_path,
        #     min_snr_db=min_snr_db,
        #     max_snr_db=max_snr_db,
        # )

        self.noise_perturber = WhiteNoisePerturbation()

        self.max_same_speaker_audios = max_same_speaker_audios

        # If True, use the 'context' key as the same speaker reference audio,
        # otherwise randomly choose from the same speaker audios

        self.use_context_as_same_speaker_audio = use_context_as_same_speaker_audio
        self.pad_multiple = pad_multiple

        if self.use_context_as_same_speaker_audio:
            logging.info("Using context as same speaker audio")
            self.add_context_records_to_manifest()

        self.base_data_dir = get_base_dir([item["audio_filepath"] for item in self.data])
        # self.filter_invalid_records()
        # if sup_data_dir is not None:
        #     self.sup_data_dir = sup_data_dir
        # else:
        #     self.sup_data_dir = os.path.join(self.base_data_dir, "sup_data")
        # if not os.path.exists(self.sup_data_dir):
        #     os.makedirs(self.sup_data_dir)

    def _is_record_valid(self, record):
        return True
        try:
            sf.read(record["audio_filepath"])
            # sf.read(record["context"])
            return True
        except:
            print("Skipping invalid record", record["audio_filepath"])
            return False
        
    def filter_invalid_records(self):
        filtered_data = []
        for ridx, record in enumerate(self.data):
            if ridx % 1000 == 0:
                print("Filtering", ridx, "of", len(self.data))
            try:
                sf.read(record["audio_filepath"])
                sf.read(record["context"])
            except:
                print("Skipping invalid record", record["audio_filepath"])
                continue
            filtered_data.append(record)
        print("Original data size", len(self.data))
        print("Filtered data size", len(filtered_data))
        self.data = filtered_data

    def add_context_records_to_manifest(self):
        # Add dummy records with audio_filepath as context
        # to ensure all context file paths have their codes extracted and saved.
        context_paths = {}
        target_paths = {}
        
        for record in self.data:
            if 'context' in record:
                if 'context_duration' not in record:
                    # Get duration from the context audio file
                    record['context_duration'] = float(sf.info(record['context']).duration)

                context_paths[record['context']] = {
                    'speaker': record['speaker'],
                    'duration': record['context_duration'],
                }
            if 'answer' in record:
                target_paths[record['audio_filepath']] = True

        for context_path in context_paths:
            if context_path not in target_paths:
                self.data.append(
                    {
                        "audio_filepath": context_path,
                        "context": context_path,
                        "duration": context_paths[context_path]['duration'],
                        "answer_duration": context_paths[context_path]['duration'],
                        "context_duration": context_paths[context_path]['duration'],
                        "text": "<dummy>",  # Indicates that this is a dummy record
                        "question": "<dummy>",
                        "speaker": context_paths[context_path]['speaker'],
                    }
                )

    def __len__(self):
        return len(self.data)

    def _get_wav_from_filepath(self, audio_filepath, perturb=False):
        if self.audio_type == "noise" or self.audio_type == "silence":
            # Create a 6 second noise audio
            if self.audio_type == "noise":
                audio_samples = np.random.normal(0, 1, 6 * self.sample_rate)
            else:
                audio_samples = np.zeros(6 * self.sample_rate)
            audio = torch.tensor(audio_samples).float()
            audio = torch.nn.functional.pad(audio, (0, self.pad_multiple - audio.size(0) % self.pad_multiple), value=0)
            audio_length = torch.tensor(audio.size(0)).long()

            perturbed_audio = None
            perturbed_audio_length = None
            if perturb:
                perturbed_audio = audio * 1.0
                perturbed_audio_length = (audio_length * 1.0).long()
            
            return audio, audio_length, perturbed_audio, perturbed_audio_length
        elif self.audio_type == "actual":
            features = AudioSegment.segment_from_file(
                audio_filepath, target_sr=self.sample_rate, n_segments=-1, trim=False,
            )
            audio_samples = features.samples
            audio = torch.tensor(audio_samples)
            audio = torch.nn.functional.pad(audio, (0, self.pad_multiple - audio.size(0) % self.pad_multiple), value=0)
            audio_length = torch.tensor(audio.size(0)).long()

            perturbed_audio = None
            perturbed_audio_length = None
            if perturb:
                features_copy = copy.deepcopy(features)
                self.noise_perturber.perturb(features_copy)
                perturbed_audio_samples = features_copy.samples
                perturbed_audio = torch.tensor(perturbed_audio_samples)
                perturbed_audio = torch.nn.functional.pad(
                    perturbed_audio, (0, self.pad_multiple - perturbed_audio.size(0) % self.pad_multiple), value=0
                )
                perturbed_audio_length = torch.tensor(perturbed_audio.size(0)).long()
                # import ipdb; ipdb.set_trace()

            return audio, audio_length, perturbed_audio, perturbed_audio_length
        
        else:
            raise ValueError("Unknown audio type {}".format(self.audio_type))

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

        perturbed_audios_padded = []
        max_perturbed_audio_len = max([_audio_len.item() for _audio_len in final_batch["perturbed_audio_len"]])
        for audio in final_batch["perturbed_audio"]:
            audio_padded = torch.nn.functional.pad(audio, (0, max_perturbed_audio_len - audio.size(0)), value=0)
            perturbed_audios_padded.append(audio_padded)

        final_batch["perturbed_audio"] = perturbed_audios_padded

        mixed_audios_padded = []
        max_mixed_audio_len = max([_audio_len.item() for _audio_len in final_batch["mixed_audio_len"]])
        for audio in final_batch["mixed_audio"]:
            audio_padded = torch.nn.functional.pad(audio, (0, max_mixed_audio_len - audio.size(0)), value=0)
            mixed_audios_padded.append(audio_padded)

        final_batch["mixed_audio"] = mixed_audios_padded

        non_tensor_keys = [
            "audio_filepath",
            "question",
            "text",
            "context",
            "old_speaker_id",
            "duration",
            "context_duration",
            "rel_audio_path_as_text_id",
            "samespeaker_audioids",
            "samespeaker_wavpaths",
            "speaker"
        ]

        for key in final_batch:
            if key not in non_tensor_keys:
                final_batch[key] = torch.stack(final_batch[key])

        return final_batch

    def __getitem__(self, index):
        sample = self.data[index]
        rel_audio_path = Path(sample["audio_filepath"]).relative_to(self.base_data_dir).with_suffix("")
        rel_audio_path_as_text_id = str(rel_audio_path).replace("/", "_")
        # speaker = torch.tensor(sample["speaker"]).long()
        speaker = sample['speaker']

        # Avoid fixed seed
        random.seed(time.time())
        alternate_speaker = random.choice(self.speaker_list)
        _ctr = 0
        while (alternate_speaker == speaker) and (_ctr < 10):
            random.seed(time.time())
            alternate_speaker = random.choice(self.speaker_list)
            _ctr += 1

        random.seed(time.time())
        alternate_wavpath = random.choice(self.speakerwise_records[alternate_speaker])["audio_filepath"]

        if not self.use_context_as_same_speaker_audio:
            random.shuffle(self.speakerwise_records[sample["speaker"]])
            samespeaker_wavpaths = []
            context_duration = 0.0
            for _record in self.speakerwise_records[sample["speaker"]][: self.max_same_speaker_audios]:
                if _record["audio_filepath"] != sample["audio_filepath"]:
                    samespeaker_wavpath = _record["audio_filepath"]
                    samespeaker_wavpaths.append(samespeaker_wavpath)
                    context_duration += _record["answer_duration"]

            if len(samespeaker_wavpaths) == 0:
                # Use the same audio if no other audio is available from the same speaker
                samespeaker_wavpaths = [sample["audio_filepath"]]
                context_duration = sample["answer_duration"]
        else:
            samespeaker_wavpaths = [sample["context"]]
            context_duration = sample["context_duration"]

        samespeaker_audioids = []
        for samespeaker_wavpath in samespeaker_wavpaths:
            samespeaker_rel_audio_path = Path(samespeaker_wavpath).relative_to(self.base_data_dir).with_suffix("")
            samespeaker_rel_audio_path_as_text_id = str(samespeaker_rel_audio_path).replace("/", "_")
            samespeaker_audioids.append(samespeaker_rel_audio_path_as_text_id)

        alternate_audio, alternate_audio_length, _, _ = self._get_wav_from_filepath(alternate_wavpath, perturb=False)
        audio, audio_length, perturbed_audio, perturbed_audio_length = self._get_wav_from_filepath(
            sample["audio_filepath"], perturb=True
        )

        # Mix audio and alternate audio
        if audio_length > alternate_audio_length:
            # Repeat alternate audio
            alternate_audio = alternate_audio.repeat(audio_length // alternate_audio_length + 1)
            alternate_audio = alternate_audio[:audio_length]
            mixed_audio = 0.5 * (audio + alternate_audio)
        elif audio_length <= alternate_audio_length:
            alternate_audio = alternate_audio[:audio_length]
            mixed_audio = 0.5 * (audio + alternate_audio)

        mixed_audio_length = audio_length

        if "question" not in sample:
            sample['question'] = "Text to speech this " + sample['text']

        return {
            "audio": audio,
            "audio_len": audio_length,
            "perturbed_audio": perturbed_audio,
            "perturbed_audio_len": perturbed_audio_length,
            "mixed_audio": mixed_audio,
            "mixed_audio_len": mixed_audio_length,
            "rel_audio_path_as_text_id": rel_audio_path_as_text_id,
            "samespeaker_audioids": samespeaker_audioids,
            "samespeaker_wavpaths": samespeaker_wavpaths,
            "audio_filepath": sample["audio_filepath"],
            "question": sample["question"],
            "text": sample["text"],
            "context": sample.get("context", None),
            "old_speaker_id": sample.get("old_speaker_id", None),
            "duration": sample["answer_duration"],
            "context_duration": context_duration,
            "speaker": speaker,
        }


def save_batch_audios(batch, bidx, temp_dir, codec_model, codec_model_type='encodec', codec_model_sample_rate=24000):
    for sidx in range(batch["audio"].shape[0]):
        sample_audio = batch["audio"][sidx]
        sample_audio_len = batch["audio_len"][sidx].item()
        sample_audio = sample_audio[:sample_audio_len]

        # Save sample_audio
        sample_audio_path = os.path.join(temp_dir, f"{bidx}_{sidx}_sample.wav")
        torchaudio.save(sample_audio_path, sample_audio[None].cpu(), codec_model_sample_rate)

        # Save perturbed_audio
        perturbed_audio = batch["perturbed_audio"][sidx]
        perturbed_audio_len = batch["perturbed_audio_len"][sidx].item()
        perturbed_audio = perturbed_audio[:perturbed_audio_len]
        perturbed_audio_path = os.path.join(temp_dir, f"{bidx}_{sidx}_perturbed.wav")
        torchaudio.save(perturbed_audio_path, perturbed_audio[None].cpu(), codec_model_sample_rate)

        # Save mixed_audio
        mixed_audio = batch["mixed_audio"][sidx]
        mixed_audio_len = batch["mixed_audio_len"][sidx].item()
        mixed_audio = mixed_audio[:mixed_audio_len]
        mixed_audio_path = os.path.join(temp_dir, f"{bidx}_{sidx}_mixed.wav")
        torchaudio.save(mixed_audio_path, mixed_audio[None].cpu(), codec_model_sample_rate)

        with torch.no_grad():
            for key in batch:
                if "CODEC" in key:
                    codec = batch[key][sidx]  # (8, T)
                    if codec_model_type == 'encodec':
                        codec_decoded_audio = codec_model.decode([[codec.unsqueeze(0), None]])[0][0]
                    elif codec_model_type == 'uniaudio_codec':
                        codec_decoded_audio = codec_model.decode(codec.unsqueeze(0))[0][0]
                    elif codec_model_type == 'dac':
                        _z = codec_model.quantizer.from_codes(codec.unsqueeze(0))[0]
                        codec_decoded_audio = codec_model.decoder(_z)[0][0]
                    elif codec_model_type in ['nemo_codec', 'nemo_codec21', 'nemo_codec211k', 'nemo_codec214k']:
                        codec_len = torch.Tensor([codec.shape[1]]).long().cuda()
                        codec_decoded_audio, _ = codec_model.decode(tokens=codec.unsqueeze(0), tokens_len=codec_len)
                        codec_decoded_audio = codec_decoded_audio[0]

                    codec_decoded_audio_path = os.path.join(temp_dir, f"{bidx}_{sidx}_{key}_decoded.wav")
                    torchaudio.save(codec_decoded_audio_path, codec_decoded_audio[None].cpu(), codec_model_sample_rate)


def estimate_duration_from_codeclen(codec_len, codec_downsampling_factor=320.0, codec_model_sample_rate=24000):
    num_audio_samples = codec_len * codec_downsampling_factor
    duration = num_audio_samples / codec_model_sample_rate
    return round(duration, 2)


def save_manifest(records, manifest_path):
    with open(manifest_path, "w") as f:
        file_str = ""
        for record in records:
            file_str += json.dumps(record) + "\n"
        file_str = file_str.strip()
        f.write(file_str)
    print("Saved manifest to {}".format(manifest_path))


def main():
    parser = argparse.ArgumentParser(description='Create multiple tasks')
    parser.add_argument("--noise_manifest", type=str, default="/datap/misc/noisedata/train_manifest.json")
    parser.add_argument(
        '--manifest_paths',
        type=str,
        default="/Data/manifests_libri_local/train_clean_300_speechlm_ttstasks_with3sec_ref_all_random.json",
    )
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--out_dir', type=str, default='/Data/CodecDatasets/speechllm_codecdatasets/')
    parser.add_argument('--dataset_name', type=str, default='LibriTTSCorrectContext_train')
    parser.add_argument('--codec_model_path', type=str, default='/Data/Checkpoints/rlang_codec/SpeechCodec.nemo')
    parser.add_argument('--codec_bw', type=float, default=6.0)  # 6 for 8 codebooks, 1.5 for 3 codebooks
    parser.add_argument('--codec_model', type=str, default='nemo_codec')  # encodec, uniaudio_codec, dac, nemo_codec, nemo_codec21, nemo_codec211k, nemo_codec214k
    parser.add_argument('--use_context_as_same_speaker_audio', action='store_true')
    parser.add_argument('--save_only_tts_records', action='store_true')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--split_into_train_val', action='store_true')
    parser.add_argument('--num_val_records', type=int, default=500)
    parser.add_argument('--audio_type', type=str, default='actual')  # actual, noise or silence
    args = parser.parse_args()

    if args.codec_model == 'encodec':
        codec_model = EncodecModel.encodec_model_24khz()
        codec_model.set_target_bandwidth(6.0)
        codec_model.cuda()
        codec_model.eval()
        codec_model_sample_rate = 24000
        codec_model_downsampling_factor = 320.0
    elif args.codec_model == 'uniaudio_codec':
        codec_config_path = os.path.join(os.path.dirname(args.codec_model_path), 'config.yaml')
        codec_config = OmegaConf.load(codec_config_path)
        codec_model = eval(codec_config.generator.name)(**codec_config.generator.config)
        codec_parameter_dict = torch.load(args.codec_model_path)
        codec_model.load_state_dict(codec_parameter_dict['codec_model'])  # load model
        codec_model = codec_model.cuda()
        # codec_model.eval()
        codec_model_sample_rate = 16000
        codec_model_downsampling_factor = 320.0
    elif args.codec_model == 'dac':
        model_path = args.codec_model_path
        codec_model = dac.DAC.load(model_path)
        codec_model.to('cuda')
        codec_model_sample_rate = 44100
        codec_model_downsampling_factor = 512.0
    elif args.codec_model == 'nemo_codec':
        model_path = args.codec_model_path
        codec_model = AudioCodecModel.restore_from(model_path)
        codec_model.to('cuda')
        codec_model.eval()
        codec_model_sample_rate = 22050
        codec_model_downsampling_factor = 256.0
    elif args.codec_model in ['nemo_codec21', 'nemo_codec211k', 'nemo_codec214k']:
        model_path = args.codec_model_path
        codec_model = AudioCodecModel.restore_from(model_path)
        codec_model.to('cuda')
        codec_model.eval()
        codec_model_sample_rate = 22050
        codec_model_downsampling_factor = 1024.0
    else:
        raise ValueError("Unknown codec model {}".format(args.codec_model))

    dataset = AudioDataset(
        manifest_paths=[args.manifest_paths],
        sample_rate=codec_model_sample_rate,
        noise_manifest_path=args.noise_manifest,
        use_context_as_same_speaker_audio=args.use_context_as_same_speaker_audio,
        pad_multiple=int(codec_model_downsampling_factor),
        audio_type=args.audio_type,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=args.batch_size, collate_fn=dataset.pad_collate_fn, shuffle=False, num_workers=8,
    )

    _exp_name = "{}_{}_bw_{}".format(args.dataset_name, args.codec_model, args.codec_bw)
    temp_dir = os.path.join(args.out_dir, "temp_{}".format(_exp_name))
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    codec_base_dir = os.path.join(args.out_dir, "codecs")
    manifest_dir = os.path.join(args.out_dir, "manifests")

    audiocodec_out_dir = os.path.join(codec_base_dir, _exp_name)

    if not os.path.exists(audiocodec_out_dir):
        os.makedirs(audiocodec_out_dir)

    if not os.path.exists(manifest_dir):
        os.makedirs(manifest_dir)

    all_tasks_records = []
    phoneme_tts_records = []
    sentencepiece_tts_records = []
    phoneme_plus_sentencepiece_tts_records = []

    for bidx, batch in enumerate(tqdm(dataloader)):
        # print("bidx", bidx+1, "of", len(dataloader))

        audio_len_mask = mask_from_lens(batch["audio_len"])

        cuda_keys = ['audio', 'perturbed_audio', 'mixed_audio', 'audio_len', 'perturbed_audio_len', 'mixed_audio_len']
        for key in cuda_keys:
            batch[key] = batch[key].cuda()
        with torch.no_grad():
            if args.codec_model == 'encodec':
                original_codec_codes = codec_model.encode(batch["audio"].unsqueeze(1))[0][0]
                if not args.save_only_tts_records:
                    perturbed_codec_codes = codec_model.encode(batch["perturbed_audio"].unsqueeze(1))[0][0]
                    mixed_codec_codes = codec_model.encode(batch["mixed_audio"].unsqueeze(1))[0][0]
            elif args.codec_model == 'uniaudio_codec':
                original_codec_codes = codec_model.encode(
                    batch["audio"].unsqueeze(1) * codec_config.audio_norm_scale, target_bw=args.codec_bw
                ).permute(1, 0, 2)
                if not args.save_only_tts_records:
                    perturbed_codec_codes = codec_model.encode(
                        batch["perturbed_audio"].unsqueeze(1) * codec_config.audio_norm_scale, target_bw=args.codec_bw
                    ).permute(1, 0, 2)
                    mixed_codec_codes = codec_model.encode(
                        batch["mixed_audio"].unsqueeze(1) * codec_config.audio_norm_scale, target_bw=args.codec_bw
                    ).permute(1, 0, 2)
            elif args.codec_model == 'dac':
                # z, codes, latents, _, _ = model.encode(x)
                _, original_codec_codes, _, _, _ = codec_model.encode(batch["audio"].unsqueeze(1))
                if not args.save_only_tts_records:
                    _, perturbed_codec_codes, _, _, _ = codec_model.encode(batch["perturbed_audio"].unsqueeze(1))
                    _, mixed_codec_codes, _, _, _ = codec_model.encode(batch["mixed_audio"].unsqueeze(1))
            elif args.codec_model in ['nemo_codec', 'nemo_codec21', 'nemo_codec211k', 'nemo_codec214k']:
                original_codec_codes, _ = codec_model.encode(audio=batch["audio"], audio_len=batch["audio_len"])
                if not args.save_only_tts_records:
                    perturbed_codec_codes, _ = codec_model.encode(
                        audio=batch["perturbed_audio"], audio_len=batch["perturbed_audio_len"]
                    )
                    mixed_codec_codes, _ = codec_model.encode(
                        audio=batch["mixed_audio"], audio_len=batch["mixed_audio_len"]
                    )
            else:
                raise ValueError("Unknown codec model {}".format(args.codec_model))
        
        if args.save_only_tts_records:
            perturbed_codec_codes = original_codec_codes # Dummy values to not break the code
            mixed_codec_codes = original_codec_codes # Dummy values to not break the code

        # codec_codes = transformer_encodec_model.encode(batch["audio"].unsqueeze(1), audio_len_mask, bandwidth=6.0)
        target_codecs = []
        mixed_codecs = []
        perturbed_codecs = []
        for sidx in range(batch['audio'].shape[0]):

            codec_len = math.ceil(batch['audio_len'][sidx].item() / codec_model_downsampling_factor)
            sample_codec_codes = original_codec_codes[sidx][:, :codec_len]
            target_codecs.append(sample_codec_codes)

            perturbed_codec_len = math.ceil(
                batch['perturbed_audio_len'][sidx].item() / codec_model_downsampling_factor
            )
            perturbed_sample_codec_codes = perturbed_codec_codes[sidx][:, :perturbed_codec_len]
            perturbed_codecs.append(perturbed_sample_codec_codes)

            mixed_codec_len = math.ceil(batch['mixed_audio_len'][sidx].item() / codec_model_downsampling_factor)
            mixed_sample_codec_codes = mixed_codec_codes[sidx][:, :mixed_codec_len]
            mixed_codecs.append(mixed_sample_codec_codes)

            example_name = batch['rel_audio_path_as_text_id'][sidx]

            target_codec_filepath = os.path.join(audiocodec_out_dir, "target_codes_{}.pt".format(example_name))
            torch.save(sample_codec_codes.cpu().type(torch.int16), target_codec_filepath)

            if batch['text'][sidx] == "<dummy>":
                # Only save target codes for dummy records
                # Don't need to add dummy records to manifest
                continue

            perturbed_codec_filepath = os.path.join(audiocodec_out_dir, "perturbed_codes_{}.pt".format(example_name))
            mixed_codec_filepath = os.path.join(audiocodec_out_dir, "mixed_codes_{}.pt".format(example_name))
            if not args.save_only_tts_records:
                torch.save(perturbed_sample_codec_codes.cpu().type(torch.int16), perturbed_codec_filepath)
                torch.save(mixed_sample_codec_codes.cpu().type(torch.int16), mixed_codec_filepath)

            tts_contextpath = ""
            for samespeaker_audioid in batch['samespeaker_audioids'][sidx]:
                tts_contextpath += os.path.join(audiocodec_out_dir, "target_codes_{}.pt".format(samespeaker_audioid))
                tts_contextpath += ";"
            tts_contextpath = tts_contextpath[:-1]

            tts_record = {
                "audio_filepath": batch['audio_filepath'][sidx],
                "text": batch['text'][sidx],
                "question": batch['question'][sidx].replace("Phoneme TTS", "Text to speech this"),
                "answer": target_codec_filepath,
                "context": tts_contextpath,
                "question_type": "TEXT",
                "answer_type": "AUDIOCODEC",
                "context_type": "REFSPEAKERCODEC",
                "context_duration": batch['context_duration'][sidx],
                "answer_duration": batch['duration'][sidx],
                "taskname": "squad",
                "speaker": batch['speaker'][sidx].item() if torch.is_tensor(batch['speaker'][sidx]) else batch['speaker'][sidx],
            }

            phoneme_tts_record = {key: value for key, value in tts_record.items()}
            phoneme_tts_record["question"] = phoneme_tts_record["question"].replace(
                "Text to speech this", "Phoneme TTS"
            )

            speechenhancement_record = {
                "audio_filepath": batch['audio_filepath'][sidx],
                "text": batch['text'][sidx],
                "question": "Remove Noise",
                "answer": target_codec_filepath,
                "context": perturbed_codec_filepath,
                "question_type": "TEXT",
                "answer_type": "AUDIOCODEC",
                "context_type": "AUDIOCODEC",
                "context_duration": estimate_duration_from_codeclen(
                    perturbed_codec_len, codec_model_downsampling_factor, codec_model_sample_rate
                ),
                "answer_duration": batch['duration'][sidx],
                "taskname": "squad",
            }

            speechseparation_record = {
                "audio_filepath": batch['audio_filepath'][sidx],
                "text": batch['text'][sidx],
                "question": "Extract Speaker Audio",
                "answer": target_codec_filepath,
                "context": "{},{}".format(mixed_codec_filepath, tts_contextpath),
                "question_type": "TEXT",
                "answer_type": "AUDIOCODEC",
                "context_type": "SEPARATIONCODECS",
                "context_duration": estimate_duration_from_codeclen(
                    mixed_codec_len, codec_model_downsampling_factor, codec_model_sample_rate
                ),
                "answer_duration": batch['duration'][sidx],
                "taskname": "squad",
            }

            speechediting_record = {
                "audio_filepath": batch['audio_filepath'][sidx],
                "text": batch['text'][sidx],
                "question": batch['question'][sidx].replace("Text to speech this", "Edit Speech"),
                "answer": target_codec_filepath,
                "context": target_codec_filepath,
                "question_type": "TEXT",
                "answer_type": "AUDIOCODEC",
                "context_type": "EDITINGCODECS",
                "context_duration": batch['duration'][sidx] + 3,  # 3 sec for speaker context
                "answer_duration": batch['duration'][sidx],
                "taskname": "squad",
            }

            phoneme_tts_records.append(phoneme_tts_record)
            sentencepiece_tts_records.append(tts_record)

            phoneme_plus_sentencepiece_tts_records.append(phoneme_tts_record)
            phoneme_plus_sentencepiece_tts_records.append(tts_record)

            all_tasks_records.append(tts_record)
            all_tasks_records.append(phoneme_tts_record)
            all_tasks_records.append(speechenhancement_record)
            all_tasks_records.append(speechseparation_record)
            all_tasks_records.append(speechediting_record)

        batch['target_CODEC'] = target_codecs
        batch['perturbed_CODEC'] = perturbed_codecs
        batch['mixed_CODEC'] = mixed_codecs

        if bidx == 0:
            save_batch_audios(batch, bidx, temp_dir, codec_model, args.codec_model, codec_model_sample_rate)

    if args.shuffle:
        # To ensure same split for encodec and uniaudio_codec
        random.seed(21)
        random.shuffle(all_tasks_records)
        random.shuffle(phoneme_tts_records)
        random.shuffle(sentencepiece_tts_records)
        random.shuffle(phoneme_plus_sentencepiece_tts_records)

    if args.split_into_train_val:
        # Shuffle compulsory for splitting into train and val
        # To ensure same split for encodec and uniaudio_codec
        random.seed(21)
        random.shuffle(all_tasks_records)
        random.shuffle(phoneme_tts_records)
        random.shuffle(sentencepiece_tts_records)
        # random.shuffle(phoneme_plus_sentencepiece_tts_records)
        phoneme_plus_sentencepiece_tts_records = []
        for idx in range(len(phoneme_tts_records)):
            phoneme_plus_sentencepiece_tts_records.append(phoneme_tts_records[idx])
            phoneme_plus_sentencepiece_tts_records.append(sentencepiece_tts_records[idx])

        num_val_records = args.num_val_records
        train_phoneme_tts_records = phoneme_tts_records[num_val_records:]
        val_phoneme_tts_records = phoneme_tts_records[:num_val_records]

        train_sentencepiece_tts_records = sentencepiece_tts_records[num_val_records:]
        val_sentencepiece_tts_records = sentencepiece_tts_records[:num_val_records]

        train_phoneme_plus_sentencepiece_tts_records = phoneme_plus_sentencepiece_tts_records[num_val_records:]
        val_phoneme_plus_sentencepiece_tts_records = phoneme_plus_sentencepiece_tts_records[:num_val_records]
        # Shuffle train mixed records
        random.shuffle(train_phoneme_plus_sentencepiece_tts_records)

        train_all_tasks_records = all_tasks_records[num_val_records:]
        val_all_tasks_records = all_tasks_records[:num_val_records]

        manifest_base_name = _exp_name
        phoneme_tts_train_manifest_path = os.path.join(
            manifest_dir, "{}_train_phoneme_tts.json".format(manifest_base_name)
        )
        phoneme_tts_val_manifest_path = os.path.join(
            manifest_dir, "{}_val_phoneme_tts.json".format(manifest_base_name)
        )
        save_manifest(train_phoneme_tts_records, phoneme_tts_train_manifest_path)
        save_manifest(val_phoneme_tts_records, phoneme_tts_val_manifest_path)

        sentencepiece_tts_train_manifest_path = os.path.join(
            manifest_dir, "{}_train_sentencepiece_tts.json".format(manifest_base_name)
        )
        sentencepiece_tts_val_manifest_path = os.path.join(
            manifest_dir, "{}_val_sentencepiece_tts.json".format(manifest_base_name)
        )
        save_manifest(train_sentencepiece_tts_records, sentencepiece_tts_train_manifest_path)
        save_manifest(val_sentencepiece_tts_records, sentencepiece_tts_val_manifest_path)

        sp_plus_phoneme_tts_train_manifest_path = os.path.join(
            manifest_dir, "{}_train_phoneme_plus_sentencepiece_tts.json".format(manifest_base_name)
        )
        sp_plus_phoneme_tts_val_manifest_path = os.path.join(
            manifest_dir, "{}_val_phoneme_plus_sentencepiece_tts.json".format(manifest_base_name)
        )
        save_manifest(train_phoneme_plus_sentencepiece_tts_records, sp_plus_phoneme_tts_train_manifest_path)
        save_manifest(val_phoneme_plus_sentencepiece_tts_records, sp_plus_phoneme_tts_val_manifest_path)

        if not args.save_only_tts_records:
            all_tasks_train_manifest_path = os.path.join(
                manifest_dir, "{}_train_all_tasks.json".format(args.dataset_name)
            )
            all_tasks_val_manifest_path = os.path.join(manifest_dir, "{}_val_all_tasks.json".format(args.dataset_name))
            save_manifest(train_all_tasks_records, all_tasks_train_manifest_path)
            save_manifest(val_all_tasks_records, all_tasks_val_manifest_path)
    else:
        manifest_base_name = _exp_name
        phoneme_tts_manifest_path = os.path.join(manifest_dir, "{}_phoneme_tts.json".format(manifest_base_name))
        save_manifest(phoneme_tts_records, phoneme_tts_manifest_path)

        sentencepiece_tts_manifest_path = os.path.join(
            manifest_dir, "{}_sentencepiece_tts.json".format(manifest_base_name)
        )
        save_manifest(sentencepiece_tts_records, sentencepiece_tts_manifest_path)

        phoneme_plus_sentencepiece_tts_manifest_path = os.path.join(
            manifest_dir, "{}_phoneme_plus_sentencepiece_tts.json".format(manifest_base_name)
        )
        save_manifest(phoneme_plus_sentencepiece_tts_records, phoneme_plus_sentencepiece_tts_manifest_path)

        if not args.save_only_tts_records:
            all_manifest_path = os.path.join(manifest_dir, "{}_all_tasks.json".format(args.dataset_name))
            save_manifest(all_tasks_records, all_manifest_path)


if __name__ == '__main__':
    main()
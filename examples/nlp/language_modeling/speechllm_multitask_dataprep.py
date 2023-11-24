import json
import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.tts.parts.utils.tts_dataset_utils import get_base_dir
from nemo.core.classes import Dataset
from nemo.utils import logging
from nemo.collections.asr.parts.preprocessing.perturb import NoisePerturbation, WhiteNoisePerturbation
import copy
import argparse
import time
import random
import torchaudio
from encodec import EncodecModel
from transformers import EncodecModel as TransformerEncodecModel
from nemo.collections.tts.modules.transformer import mask_from_lens
import math

class AudioDataset(Dataset):
    def __init__(
        self,
        manifest_paths,
        min_duration=1.5,
        max_duration=20.0,
        sample_rate=24000,
        sup_data_dir=None,
        noise_manifest_path=None,
        min_snr_db=0,
        max_snr_db=5,
        max_same_speaker_audios=5,
    ):
        self.data = []
        speakerwise_records = {}
        for manifest_path in manifest_paths:
            with open(manifest_path, "r") as f:
                for line in f:
                    record = json.loads(line)
                    
                    if record['duration'] < min_duration or record['duration'] > max_duration:
                        continue

                    self.data.append(json.loads(line))
                    if record['speaker'] not in speakerwise_records:
                        speakerwise_records[record['speaker']] = []
                    speakerwise_records[record['speaker']].append(record)

        self.speakerwise_records = speakerwise_records
        self.speaker_list = list(self.speakerwise_records.keys())

        self.base_data_dir = get_base_dir([item["audio_filepath"] for item in self.data])
        if sup_data_dir is not None:
            self.sup_data_dir = sup_data_dir
        else:
            self.sup_data_dir = os.path.join(self.base_data_dir, "sup_data")
        if not os.path.exists(self.sup_data_dir):
            os.makedirs(self.sup_data_dir)

        self.sample_rate = sample_rate
        self.noise_perturber = NoisePerturbation(
            manifest_path=noise_manifest_path,
            min_snr_db=min_snr_db,
            max_snr_db=max_snr_db,
        )

        self.white_noise_perturber = WhiteNoisePerturbation()

        self.max_same_speaker_audios = max_same_speaker_audios

    def __len__(self):
        return len(self.data)

    def _get_wav_from_filepath(self, audio_filepath, perturb=False):
        features = AudioSegment.segment_from_file(
            audio_filepath, target_sr=self.sample_rate, n_segments=-1, trim=False,
        )
        audio_samples = features.samples
        audio, audio_length = torch.tensor(audio_samples), torch.tensor(audio_samples.shape[0]).long()
        
        perturbed_audio = None
        perturbed_audio_length = None
        if perturb:
            features_copy = copy.deepcopy(features)
            self.noise_perturber.perturb(features_copy)
            # self.white_noise_perturber.perturb(features_copy)
            perturbed_audio_samples = features_copy.samples
            perturbed_audio, perturbed_audio_length = torch.tensor(perturbed_audio_samples), torch.tensor(perturbed_audio_samples.shape[0]).long()
            # import ipdb; ipdb.set_trace()

        return audio, audio_length, perturbed_audio, perturbed_audio_length

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
        ]

        for key in final_batch:
            if key not in non_tensor_keys:
                final_batch[key] = torch.stack(final_batch[key])

        return final_batch

    def __getitem__(self, index):
        sample = self.data[index]
        rel_audio_path = Path(sample["audio_filepath"]).relative_to(self.base_data_dir).with_suffix("")
        rel_audio_path_as_text_id = str(rel_audio_path).replace("/", "_")
        speaker = torch.tensor(sample["speaker"]).long()

        alternate_speaker = random.choice(self.speaker_list)
        _ctr = 0
        while (alternate_speaker == speaker) and (_ctr < 10):
            alternate_speaker = random.choice(self.speaker_list)
            _ctr += 1
        
        alternate_wavpath = random.choice(self.speakerwise_records[alternate_speaker])["audio_filepath"]

        random.shuffle(self.speakerwise_records[sample["speaker"]])
        samespeaker_wavpaths = []
        
        for _record in self.speakerwise_records[sample["speaker"]][:self.max_same_speaker_audios]:
            if _record["audio_filepath"] != sample["audio_filepath"]:
                samespeaker_wavpath = _record["audio_filepath"]
                samespeaker_wavpaths.append(samespeaker_wavpath)
        
        if len(samespeaker_wavpaths) == 0:
            # Use the same audio if no other audio is available from the same speaker
            samespeaker_wavpaths = [sample["audio_filepath"]]

        samespeaker_audioids = []
        for samespeaker_wavpath in samespeaker_wavpaths:
            samespeaker_rel_audio_path = Path(samespeaker_wavpath).relative_to(self.base_data_dir).with_suffix("")
            samespeaker_rel_audio_path_as_text_id = str(samespeaker_rel_audio_path).replace("/", "_")
            samespeaker_audioids.append(samespeaker_rel_audio_path_as_text_id)

        alternate_audio, alternate_audio_length, _, _ = self._get_wav_from_filepath(alternate_wavpath, perturb=False)
        audio, audio_length, perturbed_audio, perturbed_audio_length = self._get_wav_from_filepath(sample["audio_filepath"], perturb=True)

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

        return {
            "audio": audio,
            "audio_len": audio_length,
            "perturbed_audio" : perturbed_audio,
            "perturbed_audio_len" : perturbed_audio_length,
            "mixed_audio": mixed_audio,
            "mixed_audio_len": mixed_audio_length,
            "rel_audio_path_as_text_id": rel_audio_path_as_text_id,
            "samespeaker_audioids" : samespeaker_audioids,
            "samespeaker_wavpaths": samespeaker_wavpaths,
            "audio_filepath": sample["audio_filepath"],
            "question": sample["question"],
            "text" : sample["text"],
            "context" : sample["context"],
            "old_speaker_id" : sample["old_speaker_id"],
            "duration" : sample["duration"],
            "context_duration" : sample["context_duration"],
            "speaker": speaker,
        }


def save_batch_audios(batch, bidx, temp_dir, encodec_model):
    for sidx in range(batch["audio"].shape[0]):
        sample_audio = batch["audio"][sidx]
        sample_audio_len = batch["audio_len"][sidx].item()
        sample_audio = sample_audio[:sample_audio_len]

        # Save sample_audio
        sample_audio_path = os.path.join(temp_dir, f"{bidx}_{sidx}_sample.wav")
        torchaudio.save(sample_audio_path, sample_audio[None].cpu(), 24000)

        # Save perturbed_audio
        perturbed_audio = batch["perturbed_audio"][sidx]
        perturbed_audio_len = batch["perturbed_audio_len"][sidx].item()
        perturbed_audio = perturbed_audio[:perturbed_audio_len]
        perturbed_audio_path = os.path.join(temp_dir, f"{bidx}_{sidx}_perturbed.wav")
        torchaudio.save(perturbed_audio_path, perturbed_audio[None].cpu(), 24000)

        # Save mixed_audio
        mixed_audio = batch["mixed_audio"][sidx]
        mixed_audio_len = batch["mixed_audio_len"][sidx].item()
        mixed_audio = mixed_audio[:mixed_audio_len]
        mixed_audio_path = os.path.join(temp_dir, f"{bidx}_{sidx}_mixed.wav")
        torchaudio.save(mixed_audio_path, mixed_audio[None].cpu(), 24000)

        for key in batch:
            if "CODEC" in key:
                codec = batch[key][sidx] # (8, T)
                codec_decoded_audio = encodec_model.decode([[codec.unsqueeze(0), None]])[0][0]
                codec_decoded_audio_path = os.path.join(temp_dir, f"{bidx}_{sidx}_{key}_decoded.wav")
                torchaudio.save(codec_decoded_audio_path, codec_decoded_audio[None].cpu(), 24000)

def estimate_duration_from_codeclen(codec_len):
    num_audio_samples = codec_len * 320
    duration = num_audio_samples / 24000.0
    return round(duration, 2)

def main():
    parser = argparse.ArgumentParser(description='Create multiple tasks')
    parser.add_argument("--noise_manifest", type=str, default="/datap/misc/noisedata/train_manifest.json")
    parser.add_argument('--manifest_paths', type=str, default="/datap/misc/manifests/manifests/libritts/val_clean_300_speechlm_ttstasks.json")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--audiocodec_out_dir', type=str, default='/datap/misc/multitask_audiocodec')
    parser.add_argument('--out_manifest_path', type=str, default='/datap/misc/speechllm_multitask_val.json')
    args = parser.parse_args()

    encodec_model = EncodecModel.encodec_model_24khz()
    encodec_model.set_target_bandwidth(6.0)
    encodec_model.cuda()
    encodec_model.eval()

    # transformer_encodec_model = TransformerEncodecModel.from_pretrained("facebook/encodec_24khz")

    dataset = AudioDataset(
        manifest_paths=[args.manifest_paths],
        sample_rate=24000,
        noise_manifest_path=args.noise_manifest,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.pad_collate_fn,
        shuffle=False,
        num_workers=8,
    )
    
    temp_dir = "/datap/misc/temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    encodec_downsampling_factor = 320.0

    if not os.path.exists(args.audiocodec_out_dir):
        os.makedirs(args.audiocodec_out_dir)

    manifest_records = []

    speaker_context_len_range = [240, 400] # 3 to 5 seconds

    for bidx, batch in enumerate(tqdm(dataloader)):
        # print("bidx", bidx+1, "of", len(dataloader))
        
        audio_len_mask = mask_from_lens(batch["audio_len"])
        # import ipdb; ipdb.set_trace()
        cuda_keys = ['audio', 'perturbed_audio', 'mixed_audio']
        for key in cuda_keys:
            batch[key] = batch[key].cuda()
        with torch.no_grad():
            original_encodec_codes = encodec_model.encode(batch["audio"].unsqueeze(1))[0][0]
            perturbed_encodec_codes = encodec_model.encode(batch["perturbed_audio"].unsqueeze(1))[0][0]
            mixed_encodec_codes = encodec_model.encode(batch["mixed_audio"].unsqueeze(1))[0][0]
            
        # encodec_codes = transformer_encodec_model.encode(batch["audio"].unsqueeze(1), audio_len_mask, bandwidth=6.0)
        target_codecs = []
        mixed_codecs = []
        perturbed_codecs = []
        for sidx in range(batch['audio'].shape[0]):
            
            codec_len = math.ceil(batch['audio_len'][sidx].item() / encodec_downsampling_factor)
            sample_encodec_codes = original_encodec_codes[sidx][:,:codec_len]
            target_codecs.append(sample_encodec_codes)

            perturbed_codec_len = math.ceil(batch['perturbed_audio_len'][sidx].item() / encodec_downsampling_factor)
            perturbed_sample_encodec_codes = perturbed_encodec_codes[sidx][:,:perturbed_codec_len]
            perturbed_codecs.append(perturbed_sample_encodec_codes)

            mixed_codec_len = math.ceil(batch['mixed_audio_len'][sidx].item() / encodec_downsampling_factor)
            mixed_sample_encodec_codes = mixed_encodec_codes[sidx][:,:mixed_codec_len]
            mixed_codecs.append(mixed_sample_encodec_codes)

            
            example_name = batch['rel_audio_path_as_text_id'][sidx]

            target_codec_filepath = os.path.join(args.audiocodec_out_dir, "target_codes_{}.pt".format(example_name))
            torch.save(sample_encodec_codes.cpu().type(torch.int16), target_codec_filepath)

            perturbed_codec_filepath = os.path.join(args.audiocodec_out_dir, "perturbed_codes_{}.pt".format(example_name))
            torch.save(perturbed_sample_encodec_codes.cpu().type(torch.int16), perturbed_codec_filepath)

            mixed_codec_filepath = os.path.join(args.audiocodec_out_dir, "mixed_codes_{}.pt".format(example_name))
            torch.save(mixed_sample_encodec_codes.cpu().type(torch.int16), mixed_codec_filepath)

            tts_contextpath = ""
            for samespeaker_audioid in batch['samespeaker_audioids'][sidx]:
                tts_contextpath += os.path.join(args.audiocodec_out_dir, "target_codes_{}.pt".format(samespeaker_audioid))
                tts_contextpath += ";"
            tts_contextpath = tts_contextpath[:-1]


            tts_record = {
                "audio_filepath" : batch['audio_filepath'][sidx],
                "text" : batch['text'][sidx],
                "question" : batch['question'][sidx],
                "answer" : target_codec_filepath,
                "context" : tts_contextpath,
                "question_type" : "TEXT",
                "answer_type" : "AUDIOCODEC",
                "context_type" : "REFSPEAKERCODEC",
                "context_duration" : batch['context_duration'][sidx],
                "answer_duration" : batch['duration'][sidx],
                "taskname" : "squad",
            }

            speechenhancement_record = {
                "audio_filepath" : batch['audio_filepath'][sidx],
                "text" : batch['text'][sidx],
                "question" : "Remove Noise",
                "answer" : target_codec_filepath,
                "context" : perturbed_codec_filepath,
                "question_type" : "TEXT",
                "answer_type" : "AUDIOCODEC",
                "context_type" : "AUDIOCODEC",
                "context_duration" : estimate_duration_from_codeclen(perturbed_codec_len),
                "answer_duration" : batch['duration'][sidx],
                "taskname" : "squad",
            }

            speechseparation_record = {
                "audio_filepath" : batch['audio_filepath'][sidx],
                "text" : batch['text'][sidx],
                "question" : "Extract Speaker Audio",
                "answer" : target_codec_filepath,
                "context" : "{},{}".format(mixed_codec_filepath, tts_contextpath),
                "question_type" : "TEXT",
                "answer_type" : "AUDIOCODEC",
                "context_type" : "SEPARATIONCODECS",
                "context_duration" : estimate_duration_from_codeclen(mixed_codec_len),
                "answer_duration" : batch['duration'][sidx],
                "taskname" : "squad",
            }

            speechediting_record = {
                "audio_filepath" : batch['audio_filepath'][sidx],
                "text" : batch['text'][sidx],
                "question" : batch['question'][sidx].replace("Text to speech this", "Edit Speech"),
                "answer" : target_codec_filepath,
                "context" : target_codec_filepath,
                "question_type" : "TEXT",
                "answer_type" : "AUDIOCODEC",
                "context_type" : "EDITINGCODECS",
                "context_duration" : batch['duration'][sidx] + 3, # 3 sec for speaker context
                "answer_duration" : batch['duration'][sidx],
                "taskname" : "squad",
            }

            manifest_records.append(tts_record)
            manifest_records.append(speechenhancement_record)
            manifest_records.append(speechseparation_record)
            manifest_records.append(speechediting_record)
        
        batch['target_CODEC'] = target_codecs
        batch['perturbed_CODEC'] = perturbed_codecs
        batch['mixed_CODEC'] = mixed_codecs

        if bidx == 0:
            save_batch_audios(batch, bidx, temp_dir, encodec_model)

    random.shuffle(manifest_records)
    
    with open(args.out_manifest_path, "w") as f:
        file_str = ""
        for record in manifest_records:
            file_str += json.dumps(record) + "\n"
        file_str = file_str.strip()
        f.write(file_str)
    
    print("Saved manifest to {}".format(args.out_manifest_path))

if __name__ == '__main__':
    main()
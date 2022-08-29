
from email.mime import audio
import json
from locale import normalize
from multiprocessing import Pool
import os
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union

import hydra.utils
import librosa
import numpy as np
import torch
from omegaconf import open_dict
from tqdm import tqdm

from nemo.collections.asr.models import ssl_models
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.tts.models import ssl_tts
from nemo.collections.tts.torch.helpers import get_base_dir
from nemo.collections.tts.torch.tts_tokenizers import EnglishCharsTokenizer
from nemo.core.classes import Dataset
from nemo.utils import logging
import argparse
import time

SUP_DATA_DIR = None

wav_featurizer = WaveformFeaturizer(sample_rate=22050, int_values=False, augmentor=None)

class AudioDataset(Dataset):
    def __init__(self, manifest_path, min_duration=0.5, max_duration=16.0):
        global SUP_DATA_DIR
        self.manifest_path = manifest_path
        self.data = []
        with open(manifest_path, "r") as f:
            for line in f:
                record = json.loads(line)
                if record['duration'] < min_duration or record['duration'] > max_duration:
                    continue
                self.data.append(json.loads(line))

        self.base_data_dir = get_base_dir([item["audio_filepath"] for item in self.data])
        self.sup_data_dir = os.path.join(self.base_data_dir, "sup_data")
        if not os.path.exists(self.sup_data_dir):
            os.makedirs(self.sup_data_dir)
        SUP_DATA_DIR = self.sup_data_dir
        self.pad_multiple = 1024
        self.sample_rate = 22050
    
    def __len__(self):
        return len(self.data)

    def _get_wav_from_filepath(self, audio_filepath):
        features = AudioSegment.segment_from_file(
            audio_filepath, target_sr=self.sample_rate, n_segments=-1, trim=False,
        )
        audio_samples = features.samples
        audio, audio_length = torch.tensor(audio_samples), torch.tensor(audio_samples.shape[0]).long()

        # pad audio to a multiple of self.pad_multiple
        if audio.shape[0] % self.pad_multiple != 0:
            audio = torch.cat(
                [audio, torch.zeros(self.pad_multiple - audio.shape[0] % self.pad_multiple, dtype=torch.float)]
            )
            audio_length = torch.tensor(audio.shape[0]).long()

        # print("audio", audio.shape)
        # print("audio_length", audio_length)
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
        for key in final_batch:
            if key not in ["rel_audio_path_as_text_id", "wav_path"]:
                final_batch[key] = torch.stack(final_batch[key])

        return final_batch

    def __getitem__(self, index):
        sample = self.data[index]
        rel_audio_path = Path(sample["audio_filepath"]).relative_to(self.base_data_dir).with_suffix("")
        rel_audio_path_as_text_id = str(rel_audio_path).replace("/", "_")
        speaker = torch.tensor(sample["speaker"]).long()

        audio, audio_length = self._get_wav_from_filepath(sample["audio_filepath"])

        return {
            "audio": audio,
            "audio_len": audio_length,
            "rel_audio_path_as_text_id": rel_audio_path_as_text_id,
            "wav_path": sample["audio_filepath"],
        }

def segment_wav(wav, segment_length=44100, hop_size=22050, min_segment_size=22050):
    if len(wav) < segment_length:
        pad = torch.zeros(segment_length - len(wav))
        segment = torch.cat([wav, pad])
        return [segment]
    else:
        si = 0
        segments = []
        while si < len(wav) - min_segment_size:
            segment = wav[si:si+segment_length]
            if len(segment) < segment_length:
                pad = torch.zeros(segment_length - len(segment))
                segment = torch.cat([segment, pad])
            segments.append(segment)
            si += hop_size
        return segments

def segment_batch(batch):
    all_segments = []
    segment_indices = []
    si = 0
    for bidx in range(len(batch['audio'])):
        audio = batch['audio'][bidx]
        audio_length = batch['audio_len'][bidx]
        audio_actual = audio[:audio_length]
        audio_segments = segment_wav(audio_actual)
        all_segments += audio_segments
        segment_indices.append( (si, si + len(audio_segments)-1) )
        si += len(audio_segments)
    
    return torch.stack(all_segments), segment_indices

def get_mel_spectrogram(fb, wav):
    EPSILON = 1e-9
    window_fn = torch.hann_window

    spec = torch.stft(
        input=wav,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        window=window_fn(1024, periodic=False).to(torch.float).to('cuda') if window_fn else None,
        return_complex=True,
        center=True,
    )
    # print("spec", s`pec.shape)
    if spec.dtype in [torch.cfloat, torch.cdouble]:
        spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + EPSILON)

    mel = torch.matmul(fb.to(spec.dtype), spec)
    log_mel = torch.log(torch.clamp(mel, min=torch.finfo(mel.dtype).tiny))

    return log_mel

def load_wav(wav_path, pad_multiple=1024):
    wav = AudioSegment.segment_from_file(
        wav_path, target_sr=22050, n_segments=-1, trim=False,
    ).samples
    
    if wav.shape[0] % pad_multiple != 0:
        wav = np.concatenate(
                [wav, np.zeros(pad_multiple - wav.shape[0] % pad_multiple)]
            )
    wav = wav[:-1]
    
    return wav

def save_pitch_contour(wav_and_id):
    sup_data_dir = SUP_DATA_DIR
    wav_path, wav_text_id = wav_and_id
    wav = load_wav(wav_path)
    pitch_contour_fn = f"pitch_contour_{wav_text_id}.pt"
    pitch_contour_fp = os.path.join(sup_data_dir, pitch_contour_fn)
    
    f0, _, _ = librosa.pyin(
        wav,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        frame_length=1024,
        hop_length=256,
        sr=22050,
        center=True,
        fill_na=0.0,
    )
    
    pitch_contour = torch.tensor(f0, dtype=torch.float32)
    torch.save(pitch_contour, pitch_contour_fp)
    print("saved", pitch_contour_fp)
    
    return pitch_contour

def main():
    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument('--ssl_model_ckpt_path', type=str, default="/home/pneekhara/NeMo2022/SSLCheckPoints/SSLConformer22050_Epoch37.ckpt")
    parser.add_argument('--manifest_path', type=str, default="/home/pneekhara/NeMo2022/libri_val_formatted.json")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--ssl_content_emb_type', type=str, default="embedding_and_probs")
    parser.add_argument('--use_unique_tokens', type=int, default=0)
    parser.add_argument('--pool_workers', type=int, default=60)

    args = parser.parse_args()

    device = "cuda"
    manifest_path = args.manifest_path
    ssl_model_ckpt_path = args.ssl_model_ckpt_path

    dataset = AudioDataset(manifest_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.pad_collate_fn, num_workers=8)
    
    ssl_model = ssl_tts.SSLDisentangler.load_from_checkpoint(ssl_model_ckpt_path, strict=False)
    with open_dict(ssl_model.cfg):
        ssl_model.cfg.preprocessor.exact_pad = True
    ssl_model.preprocessor = hydra.utils.instantiate(ssl_model.cfg.preprocessor)
    ssl_model.preprocessor_disentangler = ssl_model.preprocessor
    ssl_model.eval()
    ssl_model.to(device)

    fb = torch.tensor(
        librosa.filters.mel(sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=8000),
        dtype=torch.float,
    ).unsqueeze(0).to(device)

    st = time.time()
    bidx = 0
    wav_and_id_list = []
    for batch in tqdm(dataloader):
        bidx += 1
        with torch.no_grad():
            (
                _,
                _,
                batch_content_embedding,
                batch_content_log_probs,
                batch_encoded_len,
            ) = ssl_model.forward_for_export(
                input_signal=batch['audio'].to(device),
                input_signal_length=batch['audio_len'].to(device),
                normalize_content=True,
            )

            batch_mel_specs = get_mel_spectrogram(fb, batch['audio'][:,:-1].to(device))
            audio_segmented, segment_indices = segment_batch(batch)
            audio_seg_len = torch.tensor([len(segment) for segment in audio_segmented]).to(device).long()

            _, batch_speaker_embeddings, _, _, _ = ssl_model.forward_for_export(
                input_signal=audio_segmented.to(device),
                input_signal_length=audio_seg_len,
                normalize_content=True,
            ) 

            
            for idx in range(batch['audio'].shape[0]):
                wav_path = batch['wav_path'][idx]
                wav_id = batch['rel_audio_path_as_text_id'][idx]
                wav_and_id_list.append( (wav_path, wav_id) )
                content_embedding = batch_content_embedding[idx].detach()
                content_log_probs = batch_content_log_probs[:, idx, :].detach()  # (content lob prob is (t, b, c))
                encoded_len = batch_encoded_len[idx].detach()
                content_embedding = content_embedding[: encoded_len.item()]
                content_embedding = content_embedding.t()
                content_log_probs = content_log_probs[: encoded_len.item()]
                content_log_probs = content_log_probs.t()
                content_probs = torch.exp(content_log_probs)

                duration = torch.ones(content_embedding.shape[1]) * 4.0

                bsi_start = segment_indices[idx][0]
                bsi_end = segment_indices[idx][1]
                speaker_embedding = torch.mean(batch_speaker_embeddings[bsi_start:bsi_end+1], dim=0)
                
                l2_norm = torch.norm(speaker_embedding, p=2)
                speaker_embedding = speaker_embedding / l2_norm

                if args.ssl_content_emb_type == "probs":
                    final_content_embedding = content_probs
                elif args.ssl_content_emb_type == "embedding":
                    final_content_embedding = content_embedding
                elif args.ssl_content_emb_type == "log_probs":
                    final_content_embedding = content_log_probs
                elif args.ssl_content_emb_type == "embedding_and_probs":
                    final_content_embedding = torch.cat([content_embedding, content_probs], dim=0)

                if args.use_unique_tokens == 1:
                    token_predictions = torch.argmax(content_probs, dim=0)
                    # print("token predictions:", token_predictions)
                    content_buffer = [final_content_embedding[:, 0]]
                    unique_content_embeddings = []
                    unique_tokens = []
                    durations = []
                    for _t in range(1, final_content_embedding.shape[1]):
                        if token_predictions[_t] == token_predictions[_t - 1]:
                            content_buffer.append(final_content_embedding[:, _t])
                        else:
                            durations.append(len(content_buffer) * 4)
                            unique_content_embeddings.append(torch.mean(torch.stack(content_buffer), dim=0))
                            content_buffer = [final_content_embedding[:, _t]]
                            unique_tokens.append(token_predictions[_t].item())

                    if len(content_buffer) > 0:
                        durations.append(len(content_buffer) * 4)
                        unique_content_embeddings.append(torch.mean(torch.stack(content_buffer), dim=0))
                        unique_tokens.append(token_predictions[_t].item())

                    unique_content_embedding = torch.stack(unique_content_embeddings)
                    final_content_embedding = unique_content_embedding.t()
                    duration = torch.tensor(durations).float()
                    # print("duration ds", duration)
                    encoded_len = torch.tensor(final_content_embedding.shape[1]).long()

                mel_len = int(batch['audio_len'][idx].item()/256.0)
                item_mel = batch_mel_specs[idx][:,:mel_len]
                # print("item mel shape: ", item_mel.shape)

                wav_text_id = batch["rel_audio_path_as_text_id"][idx]
                content_emb_fn = f"{args.ssl_content_emb_type}_content_embedding_{wav_text_id}.pt"
                speaker_emb_fn = f"speaker_embedding_{wav_text_id}.pt"
                duration_fn = f"duration_embedding_{wav_text_id}.pt"  # embedding just for namesake
                content_emb_fp = os.path.join(dataset.sup_data_dir, content_emb_fn)
                speaker_emb_fp = os.path.join(dataset.sup_data_dir, speaker_emb_fn)
                duration_fp = os.path.join(dataset.sup_data_dir, duration_fn)

                mel_spec_fn = f"mel_spec_{wav_text_id}.pt"
                mel_spec_fp = os.path.join(dataset.sup_data_dir, mel_spec_fn)

                torch.save(item_mel.cpu(), mel_spec_fp)
                torch.save(final_content_embedding.cpu(), content_emb_fp)
                torch.save(speaker_embedding.cpu(), speaker_emb_fp)
                torch.save(duration.cpu(), duration_fp)
            
            et = time.time()
            print("Time per batch", bidx, len(dataloader), (et - st)/bidx)


    with Pool(args.pool_workers) as p:
        p.map(save_pitch_contour, wav_and_id_list)            
            
            

            


if __name__ == '__main__':
    main()
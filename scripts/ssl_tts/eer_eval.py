import json
import os
import time

import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, roc_curve

from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.tts.models import ssl_tts

device = "cuda:1"


def segment_wav(wav, segment_length=44100, hop_size=22050, min_segment_size=22050):
    if len(wav) < segment_length:
        pad = torch.zeros(segment_length - len(wav))
        segment = torch.cat([wav, pad])
        return [segment]
    else:
        si = 0
        segments = []
        while si < len(wav) - min_segment_size:
            segment = wav[si : si + segment_length]
            if len(segment) < segment_length:
                pad = torch.zeros(segment_length - len(segment))
                segment = torch.cat([segment, pad])

            segments.append(segment)
            si += hop_size
        return segments


def sv_emb_from_audio_segmented(audio_path, ssl_model):
    wav = wav_featurizer.process(audio_path)

    wav_segments = segment_wav(wav)
    signal_batch = torch.stack(wav_segments).to(device)
    signal_length_batch = torch.stack([torch.tensor(signal_batch.shape[1]) for _i in range(len(wav_segments))]).to(
        device
    )
    _, speaker_embeddings, _, _, _ = ssl_model.forward_for_export(
        input_signal=signal_batch, input_signal_length=signal_length_batch, normalize_content=True
    )

    speaker_embedding = torch.mean(speaker_embeddings, dim=0)
    l2_norm = torch.norm(speaker_embedding, p=2)
    speaker_embedding = speaker_embedding / l2_norm

    return speaker_embedding[None]


def sv_emb_from_audio(audio_path, ssl_model):
    wav = wav_featurizer.process(audio_path)
    audio_signal = wav[None].to(device)
    audio_signal_length = torch.tensor([wav.shape[0]]).to(device)
    _, speaker_embedding, _, _, _ = ssl_model.forward_for_export(
        input_signal=audio_signal, input_signal_length=audio_signal_length, normalize_content=True
    )
    return speaker_embedding


def get_similarity(audio_path_1, audio_path_2, ssl_model):
    sv_emb1 = sv_emb_from_audio_segmented(audio_path_1, ssl_model)
    sv_emb2 = sv_emb_from_audio_segmented(audio_path_2, ssl_model)
    similarity = F.cosine_similarity(sv_emb1, sv_emb2)
    similarity = similarity.item()
    return similarity


def get_checkpoint(folder_path):

    ckpt_path = None
    for filename in os.listdir(folder_path):
        if filename.endswith('last.ckpt'):
            ckpt_path = os.path.join(folder_path, filename)

    return ckpt_path


ssl_model_ckpt_path = "/home/shehzeenh/Conformer-SSL/ConformerEpoch1.ckpt"
ssl_model = ssl_tts.SSLDisentangler.load_from_checkpoint(ssl_model_ckpt_path, strict=False)
ssl_model = ssl_model.to(device)
ssl_model.eval()

wav_featurizer = WaveformFeaturizer(sample_rate=22050, int_values=False, augmentor=None)

y_score = []
y_true = []

with open('/home/shehzeenh/datasets/speaker_verification_full/vox_o_trial.txt') as f:
    lines = f.readlines()  # list containing lines of file
    ssl_model.eval()

    st = time.time()
    for lidx, line in enumerate(lines):
        line = line.strip()  # remove leading/trailing white spaces
        #         print(line)
        label, wav1, wav2 = line.split(' ')

        wav1_path = '/home/shehzeenh/datasets/speaker_verification_full/test_wav/' + str(wav1)
        wav2_path = '/home/shehzeenh/datasets/speaker_verification_full/test_wav/' + str(wav2)

        with torch.no_grad():
            sim_score = get_similarity(wav1_path, wav2_path, ssl_model)

        y_score.append(sim_score)
        y_true.append(int(label))

        if lidx > 1 and lidx % 50 == 0:
            et = time.time()
            time_elapsed = et - st
            time_per_iter = time_elapsed / (lidx + 1)
            time_remaining = (len(lines) - lidx) * time_per_iter
            time_remaining_mins = time_remaining / 60.0
            print("processed {} of {}. Time remaining {} mins.".format(lidx, len(lines), time_remaining_mins))


fpr, tpr, thresholds = roc_curve(y_true, y_score)
_auc = auc(fpr, tpr)
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
eer_verify = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

assert abs(eer - eer_verify) < 1.0
print("eer", eer)
print("auc", _auc)

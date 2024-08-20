from pathlib import Path
import json
import argparse
import os
import shutil
import csv
import soundfile as sf
from nemo.collections.tts.models import AudioCodecModel
import librosa
import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


def read_result_dir(result_path, file_prefix, codec_model):
    with open(os.path.join(result_path, file_prefix+'inputs.jsonl'), 'r') as f:
        n_samples = len(f.readlines())

    for i in tqdm(range(n_samples)):
        answer = np.load(os.path.join(result_path, 'npy', 'speech_answer', file_prefix+f"speech_answer_{i}.npy"))
        pred = np.load(os.path.join(result_path, 'npy', 'speech_pred', file_prefix+f"speech_pred_{i}.npy"))
        # speaker_context = np.load(os.path.join(result_path, 'npy', 'speaker_contexts', file_prefix+f"speaker_context_{i}.npy"))

        decode_savewav(answer, os.path.join(result_path, 'wav', 'answer', f"answer_{i}.wav"), codec_model)
        decode_savewav(pred, os.path.join(result_path, 'wav', 'pred', f"pred_{i}.wav"), codec_model)
        # decode_savewav(speaker_context, os.path.join(result_path, 'wav', 'speaker_contexts', f"speaker_context_{i}.wav"), codec_model)

        self_attn = np.load(os.path.join(result_path, 'npy', 'self_attn', file_prefix+f"self_attn_{i}.npy"))
        cross_attn = np.load(os.path.join(result_path, 'npy', 'cross_attn', file_prefix+f"cross_attn_{i}.npy"))
        for j in range(self_attn.shape[0]):
            self_attn[j][j] = 0

        plot(self_attn, os.path.join(result_path, 'png', 'self_attn', f"self_attn_{i}.png"))
        plot(cross_attn, os.path.join(result_path, 'png', 'cross_attn', f"cross_attn_{i}.png"))

def decode_savewav(codes, name, codec_model):
    sample_rate = 22050
    os.makedirs(os.path.dirname(name), exist_ok=True)

    codes = torch.tensor(codes).to(codec_model.device).T
    codec_len = torch.Tensor([codes.shape[1]]).long().to(codec_model.device)
    wav, _ = codec_model.decode(tokens=codes.unsqueeze(0), tokens_len=codec_len)
    wav = wav[0]

    sf.write(name, wav.detach().cpu().numpy(), sample_rate)


def plot(attn_weights, name):
    os.makedirs(os.path.dirname(name), exist_ok=True)
    plt.imshow(attn_weights)
    plt.savefig(name)
    plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--result_path',
        type=str,
        default="/workspace/Results/s2st/",
    )
    parser.add_argument(
        '--file_prefix',
        type=str,
        default="test_validation_s2st_",
    )
    parser.add_argument(
        '--codec_model_ckpt',
        type=str,
        default="/workspace/model/SpeechCodec_2402.nemo",
    )
    args = parser.parse_args()

    codec_model = AudioCodecModel.restore_from(args.codec_model_ckpt)
    codec_model.to('cuda')
    codec_model.eval()
    read_result_dir(args.result_path, args.file_prefix, codec_model)

"""generate some audio from a TTS model

example usage:
python synthesize_examples.py \
    --last_ckpt "nemo_results/FastPitch/nichole_20230316/checkpoints/FastPitch--val_loss=1.2200-epoch=174.ckpt"
    --dest 'nemo_results/FastPitch/nichole_20230316/generated_recordings'
    --data train:data/nichole/data.json val:data/nichole/validation_manifest_0316.json


"""

from nemo.collections.tts.models import HifiGanModel
from argparse import ArgumentParser
from nemo.collections.tts.models import FastPitchModel
import soundfile as sf
import os
import json
import torch
import time
from tqdm import tqdm


def infer(spec_gen_model, vocoder_model, str_input, speaker=None):
    """
    Synthesizes spectrogram and audio from a text string given a spectrogram synthesis and vocoder model.

    Args:
    spec_gen_model: Spectrogram generator model (FastPitch in our case)
    vocoder_model: Vocoder model (HiFiGAN in our case)
    str_input: Text input for the synthesis
    speaker: Speaker ID

    Returns:
    spectrogram and waveform of the synthesized audio.
    """
    with torch.no_grad():
        parsed = spec_gen_model.parse(str_input)
        if speaker is not None:
            speaker = torch.tensor([speaker]).long().to(device=spec_gen_model.device)
        spectrogram = spec_gen_model.generate_spectrogram(tokens=parsed, speaker=speaker)
        audio = vocoder_model.convert_spectrogram_to_audio(spec=spectrogram)

    if spectrogram is not None:
        if isinstance(spectrogram, torch.Tensor):
            spectrogram = spectrogram.to('cpu').numpy()
        if len(spectrogram.shape) == 3:
            spectrogram = spectrogram[0]
    if isinstance(audio, torch.Tensor):
        audio = audio.to('cpu').numpy()
    return spectrogram, audio


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--tts_ckpt", help="path to tts model", type=str)
    parser.add_argument("--dest", help="where to save generated files", type=str)
    parser.add_argument(
        "--data", nargs='+',
        help="list of manifests to generate from in format <name>:<path to manifest>"
    )
    args = parser.parse_args()
    vocoder = HifiGanModel.from_pretrained("tts_hifigan")
    vocoder = vocoder.eval()

    last_ckpt = args.tts_ckpt
    dest = args.dest

    spec_model = FastPitchModel.load_from_checkpoint(last_ckpt)
    spec_model.eval()

    folder_names, manifest_names = [], []
    for dataset_str in args.data:
        folder_name, manifest_name = dataset_str.split(':')
        folder_names += [folder_name]
        manifest_names += [manifest_name]

    for folder_name, manifest_name in zip(folder_names, manifest_names):
        folder_dest = f'{dest}/{folder_name}'
        os.makedirs(folder_dest, exist_ok=True)
        with open(manifest_name, "r") as f:
            records = [json.loads(line) for line in f.readlines()]

        for record in tqdm(records):
            filepath = record['audio_filepath']
            filename = record['text'][:200] + '.wav'
            file_path = f'{folder_dest}/{filename}'
            start = time.time()
            spec, audio = infer(spec_model, vocoder, record['text'])
            sf.write(file_path, audio[0], 22050)

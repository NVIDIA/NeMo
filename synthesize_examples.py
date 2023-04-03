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
import whisper
from collections import defaultdict
from torchmetrics import WordErrorRate


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


def calculate_wer(eval_groups, asr_models):
    wer = WordErrorRate()
    end_result = defaultdict(dict)

    for eval_group_name, eval_examples in eval_groups.items():
        for model_name, asr_model in asr_models.items():
            transcriptions = [
                asr_model.transcribe(file_path)['text']
                for file_path, text in
                tqdm(
                    eval_examples,
                    desc=f'transcribing {eval_group_name} using {model_name}'
                )
            ]
            error_rate = wer(
                preds=transcriptions,
                target=[text for file_path, text in eval_examples]
            )

            error_rate = float(error_rate.cpu().numpy())

            end_result[eval_group_name][model_name] = {
                'error_rate': error_rate, 'transcriptions': transcriptions}

    return end_result


def synthesize_examples(
    tts_ckpt,
    dest,
    data,
    run_wer_calculations=False,
):
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    vocoder = HifiGanModel.from_pretrained("tts_hifigan")
    vocoder = vocoder.eval()

    spec_model = FastPitchModel.load_from_checkpoint(tts_ckpt)
    spec_model.eval()

    folder_names, manifest_names = [], []
    for dataset_str in data:
        folder_name, manifest_name = dataset_str.split(':')
        folder_names += [folder_name]
        manifest_names += [manifest_name]

    eval_groups = {}
    for folder_name, manifest_name in zip(folder_names, manifest_names):

        folder_dest = f'{dest}/{folder_name}'
        os.makedirs(folder_dest, exist_ok=True)
        with open(manifest_name, "r") as f:
            records = [json.loads(line) for line in f.readlines()]

        eval_groups[f'{folder_name}/ground_truth'] = [
            (record['audio_filepath'], record['text']) for record in records]

        generated_eval_group = []
        for i, record in enumerate(tqdm(records)):
            filename = record['text'][:200] + '.wav'
            # filename = f'{i}'.wav
            file_path = f'{folder_dest}/{filename}'
            spec, audio = infer(spec_model, vocoder, record['text'])
            generated_eval_group += [(file_path, record['text'])]
            sf.write(file_path, audio[0], 22050)

        eval_groups[f'{folder_name}/generated'] = generated_eval_group

    if run_wer_calculations:
        asr_models = {
            'tiny': whisper.load_model("tiny.en", device=device),
            # 'base': whisper.load_model("base.en", device=device),
            # 'medium': whisper.load_model("medium.en", device=device),
        }
        wer_results = calculate_wer(eval_groups, asr_models)

        with open(f'{dest}/wer_results.json', 'w') as f:
            json.dump(wer_results, f, indent=4)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--tts_ckpt", help="path to tts model", type=str)
    parser.add_argument(
        "--dest", help="where to save generated files", type=str)
    parser.add_argument(
        "--data", nargs='+',
        help="list of manifests to generate from in format <name>:<path to manifest>"
    )
    parser.add_argument(
        '--calculate_wer', action='store_true',
        help='if set calculate wer on recordings'
    )
    args = parser.parse_args()
    synthesize_examples(
        args.tts_ckpt,
        args.dest,
        args.data,
        args.calculate_wer,
    )

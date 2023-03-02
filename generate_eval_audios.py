"""
example run:
python generate_eval_audios.py         --inpainter_checkpoint nemo_results/Inpainter/inpainter_20230301/checkpoints/Inpainter--validation_mean_loss=9.1062-epoch=1-last.ckpt         --vocoder_checkpoint  nemo_results/HifiGan/second_try_more_training/checkpoints/HifiGan--val_loss=0.2877-epoch=39-last.ckpt         --dest  recordings_no_disc

"""
import soundfile as sf
import hashlib
import argparse
from nemo.collections.tts.models import InpainterModel
from nemo.collections.tts.models import HifiGanModel
import torch
import librosa
import os
import json
import random
from tqdm import tqdm

os.environ['DATA_CAP'] = '0'


def get_all_replacements(text):
    # assumes text is in correct format
    clean_str = ''
    replacements = []
    for ch in text:
        if ch == '[':
            new_replacement = clean_str + ch
            replacements = [new_replacement] + replacements
        elif ch == ']':
            # add it to the most recent replacement
            replacements[0] += ch
        else:
            clean_str += ch
            for i in range(len(replacements)):
                replacements[i] += ch

    return replacements


expected_sample_rate = 22050


def apply_repalcements(file_path, replacement_phrase, dest, inpainter, vocoder):
    data, sample_rate = sf.read(file_path)
    if sample_rate != expected_sample_rate:
        data = librosa.core.resample(
            data, orig_sr=sample_rate, target_sr=expected_sample_rate)

    original_spectrogram = inpainter.make_spectrogram(data)
    replacements = get_all_replacements(replacement_phrase)

    spectrogram = original_spectrogram
    for replacement in replacements:
        (
            full_replacement,
            partial_replacement,
            mcd_full,
            mcd_partial
        ) = inpainter.regenerate_audio(
            spectrogram,
            replacement,
        )
        spectrogram = partial_replacement

    with torch.inference_mode():
        audio = vocoder.convert_spectrogram_to_audio(
            spec=spectrogram.T.unsqueeze(0))

    os.makedirs(os.path.dirname(dest), exist_ok=True)

    inpainter_data = audio.to('cpu').numpy()[0]
    inpainter_downsampled = librosa.resample(
        inpainter_data, orig_sr=22050, target_sr=8000)

    orignal_downsampled = librosa.resample(
        data, orig_sr=22050, target_sr=8000)

    sf.write(
        f'{dest}.inpainter.wav', inpainter_downsampled, 8000, format='wav')
    sf.write(
        f'{dest}.original.wav', orignal_downsampled, 8000, format='wav')


def deterministic_hash(s):
    """deterministic hash for a string"""
    return int(hashlib.sha224(bytes(s, encoding='utf-8')).hexdigest(), 16)


def deterministic_blank(phrase):
    seed = deterministic_hash(phrase)
    random.seed(seed)

    words = phrase.split()

    word_to_blank = random.randint(0, len(words) - 1)
    blank_two_words = bool(random.randint(0, 1))

    second_blank_direction, = random.sample([-1, 1], 1)
    if word_to_blank == 0:
        second_blank_direction = 1
    if word_to_blank == len(words) - 1:
        second_blank_direction = -1

    if not blank_two_words:
        second_blank_direction == 0

    start_index, end_index = sorted([
        word_to_blank, word_to_blank + second_blank_direction])

    end_index += 1

    left = ' '.join(words[:start_index])
    middle = ' '.join(words[start_index:end_index])
    end = ' '.join(words[end_index:])

    return f'{left} [{middle}] {end}'.strip()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate audio for evaluation')
    parser.add_argument(
        "--inpainter_checkpoint",
        type=str,
        help="path to inpainter model",
    )
    parser.add_argument(
        "--vocoder_checkpoint",
        type=str,
        help="path to vocoder model",
    )
    parser.add_argument(
        "--dest",
        type=str,
        help="directory to output files",
    )
    args = parser.parse_args()
    manifests = [
        'data/NickyData/test_manifest.json',
        'data/NickyData/validation_manifest.json'
    ]
    data_dicts = []
    for manifest in manifests:
        with open(manifest) as f:
            data_dicts += [json.loads(line) for line in f.readlines()]

    replacement_phrases = [
        deterministic_blank(dd['text']) for dd in data_dicts]

    inpainter = InpainterModel.load_from_checkpoint(args.inpainter_checkpoint)
    vocoder = HifiGanModel.load_from_checkpoint(args.vocoder_checkpoint)

    files_to_replacements = {
        dd['audio_filepath']: replacement_phrase
        for dd, replacement_phrase in zip(data_dicts, replacement_phrases)
    }

    for file_path, replacement_phrase in tqdm(files_to_replacements.items()):
        dest = os.path.join(args.dest, replacement_phrase[:200])
        apply_repalcements(
            file_path, replacement_phrase, dest, inpainter, vocoder)

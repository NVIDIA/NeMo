import soundfile as sf
from nemo.collections.tts.models import InpainterModel
from nemo.collections.tts.models import HifiGanModel
import torch
import librosa
import sys
import os
from tqdm import tqdm

os.environ['DATA_CAP'] = '1'

args = sys.argv[1:]

try:
    checkpoint_path, recordings_path, output_dest = args
except:
    exit('usage: <checkpoint_path> <recordings_path> <output_dest>')

os.makedirs(output_dest, exist_ok=True)

model = InpainterModel.load_from_checkpoint(checkpoint_path)

files_to_replacements = {
    'test_1.wav': 'Would you like to speak with [joe burgarino] from [management team]?',
    'test_2.wav': 'Would you like to speak with [linda toney] from [loss prevention]?',
    'test_3.wav': 'Would you like to speak with [mary nelson] from [loss prevention]?',
    'test_4.wav': 'Would you like to speak with [mary nelson] from [management team]?',
    'test_5.wav': 'Would you like to speak with [joe burgarino] from [human resources]?',
    'test_6.wav': 'Would you like to speak with [james brucken] from [human resources]?',
    'test_7.wav': 'Ok that was [two blankets] and [two pillows] right?',
    'test_8.wav': 'Ok that was a [toothbrush] and [a tooth] paste?',
    'test_9.wav': 'Ok that was [three pillows] right?',
    'test_10.wav': 'Good evening! Thanks for your loyalty as a [gold] member. How can I help?',
}


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

vocoder = HifiGanModel.from_pretrained(model_name="nvidia/tts_hifigan")


def apply_repalcements(file_path, replacement_phrase, dest):
    data, sample_rate = sf.read(file_path)
    if sample_rate != expected_sample_rate:
        data = librosa.core.resample(
            data, orig_sr=sample_rate, target_sr=expected_sample_rate)

    original_spectrogram = model.make_spectrogram(data)
    replacements = get_all_replacements(replacement_phrase)

    spectrogram = original_spectrogram
    for replacement in replacements:
        full_replacement, partial_replacement, mcd_full, mcd_partial = model.regnerate_audio(
            spectrogram,
            replacement,
        )
        spectrogram = partial_replacement

    with torch.inference_mode():
        audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram.T.unsqueeze(0))

    sf.write(dest, audio.to('cpu').numpy()[0], 22050, format='wav')


for filename, replacement_phrase in tqdm(files_to_replacements.items()):
    file_path = os.path.join(recordings_path, filename)
    dest = os.path.join(output_dest, filename)
    apply_repalcements(file_path, replacement_phrase, dest)

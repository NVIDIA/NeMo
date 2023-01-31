import numpy as np
import soundfile as sf
from nemo.collections.tts.models import InpainterModel, FastPitchModel
from nemo.collections.tts.models import HifiGanModel
import torch

model = InpainterModel.load_from_checkpoint('nemo_results/Inpainter/aligner_pitch_obj_1/checkpoints/Inpainter--validation_mean_loss=1147.4573-epoch=19-last.ckpt')

data_info = {
    "audio_filepath": "data/LJSpeech-1.1/wavs/LJ012-0161.wav",
    "duration": 2.990884,
    "text": "he was reported to have fallen away to a shadow.",
    "normalized_text": "he was reported to have fallen away to a shadow."
}


expected_sample_rate = 22050

data, samplerate = sf.read(data_info['audio_filepath'])


original_spectrogram = model.make_spectrogram(data)
text = data_info['text']

full_replacement, partial_replacement = model.edit_recording(
    spectrogram=original_spectrogram,
    full_transcript=text,
    new_transcript="he was reported to have faded to nothing but shadow."
)
full_replacement = full_replacement.detach().numpy()
partial_replacement = partial_replacement.detach().numpy()



model = HifiGanModel.from_pretrained(model_name="nvidia/tts_hifigan")
#
with torch.inference_mode():
    audio = model.convert_spectrogram_to_audio(spec=torch.tensor([full_replacement.T]))

sf.write("full_replacement.wav", audio.to('cpu').numpy()[0], 22050, format='wav')

with torch.inference_mode():
    audio = model.convert_spectrogram_to_audio(spec=torch.tensor([partial_replacement.T]))

sf.write("partial_replacement.wav", audio.to('cpu').numpy()[0], 22050, format='wav')

#
# from mel_cepstral_distance import get_metrics_mels
#
# mcd_full, _, _ = get_metrics_mels(original_spectrogram, full_replacement)
# mcd_partial, _, _ = get_metrics_mels(original_spectrogram, partial_replacement)
#
# print(mcd_full)
# print(partial)

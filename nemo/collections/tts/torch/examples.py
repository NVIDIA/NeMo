import torch

from nemo.collections.tts.torch.data import TTSDataset
from nemo.collections.tts.data.vocabs import EnglishPhonemes
from nemo.collections.tts.data.g2p_modules import EnglishG2p

g2p = EnglishG2p(
    phoneme_dict="/home/otatanov/projects/tts_text_processing/cmudict-0.7b",
    heteronyms="/home/otatanov/projects/tts_text_processing/heteronyms"
)
parser = EnglishPhonemes(
    punct=True,
    stresses=True,
    spaces=True,
    chars=True,
    space=' ',
    silence=None,
    apostrophe=True,
    sep='|',
    add_blank_at=None,
    pad_with_space=False,
    g2p=g2p,
)

dataset = TTSDataset(
  manifest_filepath="train_manifest.json",  # Path to file that describes the location of audio and text
  text_parser=parser,
  text_parser_pad_id=parser.pad,
  sup_data_folder="sup_data_test",  # An additional folder that will store sup data
  sup_data_types=["pitch", "duration_prior", "mel", "energy"],
  sample_rate=22050,
  n_fft=1024,
  win_length=1024,
  hop_length=256,
  window="hann",
  n_mels=80,  # Number of mel filters
  lowfreq=0,  # lowfreq for mel filters
  highfreq=8000,  # highfreq for mel filters
  max_duration=20.,  # Max duration of samples in seconds
  min_duration=0.1,  # Min duration of samples in seconds
  ignore_file=None,
  trim=False,  # Whether to use librosa.effects.trim
  pitch_fmin=80,
  pitch_fmax=640,
)

dataloader = torch.utils.data.DataLoader(dataset, 10, collate_fn=dataset._collate_fn)

for batch in dataloader:
    tokens, tokens_lengths, audios, audio_lengths, pitches, duration_priors, mels, mel_lengths, energies = batch
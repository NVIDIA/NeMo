import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

cfg = OmegaConf.load("nemo/collections/tts/torch/tts_dataset.yaml")
dataset = instantiate(cfg.tts_dataset)
dataloader = torch.utils.data.DataLoader(dataset, 1, collate_fn=dataset._collate_fn)

for batch in dataloader:
    tokens, tokens_lengths, audios, audio_lengths, pitches, duration_priors, mels, mel_lengths, energies = batch

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

cfg = OmegaConf.load("nemo/collections/tts/torch/tts_dataset.yaml")
dataset = instantiate(cfg.tts_dataset)
dataloader = torch.utils.data.DataLoader(dataset, 1, collate_fn=dataset._collate_fn, num_workers=20)

pitch_list = []
for batch in tqdm(dataloader, total=len(dataloader)):
    tokens, tokens_lengths, audios, audio_lengths, pitches, pitches_lengths, duration_priors, nlp_tokens = batch
    # tokens, tokens_lengths, audios, audio_lengths, pitches, pitches_lengths, duration_priors = batch
    # tokens, tokens_lengths, audios, audio_lengths, duration_priors = batch

    pitch = pitches.squeeze(0)
    pitch_list.append(pitch[pitch != 0])

pitch_tensor = torch.cat(pitch_list)
print(f"PITCH_MEAN, PITCH_STD = {pitch_tensor.mean().item()}, {pitch_tensor.std().item()}")

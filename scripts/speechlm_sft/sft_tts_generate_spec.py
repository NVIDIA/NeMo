import os
import torch
import random
from tqdm import tqdm
import json
from tts_normalization_utils import get_normalizer, normalize
from nemo.collections.asr.parts.preprocessing.features import normalize_batch, clean_spectrogram_batch
from nemo.core.classes import typecheck

from nemo.collections.tts.models import FastPitchModel, SpectrogramEnhancerModel

manifest_file = "squadv2_train_not_normalized.json"
output_dir = f"./features_{manifest_file}"
normalize_type = "per_feature"
do_normalize = False
do_lowercase = False
use_enhancer = False  # TODO: fix it. It has a bug!


def generate_spec(tts_model, text, normalizer=None, do_lowercase=False, enhancer_model=None):
    if normalizer:
        text = normalize(text=text, normalizer=normalizer, do_lowercase=do_lowercase)

    src_ids = tts_model.vocab.encode(text)
    src_ids = torch.tensor([src_ids]).to(tts_model.device)

    with torch.no_grad():
        speaker_id = random.randint(0, n_speakers - 1)
        speaker_id = torch.tensor([speaker_id]).to(src_ids.device)
        signal, signal_len, *_ = tts_model(text=src_ids, durs=None, pitch=None, speaker=speaker_id, pace=1.0)
        if enhancer_model:
            with typecheck.disable_checks():
                signal = enhancer_model.forward(input_spectrograms=signal, lengths=signal_len)

        signal, *_ = normalize_batch(x=signal, seq_len=signal_len, normalize_type=normalize_type)
        signal = clean_spectrogram_batch(signal, signal_len)

    return signal


tts_model = FastPitchModel.from_pretrained(model_name="tts_en_fastpitch_for_asr_finetuning")
n_speakers = tts_model.cfg.n_speakers
tts_model.eval().cuda()

if use_enhancer:
    enhancer_model = SpectrogramEnhancerModel.from_pretrained(model_name="tts_en_spectrogram_enhancer_for_asr_finetuning")
    enhancer_model.eval().cuda()
else:
    enhancer_model = None

if do_normalize:
    normalizer = get_normalizer()
else:
    normalizer = None

os.makedirs(output_dir, exist_ok=True)

with open(manifest_file, 'r') as f:
    for line in tqdm(f):
        sample = json.loads(line)
        spec = generate_spec(tts_model, text=sample["context"], normalizer=normalizer, do_lowercase=do_lowercase, enhancer_model=enhancer_model)
        output_file_path = os.path.join(output_dir, sample["sample_id"]) + ".pt"
        torch.save(spec, output_file_path)
        print("Saved:", output_file_path)

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ipdb
import torch
import soundfile as sf

from nemo.collections.tts.models.fastspeech2s import FastSpeech2SModel
from nemo.core.config import hydra_runner


def denoise(audio_batch, bias_spect, strength=0.01):
    window = torch.hann_window(1024).to(audio_batch.device)
    audio_spect = torch.stft(audio_batch, n_fft=1024, window=window, return_complex=False)
    audio_angles = torch.atan2(audio_spect[..., -1], audio_spect[..., 0])
    audio_spect = torch.sqrt(torch.sum(torch.square(audio_spect), dim=-1))
    audio_spect_denoised = audio_spect - bias_spect.to(audio_batch.device) * strength
    audio_spect_denoised = torch.clamp(audio_spect_denoised, 0.0)
    audio_denoised = torch.istft(
        torch.stack(
            (audio_spect_denoised * torch.cos(audio_angles), audio_spect_denoised * torch.sin(audio_angles)), dim=-1
        ),
        n_fft=1024,
        window=window,
    )
    return audio_denoised


@hydra_runner(config_path="conf", config_name="fastspeech2s")
def main(cfg):
    # model = FastSpeech2SModel.load_from_checkpoint(
    #     ""
    # )
    model = FastSpeech2SModel.restore_from(
        "/home/jasoli/nemo/NeMo/examples/tts/nemo_experiments/FastSpeech2S/2020-02-22_31_GH_GenUpsample8442/checkpoints/FastSpeech2S.nemo"
    )
    model.cuda()
    model.setup_validation_data(cfg.model.validation_ds)
    batch = next(iter(model._validation_dl))
    f, fl, t, tl, duration = batch
    f = f.cuda()
    fl = fl.cuda()
    t = t.cuda()
    tl = tl.cuda()
    bias_spect = torch.load("bias_spect.pt").to(f.device)
    with torch.no_grad():
        spec, spec_len = model.audio_to_melspec_precessor(f, fl)
        audio_pred, _, log_duration_prediction = model(
            spec=spec, spec_len=spec_len, text=t, text_length=tl, splice=False,
        )
        audio_pred = audio_pred.squeeze()
        # audio_pred = denoise(audio_pred, bias_spect)
        duration_rounded = torch.clamp_min(torch.exp(log_duration_prediction) - 1, 0).long()
        duration_rounded = torch.sum(duration_rounded, dim=1)
        for i, aud in enumerate(audio_pred):
            aud = aud.cpu().numpy()
            aud = aud[: duration_rounded[i] * 256]
            sf.write(f"{i}.wav", aud, 22050)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter

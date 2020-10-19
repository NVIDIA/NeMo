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

import pytorch_lightning as pl

from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.collections.tts.models import WaveGlowModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="waveglow")
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = WaveGlowModel(cfg=cfg.model, trainer=trainer)
    epoch_time_logger = LogEpochTimeCallback()
    trainer.callbacks.extend([epoch_time_logger])
    trainer.fit(model)


@hydra_runner(config_path="conf", config_name="waveglow")
def infer_batch(cfg):
    import torch
    import librosa
    from torch_stft import STFT

    model = WaveGlowModel.from_pretrained("WaveGlow-22050Hz")
    model.setup_validation_data(cfg.model.validation_ds)
    model.cuda()
    model.eval()

    class Denoiser(torch.nn.Module):
        """ Removes model bias from audio produced with waveglow """

        def __init__(self, waveglow, filter_length=1024, n_overlap=4, win_length=1024, mode='zeros'):
            super(Denoiser, self).__init__()
            self.stft = STFT(
                filter_length=filter_length, hop_length=int(filter_length / n_overlap), win_length=win_length
            ).cuda()
            if mode == 'zeros':
                mel_input = torch.zeros((1, 80, 88)).cuda()
            elif mode == 'normal':
                mel_input = torch.randn((1, 80, 88)).cuda()
            else:
                raise Exception("Mode {} if not supported".format(mode))

            with torch.no_grad():
                bias_audio = waveglow.convert_spectrogram_to_audio(spec=mel_input, sigma=0.0).float()
                bias_spec, _ = self.stft.transform(bias_audio)

            self.register_buffer('bias_spec', bias_spec[:, :, 0][:, :, None])

        def forward(self, audio, strength=0.1):
            audio_spec, audio_angles = self.stft.transform(audio.cuda().float())
            audio_spec_denoised = audio_spec - self.bias_spec * strength
            audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
            audio_denoised = self.stft.inverse(audio_spec_denoised, audio_angles)
            return audio_denoised

    denoiser = Denoiser(model)
    denoiser.cuda()
    denoiser.eval()

    for batch in model._validation_dl:
        audio, audio_len = batch
        audio = audio.to("cuda")
        audio_len = audio_len.to("cuda")
        with torch.no_grad():
            spec, _ = model.audio_to_melspec_precessor(audio, audio_len)
            audio_pred = model.convert_spectrogram_to_audio(spec=spec)
            # audio_pred = denoiser(audio_pred)
        for i, single_audio in enumerate(audio_pred):
            print(single_audio.cpu().numpy().squeeze())
            librosa.output.write_wav(
                f"WaveGlow_{i}.wav", single_audio.cpu().numpy().squeeze()[: audio_len[i]], sr=22050
            )
        for i, single_audio in enumerate(audio):
            librosa.output.write_wav(f"Real_{i}.wav", single_audio.cpu().numpy().squeeze()[: audio_len[i]], sr=22050)
        break


if __name__ == '__main__':
    # main()  # noqa pylint: disable=no-value-for-parameter
    infer_batch()

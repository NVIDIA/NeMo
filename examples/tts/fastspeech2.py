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
import matplotlib.pyplot as plt


from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.collections.tts.models import FastSpeech2Model
from nemo.collections.tts.data.datalayers import FastSpeechWithDurs
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="fastspeech2")
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = FastSpeech2Model(cfg=cfg.model, trainer=trainer)
    lr_logger = pl.callbacks.LearningRateMonitor()
    epoch_time_logger = LogEpochTimeCallback()
    trainer.callbacks.extend([lr_logger, epoch_time_logger])
    trainer.fit(model)

    # model = FastSpeech2Model.load_from_checkpoint(
    #     "/home/jasoli/nemo/NeMo/examples/tts/nemo_experiments/FastSpeech2/2020-12-02_20-19-06/checkpoints/FastSpeech2-last.ckpt"
    # )
    # dataset = FastSpeechWithDurs(
    #     "/mnt/ssd1/data/LJSpeech-1.1/nvidia_ljspeech_val.json",
    #     22050,
    #     "/mnt/ssd1/data/LJSpeech-1.1/supplementary",
    #     ignore_file="/mnt/ssd1/data/LJSpeech-1.1/wavs_to_ignore",
    # )
    # f, fl, t, tl, durations = dataset[0]
    # f = f.unsqueeze(0)
    # fl = fl.unsqueeze(0)
    # t = t.unsqueeze(0)
    # tl = tl.unsqueeze(0)
    # durations = durations.unsqueeze(0)

    # spec, spec_len = model.audio_to_melspec_precessor(f, fl)
    # mel = model(spec_len=spec_len, text=t, text_length=tl, durations=durations)
    # fig = plt.Figure()
    # ax = fig.add_subplot(111)
    # im = ax.imshow(mel.detach().cpu().numpy().squeeze().T, aspect="auto", origin="lower", interpolation='none')
    # plt.colorbar(im, ax=ax)
    # plt.xlabel("Frames")
    # plt.ylabel("Channels")
    # plt.tight_layout()
    # fig.savefig('spec.png')


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter

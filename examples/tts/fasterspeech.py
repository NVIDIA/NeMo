# Copyright 2020 NVIDIA. All Rights Reserved.
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

import argparse
import math
import os
from typing import Any, Mapping

import librosa
import numpy as np
import torch
from ruamel import yaml

import nemo
from nemo.collections import asr as nemo_asr
from nemo.collections import tts as nemo_tts
from nemo.utils import argparse as nm_argparse
from nemo.utils import lr_policies

logging = nemo.logging


def parse_args():
    parser = argparse.ArgumentParser(
        description='FasterSpeech Training Pipeline.',
        parents=[nm_argparse.NemoArgParser()],
        conflict_handler='resolve',  # For parents common flags.
    )
    parser.set_defaults(
        # `x = 30000`
        amp_opt_level='O0',
        model_config='configs/fasterspeech.yaml',
        batch_size=32,
        eval_batch_size=32,
        train_freq=300,  # 1/100x
        eval_freq=10000,  # 1/3x (Sampling process may take time.)
        optimizer='novograd',
        weight_decay=1e-6,
        warmup=3000,
        num_epochs=150,  # Couple of epochs for testing.
        lr=1e-2,  # Goes good with Adam.
        min_lr=1e-5,  # Goes good with cosine policy.
        work_dir='work',
        checkpoint_save_freq=10000,  # 1/3x
    )

    # Required: train_dataset
    # Optional: eval_names, eval_datasets

    # Durations from ASR CTC Model
    parser.add_argument('--train_durs', type=str, required=True, help="Train dataset durations directory path.")
    parser.add_argument('--eval_durs', type=str, nargs="*", default=[], help="Eval datasets durations")
    parser.add_argument('--durs_type', type=str, choices=['pad', 'full-pad'], default='full-pad', help="Durs type.")

    # Speakers
    parser.add_argument('--speakers', type=str, required=True, help="LibriTTS speakers TSV File")
    parser.add_argument('--d_speaker', type=int, default=64, help="Size of Speaker Embedding")

    # Model
    parser.add_argument('--d_char', type=int, default=64, help="Size of input char embedding")
    parser.add_argument('--loss_reduction', type=str, choices=['batch', 'all'], default='all', help="Loss Reduction")

    args = parser.parse_args()

    return args


class AudioInspector(nemo.core.Metric):
    def __init__(
        self, sample_rate, n_fft, hop_length, win_length, window, power, k=3, vocoder='griffin-lim',
    ):
        super().__init__()

        if vocoder != 'griffin-lim':
            raise ValueError("Only griffin-lim supported at this point.")

        self._sample_rate = sample_rate
        self._n_fft = n_fft
        self._hop_length = hop_length
        self._win_length = win_length
        self._window = window
        self._power = power
        self._k = k

        self._samples = None

    def _mel_to_audio(self, mel):
        return librosa.feature.inverse.mel_to_audio(
            M=np.exp(mel),
            sr=self._sample_rate,
            n_fft=self._n_fft,
            hop_length=self._hop_length,
            win_length=self._win_length,
            window=self._window,
            power=self._power,
            n_iter=50,
        )

    def clear(self) -> None:
        self._samples = ()

    def batch(self, tensors) -> None:
        if len(self._samples):
            return

        logging.info("Start vocoding process...")

        # Audio
        audios = []
        for audio1, audio_len1 in zip(tensors.audio[: self._k], tensors.audio_len[: self._k]):
            audios.append(audio1.cpu().numpy()[: audio_len1.cpu().numpy().item()])

        # True Mels
        vocodeds_true, mels_true = [], []
        for mel1, mel_len1 in zip(tensors.mel_true[: self._k], tensors.mel_len[: self._k]):
            mel = mel1.cpu().numpy()[:, : mel_len1.cpu().numpy().item()]
            vocodeds_true.append(self._mel_to_audio(mel))
            mels_true.append(mel)

        # Pred Mels
        vocodeds_pred, mels_pred = [], []
        for mel1, mel_len1 in zip(tensors.mel_pred[: self._k], tensors.mel_len[: self._k]):
            mel1 = mel1.t()  # Pred tensors have another order.
            mel = mel1.cpu().numpy()[:, : mel_len1.cpu().numpy().item()]
            vocodeds_pred.append(self._mel_to_audio(mel))
            mels_pred.append(mel)

        logging.info("End vocoding process.")

        self._samples = audios, vocodeds_true, vocodeds_pred, mels_true, mels_pred

    def log(self, tb_writer, step, prefix, samples=None) -> None:
        def log_audio(key, signal):
            tb_writer.add_audio(key, torch.tensor(signal).unsqueeze(0), step, self._sample_rate)

        def log_mel(key, mel):
            tb_writer.add_image(
                tag=key,
                img_tensor=nemo_tts.parts.helpers.plot_spectrogram_to_numpy(mel),
                global_step=step,
                dataformats='HWC',
            )

        for i, (audio, vocoded_true, vocoded_pred, mel_true, mel_pred) in enumerate(zip(*self._samples)):
            log_audio(f'{prefix}/sample{i}_audio', audio)
            log_audio(f'{prefix}/sample{i}_vocoded-true', vocoded_true)
            log_audio(f'{prefix}/sample{i}_vocoded-pred', vocoded_pred)

            log_mel(f'{prefix}/sample{i}_mel-true', mel_true)
            log_mel(f'{prefix}/sample{i}_mel-pred', mel_pred)


class FasterSpeechGraph:
    def __init__(self, args, engine, config):
        labels = config.labels
        pad_id, labels = len(labels), labels + ['<PAD>']
        blank_id, labels = len(labels), labels + ['<BLANK>']

        self.train_dl = nemo_tts.FasterSpeechDataLayer(
            manifests=args.train_dataset,
            durs_file=args.train_durs,
            labels=labels,
            durs_type=args.durs_type,
            speakers=args.speakers,
            batch_size=args.batch_size,
            pad_id=pad_id,
            blank_id=blank_id,
            num_workers=max(int(os.cpu_count() / engine.world_size), 1),
            **config.FasterSpeechDataLayer_train,  # Including sample rate.
        )

        self.eval_dls = {}
        for name, eval_dataset, eval_durs1 in zip(args.eval_names, args.eval_datasets, args.eval_durs):
            self.eval_dls[name] = nemo_tts.FasterSpeechDataLayer(
                manifests=eval_dataset,
                durs_file=eval_durs1,
                labels=labels,
                durs_type=args.durs_type,
                speakers=args.speakers,
                batch_size=args.eval_batch_size,
                pad_id=pad_id,
                blank_id=blank_id,
                num_workers=max(int(os.cpu_count() / engine.world_size), 1),
                **config.FasterSpeechDataLayer_eval,
            )

        self.preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor(**config.AudioToMelSpectrogramPreprocessor)
        self._audio_inspector = AudioInspector(
            sample_rate=config.sample_rate,
            n_fft=self.preprocessor.featurizer.n_fft,
            hop_length=self.preprocessor.featurizer.hop_length,
            win_length=self.preprocessor.featurizer.win_length,
            window=config.AudioToMelSpectrogramPreprocessor['window'],
            power=self.preprocessor.featurizer.mag_power,
        )

        self.model = nemo_tts.FasterSpeech(
            n_vocab=len(labels),
            d_char=args.d_char,
            pad_id=pad_id,
            jasper_kwargs=config.JasperEncoder,
            d_out=config.n_mels,
            n_speakers=self.train_dl.n_speakers,
            d_speaker=args.d_speaker,
        )

        self.loss = nemo_tts.FasterSpeechMelLoss(reduction=args.loss_reduction)

    def build(self, args, engine):
        train_loss, callbacks = None, []

        # Train
        data = self.train_dl()
        mel_true, mel_len = self.preprocessor(input_signal=data.audio, length=data.audio_len)
        output = self.model(
            text=data.text,
            text_mask=data.text_mask,
            text_rep=data.text_rep,
            text_rep_mask=data.text_rep_mask,
            speaker=data.speaker,
        )
        train_loss = self.loss(
            mel_true=mel_true,
            mel_pred=output.pred,
            mel_len=mel_len,
            dur_true=data.dur,
            text_rep_mask=data.text_rep_mask,
        )
        callbacks.append(
            nemo.core.TrainLogger(
                tensors=dict(loss=train_loss), metrics=['loss'], freq=args.train_freq, tb_writer=engine.tb_writer,
            )
        )

        # Eval
        for name, eval_dl in self.eval_dls.items():
            data = eval_dl()
            mel_true, mel_len = self.preprocessor(input_signal=data.audio, length=data.audio_len)
            output = self.model(
                text=data.text,
                text_mask=data.text_mask,
                text_rep=data.text_rep,
                text_rep_mask=data.text_rep_mask,
                speaker=data.speaker,
            )
            loss = self.loss(
                mel_true=mel_true,
                mel_pred=output.pred,
                mel_len=mel_len,
                dur_true=data.dur,
                text_rep_mask=data.text_rep_mask,
            )
            callbacks.append(
                nemo.core.EvalLogger(
                    tensors=dict(
                        loss=loss,
                        audio=data.audio,
                        audio_len=data.audio_len,
                        mel_true=mel_true,
                        mel_pred=output.pred,
                        mel_len=mel_len,
                    ),
                    metrics=['loss', self._audio_inspector],
                    freq=args.eval_freq,
                    tb_writer=engine.tb_writer,
                    prefix=name,
                )
            )

        callbacks.append(
            nemo.core.CheckpointCallback(folder=engine.checkpoint_dir, step_freq=args.checkpoint_save_freq)
        )

        return train_loss, callbacks


def main():
    args = parse_args()
    logging.info('Args: %s', args)

    engine = nemo.core.NeuralModuleFactory(
        local_rank=args.local_rank,
        optimization_level=args.amp_opt_level,
        cudnn_benchmark=args.cudnn_benchmark,
        log_dir=args.work_dir,
        tensorboard_dir=args.tensorboard_dir,
        create_tb_writer=True,
        files_to_copy=[args.model_config, __file__],
    )
    # noinspection PyProtectedMember
    logging.info('Engine: %s', vars(engine._exp_manager))

    yaml_loader = yaml.YAML(typ='safe')
    with open(args.model_config) as f:
        config = argparse.Namespace(**yaml_loader.load(f))
    logging.info('Config: %s', config)

    graph = FasterSpeechGraph(args, engine, config)

    loss, callbacks = graph.build(args, engine)

    total_steps = (
        args.max_steps
        if args.max_steps is not None
        else args.num_epochs * math.ceil(len(graph.train_dl) / (args.batch_size * engine.world_size))
    )
    engine.train(
        tensors_to_optimize=[loss],
        optimizer=args.optimizer,
        optimization_params=dict(
            num_epochs=args.num_epochs, max_steps=total_steps, lr=args.lr, weight_decay=args.weight_decay,
        ),
        callbacks=callbacks,
        lr_policy=lr_policies.CosineAnnealing(total_steps=total_steps, min_lr=args.min_lr, warmup_steps=args.warmup),
    )


if __name__ == '__main__':
    main()

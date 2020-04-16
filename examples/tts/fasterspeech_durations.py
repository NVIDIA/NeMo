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

import torch
from ruamel import yaml

import nemo
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
        model_config='configs/fasterspeech_durations.yaml',
        batch_size=64,
        eval_batch_size=64,
        train_freq=300,  # 1/100x
        eval_freq=3000,  # 10x train freq.
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

    # Model
    parser.add_argument('--d_char', type=int, default=64, help="Size of input char embedding")
    parser.add_argument('--loss_dmld_hidden', type=int, default=5, help="1/3 of d_hidden for dmld (log2(num_classes)")
    parser.add_argument('--loss_num_classes', type=int, default=32, help="'n_classes' for dmld or xe (32 covers 98%).")
    parser.add_argument('--loss_reduction', type=str, choices=['batch', 'all'], default='all', help="Loss Reduction")
    parser.add_argument(
        '--loss_method',
        type=str,
        choices=['l2-log', 'l2', 'dmld-log', 'dmld', 'xe', 'xe-steps'],
        default='xe-steps',
        help="Method for Loss Calculation",
    )

    args = parser.parse_args()

    return args


class DurMetric(nemo.core.Metric):
    def __init__(self, preprocessing):
        super().__init__()

        self._preprocessing = preprocessing

        self._ss, self._hit0, self._hit1, self._hit2, self._hit3, self._total = (None,) * 6
        self._hit10m, self._total10m = None, None

    def clear(self) -> None:
        self._ss, self._hit0, self._hit1, self._hit2, self._hit3, self._total = 0, 0, 0, 0, 0, 0
        self._hit10m, self._total10m = 0, 0

    def batch(self, tensors) -> None:
        tensors = self._preprocessing(tensors)

        for dur_true1, dur_pred1, mask1 in zip(tensors.dur_true, tensors.dur_pred, tensors.mask):
            prefix = mask1.sum().item()
            dur_true1 = dur_true1[:prefix]
            dur_pred1 = dur_pred1[:prefix]
            assert dur_true1.shape == dur_true1.shape

            self._ss += ((dur_true1 - dur_pred1) ** 2).sum().item()
            self._hit0 += (dur_true1 == dur_pred1).sum().item()
            self._hit1 += ((dur_true1 - dur_pred1).abs() <= 1).sum().item()
            self._hit2 += ((dur_true1 - dur_pred1).abs() <= 2).sum().item()
            self._hit3 += ((dur_true1 - dur_pred1).abs() <= 3).sum().item()
            self._total += prefix
            self._hit10m += ((dur_true1 == dur_pred1) & (dur_true1 >= 10)).sum().item()
            self._total10m += (dur_true1 >= 10).sum().item()

    def final(self) -> Any:
        mse = self._ss / self._total
        acc = self._hit0 / self._total * 100
        d1 = self._hit1 / self._total * 100
        d2 = self._hit2 / self._total * 100
        d3 = self._hit3 / self._total * 100
        acc10m = self._hit10m / self._total10m * 100

        return dict(mse=mse, acc=acc, d1=d1, d2=d2, d3=d3, acc10m=acc10m)


class FasterSpeechGraph:
    def __init__(self, args, engine, config):
        labels = config.labels
        pad_id, labels = len(labels), labels + ['<PAD>']
        blank_id, labels = len(labels), labels + ['<BLANK>']

        self.train_dl = nemo_tts.FasterSpeechDataLayer(
            data=args.train_dataset,
            durs=args.train_durs,
            labels=labels,
            durs_type=args.durs_type,
            batch_size=args.batch_size,
            pad_id=pad_id,
            blank_id=blank_id,
            load_audio=False,  # It's just durations predictor, so we won't need audio.
            num_workers=max(int(os.cpu_count() / engine.world_size), 1),
            **config.FasterSpeechDataLayer_train,
        )

        self.eval_dls = {}
        for name, eval_dataset, eval_durs1 in zip(args.eval_names, args.eval_datasets, args.eval_durs):
            self.eval_dls[name] = nemo_tts.FasterSpeechDataLayer(
                data=eval_dataset,
                durs=eval_durs1,
                labels=labels,
                durs_type=args.durs_type,
                batch_size=args.eval_batch_size,
                pad_id=pad_id,
                blank_id=blank_id,
                load_audio=False,
                num_workers=max(int(os.cpu_count() / engine.world_size), 1),
                **config.FasterSpeechDataLayer_eval,
            )

        self.loss = nemo_tts.FasterSpeechDurLoss(
            method=args.loss_method,
            num_classes=args.loss_num_classes,
            dmld_hidden=args.loss_dmld_hidden,
            reduction=args.loss_reduction,
        )

        self.model = nemo_tts.FasterSpeech(
            n_vocab=len(labels),
            d_char=args.d_char,
            pad_id=pad_id,
            jasper_kwargs=config.JasperEncoder,
            d_out=self.loss.d_out,
        )

    def build(self, args, engine):
        train_loss, callbacks = None, []
        metrics = ['loss', DurMetric(self.loss.preprocessing)]

        # Train
        data = self.train_dl()
        output = self.model(text=data.text, text_mask=data.text_mask)
        train_loss = self.loss(dur_true=data.dur, dur_pred=output.pred, text_mask=data.text_mask)
        callbacks.append(
            nemo.core.TrainLogger(
                tensors=dict(loss=train_loss, dur_true=data.dur, dur_pred=output.pred, mask=data.text_mask),
                metrics=metrics,
                freq=args.train_freq,
                tb_writer=engine.tb_writer,
            )
        )

        # Eval
        for name, eval_dl in self.eval_dls.items():
            data = eval_dl()
            output = self.model(text=data.text, text_mask=data.text_mask)
            loss = self.loss(dur_true=data.dur, dur_pred=output.pred, text_mask=data.text_mask)
            callbacks.append(
                nemo.core.EvalLogger(
                    tensors=dict(loss=loss, dur_true=data.dur, dur_pred=output.pred, mask=data.text_mask),
                    metrics=metrics,
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

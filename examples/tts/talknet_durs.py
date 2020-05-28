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
import datetime
import math
import os
from typing import Any

import wandb
from ruamel import yaml

import nemo
from nemo.collections import tts as nemo_tts
from nemo.utils import argparse as nm_argparse
from nemo.utils import lr_policies

logging = nemo.logging

MODEL_WEIGHTS_UPPER_BOUND = 3_000_000


def parse_args():
    parser = argparse.ArgumentParser(
        description='TalkNet Durs Predictor Training Pipeline',
        parents=[nm_argparse.NemoArgParser()],
        conflict_handler='resolve',  # For parents common flags.
    )
    parser.set_defaults(
        amp_opt_level='O0',  # O1/O2 works notably faster, O3 usually produces NaNs.
        model_config='configs/talknet-durs-lj.yaml',
        batch_size=64,
        eval_batch_size=64,
        train_freq=300,
        eval_freq=3000,  # 10x train freq
        optimizer='adam',
        weight_decay=1e-6,
        grad_norm_clip=1.0,
        warmup=3000,
        num_epochs=100,  # Couple of epochs for testing.
        lr=1e-3,  # Goes good with Adam.
        min_lr=1e-5,  # Goes good with cosine policy.
        work_dir='work/' + str(datetime.datetime.now()).replace(' ', '_'),
        checkpoint_save_freq=10000,
        wdb_project='fast-tts',
        wdb_name='test_' + str(datetime.datetime.now()).replace(' ', '_'),
        wdb_tags=['durs', 'test', 'to-delete'],
    )

    # Required: train_dataset
    # Optional: eval_names, eval_datasets

    # Durations
    parser.add_argument('--train_durs', type=str, required=True, help="Train dataset durations directory path.")
    parser.add_argument('--eval_durs', type=str, nargs='*', default=[], help="Eval datasets durations")
    parser.add_argument('--durs_type', type=str, choices=['pad', 'full-pad'], default='full-pad', help="Durs type")

    args = parser.parse_args()

    return args


class DurMetric(nemo.core.Metric):
    def __init__(self, preprocessing, k=1 + 5):
        super().__init__()

        self._preprocessing = preprocessing
        self._k = k

        self._hits = None
        self._ss, self._total = None, None
        self._hit10m, self._total10m = None, None
        self._durs_hit, self._durs_total = None, None
        self._blanks_hit, self._blanks_total = None, None

    def clear(self) -> None:
        self._hits = [0] * self._k
        self._ss, self._total = 0, 0
        self._hit10m, self._total10m = 0, 0
        self._durs_hit, self._durs_total = 0, 0
        self._blanks_hit, self._blanks_total = 0, 0

    def batch(self, tensors) -> None:
        tensors = self._preprocessing(tensors)

        for dur_true1, dur_pred1, mask1 in zip(tensors.dur_true, tensors.dur_pred, tensors.mask):
            prefix = mask1.sum().item()
            dur_true1 = dur_true1[:prefix]
            dur_pred1 = dur_pred1[:prefix]
            assert dur_true1.shape == dur_true1.shape

            self._ss += ((dur_true1 - dur_pred1) ** 2).sum().item()
            for k in range(self._k):
                self._hits[k] += ((dur_true1 - dur_pred1).abs() <= k).sum().item()
            self._total += prefix
            self._hit10m += ((dur_true1 == dur_pred1) & (dur_true1 >= 10)).sum().item()
            self._total10m += (dur_true1 >= 10).sum().item()
            self._durs_hit += (dur_true1 == dur_pred1)[1::2].sum().item()
            self._durs_total += prefix // 2
            self._blanks_hit += (dur_true1 == dur_pred1)[::2].sum().item()
            self._blanks_total += (prefix // 2) + 1

    def final(self) -> Any:
        mse = self._ss / self._total
        acc = self._hits[0] / self._total * 100
        dx = {f'd{k}': self._hits[k] / self._total * 100 for k in range(1, self._k)}
        acc10m = self._hit10m / self._total10m * 100
        durs_acc = self._durs_hit / self._durs_total * 100
        blanks_acc = self._blanks_hit / self._blanks_total * 100

        return dict(mse=mse, acc=acc, acc10m=acc10m, durs_acc=durs_acc, blanks_acc=blanks_acc, **dx)


class TalkNetGraph:
    def __init__(self, args, engine, config):
        labels = config.labels
        pad_id, labels = len(labels), labels + ['<PAD>']
        blank_id, labels = len(labels), labels + ['<BLANK>']

        self.train_dl = nemo_tts.TalkNetDataLayer(
            data=args.train_dataset,
            durs=args.train_durs,
            labels=labels,
            durs_type=args.durs_type,
            batch_size=args.batch_size,
            pad_id=pad_id,
            blank_id=blank_id,
            load_audio=False,  # It's just durations predictor, so we won't need audio.
            num_workers=max(int(os.cpu_count() / engine.world_size), 1),
            **config.TalkNetDataLayer_train,  # Including sample rate.
        )

        self.eval_dls = {}
        for name, eval_dataset, eval_durs1 in zip(args.eval_names, args.eval_datasets, args.eval_durs):
            self.eval_dls[name] = nemo_tts.TalkNetDataLayer(
                data=eval_dataset,
                durs=eval_durs1,
                labels=labels,
                durs_type=args.durs_type,
                batch_size=args.eval_batch_size,
                pad_id=pad_id,
                blank_id=blank_id,
                load_audio=False,
                num_workers=max(int(os.cpu_count() / engine.world_size), 1),
                **config.TalkNetDataLayer_eval,
            )

        # Need to calculate 'd_out' for model.
        self.loss = nemo_tts.TalkNetDursLoss(**config.TalkNetDursLoss)

        self.model = nemo_tts.TalkNet(
            n_vocab=len(labels),
            pad_id=pad_id,
            jasper_kwargs=config.JasperEncoder,
            d_out=self.loss.d_out,
            **config.TalkNet,
        )
        if args.local_rank is None or args.local_rank == 0:
            # There is a bug in WanDB with logging gradients.
            # wandb.watch(self.model, log='all')
            wandb.config.total_weights = self.model.num_weights
            nemo.logging.info('Total weights: %s', self.model.num_weights)
            assert self.model.num_weights < MODEL_WEIGHTS_UPPER_BOUND

    def build(self, args, engine):  # noqa
        train_loss, callbacks = None, []
        metrics = ['loss', DurMetric(self.loss.preprocessing)]

        # Train
        data = self.train_dl()
        output = self.model(text=data.text, text_mask=data.text_mask)
        train_loss = self.loss(dur_true=data.dur, dur_pred=output.pred, text_mask=data.text_mask)
        callbacks.extend(
            [
                nemo.core.TrainLogger(
                    tensors=dict(loss=train_loss, dur_true=data.dur, dur_pred=output.pred, mask=data.text_mask),
                    metrics=metrics,
                    freq=args.train_freq,
                    batch_p=args.batch_size / (len(self.train_dl) / engine.world_size),
                ),
                nemo.core.WandbCallback(update_freq=args.train_freq),
            ]
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
                    prefix=name,
                    single_gpu=isinstance(eval_dl._dataloader.sampler, nemo_tts.LenSampler),  # noqa
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
        files_to_copy=[args.model_config, __file__],
    )
    logging.info('Engine: %s', vars(engine._exp_manager))  # noqa

    yaml_loader = yaml.YAML(typ='safe')
    with open(args.model_config) as f:
        config = argparse.Namespace(**yaml_loader.load(f))
    logging.info('Config: %s', config)

    if args.local_rank is None or args.local_rank == 0:
        wandb.init(
            name=args.wdb_name,
            config=dict(args=vars(args), engine=vars(engine._exp_manager), config=vars(config)),  # noqa
            project=args.wdb_project,
            tags=args.wdb_tags,
        )
        wandb.save(args.model_config)

    graph = TalkNetGraph(args, engine, config)
    loss, callbacks = graph.build(args, engine)
    total_steps = (
        args.max_steps
        if args.max_steps is not None
        else args.num_epochs * math.floor(len(graph.train_dl) / (args.batch_size * engine.world_size))
    )
    if args.local_rank is None or args.local_rank == 0:
        wandb.config.total_steps = total_steps
        nemo.logging.info('Total steps: %s', total_steps)

    engine.train(
        tensors_to_optimize=[loss],
        optimizer=args.optimizer,
        optimization_params=dict(
            num_epochs=args.num_epochs,
            max_steps=total_steps,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_norm_clip=args.grad_norm_clip,
        ),
        callbacks=callbacks,
        lr_policy=lr_policies.CosineAnnealing(total_steps=total_steps, min_lr=args.min_lr, warmup_steps=args.warmup),
    )


if __name__ == '__main__':
    main()

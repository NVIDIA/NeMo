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

from ruamel import yaml

import nemo
from nemo.collections import tts as nemo_tts
from nemo.utils import argparse as nm_argparse
from nemo.utils import lr_policies

logging = nemo.logging


def parse_args():
    parser = argparse.ArgumentParser(
        description='TalkNet Durs Predictor Training Pipeline',
        parents=[nm_argparse.NemoArgParser()],
        conflict_handler='resolve',  # For parents common flags.
    )
    parser.add_argument('--eval_names', type=str, nargs="*", default=[], help="Eval datasets names.")
    parser.add_argument("--eval_datasets", type=str, nargs="*", default=[], help="Evaluation datasets paths.")
    parser.add_argument('--train_freq', type=int, default=300, help="Train metrics logging frequency.")
    parser.add_argument('--grad_norm_clip', type=float, help="grad norm clip")
    parser.add_argument('--warmup', type=int, default=3000, help="Number of steps for warmup.")
    parser.add_argument('--min_lr', type=float, default=1e-5, help="Minimum learning rate to decay to.")
    parser.add_argument('--wdb_project', type=str, help="WanDB run project")
    parser.add_argument('--wdb_name', type=str, help="WanDB run name")
    parser.add_argument('--wdb_tags', type=str, nargs="*", default=[], help="WanDB run tags")
    parser.set_defaults(
        amp_opt_level='O0',  # O1/O2 works notably faster, O3 usually produces NaNs.
        model_config='configs/talknet-durs-lj.yaml',
        batch_size=64,
        eval_batch_size=64,
        train_freq=10,
        eval_freq=100,  # 10x train freq
        optimizer='adam',
        weight_decay=1e-6,
        grad_norm_clip=1.0,
        warmup=300,
        num_epochs=100,
        lr=1e-3,  # Goes good with Adam.
        min_lr=1e-5,  # Goes good with cosine policy.
        work_dir='work/' + str(datetime.datetime.now()).replace(' ', '_'),
        checkpoint_save_freq=10000,
        wdb_project='talknet',
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
            nemo.logging.info('Total weights: %s', self.model.num_weights)

    def build(self, args, engine):  # noqa
        train_loss, callbacks = None, []

        # Train
        data = self.train_dl()
        output = self.model(text=data.text, text_mask=data.text_mask)
        train_loss = self.loss(dur_true=data.dur, dur_pred=output.pred, text_mask=data.text_mask)
        callbacks.append(
            nemo.core.SimpleLossLoggerCallback(
                tensors=[train_loss],
                print_func=lambda x: logging.info('Train loss: %s', x[0].item()),
                step_freq=args.train_freq,
            )
        )

        # Eval
        for name, eval_dl in self.eval_dls.items():
            data = eval_dl()
            output = self.model(text=data.text, text_mask=data.text_mask)
            loss = self.loss(dur_true=data.dur, dur_pred=output.pred, text_mask=data.text_mask)

            def user_iter_callback(vd, gvd):
                for k, v in vd.items():
                    if k.startswith('loss'):
                        gvd.setdefault('loss', []).extend(v)

            def user_epochs_done_callback(gvd):
                logging.info('Eval loss: %s', (sum(gvd['loss']) / len(gvd['loss'])).item())

            callbacks.append(
                nemo.core.EvaluatorCallback(
                    eval_tensors=[loss],
                    user_iter_callback=user_iter_callback,
                    user_epochs_done_callback=user_epochs_done_callback,
                    eval_step=args.eval_freq,
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

    graph = TalkNetGraph(args, engine, config)
    loss, callbacks = graph.build(args, engine)
    total_steps = (
        args.max_steps
        if args.max_steps is not None
        else args.num_epochs * math.floor(len(graph.train_dl) / (args.batch_size * engine.world_size))
    )
    if args.local_rank is None or args.local_rank == 0:
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

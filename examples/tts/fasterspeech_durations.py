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

import attrdict
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
        amp_opt_level='O0',
        model_config='configs/fasterspeech_durations.yaml',
        batch_size=32,
        optimizer='novograd',
        weight_decay=1e-6,
        num_epochs=150,  # Couple of epochs for testing.
        lr=1e-2,  # Goes good with Adam.
        work_dir='work',
        checkpoint_save_freq=15000,
    )

    # Default training things.
    parser.add_argument('--train_log_freq', type=int, default=50, help="Train metrics logging frequency.")
    parser.add_argument('--min_lr', type=float, default=1e-5, help="Minimum learning rate to decay to.")
    parser.add_argument('--warmup', type=int, default=3000, help="Number of steps for warmup.")

    # Durations from ASR CTC model.
    parser.add_argument('--durs_dir', type=str, required=True, help="Train dataset durations directory path.")

    # Model.
    parser.add_argument('--d_emb', type=int, default=128, help="Size of input char embedding.")

    args = parser.parse_args()

    return args


class FasterSpeechGraph:
    def __init__(self, args, engine, config):
        self.data_layer = nemo_tts.FasterSpeechDataLayer(
            manifest_filepath=args.train_dataset,
            durs_dir=args.durs_dir,
            labels=config.labels,
            sample_rate=config.sample_rate,
            batch_size=args.batch_size,
            num_workers=max(int(os.cpu_count() / engine.world_size), 1),
            shuffle=True,
            **config.FasterSpeechDataLayer,
        )

        self.model = nemo_tts.FasterSpeech(
            n_vocab=len(config.labels), d_emb=args.d_emb, pad_id=None, jasper_kwargs=config.JasperEncoder, d_out=1,
        )

        self.loss = nemo_tts.FasterSpeechDurLoss()

    def build(self, args, engine):
        data = self.data_layer()
        output = self.model(text=data.text, text_mask=data.text_mask)
        loss = self.loss(dur_true=data.dur, dur_pred=output.pred, text_mask=data.text_mask)

        callbacks = [
            nemo.core.TrainLogger(
                tensors=dict(loss=loss, dur_true=data.dur, dur_pred=output.pred, mask=data.text_mask),
                metrics=[self._train_metrics],
                freq=args.train_log_freq,
                tb_writer=engine.tb_writer,
            ),
            nemo.core.CheckpointCallback(folder=engine.checkpoint_dir, step_freq=args.checkpoint_save_freq),
        ]

        total_steps = (
            args.max_steps
            if args.max_steps is not None
            else args.num_epochs * math.ceil(len(self.data_layer) / (args.batch_size * engine.world_size))
        )

        return loss, callbacks, total_steps

    @staticmethod
    def _train_metrics(tensors):
        # Loss.
        loss = tensors.loss.item()
        logging.info(f'Loss: {loss:.5f}')

        # Acc.
        hit, total = 0, 0
        for dur_true1, dur_pred1, mask1 in zip(tensors.dur_true, tensors.dur_pred, tensors.mask):
            prefix = mask1.sum().item()
            dur_true1, dur_pred1 = dur_true1[:prefix], dur_pred1.squeeze(-1)[:prefix]
            assert dur_true1.shape == dur_true1.shape

            # Preprocessing.
            dur_pred1 = dur_pred1.exp() - 1
            dur_pred1[dur_pred1 < 0.0] = 0.0
            dur_pred1 = dur_pred1.round().long()

            hit += (dur_true1 == dur_pred1).sum().item()
            total += prefix

        # Example.
        # noinspection PyUnboundLocalVariable
        logging.info(f'dur_true: {dur_true1.data}')
        # noinspection PyUnboundLocalVariable
        logging.info(f'dur_pred: {dur_pred1.data}')

        acc = hit / total * 100
        assert 0 <= acc <= 100
        logging.info(f'Acc: {acc:.3f}%')

        return dict(loss=loss, acc=acc)


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
    # noinspection PyProtectedMember
    logging.info('Engine: %s', vars(engine._exp_manager))

    yaml_loader = yaml.YAML(typ="safe")
    with open(args.model_config) as f:
        config = attrdict.AttrDict(yaml_loader.load(f))
    logging.info('Config: %s', config)

    graph = FasterSpeechGraph(args, engine, config)

    loss, callbacks, total_steps = graph.build(args, engine)

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

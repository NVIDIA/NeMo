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
from pathlib import Path

import attrdict
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
        amp_opt_level='O0',
        model_config='configs/fasterspeech.yaml',
        batch_size=32,
        optimizer='adam',
        weight_decay=1e-6,
        num_epochs=3,  # Couple of epochs for testing.
        lr=1e-3,  # Goes good with Adam.
        work_dir='work',
    )

    # To be able to discern experiments in the future.
    parser.add_argument('--id', type=str, required=True, help="Experiment identificator for clarity.")

    # Cosine policy.
    parser.add_argument('--min_lr', type=float, default=1e-5, help="Minimum learning rate to decay to.")
    parser.add_argument('--warmup', type=int, default=3000, help="Number of steps for warmup.")

    # Durations from ASR CTC model.
    parser.add_argument('--durs_dir', type=str, required=True, help="Train dataset durations directory path.")

    args = parser.parse_args()

    return args


class FasterSpeechGraph:
    def __init__(self, args, config):
        self.data_layer = nemo_tts.FastSpeechDataLayer.import_from_config(
            args.model_config,
            'FastSpeechDataLayer',
            overwrite_params=dict(
                manifest_filepath=args.train_dataset,
                durs_dir=args.durations_dir,
                bos_id=len(config.labels),
                eos_id=len(config.labels) + 1,
                pad_id=len(config.labels) + 2,
                batch_size=args.batch_size,
                num_workers=max(int(os.cpu_count() / args.world_size), 1),
            ),
        )

        self.data_preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor.import_from_config(
            args.model_config, 'AudioToMelSpectrogramPreprocessor', overwrite_params=dict(pad_to=0),
        )

        self.fastspeech = nemo_tts.FastSpeech.import_from_config(
            args.model_config,
            'FastSpeech',
            overwrite_params=dict(n_src_vocab=len(config.labels) + 3, pad_id=len(config.labels) + 2),
        )

        self.loss = nemo_tts.FastSpeechLoss()

    def build(self):
        data = self.data_layer()
        mel_true, _ = self.data_preprocessor(input_signal=data.audio, length=data.audio_len)
        mel_pred, dur_pred = self.fastspeech(
            text=data.text, text_pos=data.text_pos, mel_true=mel_true, dur_true=data.dur_true,
        )
        loss = self.loss(
            mel_true=mel_true, mel_pred=mel_pred, dur_true=data.dur_true, dur_pred=dur_pred, text_pos=data.text_pos,
        )

        callbacks = [
            nemo.core.SimpleLossLoggerCallback([loss], print_func=lambda x: logging.info(f'Loss: {x[0].data}'))
        ]

        return loss, callbacks


def main():
    args = parse_args()
    logging.info('Args: %s', args)

    work_dir = Path(args.work_dir) / args.id
    engine = nemo.core.NeuralModuleFactory(
        local_rank=args.local_rank,
        optimization_level=args.amp_opt_level,
        cudnn_benchmark=args.cudnn_benchmark,
        log_dir=work_dir / 'log',
        files_to_copy=[__file__],
    )
    args.world_size = engine.world_size

    yaml_loader = yaml.YAML(typ="safe")
    with open(args.model_config) as f:
        config = attrdict.AttrDict(yaml_loader.load(f))
    logging.info('Config: %s', config)

    graph = FasterSpeechGraph(args, config)

    exit(0)

    steps_per_epoch = math.ceil(len(graph.data_layer) / (args.batch_size * engine.world_size))
    total_steps = args.max_steps if args.max_steps is not None else args.num_epochs * steps_per_epoch
    loss, callbacks = graph.build()
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

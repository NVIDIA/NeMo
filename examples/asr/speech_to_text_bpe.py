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

# TODO: This is WIP and needs a lot of polishing
# python speech_to_text_bpe.py \
#         --asr_model "./experimental/configs/contextnet_128_v2.yaml" \
#         --train_dataset "./an4/train_manifest.json" \
#         --eval_dataset "./an4/test_manifest.json" \
#         --tokenizer_path "./an4/tokenizer/LibriSpeechTokenizer/librispeech_tokenizer_bpe_v1024/"
#         --gpus 2 \
#         --distributed_backend "ddp" \
#         --max_epochs 100 \
#         --optimizer adamw \
#         --lr 0.1 \
#         --opt_args weight_decay=1e-4 betas=0.9,0.999 \
#         ---warmup_ratio=0.05 --min_lr 1e-6

from argparse import ArgumentParser

import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from ruamel.yaml import YAML

from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.core.optim.lr_scheduler import CosineAnnealing
from nemo.utils.arguments import add_asr_args, add_optimizer_args, add_scheduler_args


def main(args):

    yaml = YAML(typ="safe")
    with open(args.asr_model) as f:
        model_config = yaml.load(f)

    asr_model = EncDecCTCModelBPE(
        preprocessor_params=model_config['AudioToMelSpectrogramPreprocessor'],
        encoder_params=model_config['ContextNetEncoder'],
        decoder_params=model_config['ContextNetDecoder'],
        tokenizer_path=args.tokenizer_path,
        spec_augment_params=model_config.get('SpectrogramAugmentation', None),
    )

    # Setup where your training data is
    model_config['AudioToBPELayer']['manifest_filepath'] = args.train_dataset
    model_config['AudioToBPELayer_eval']['manifest_filepath'] = args.eval_dataset
    asr_model.setup_training_data(model_config['AudioToBPELayer'])
    asr_model.setup_validation_data(model_config['AudioToBPELayer_eval'])

    # Setup optimizer and scheduler
    scheduler_args = {
        'monitor': 'val_loss',  # pytorch lightning requires this value
        'warmup_ratio': args.warmup_ratio,
        'warmup_steps': args.warmup_steps,
        'min_lr': args.min_lr,
        'last_epoch': args.last_epoch,
    }

    if args.max_steps is None:
        if args.gpus == 0:
            # training on CPU
            iters_per_batch = args.max_epochs / float(args.num_nodes * args.accumulate_grad_batches)
        else:
            iters_per_batch = args.max_epochs / float(args.gpus * args.num_nodes * args.accumulate_grad_batches)
        scheduler_args['iters_per_batch'] = iters_per_batch
    else:
        scheduler_args['max_steps'] = args.max_steps

    asr_model.setup_optimization(
        optim_params={
            'optimizer': args.optimizer,
            'lr': args.lr,
            'opt_args': args.opt_args,
            'scheduler': CosineAnnealing,
            'scheduler_args': scheduler_args,
        }
    )

    trainer = pl.Trainer.from_argparse_args(args)

    if args.experiment_name is not None and args.project_name is not None:
        logger = pl_loggers.WandbLogger(name=args.experiment_name, project=args.project_name)
        trainer.configure_logger(logger)

    trainer.fit(asr_model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = add_asr_args(parser)
    parser = add_optimizer_args(parser, optimizer='novograd', default_lr=0.01, default_opt_args={'betas': (0.95, 0.5)})
    parser = add_scheduler_args(parser)

    # Additional arguments
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer directory")
    parser.add_argument('--experiment_name', default=None, help='Name of experiment for WandB logger.')
    parser.add_argument('--project_name', default=None, help='Name of project for WandB logger.')

    args = parser.parse_args()

    main(args)

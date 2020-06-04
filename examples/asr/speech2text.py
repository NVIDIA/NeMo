# Copyright (c) 2019-, NVIDIA CORPORATION. All rights reserved.
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
import os
from argparse import ArgumentParser
from functools import partial

import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.helpers import monitor_asr_train_progress, process_evaluation_batch, process_evaluation_epoch
from nemo.utils import logging
from nemo.utils.lr_policies import CosineAnnealing


def main():
    # Usage and Command line arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--asr_model",
        type=str,
        default="QuartzNet15x5-En",
        required=True,
        help="Pass: 'QuartzNet15x5', 'QuartzNet15x5-Zh', or 'JasperNet10x5-En' to train from pre-trained models. To train from scratch pass path to modelfile ending with .yaml.",
    )
    parser.add_argument(
        "--amp_opt_level",
        default="O0",
        type=str,
        choices=["O0", "O1", "O2", "O3"],
        help="See: https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--train_dataset", type=str, required=True, default=None, help="training dataset path")
    parser.add_argument("--eval_datasets", type=str, nargs="*", help="evaluation datasets paths")
    parser.add_argument("--eval_freq", default=1000, type=int, help="Evaluation frequency")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="batch size to use for evaluation")
    parser.add_argument("--local_rank", default=None, type=int, help="node rank for distributed training")
    parser.add_argument("--stats_freq", default=25, type=int, help="frequency with which to update train stats")
    parser.add_argument("--checkpoint_dir", default=None, type=str, help="Folder where to save checkpoints")
    parser.add_argument("--checkpoint_save_freq", required=False, type=int, help="how often to checkpoint")
    parser.add_argument("--optimizer", default="novograd", type=str)
    parser.add_argument("--warmup_ratio", default=0.02, type=float, help="learning rate warmup ratio")
    parser.add_argument("--batch_size", required=True, type=int, help="train batch size per GPU")
    parser.add_argument("--num_epochs", default=5, type=int, help="number of epochs to train")
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--beta1", default=0.95, type=float)
    parser.add_argument("--beta2", default=0.5, type=float)
    parser.add_argument("--weight_decay", default=0.001, type=float)
    parser.add_argument("--iter_per_step", default=1, type=int, help="number of grad accumulations per batch")
    parser.add_argument("--wandb_exp_name", default=None, type=str)
    parser.add_argument("--wandb_project", default=None, type=str)
    parser.add_argument("--max_train_audio_len", default=16.7, type=float, help="max audio length")
    parser.add_argument("--trim_silence", default=True, type=bool, help="trim audio from silence or not")
    args = parser.parse_args()

    # Setup NeuralModuleFactory to control training
    # instantiate Neural Factory with supported backend
    nf = nemo.core.NeuralModuleFactory(
        local_rank=args.local_rank,  # This is necessary for distributed training
        optimization_level=args.amp_opt_level,  # This is necessary for mixed precision optimization
        cudnn_benchmark=True,
    )

    # Instantiate the model which we'll train
    if args.asr_model.endswith('.yaml'):
        logging.info(f"Speech2Text: Will train from scratch using config from {args.asr_model}")
        asr_model = nemo_asr.models.ASRConvCTCModel.import_from_config(args.asr_model)
    else:
        logging.info(f"Speech2Text: Will fine-tune from {args.asr_model}")
        asr_model = nemo_asr.models.ASRConvCTCModel.from_pretrained(
            model_info=args.asr_model, local_rank=args.local_rank
        )
    logging.info("\n\n")
    logging.info(f"Speech2Text: Training on {nf.world_size} GPUs.")
    logging.info(f"Training {type(asr_model)} model.")
    logging.info(f"Training CTC model with alphabet {asr_model.vocabulary}.")
    logging.info(f"Training CTC model with {asr_model.num_weights} weights.\n\n")

    train_data_layer = nemo_asr.AudioToTextDataLayer(
        manifest_filepath=args.train_dataset,
        labels=asr_model.vocabulary,
        batch_size=args.batch_size,
        trim_silence=args.trim_silence,
        max_duration=args.max_train_audio_len,
        shuffle=True,
    )
    ctc_loss = nemo_asr.CTCLossNM(num_classes=len(asr_model.vocabulary))
    greedy_decoder = nemo_asr.GreedyCTCDecoder()

    audio_signal, audio_signal_len, transcript, transcript_len = train_data_layer()
    log_probs, encoded_len = asr_model(input_signal=audio_signal, length=audio_signal_len)
    predictions = greedy_decoder(log_probs=log_probs)
    loss = ctc_loss(log_probs=log_probs, targets=transcript, input_length=encoded_len, target_length=transcript_len)

    # Callbacks which we'll be using:
    callbacks = []
    # SimpleLossLogger prints basic training stats (e.g. loss) to console
    train_callback = nemo.core.SimpleLossLoggerCallback(
        tensors=[loss, predictions, transcript, transcript_len],
        step_freq=args.stats_freq,
        print_func=partial(monitor_asr_train_progress, labels=asr_model.vocabulary),
    )
    callbacks.append(train_callback)
    if args.checkpoint_dir is not None and args.checkpoint_save_freq is not None:
        # Checkpoint callback saves checkpoints periodically
        checkpointer_callback = nemo.core.CheckpointCallback(
            folder=args.checkpoint_dir, step_freq=args.checkpoint_save_freq
        )
        callbacks.append(checkpointer_callback)

    if args.wandb_exp_name is not None and args.wandb_project is not None:
        # WandbCallback saves stats to Weights&Biases
        wandb_callback = nemo.core.WandBLogger(
            step_freq=args.stats_freq, wandb_name=args.wandb_exp_name, wandb_project=args.wandb_project, args=args
        )
        callbacks.append(wandb_callback)

    # Evaluation
    if args.eval_datasets is not None and args.eval_freq is not None:
        asr_model.eval()  # switch model to evaluation mode
        logging.info(f"Will perform evaluation every {args.eval_freq} steps.")
        for ind, eval_dataset in enumerate(args.eval_datasets):
            eval_data_layer = nemo_asr.AudioToTextDataLayer(
                manifest_filepath=eval_dataset, labels=asr_model.vocabulary, batch_size=args.eval_batch_size
            )
            audio_signal, audio_signal_len, transcript, transcript_len = eval_data_layer()
            log_probs, encoded_len = asr_model(input_signal=audio_signal, length=audio_signal_len)
            eval_predictions = greedy_decoder(log_probs=log_probs)
            eval_loss = ctc_loss(
                log_probs=log_probs, targets=transcript, input_length=encoded_len, target_length=transcript_len
            )
            tag_name = os.path.basename(eval_dataset).split(".")[0]
            eval_callback = nemo.core.EvaluatorCallback(
                eval_tensors=[eval_loss, eval_predictions, transcript, transcript_len],
                user_iter_callback=partial(process_evaluation_batch, labels=asr_model.vocabulary),
                user_epochs_done_callback=partial(process_evaluation_epoch, tag=tag_name),
                eval_step=args.eval_freq,
                wandb_name=args.wandb_exp_name,
                wandb_project=args.wandb_project,
            )
            callbacks.append(eval_callback)

    steps_in_epoch = len(train_data_layer) / (args.batch_size * args.iter_per_step * nf.world_size)
    lr_policy = CosineAnnealing(total_steps=args.num_epochs * steps_in_epoch, warmup_ratio=args.warmup_ratio)

    nf.train(
        tensors_to_optimize=[loss],
        callbacks=callbacks,
        optimizer=args.optimizer,
        optimization_params={
            "num_epochs": args.num_epochs,
            "lr": args.lr,
            "betas": (args.beta1, args.beta2),
            "weight_decay": args.weight_decay,
        },
        batches_per_step=args.iter_per_step,
        lr_policy=lr_policy,
    )


if __name__ == '__main__':
    main()

# Copyright (c) 2019 NVIDIA Corporation
import argparse
import copy
import glob
import math
import os
from datetime import datetime
from functools import partial

from ruamel.yaml import YAML

import nemo
import nemo.collections.asr as nemo_asr
import nemo.utils.argparse as nm_argparse
from nemo.collections.asr.helpers import (
    monitor_classification_training_progress,
    process_classification_evaluation_batch,
    process_classification_evaluation_epoch,
)
from nemo.utils.lr_policies import CosineAnnealing, PolynomialDecayAnnealing, PolynomialHoldDecayAnnealing

logging = nemo.logging


def parse_args():
    parser = argparse.ArgumentParser(
        parents=[nm_argparse.NemoArgParser()], description='Jasper Speech Commands', conflict_handler='resolve',
    )
    parser.set_defaults(
        checkpoint_dir=None,
        optimizer="sgd",
        batch_size=128,
        eval_batch_size=128,
        lr=0.1,
        amp_opt_level="O1",
        create_tb_writer=True,
    )

    # Overwrite default args
    parser.add_argument(
        "--max_steps", type=int, default=None, required=False, help="max number of steps to train",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=None, required=False, help="number of epochs to train",
    )
    parser.add_argument(
        "--model_config", type=str, required=True, help="model configuration file: model.yaml",
    )

    # Create new args
    parser.add_argument("--exp_name", default="Jasper_Speech_Commands", type=str)
    parser.add_argument('--min_lr', default=1e-3, type=float)
    parser.add_argument("--beta1", default=0.95, type=float)
    parser.add_argument("--beta2", default=0.5, type=float)
    parser.add_argument("--warmup_ratio", default=0.0, type=float)
    parser.add_argument("--hold_ratio", default=0.0, type=float)
    parser.add_argument(
        "--load_dir", default=None, type=str, help="directory with pre-trained checkpoint",
    )

    args = parser.parse_args()

    if args.max_steps is not None and args.num_epochs is not None:
        raise ValueError("Either max_steps or num_epochs should be provided.")
    return args


def construct_name(name, lr, batch_size, max_steps, num_epochs, wd, optimizer, iter_per_step):
    if max_steps is not None:
        return "{0}-lr_{1}-bs_{2}-s_{3}-wd_{4}-opt_{5}-ips_{6}".format(
            name, lr, batch_size, max_steps, wd, optimizer, iter_per_step
        )
    else:
        return "{0}-lr_{1}-bs_{2}-e_{3}-wd_{4}-opt_{5}-ips_{6}".format(
            name, lr, batch_size, num_epochs, wd, optimizer, iter_per_step
        )


def create_all_dags(args, neural_factory):
    yaml = YAML(typ="safe")
    with open(args.model_config) as f:
        jasper_params = yaml.load(f)

    labels = jasper_params['labels']  # Vocab of tokens
    sample_rate = jasper_params['sample_rate']

    # Calculate num_workers for dataloader
    total_cpus = os.cpu_count()
    cpu_per_traindl = max(int(total_cpus / neural_factory.world_size), 1)

    # perturb_config = jasper_params.get('perturb', None)
    train_dl_params = copy.deepcopy(jasper_params["AudioToSpeechLabelDataLayer"])
    train_dl_params.update(jasper_params["AudioToSpeechLabelDataLayer"]["train"])
    del train_dl_params["train"]
    del train_dl_params["eval"]
    # del train_dl_params["normalize_transcripts"]

    # Look for augmentations
    audio_augmentor = jasper_params.get('AudioAugmentor', None)

    data_layer = nemo_asr.AudioToSpeechLabelDataLayer(
        manifest_filepath=args.train_dataset,
        labels=labels,
        sample_rate=sample_rate,
        batch_size=args.batch_size,
        num_workers=cpu_per_traindl,
        augmentor=audio_augmentor,
        **train_dl_params,
    )

    crop_pad_augmentation = nemo_asr.CropOrPadSpectrogramAugmentation(audio_length=128)

    N = len(data_layer)
    steps_per_epoch = math.ceil(N / (args.batch_size * args.iter_per_step * args.num_gpus))
    print("Steps per epoch :", steps_per_epoch)
    logging.info('Have {0} examples to train on.'.format(N))

    data_preprocessor = nemo_asr.AudioToMFCCPreprocessor(
        sample_rate=sample_rate, **jasper_params["AudioToMFCCPreprocessor"],
    )

    spectr_augment_config = jasper_params.get('SpectrogramAugmentation', None)
    if spectr_augment_config:
        data_spectr_augmentation = nemo_asr.SpectrogramAugmentation(**spectr_augment_config)

    eval_dl_params = copy.deepcopy(jasper_params["AudioToSpeechLabelDataLayer"])
    eval_dl_params.update(jasper_params["AudioToSpeechLabelDataLayer"]["eval"])
    del eval_dl_params["train"]
    del eval_dl_params["eval"]
    data_layers_eval = []

    if args.eval_datasets:
        for eval_datasets in args.eval_datasets:
            data_layer_eval = nemo_asr.AudioToSpeechLabelDataLayer(
                manifest_filepath=eval_datasets,
                sample_rate=sample_rate,
                labels=labels,
                batch_size=args.eval_batch_size,
                num_workers=cpu_per_traindl,
                **eval_dl_params,
            )

            data_layers_eval.append(data_layer_eval)
    else:
        logging.warning("There were no val datasets passed")

    jasper_encoder = nemo_asr.JasperEncoder(**jasper_params["JasperEncoder"],)

    jasper_decoder = nemo_asr.JasperDecoderForClassification(
        feat_in=jasper_params["JasperEncoder"]["jasper"][-1]["filters"],
        num_classes=len(labels),
        **jasper_params['JasperDecoderForClassification'],
    )

    ce_loss = nemo_asr.CrossEntropyLossNM()

    logging.info('================================')
    logging.info(f"Number of parameters in encoder: {jasper_encoder.num_weights}")
    logging.info(f"Number of parameters in decoder: {jasper_decoder.num_weights}")
    logging.info(
        f"Total number of parameters in model: " f"{jasper_decoder.num_weights + jasper_encoder.num_weights}"
    )
    logging.info('================================')

    # Train DAG
    # --- Assemble Training DAG --- #
    audio_signal, audio_signal_len, commands, command_len = data_layer()

    processed_signal, processed_signal_len = data_preprocessor(input_signal=audio_signal, length=audio_signal_len)

    processed_signal, processed_signal_len = crop_pad_augmentation(
        input_signal=processed_signal, length=audio_signal_len
    )

    if spectr_augment_config:
        processed_signal = data_spectr_augmentation(input_spec=processed_signal)

    encoded, encoded_len = jasper_encoder(audio_signal=processed_signal, length=processed_signal_len)

    decoded = jasper_decoder(encoder_output=encoded)

    loss = ce_loss(logits=decoded, labels=commands)

    # Callbacks needed to print info to console and Tensorboard
    train_callback = nemo.core.SimpleLossLoggerCallback(
        # Notice that we pass in loss, predictions, and the transcript info.
        # Of course we would like to see our training loss, but we need the
        # other arguments to calculate the WER.
        tensors=[loss, decoded, commands],
        # The print_func defines what gets printed.
        print_func=partial(monitor_classification_training_progress, eval_metric=None),
        get_tb_values=lambda x: [("loss", x[0])],
        tb_writer=neural_factory.tb_writer,
    )

    chpt_callback = nemo.core.CheckpointCallback(
        folder=neural_factory.checkpoint_dir, load_from_folder=args.load_dir, step_freq=args.checkpoint_save_freq,
    )

    callbacks = [train_callback, chpt_callback]

    # assemble eval DAGs
    for i, eval_dl in enumerate(data_layers_eval):
        # --- Assemble Training DAG --- #
        test_audio_signal, test_audio_signal_len, test_commands, test_command_len = eval_dl()

        test_processed_signal, test_processed_signal_len = data_preprocessor(
            input_signal=test_audio_signal, length=test_audio_signal_len
        )

        test_processed_signal, test_processed_signal_len = crop_pad_augmentation(
            input_signal=test_processed_signal, length=test_processed_signal_len
        )

        test_encoded, test_encoded_len = jasper_encoder(
            audio_signal=test_processed_signal, length=test_processed_signal_len
        )

        test_decoded = jasper_decoder(encoder_output=test_encoded)

        test_loss = ce_loss(logits=test_decoded, labels=test_commands)

        # create corresponding eval callback
        tagname = os.path.basename(args.eval_datasets[i]).split(".")[0]
        eval_callback = nemo.core.EvaluatorCallback(
            eval_tensors=[test_loss, test_decoded, test_commands],
            user_iter_callback=partial(process_classification_evaluation_batch, top_k=1),
            user_epochs_done_callback=partial(process_classification_evaluation_epoch, eval_metric=1, tag=tagname),
            eval_step=args.eval_freq,  # How often we evaluate the model on the test set
            tb_writer=neural_factory.tb_writer,
        )

        callbacks.append(eval_callback)
    return loss, callbacks, steps_per_epoch


def main():
    args = parse_args()
    name = construct_name(
        args.exp_name,
        args.lr,
        args.batch_size,
        args.max_steps,
        args.num_epochs,
        args.weight_decay,
        args.optimizer,
        args.iter_per_step,
    )

    # time stamp
    date_time = datetime.now().strftime("%m-%d-%Y -- %H-%M-%S")

    log_dir = name
    if args.work_dir:
        log_dir = os.path.join(args.work_dir, name)

    if args.tensorboard_dir is None:
        tensorboard_dir = os.path.join(name, 'tensorboard', date_time)
    else:
        tensorboard_dir = args.tensorboard_dir

    if args.checkpoint_dir is None:
        checkpoint_dir = os.path.join(name, date_time)
    else:
        base_checkpoint_dir = args.checkpoint_dir
        if len(glob.glob(os.path.join(base_checkpoint_dir, '*.pt'))) > 0:
            checkpoint_dir = base_checkpoint_dir
        else:
            checkpoint_dir = os.path.join(args.checkpoint_dir, date_time)

    # instantiate Neural Factory with supported backend
    neural_factory = nemo.core.NeuralModuleFactory(
        backend=nemo.core.Backend.PyTorch,
        local_rank=args.local_rank,
        optimization_level=args.amp_opt_level,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        create_tb_writer=args.create_tb_writer,
        files_to_copy=[args.model_config, __file__],
        cudnn_benchmark=args.cudnn_benchmark,
        tensorboard_dir=tensorboard_dir,
    )
    args.num_gpus = neural_factory.world_size

    checkpoint_dir = neural_factory.checkpoint_dir
    if args.local_rank is not None:
        logging.info('Doing ALL GPU')

    # build dags
    train_loss, callbacks, steps_per_epoch = create_all_dags(args, neural_factory)

    yaml = YAML(typ="safe")
    with open(args.model_config) as f:
        jasper_params = yaml.load(f)

    lr_schedule = jasper_params.get('lr_schedule', 'CosineAnnealing')

    if lr_schedule == 'CosineAnnealing':
        lr_policy = CosineAnnealing(
            total_steps=args.max_steps if args.max_steps is not None else args.num_epochs * steps_per_epoch,
            warmup_ratio=args.warmup_ratio,
            min_lr=args.min_lr,
        )
    elif lr_schedule == 'PolynomialDecayAnnealing':
        lr_policy = PolynomialDecayAnnealing(
            total_steps=args.max_steps if args.max_steps is not None else args.num_epochs * steps_per_epoch,
            warmup_ratio=args.warmup_ratio,
            min_lr=args.min_lr,
            power=2.0,
        )
    elif lr_schedule == 'PolynomialHoldDecayAnnealing':
        lr_policy = PolynomialHoldDecayAnnealing(
            total_steps=args.max_steps if args.max_steps is not None else args.num_epochs * steps_per_epoch,
            warmup_ratio=args.warmup_ratio,
            hold_ratio=args.hold_ratio,
            min_lr=args.min_lr,
            power=2.0,
        )
    else:
        raise ValueError("LR schedule is invalid !")

    logging.info(f"Using `{lr_policy}` Learning Rate Scheduler")

    # train model
    neural_factory.train(
        tensors_to_optimize=[train_loss],
        callbacks=callbacks,
        lr_policy=lr_policy,
        optimizer=args.optimizer,
        optimization_params={
            "num_epochs": args.num_epochs,
            "max_steps": args.max_steps,
            "lr": args.lr,
            "momentum": 0.95,
            "betas": (args.beta1, args.beta2),
            "weight_decay": args.weight_decay,
            "grad_norm_clip": None,
        },
        batches_per_step=args.iter_per_step,
    )


if __name__ == '__main__':
    main()

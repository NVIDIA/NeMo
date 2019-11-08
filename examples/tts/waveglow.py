# Copyright (c) 2019 NVIDIA Corporation
import argparse
import copy
from functools import partial
import os

from ruamel.yaml import YAML

import nemo
import nemo.utils.argparse as nm_argparse
import nemo_asr
import nemo_tts
from nemo_tts import (waveglow_log_to_tb_func,
                      waveglow_process_eval_batch,
                      # waveglow_process_final_eval,
                      waveglow_eval_log_to_tb_func)


def parse_args():
    parser = argparse.ArgumentParser(
        parents=[nm_argparse.NemoArgParser()],
        description='Waveglow',
        conflict_handler='resolve')
    parser.set_defaults(
        checkpoint_dir=None,
        optimizer="adam",
        batch_size=12,
        eval_batch_size=12,
        lr=0.0001,
        amp_opt_level="O1",
        create_tb_writer=True,
        lr_policy=None,
        weight_decay=1e-6
    )

    # Overwrite default args
    parser.add_argument("--max_steps", type=int, default=None, required=False,
                        help="max number of steps to train")
    parser.add_argument("--num_epochs", type=int, default=None, required=False,
                        help="number of epochs to train")
    parser.add_argument("--model_config", type=str, required=True,
                        help="model configuration file: model.yaml")
    parser.add_argument("--grad_norm_clip", type=float, default=65504.0,
                        help="gradient clipping")

    # Create new args
    parser.add_argument("--exp_name", default="Waveglow", type=str)

    args = parser.parse_args()

    if args.lr_policy:
        raise NotImplementedError("Waveglow does not support lr policy arg")
    if args.max_steps is not None and args.num_epochs is not None:
        raise ValueError("Either max_steps or num_epochs should be provided.")
    if args.eval_freq % 25 != 0:
        raise ValueError("eval_freq should be a multiple of 25.")

    exp_directory = [f"{args.exp_name}-lr_{args.lr}-bs_{args.batch_size}",
                     "",
                     (f"-wd_{args.weight_decay}-opt_{args.optimizer}"
                      f"-ips_{args.iter_per_step}")]
    if args.max_steps:
        exp_directory[1] = f"-s_{args.max_steps}"
    elif args.num_epochs:
        exp_directory[1] = f"-e_{args.num_epochs}"
    else:
        raise ValueError("Both max_steps and num_epochs were None.")
    return args, "".join(exp_directory)


def create_NMs(waveglow_params, logger=None):
    data_preprocessor = nemo_asr.AudioPreprocessing(
        **waveglow_params["AudioPreprocessing"])
    waveglow = nemo_tts.WaveGlowNM(**waveglow_params["WaveGlowNM"])
    waveglow_loss = nemo_tts.WaveGlowLoss()

    if logger:
        logger.info('================================')
        logger.info(f"Total number of parameters: {waveglow.num_weights}")
        logger.info('================================')
    return (data_preprocessor, waveglow, waveglow_loss)


def create_train_dag(neural_factory,
                     neural_modules,
                     waveglow_params,
                     train_dataset,
                     batch_size,
                     checkpoint_save_freq,
                     cpu_per_dl=1):
    data_preprocessor, waveglow, waveglow_loss = neural_modules

    train_dl_params = copy.deepcopy(waveglow_params["AudioDataLayer"])
    train_dl_params.update(waveglow_params["AudioDataLayer"]["train"])
    del train_dl_params["train"]
    del train_dl_params["eval"]

    data_layer = nemo_tts.AudioDataLayer(
        manifest_filepath=train_dataset,
        batch_size=batch_size,
        num_workers=cpu_per_dl,
        **train_dl_params,
    )

    N = len(data_layer)
    steps_per_epoch = int(N / (batch_size * neural_factory.world_size))
    neural_factory.logger.info('Have {0} examples to train on.'.format(N))

    # Train DAG
    audio, audio_len, = data_layer()
    spec_target, spec_target_len = data_preprocessor(
        input_signal=audio,
        length=audio_len)

    audio_pred, log_s_list, log_det_W_list = waveglow(
        mel_spectrogram=spec_target, audio=audio)
    loss_t = waveglow_loss(
        audio_pred=audio_pred,
        log_s_list=log_s_list,
        log_det_W_list=log_det_W_list)

    # Callbacks needed to print info to console and Tensorboard
    train_callback = nemo.core.SimpleLossLoggerCallback(
        tensors=[loss_t, audio_pred, spec_target, spec_target_len],
        print_func=lambda x: print(f"Loss: {x[0].data}"),
        log_to_tb_func=partial(
            waveglow_log_to_tb_func,
            log_images=False),
        tb_writer=neural_factory.tb_writer,
    )

    chpt_callback = nemo.core.CheckpointCallback(
        folder=neural_factory.checkpoint_dir,
        step_freq=checkpoint_save_freq)

    callbacks = [train_callback, chpt_callback]
    return loss_t, callbacks, steps_per_epoch


def create_eval_dags(neural_factory,
                     neural_modules,
                     waveglow_params,
                     eval_datasets,
                     eval_batch_size,
                     eval_freq,
                     cpu_per_dl=1):
    data_preprocessor, waveglow, _ = neural_modules

    eval_dl_params = copy.deepcopy(waveglow_params["AudioDataLayer"])
    eval_dl_params.update(waveglow_params["AudioDataLayer"]["eval"])
    del eval_dl_params["train"]
    del eval_dl_params["eval"]

    callbacks = []
    # assemble eval DAGs
    for eval_dataset in eval_datasets:
        data_layer_eval = nemo_tts.AudioDataLayer(
            manifest_filepath=eval_dataset,
            batch_size=eval_batch_size,
            num_workers=cpu_per_dl,
            **eval_dl_params,
        )

        audio, audio_len, = data_layer_eval()
        spec_target, spec_target_len = data_preprocessor(
            input_signal=audio,
            length=audio_len)

        audio_pred, log_s_list, log_det_W_list = waveglow(
            mel_spectrogram=spec_target, audio=audio)

        # create corresponding eval callback
        tagname = os.path.basename(eval_dataset).split(".")[0]
        eval_callback = nemo.core.EvaluatorCallback(
            eval_tensors=[audio_pred, spec_target, spec_target_len],
            user_iter_callback=waveglow_process_eval_batch,
            user_epochs_done_callback=lambda x: x,
            tb_writer_func=partial(
                waveglow_eval_log_to_tb_func,
                tag=tagname,
                mel_fb=data_preprocessor.filter_banks),
            eval_step=eval_freq,
            tb_writer=neural_factory.tb_writer)

        callbacks.append(eval_callback)
    return callbacks


def create_all_dags(neural_factory,
                    neural_modules,
                    waveglow_params,
                    train_dataset,
                    batch_size,
                    checkpoint_save_freq,
                    eval_datasets=None,
                    eval_batch_size=None,
                    eval_freq=None):
    # Calculate num_workers for dataloader
    cpu_per_dl = max(int(os.cpu_count() / neural_factory.world_size), 1)

    training_loss, training_callbacks, steps_per_epoch = create_train_dag(
        neural_factory=neural_factory,
        neural_modules=neural_modules,
        waveglow_params=waveglow_params,
        train_dataset=train_dataset,
        batch_size=batch_size,
        checkpoint_save_freq=checkpoint_save_freq,
        cpu_per_dl=cpu_per_dl)

    eval_callbacks = []
    if eval_datasets:
        eval_callbacks = create_eval_dags(
            neural_factory=neural_factory,
            neural_modules=neural_modules,
            waveglow_params=waveglow_params,
            eval_datasets=eval_datasets,
            eval_batch_size=eval_batch_size,
            eval_freq=eval_freq,
            cpu_per_dl=cpu_per_dl)
    else:
        neural_factory.logger.info("There were no val datasets passed")

    callbacks = training_callbacks + eval_callbacks
    return training_loss, callbacks, steps_per_epoch


def main():
    args, name = parse_args()

    log_dir = None
    if args.work_dir:
        log_dir = os.path.join(args.work_dir, name)

    # instantiate Neural Factory with supported backend
    neural_factory = nemo.core.NeuralModuleFactory(
        backend=nemo.core.Backend.PyTorch,
        local_rank=args.local_rank,
        optimization_level=args.amp_opt_level,
        log_dir=log_dir,
        checkpoint_dir=args.checkpoint_dir,
        create_tb_writer=args.create_tb_writer,
        files_to_copy=[args.model_config, __file__],
        cudnn_benchmark=args.cudnn_benchmark,
        tensorboard_dir=args.tensorboard_dir)

    if args.local_rank is not None:
        neural_factory.logger.info('Doing ALL GPU')

    yaml = YAML(typ="safe")
    with open(args.model_config) as file:
        waveglow_params = yaml.load(file)
    # instantiate neural modules
    neural_modules = create_NMs(waveglow_params, neural_factory.logger)

    # build dags
    train_loss, callbacks, steps_per_epoch = create_all_dags(
        neural_factory=neural_factory,
        neural_modules=neural_modules,
        waveglow_params=waveglow_params,
        train_dataset=args.train_dataset,
        batch_size=args.batch_size,
        checkpoint_save_freq=args.checkpoint_save_freq,
        eval_datasets=args.eval_datasets,
        eval_batch_size=args.eval_batch_size,
        eval_freq=args.eval_freq)

    # train model
    neural_factory.train(
        tensors_to_optimize=[train_loss],
        callbacks=callbacks,
        optimizer=args.optimizer,
        optimization_params={
            "num_epochs": args.num_epochs,
            "max_steps": args.max_steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_norm_clip": args.grad_norm_clip},
        batches_per_step=args.iter_per_step)


if __name__ == '__main__':
    main()

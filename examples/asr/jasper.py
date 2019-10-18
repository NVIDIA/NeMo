# Copyright (c) 2019 NVIDIA Corporation
import argparse
import copy
from functools import partial
import os

from ruamel.yaml import YAML

import nemo
from nemo.utils.lr_policies import CosineAnnealing
import nemo.utils.argparse as nm_argparse
import nemo_asr
from nemo_asr.helpers import monitor_asr_train_progress, \
    process_evaluation_batch, process_evaluation_epoch


def parse_args():
    parser = argparse.ArgumentParser(
        parents=[nm_argparse.NemoArgParser()],
        description='Jasper',
        conflict_handler='resolve')
    parser.set_defaults(
        checkpoint_dir=None,
        optimizer="novograd",
        batch_size=64,
        eval_batch_size=64,
        lr=0.02,
        amp_opt_level="O1",
        create_tb_writer=True
    )

    # Overwrite default args
    parser.add_argument("--max_steps", type=int, default=None, required=False,
                        help="max number of steps to train")
    parser.add_argument("--num_epochs", type=int, default=None, required=False,
                        help="number of epochs to train")
    parser.add_argument("--model_config", type=str, required=True,
                        help="model configuration file: model.yaml")

    # Create new args
    parser.add_argument("--exp_name", default="Jasper", type=str)
    parser.add_argument("--beta1", default=0.95, type=float)
    parser.add_argument("--beta2", default=0.25, type=float)
    parser.add_argument("--warmup_steps", default=0, type=int)

    args = parser.parse_args()

    if args.max_steps is not None and args.num_epochs is not None:
        raise ValueError("Either max_steps or num_epochs should be provided.")
    return args


def construct_name(name, lr, batch_size, max_steps, wd, optimizer,
                   iter_per_step):
    return ("{0}-lr_{1}-bs_{2}-s_{3}-wd_{4}-opt_{5}-ips_{6}".format(
        name, lr,
        batch_size,
        max_steps,
        wd,
        optimizer,
        iter_per_step))


def create_all_dags(args, neural_factory):
    logger = neural_factory.logger
    yaml = YAML(typ="safe")
    with open(args.model_config) as f:
        jasper_params = yaml.load(f)
    vocab = jasper_params['labels']
    sample_rate = jasper_params['sample_rate']

    # Calculate num_workers for dataloader
    total_cpus = os.cpu_count()
    cpu_per_traindl = max(int(total_cpus / neural_factory.world_size), 1)

    # perturb_config = jasper_params.get('perturb', None)
    train_dl_params = copy.deepcopy(jasper_params["AudioToTextDataLayer"])
    train_dl_params.update(jasper_params["AudioToTextDataLayer"]["train"])
    del train_dl_params["train"]
    del train_dl_params["eval"]
    # del train_dl_params["normalize_transcripts"]

    data_layer = nemo_asr.AudioToTextDataLayer(
        manifest_filepath=args.train_dataset,
        sample_rate=sample_rate,
        labels=vocab,
        batch_size=args.batch_size,
        num_workers=cpu_per_traindl,
        **train_dl_params,
        # normalize_transcripts=False
    )

    N = len(data_layer)
    steps_per_epoch = int(N / (args.batch_size * args.num_gpus))
    logger.info('Have {0} examples to train on.'.format(N))

    data_preprocessor = nemo_asr.AudioPreprocessing(
        sample_rate=sample_rate,
        **jasper_params["AudioPreprocessing"])

    multiply_batch_config = jasper_params.get('MultiplyBatch', None)
    if multiply_batch_config:
        multiply_batch = nemo_asr.MultiplyBatch(**multiply_batch_config)

    spectr_augment_config = jasper_params.get('SpectrogramAugmentation', None)
    if spectr_augment_config:
        data_spectr_augmentation = nemo_asr.SpectrogramAugmentation(
            **spectr_augment_config)

    eval_dl_params = copy.deepcopy(jasper_params["AudioToTextDataLayer"])
    eval_dl_params.update(jasper_params["AudioToTextDataLayer"]["eval"])
    del eval_dl_params["train"]
    del eval_dl_params["eval"]
    data_layers_eval = []

    if args.eval_datasets:
        for eval_datasets in args.eval_datasets:
            data_layer_eval = nemo_asr.AudioToTextDataLayer(
                manifest_filepath=eval_datasets,
                sample_rate=sample_rate,
                labels=vocab,
                batch_size=args.eval_batch_size,
                num_workers=cpu_per_traindl,
                **eval_dl_params,
            )

            data_layers_eval.append(data_layer_eval)
    else:
        neural_factory.logger.info("There were no val datasets passed")

    jasper_encoder = nemo_asr.JasperEncoder(
        feat_in=jasper_params["AudioPreprocessing"]["features"],
        **jasper_params["JasperEncoder"])

    jasper_decoder = nemo_asr.JasperDecoderForCTC(
        feat_in=jasper_params["JasperEncoder"]["jasper"][-1]["filters"],
        num_classes=len(vocab),
        factory=neural_factory)

    ctc_loss = nemo_asr.CTCLossNM(
        num_classes=len(vocab))

    greedy_decoder = nemo_asr.GreedyCTCDecoder()

    logger.info('================================')
    logger.info(
        f"Number of parameters in encoder: {jasper_encoder.num_weights}")
    logger.info(
        f"Number of parameters in decoder: {jasper_decoder.num_weights}")
    logger.info(
        f"Total number of parameters in decoder: "
        f"{jasper_decoder.num_weights + jasper_encoder.num_weights}")
    logger.info('================================')

    # Train DAG
    audio_signal_t, a_sig_length_t, \
        transcript_t, transcript_len_t = data_layer()
    processed_signal_t, p_length_t = data_preprocessor(
        input_signal=audio_signal_t,
        length=a_sig_length_t)

    if multiply_batch_config:
        processed_signal_t, p_length_t, transcript_t, transcript_len_t = \
            multiply_batch(
                in_x=processed_signal_t, in_x_len=p_length_t,
                in_y=transcript_t,
                in_y_len=transcript_len_t)

    if spectr_augment_config:
        processed_signal_t = data_spectr_augmentation(
            input_spec=processed_signal_t)

    encoded_t, encoded_len_t = jasper_encoder(
        audio_signal=processed_signal_t,
        length=p_length_t)
    log_probs_t = jasper_decoder(encoder_output=encoded_t)
    predictions_t = greedy_decoder(log_probs=log_probs_t)
    loss_t = ctc_loss(
        log_probs=log_probs_t,
        targets=transcript_t,
        input_length=encoded_len_t,
        target_length=transcript_len_t)

    # Callbacks needed to print info to console and Tensorboard
    train_callback = nemo.core.SimpleLossLoggerCallback(
        tensors=[loss_t, predictions_t, transcript_t, transcript_len_t],
        print_func=partial(
            monitor_asr_train_progress,
            labels=vocab,
            logger=logger),
        get_tb_values=lambda x: [("loss", x[0])],
        tb_writer=neural_factory.tb_writer,
    )

    chpt_callback = nemo.core.CheckpointCallback(
        folder=neural_factory.checkpoint_dir,
        step_freq=args.checkpoint_save_freq)

    callbacks = [train_callback, chpt_callback]

    # assemble eval DAGs
    for i, eval_dl in enumerate(data_layers_eval):
        audio_signal_e, a_sig_length_e, transcript_e, transcript_len_e = \
            eval_dl()
        processed_signal_e, p_length_e = data_preprocessor(
            input_signal=audio_signal_e,
            length=a_sig_length_e)
        encoded_e, encoded_len_e = jasper_encoder(
            audio_signal=processed_signal_e,
            length=p_length_e)
        log_probs_e = jasper_decoder(encoder_output=encoded_e)
        predictions_e = greedy_decoder(log_probs=log_probs_e)
        loss_e = ctc_loss(
            log_probs=log_probs_e,
            targets=transcript_e,
            input_length=encoded_len_e,
            target_length=transcript_len_e)

        # create corresponding eval callback
        tagname = os.path.basename(args.eval_datasets[i]).split(".")[0]
        eval_callback = nemo.core.EvaluatorCallback(
            eval_tensors=[loss_e, predictions_e,
                          transcript_e, transcript_len_e],
            user_iter_callback=partial(
                process_evaluation_batch,
                labels=vocab),
            user_epochs_done_callback=partial(
                process_evaluation_epoch,
                tag=tagname,
                logger=logger),
            eval_step=args.eval_freq,
            tb_writer=neural_factory.tb_writer)

        callbacks.append(eval_callback)
    return loss_t, callbacks, steps_per_epoch


def main():

    args = parse_args()
    name = construct_name(
        args.exp_name,
        args.lr,
        args.batch_size,
        args.max_steps,
        args.weight_decay,
        args.optimizer,
        args.iter_per_step)
    log_dir = name
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
    args.num_gpus = neural_factory.world_size

    logger = neural_factory.logger
    checkpoint_dir = neural_factory.checkpoint_dir
    if args.local_rank is not None:
        logger.info('Doing ALL GPU')

    # build dags
    train_loss, callbacks, steps_per_epoch = \
        create_all_dags(args, neural_factory)

    # train model
    neural_factory.train(
        tensors_to_optimize=[train_loss],
        callbacks=callbacks,
        lr_policy=CosineAnnealing(args.max_steps,
                                  warmup_steps=args.warmup_steps),
        optimizer=args.optimizer,
        optimization_params={
            "num_epochs": args.num_epochs,
            "max_steps": args.max_steps,
            "lr": args.lr,
            "betas": (
                args.beta1,
                args.beta2),
            "weight_decay": args.weight_decay,
            "grad_norm_clip": None},
        batches_per_step=args.iter_per_step)


if __name__ == '__main__':
    main()

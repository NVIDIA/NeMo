# Copyright (c) 2019 NVIDIA Corporation
import argparse
import datetime
import os
import random
from collections import namedtuple
from functools import partial
from pprint import pformat

import numpy as np
import torch
from ruamel.yaml import YAML
from tensorboardX import SummaryWriter

import nemo
from nemo.core.callbacks import ValueSetterCallback
from nemo.utils.lr_policies import SquareAnnealing
import nemo.utils.argparse as nm_argparse
from nemo.utils.misc import Config
import nemo_asr
from nemo.collections.asr.las.helpers import process_evaluation_batch, \
    process_evaluation_epoch

# Special symbols for seq2seq with cross-entropy criterion and aux CTC loss
SS = namedtuple('SS', 'id char name')
_ = 0
sss = [
    SS(_ + 0, '#', 'pad'), SS(_ + 1, '<', 'bos'), SS(_ + 2, '>', 'eos'),  # CE
    # SS(_ + 3, '@', 'ctc_blank')  # CTC
]


def parse_args():
    parser = argparse.ArgumentParser(
        parents=[nm_argparse.NemoArgParser()],
        description='GarNet RnnLM',
        conflict_handler='resolve')
    parser.set_defaults(
        checkpoint_dir=None,
        optimizer="novograd",
        batch_size=32,
        eval_batch_size=32,
        num_epochs=25,
        weight_decay=1e-5,
        lr=0.02,
        amp_opt_level="O1",
        create_tb_writer=True
    )

    # Overwrite default args
    parser.add_argument("--num_epochs", type=int, default=None, required=True,
                        help="number of epochs to train. You should specify"
                             "either num_epochs or max_steps")
    parser.add_argument("--model_config", type=str, required=True,
                        help="model configuration file: model.yaml")
    parser.add_argument("--eval_datasets", type=str, required=True,
                        help="validation dataset path")

    # Create new args
    parser.add_argument("--exp_name", default="GarNet", type=str)
    parser.add_argument("--random_seed", default=0, type=float)

    args = parser.parse_args()
    if args.max_steps is not None:
        raise ValueError("GarNet RNNLM uses num_epochs instead of max_steps")

    return args


def parse_cfg(args):
    yaml = YAML(typ="safe")
    with open(args.model_config) as f:
        cfg = Config(yaml.load(f))

    # Transforming labels with special symbols
    labels = cfg['target']['labels']
    labels = [ss.char for ss in sss] + labels
    cfg['target']['labels'] = labels
    for ss in sss:
        cfg['target'][f'{ss.name}_id'] = ss.id

    cfg['optimization']['batch_size'] = args.batch_size
    cfg['inference']['batch_size'] = args.eval_batch_size

    cfg['optimization']['optimizer'] = args.optimizer
    cfg['optimization']['params']['lr'] = args.lr
    cfg['optimization']['params']['weight_decay'] = args.weight_decay

    return cfg


def create_dag(args, cfg, num_gpus):
    # Defining nodes
    data = nemo_asr.TranscriptDataLayer(
        path=args.train_dataset,
        labels=cfg['target']['labels'],
        eos_id=cfg['target']['eos_id'],
        pad_id=cfg['target']['pad_id'],
        batch_size=cfg['optimization']['batch_size'],
        drop_last=True,
    )
    data_eval = nemo_asr.AudioToTextDataLayer(
        manifest_filepath=args.eval_datasets,
        labels=cfg['target']['labels'],
        eos_id=cfg['target']['eos_id'],
        batch_size=cfg['inference']['batch_size'],
        load_audio=False
    )
    decoder = nemo.backends.pytorch.DecoderRNN(
        voc_size=len(cfg['target']['labels']),
        bos_id=cfg['target']['bos_id'],
        **cfg['DecoderRNN']
    )
    num_data = len(data)
    batch_size = cfg['optimization']['batch_size']
    num_epochs = cfg['optimization']['params']['num_epochs']
    steps_per_epoch = int(num_data / (batch_size))
    total_steps = num_epochs * steps_per_epoch
    vsc = ValueSetterCallback
    tf_callback = ValueSetterCallback(
        decoder, 'teacher_forcing',
        policies=[
            vsc.Policy(vsc.Method.Const(1.0), start=0.0, end=1.0),
        ],
        total_steps=total_steps
    )
    seq_loss = nemo.backends.pytorch.SequenceLoss(
        pad_id=cfg['target']['pad_id'],
        smoothing_coef=cfg['optimization']['smoothing_coef']
    )
    saver_callback = nemo.core.ModuleSaverCallback(
        save_modules_list=[decoder],
        folder=args.checkpoint_dir,
        step_freq=args.checkpoint_save_freq
    )

    # Creating DAG
    texts, _ = data()
    log_probs, _ = decoder(
        targets=texts
    )
    train_loss = seq_loss(
        log_probs=log_probs,
        targets=texts
    )
    evals = []
    _, _, texts, _ = data_eval()
    log_probs, _ = decoder(
        targets=texts
    )
    eval_loss = seq_loss(
        log_probs=log_probs,
        targets=texts
    )
    evals.append((args.eval_datasets,
                  (eval_loss, log_probs, texts)))

    # Update config
    cfg['num_params'] = {'decoder': decoder.num_weights}
    cfg['num_params']['total'] = sum(cfg['num_params'].values())
    cfg['input']['train'] = {'num_data': num_data}
    cfg['optimization']['steps_per_epoch'] = steps_per_epoch
    cfg['optimization']['total_steps'] = total_steps

    return (train_loss, evals), cfg, [tf_callback, saver_callback]


def construct_name(args, cfg):
    name = '{}/{}_{}_{}'.format(
        cfg['model'],
        args.exp_name,
        'bs' + str(cfg['optimization']['batch_size']),
        'epochs' + str(cfg['optimization']['params']['num_epochs']),
    )
    if args.work_dir:
        name = os.path.join(args.work_dir, name)
    return name


def main():
    # Parse args
    args = parse_args()
    cfg = parse_cfg(args)
    name = construct_name(args, cfg)

    # instantiate Neural Factory with supported backend
    neural_factory = nemo.core.NeuralModuleFactory(
        backend=nemo.core.Backend.PyTorch,
        local_rank=args.local_rank,
        optimization_level=args.amp_opt_level,
        log_dir=name,
        checkpoint_dir=args.checkpoint_dir,
        create_tb_writer=args.create_tb_writer,
        files_to_copy=[args.model_config, __file__],
        cudnn_benchmark=args.cudnn_benchmark,
        tensorboard_dir=args.tensorboard_dir)

    logger = neural_factory.logger
    tb_writer = neural_factory.tb_writer
    args.checkpoint_dir = neural_factory.checkpoint_dir

    logger.info(f'Name:\n{name}')
    logger.info(f'Args to be passed to job #{args.local_rank}:')
    logger.info(pformat(vars(args)))

    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        logger.info(f'Using seed {args.random_seed}')

    # Defining computational graph
    (train_loss, evals), cfg, dag_callbacks = create_dag(
        args, cfg, neural_factory.world_size)
    logger.info('Config:')
    logger.info(pformat(cfg))

    num_data = cfg['input']['train']['num_data']
    steps_per_epoch = cfg['optimization']['steps_per_epoch']
    total_steps = cfg['optimization']['total_steps']
    logger.info(f'Num data: {num_data}\n'
                f'Steps per epoch: {steps_per_epoch}\n'
                f'Total steps: {total_steps}')

    dag_callbacks[0].tb_writer = tb_writer

    # Callbacks
    train_callback = nemo.core.SimpleLossLoggerCallback(
        tensors=[train_loss],
        print_func=lambda x: logger.info(f"Loss: {x[0].item()}"),
        get_tb_values=lambda x: [("loss", x[0])],
        tb_writer=tb_writer
    )
    log_callbacks = [train_callback]
    target = cfg['target']
    labels = target['labels']
    specials = {f'{ss.name}_id': target[f'{ss.name}_id'] for ss in sss}
    for name, tensors in evals:
        eval_callback = nemo.core.EvaluatorCallback(
            # TODO: Should be fixed soon, so we don't need to pass exactly list
            eval_tensors=list(tensors),
            user_iter_callback=partial(
                process_evaluation_batch,
                labels=labels,
                specials=specials,
                write_attn=False
            ),
            user_epochs_done_callback=partial(
                process_evaluation_epoch,
                tag=os.path.basename(name),
                logger=logger
            ),
            eval_step=args.eval_freq,
            tb_writer=tb_writer
        )
        log_callbacks.append(eval_callback)
    # noinspection PyTypeChecker
    callbacks = log_callbacks + dag_callbacks

    # Optimize
    neural_factory.train(
        tensors_to_optimize=[train_loss],
        callbacks=callbacks,
        lr_policy=SquareAnnealing(
            cfg['optimization']['total_steps'],
            min_lr=cfg['optimization']['min_lr'],
            warmup_steps=(
                    cfg['optimization']['warmup_epochs']
                    * cfg['optimization']['steps_per_epoch']
            )
        ),
        optimizer=cfg['optimization']['optimizer'],
        optimization_params=cfg['optimization']['params'],
        batches_per_step=args.iter_per_step
    )


if __name__ == '__main__':
    main()

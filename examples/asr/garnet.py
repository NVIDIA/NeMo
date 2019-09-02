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
from nemo.core.callbacks import ValueSetterCallback, UnfreezeCallback
import nemo.utils.argparse as nm_argparse
from nemo.utils.lr_policies import SquareAnnealing
from nemo.utils.misc import Config
import nemo_asr
from nemo_asr.las.helpers import process_evaluation_batch, \
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
        description='GarNet',
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

    # Create new args
    parser.add_argument("--exp_name", default="GarNet", type=str)
    parser.add_argument("--random_seed", default=0, type=float)
    parser.add_argument('--encoder_checkpoint', default=None, type=str)
    parser.add_argument('--decoder_checkpoint', default=None, type=str)

    args = parser.parse_args()
    if args.max_steps is not None:
        raise ValueError("GarNet uses num_epochs instead of max_steps")

    return args


def parse_cfg(args):
    yaml = YAML(typ="safe")
    with open(args.model_config) as f:
        cfg = Config(yaml.load(f))

    # Transforming labels with special symbols
    labels = cfg['target']['labels']
    labels = [ss.char for ss in sss] + labels
    cfg['target']['labels'] = labels
    cfg['target']['max_len'] += 1  # EOS symbol
    for ss in sss:
        cfg['target'][f'{ss.name}_id'] = ss.id

    cfg['optimization']['batch_size'] = args.batch_size
    # Calculating real inference batch_size
    inference_batch_size = int(
        args.eval_batch_size / cfg['inference']['beam_size']
    )
    assert inference_batch_size >= 1
    cfg['inference']['batch_size'] = inference_batch_size

    cfg['optimization']['optimizer'] = args.optimizer
    cfg['optimization']['params']['lr'] = args.lr
    cfg['optimization']['params']['weight_decay'] = args.weight_decay

    return cfg


def create_dag(args, cfg, logger, num_gpus):

    # Defining nodes
    data = nemo_asr.AudioToTextDataLayer(
        manifest_filepath=args.train_dataset,
        labels=cfg['target']['labels'],
        batch_size=cfg['optimization']['batch_size'],
        eos_id=cfg['target']['eos_id'],
        **cfg['AudioToTextDataLayer']['train']
    )
    data_evals = []
    if args.eval_datasets:
        for val_path in args.eval_datasets:
            data_evals.append(nemo_asr.AudioToTextDataLayer(
                manifest_filepath=val_path,
                labels=cfg['target']['labels'],
                batch_size=cfg['inference']['batch_size'],
                eos_id=cfg['target']['eos_id'],
                **cfg['AudioToTextDataLayer']['eval']
            ))
    else:
        logger.info("There were no val datasets passed")
    data_preprocessor = nemo_asr.AudioPreprocessing(
        **cfg['AudioPreprocessing']
    )
    data_augmentation = nemo_asr.SpectrogramAugmentation(
        **cfg['SpectrogramAugmentation']
    )
    encoder = nemo_asr.JasperEncoder(
        feat_in=cfg["AudioPreprocessing"]["features"],
        **cfg['JasperEncoder']
    )
    if args.encoder_checkpoint is not None \
            and os.path.exists(args.encoder_checkpoint):
        if cfg['JasperEncoder']['load']:
            encoder.restore_from(args.encoder_checkpoint, args.local_rank)
            logger.info(f'Loaded weights for encoder'
                        f' from {args.encoder_checkpoint}')
        if cfg['JasperEncoder']['freeze']:
            encoder.freeze()
            logger.info(f'Freeze encoder weights')
    connector = nemo_asr.JasperRNNConnector(
        in_channels=cfg['JasperEncoder']['jasper'][-1]['filters'],
        out_channels=cfg['DecoderRNN']['hidden_size']
    )
    decoder = nemo.backends.pytorch.DecoderRNN(
        voc_size=len(cfg['target']['labels']),
        bos_id=cfg['target']['bos_id'],
        **cfg['DecoderRNN']
    )
    if args.decoder_checkpoint is not None \
            and os.path.exists(args.decoder_checkpoint):
        if cfg['DecoderRNN']['load']:
            decoder.restore_from(args.decoder_checkpoint, args.local_rank)
            logger.info(f'Loaded weights for decoder'
                        f' from {args.decoder_checkpoint}')
        if cfg['DecoderRNN']['freeze']:
            decoder.freeze()
            logger.info(f'Freeze decoder weights')
            if cfg['decoder']['unfreeze_attn']:
                for name, param in decoder.attention.named_parameters():
                    param.requires_grad = True
                logger.info(f'Unfreeze decoder attn weights')
    num_data = len(data)
    batch_size = cfg['optimization']['batch_size']
    num_epochs = cfg['optimization']['params']['num_epochs']
    steps_per_epoch = int(num_data / (batch_size * num_gpus))
    total_steps = num_epochs * steps_per_epoch
    vsc = ValueSetterCallback
    tf_callback = ValueSetterCallback(
        decoder, 'teacher_forcing',
        policies=[
            vsc.Policy(vsc.Method.Const(1.0), start=0.0, end=1.0)
        ],
        total_steps=total_steps
    )
    seq_loss = nemo.backends.pytorch.SequenceLoss(
        pad_id=cfg['target']['pad_id'],
        smoothing_coef=cfg['optimization']['smoothing_coef'],
        sample_wise=cfg['optimization']['sample_wise']
    )
    se_callback = ValueSetterCallback(
        seq_loss, 'smoothing_coef',
        policies=[
            vsc.Policy(
                vsc.Method.Const(seq_loss.smoothing_coef),
                start=0.0, end=1.0
            ),
        ],
        total_steps=total_steps
    )
    beam_search = nemo.backends.pytorch.BeamSearch(
        decoder=decoder,
        pad_id=cfg['target']['pad_id'],
        bos_id=cfg['target']['bos_id'],
        eos_id=cfg['target']['eos_id'],
        max_len=cfg['target']['max_len'],
        beam_size=cfg['inference']['beam_size']
    )
    uf_callback = UnfreezeCallback(
        [encoder, decoder],
        start_epoch=cfg['optimization']['start_unfreeze']
    )
    saver_callback = nemo.core.ModuleSaverCallback(
        save_modules_list=[encoder, connector, decoder],
        folder=args.checkpoint_dir,
        step_freq=args.eval_freq
    )

    # Creating DAG
    audios, audio_lens, transcripts, _ = data()
    processed_audios, processed_audio_lens = data_preprocessor(
        input_signal=audios,
        length=audio_lens
    )
    augmented_spec = data_augmentation(input_spec=processed_audios)
    encoded, _ = encoder(
        audio_signal=augmented_spec,
        length=processed_audio_lens
    )
    encoded = connector(tensor=encoded)
    log_probs, _ = decoder(
        targets=transcripts,
        encoder_outputs=encoded
    )
    train_loss = seq_loss(
        log_probs=log_probs,
        targets=transcripts
    )
    evals = []
    for i, data_eval in enumerate(data_evals):
        audios, audio_lens, transcripts, _ = data_eval()
        processed_audios, processed_audio_lens = data_preprocessor(
            input_signal=audios,
            length=audio_lens
        )
        encoded, _ = encoder(
            audio_signal=processed_audios,
            length=processed_audio_lens
        )
        encoded = connector(tensor=encoded)
        log_probs, _ = decoder(
            targets=transcripts,
            encoder_outputs=encoded
        )
        loss = seq_loss(
            log_probs=log_probs,
            targets=transcripts
        )
        predictions, aw = beam_search(encoder_outputs=encoded)
        evals.append((args.eval_datasets[i],
                     (loss, log_probs, transcripts, predictions, aw)))

    # Update config
    cfg['num_params'] = {
        'encoder': encoder.num_weights,
        'connector': connector.num_weights,
        'decoder': decoder.num_weights
    }
    cfg['num_params']['total'] = sum(cfg['num_params'].values())
    cfg['input']['train'] = {'num_data': num_data}
    cfg['optimization']['steps_per_epoch'] = steps_per_epoch
    cfg['optimization']['total_steps'] = total_steps

    return (train_loss, evals), cfg, [tf_callback, se_callback,
                                      uf_callback, saver_callback]


def construct_name(args, cfg):
    name = '{}/{}_{}_{}'.format(
        cfg['model'],
        args.exp_name,
        'bs' + str(cfg['optimization']['batch_size']),
        'epochs' + str(cfg['optimization']['params']['num_epochs'])
    )
    name = name
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
        args, cfg, logger, neural_factory.world_size)
    logger.info('Config:')
    logger.info(pformat(cfg))

    num_data = cfg['input']['train']['num_data']
    steps_per_epoch = cfg['optimization']['steps_per_epoch']
    total_steps = cfg['optimization']['total_steps']
    logger.info(f'Num data: {num_data}\n'
                f'Steps per epoch: {steps_per_epoch}\n'
                f'Total steps: {total_steps}')

    # TODO: Workaround?
    dag_callbacks[0].tb_writer = tb_writer
    dag_callbacks[1].tb_writer = tb_writer

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
                tb_writer=tb_writer,
                write_attn=False
            ),
            user_epochs_done_callback=partial(
                process_evaluation_epoch,
                tag=os.path.basename(name),
                calc_wer=True,
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

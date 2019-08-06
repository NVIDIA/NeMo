# # Copyright (c) 2019 NVIDIA Corporation
# # TODO: Update for YAML
# # TODO: Update for new param organization
# import argparse
# import datetime
# import os
# import random
# from collections import namedtuple
# from functools import partial
# from pprint import pprint
# from shutil import copyfile

# import nemo
# import toml
# import torch
# from nemo.core.callbacks import ValueSetterCallback, Policy, Const
# from nemo.utils.lr_policies import SquareAnnealing
# from nemo.utils.misc import Config
# from tensorboardX import SummaryWriter

# from nemo_asr.las.helpers import process_evaluation_batch, \
#     process_evaluation_epoch

# # Special symbols for seq2seq with cross-entropy criterion and aux CTC loss
# SS = namedtuple('SS', 'id char name')
# _ = 0
# sss = [
# Add two spaces below
#   SS(_ + 0, '#', 'pad'), SS(_ + 1, '<', 'bos'), SS(_ + 2, '>', 'eos'),  # CE
#     # SS(_ + 3, '@', 'ctc_blank')  # CTC
# ]


# def parse_args():
#     parser = argparse.ArgumentParser(description='Decoder language model')

#     parser.add_argument('--config_path', required=True, type=str)
#     parser.add_argument('--train_path', required=True, type=str)
#     parser.add_argument('--val_path', required=True, type=str)
#     parser.add_argument('--exp_name', default='default', type=str)
#     parser.add_argument('--output_path', default='output', type=str)
#     parser.add_argument('--result_path', default='result', type=str)
#     parser.add_argument('--local_rank', default=None, type=int)
#     parser.add_argument('--num_gpu', default=1, type=int)
#     parser.add_argument('--num_workers', default=0, type=int)
#     parser.add_argument('--eval_steps', default=1000, type=int)
#     parser.add_argument('--seed', default=None, type=int)

#     args = parser.parse_args()

#     global VERBOSE
#     VERBOSE = args.local_rank is None or args.local_rank == 0

#     return args


# def parse_cfg(args):
#     cfg = Config(toml.load(args.config_path))

#     cfg['inference']['batch_size'] = cfg['optimization']['batch_size']
#     # Transforming labels with special symbols
#     labels = cfg['target']['labels']
#     labels = [ss.char for ss in sss] + labels
#     cfg['target']['labels'] = labels
#     for ss in sss:
#         cfg['target'][f'{ss.name}_id'] = ss.id

#     return cfg


# def create_dag(args, neural_factory):
#     # Making config
#     cfg = parse_cfg(args)

#     # Defining nodes
#     # data = neural_factory.get_module(
#     #     name='AudioToTextDataLayer',
#     #     params={
#     #         'manifest_filepath': args.train_path,
#     #         'labels': cfg['target']['labels'],
#     #         'eos_id': cfg['target']['eos_id'],
#     #         'batch_size': cfg['optimization']['batch_size'],
#     #         'num_workers': args.num_workers,
#     #         'drop_last': True,
#     #         'load_audio': False
#     #     },
#     #     collection='nemo_asr'
#     # )
#     data = neural_factory.get_module(
#         name='TextDataLayer',
#         params={
#             'path': args.train_path,
#             'labels': cfg['target']['labels'],
#             'eos_id': cfg['target']['eos_id'],
#             'pad_id': cfg['target']['pad_id'],
#             'batch_size': cfg['optimization']['batch_size'],
#             'drop_last': True,
#             'num_workers': args.num_workers
#         },
#         collection='common'
#     )
#     data_eval = neural_factory.get_module(
#         name='AudioToTextDataLayer',
#         params={
#             'manifest_filepath': args.val_path,
#             'labels': cfg['target']['labels'],
#             'eos_id': cfg['target']['eos_id'],
#             'batch_size': cfg['inference']['batch_size'],
#             'num_workers': args.num_workers,
#             'load_audio': False
#         },
#         collection='nemo_asr'
#     )
#     decoder = neural_factory.get_module(
#         name='DecoderRNN',
#         params={
#             'voc_size': len(cfg['target']['labels']),
#             'bos_id': cfg['target']['bos_id'],
#             **cfg['decoder']
#         },
#         collection='common'
#     )
#     num_data = len(data)
#     batch_size = cfg['optimization']['batch_size']
#     num_epochs = cfg['optimization']['params']['num_epochs']
#     steps_per_epoch = int(num_data / (batch_size * args.num_gpu))
#     total_steps = num_epochs * steps_per_epoch
#     tf_callback = ValueSetterCallback(
#         decoder, 'teacher_forcing',
#         policies=[
#             Policy(Const(1.0), start=0.0, end=1.0),
# Add two spaces below
#           # Policy(Linear(1.0, decoder.teacher_forcing), start=0.5, end=1.0)
#         ],
#         total_steps=total_steps
#     )
#     seq_loss = neural_factory.get_module(
#         name='SequenceLoss',
#         params={
#             'pad_id': cfg['target']['pad_id'],
#             # Really makes a difference!
#             'smoothing_coef': cfg['optimization']['smoothing_coef'],
#             # 'aux_ctc': True,
#             # 'ctc_initial_coef': 0.1,
#             # 'ctc_blank_id': cfg['target']['ctc_blank_id']
#         },
#         collection='common'
#     )
#     # beam_pred = neural_factory.get_module(
#     #     name='BeamSearch',
#     #     params={
#     #         'batch_size': cfg['inference']['batch_size'],
#     #         'decoder': decoder,
#     #         'pad_id': cfg['target']['pad_id'],
#     #         'bos_id': cfg['target']['bos_id'],
#     #         'eos_id': cfg['target']['eos_id'],
#     #         'max_len': cfg['target']['max_len'],
#     #         'beam_size': cfg['inference']['beam_size']
#     #     },
#     #     collection='common'
#     # )
#     saver_callback = nemo.core.ModuleSaverCallback(
#         save_modules_list=[decoder],
#         folder=args.result_path,
#         step_frequency=args.eval_steps
#     )

#     # Creating DAG
#     # _, _, texts, _ = data()
#     texts = data()
#     log_probs, _ = decoder(
#         targets=texts
#     )
#     train_loss = seq_loss(
#         log_probs=log_probs,
#         targets=texts
#     )
#     evals = []
#     _, _, transcripts, _ = data_eval()
#     log_probs, _ = decoder(
#         targets=transcripts
#     )
#     eval_loss = seq_loss(
#         log_probs=log_probs,
#         targets=transcripts
#     )
#     # predictions, _ = beam_pred(encoder_outputs=log_probs)  # Fictive edge
#     evals.append((args.val_path,
#                   (eval_loss, log_probs, transcripts)))

#     # Update config
#     cfg['num_params'] = {'decoder': decoder.num_weights}
#     cfg['num_params']['total'] = sum(cfg['num_params'].values())
#     cfg['input']['train'] = {'num_data': num_data}
#     cfg['optimization']['steps_per_epoch'] = steps_per_epoch
#     cfg['optimization']['total_steps'] = total_steps

#     return (train_loss, evals), cfg, [tf_callback, saver_callback]


# def construct_name(args, cfg):
#     name = '{}/{}_{}_{}_{}_{}'.format(
#         cfg['model'],
#         args.exp_name,
#         'bs' + str(cfg['optimization']['batch_size']),
#         'epochs' + str(cfg['optimization']['params']['num_epochs']),
#         str(int(cfg['num_params']['total'] / (10 ** 6))) + 'M',
#         str(datetime.datetime.now())[:-7].replace(' ', '-')
#     )
#     name = os.path.join(args.output_path, name)
#     return name


# def main():
#     # Parse args
#     args = parse_args()
#     if VERBOSE:
#         print(f'Args to be passed to job #{args.local_rank}:')
#         pprint(vars(args))

#     # Define factory
#     neural_factory = nemo.core.NeuralModuleFactory(
#         backend=nemo.core.Backend.PyTorch,
#         local_rank=args.local_rank,
#         optimization_level=nemo.core.Optimization.mxprO1,
#         placement=(
#             nemo.core.DeviceType.AllGpu
#             if args.local_rank is not None
#             else nemo.core.DeviceType.GPU
#         ),
#         cudnn_benchmark=True
#     )
#     # TODO: Should be just neural factory arg
#     if args.seed is not None:
#         random.seed(args.seed)
#         torch.manual_seed(args.seed)
#         if VERBOSE:
#             print(f'Using seed {args.seed}')

#     # Defining computational graph
# Add two spaces below
#   (train_loss, evals), cfg, dag_callbacks = create_dag(args, neural_factory)
#     if VERBOSE:
#         print('Config:')
#         pprint(cfg)

#     # TB, config and checkpoints folder
#     if VERBOSE:
#         num_data = cfg['input']['train']['num_data']
#         steps_per_epoch = cfg['optimization']['steps_per_epoch']
#         total_steps = cfg['optimization']['total_steps']
#         print(f'Num data: {num_data}\n'
#               f'Steps per epoch: {steps_per_epoch}\n'
#               f'Total steps: {total_steps}')

#         name = construct_name(args, cfg)
#         print(f'Name:\n{name}')

#         os.makedirs(name, exist_ok=True)
#         copyfile(args.config_path,
#                  os.path.join(name, os.path.basename(args.config_path)))
#         tb_writer = SummaryWriter(name)
#         # TODO: Workaround?
#         dag_callbacks[0].tb_writer = tb_writer
#     else:
#         tb_writer = None

#     # Callbacks
#     train_callback = nemo.core.SimpleLossLoggerCallback(
#         tensor_list2string=lambda x: str(x[0].item()),
#         tensorboard_writer=tb_writer
#     )
#     log_callbacks = [train_callback]
#     target = cfg['target']
#     labels = target['labels']
#     specials = {f'{ss.name}_id': target[f'{ss.name}_id'] for ss in sss}
#     for name, tensors in evals:
#         eval_callback = nemo.core.EvaluatorCallback(
# Add two spaces below
#           # TODO: Should be fixed soon, so we don't need to pass exactly list
#             eval_tensors=list(tensors),
#             user_iter_callback=partial(
#                 process_evaluation_batch, labels=labels, specials=specials
#             ),
#             user_epochs_done_callback=partial(
#                 process_evaluation_epoch, tag=os.path.basename(name)
#             ),
#             eval_step=args.eval_steps,
#             tensorboard_writer=tb_writer
#         )
#         log_callbacks.append(eval_callback)
#     # noinspection PyTypeChecker
#     callbacks = log_callbacks + dag_callbacks

#     # Optimize
#     optimizer = neural_factory.get_trainer(
#         params={
#             'optimizer_kind': cfg['optimization']['optimizer'],
#             'optimization_params': cfg['optimization']['params']
#         },
#         tb_writer=tb_writer
#     )
#     optimizer.train(
#         tensors_to_optimize=[train_loss],
#         callbacks=callbacks,
#         lr_policy=SquareAnnealing(
#             cfg['optimization']['total_steps'],
#             warmup_steps=(
#                     cfg['optimization']['warmup_epochs']
#                     * cfg['optimization']['steps_per_epoch']
#             )
#         )
#     )


# if __name__ == '__main__':
#     main()

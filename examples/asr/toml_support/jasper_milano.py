# Copyright (c) 2019 NVIDIA Corporation
# TODO: this probably doesn't work anymore; can we just call jasper.py under
# the hood?
import argparse
import nemo
from nemo_asr.helpers import monitor_asr_train_progress, \
    process_evaluation_batch, process_evaluation_epoch


parser = argparse.ArgumentParser(description='Jasper')
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--num_gpus", default=8, type=int)
parser.add_argument("--num_epochs", default=50, type=int)
parser.add_argument("--save_freq", default=-1, type=int)
parser.add_argument("--eval_freq", default=50000000, type=int)
parser.add_argument("--lr", default=0.02, type=float)
parser.add_argument("--weight_decay", default=0.001, type=float)
parser.add_argument("--train_manifest", type=str)
parser.add_argument("--exp_name", default="Jasper", type=str)
parser.add_argument("--val_manifest1", type=str)
parser.add_argument("--val_manifest2", type=str)
parser.add_argument("--optimizer", default="novograd", type=str)
parser.add_argument("--logdir", default=None, type=str)
parser.add_argument("--warmup_steps", default=0, type=float)
parser.add_argument("--cutout_rect_regions", default=None, type=int)
parser.add_argument("--cutout_rect_time", default=None, type=int)
parser.add_argument("--cutout_rect_freq", default=None, type=int)
parser.add_argument("--cutout_x_regions", default=None, type=int)
parser.add_argument("--cutout_x_width", default=None, type=int)
parser.add_argument("--cutout_y_regions", default=None, type=int)
parser.add_argument("--cutout_y_width", default=None, type=int)
parser.add_argument("--prolog_filters", default=None, type=int)
parser.add_argument("--prolog_kernel", default=None, type=int)
parser.add_argument("--epilog_filters1", default=None, type=int)
parser.add_argument("--epilog_filters2", default=None, type=int)
parser.add_argument("--epilog_kernel", default=None, type=int)
parser.add_argument("--epilog_dilation", default=None, type=int)
parser.add_argument("--epilog_dp=0.0", default=None, type=int)
parser.add_argument("--B1_filters=256", default=None, type=int)
parser.add_argument("--B1_kernel=33", default=None, type=int)
parser.add_argument("--B1_dp=0.0", default=None, type=float)
parser.add_argument("--B1_subblocks", default=None, type=int)
parser.add_argument("--B1_count", default=None, type=int)
parser.add_argument("--B2_filters", default=None, type=int)
parser.add_argument("--B2_kernel", default=None, type=int)
parser.add_argument("--B2_dp", default=None, type=float)
parser.add_argument("--B2_subblocks", default=None, type=int)
parser.add_argument("--B2_count", default=None, type=int)
parser.add_argument("--B3_filters", default=None, type=int)
parser.add_argument("--B3_kernel", default=None, type=int)
parser.add_argument("--B3_dp=0.0", default=None, type=float)
parser.add_argument("--B3_subblocks", default=None, type=int)
parser.add_argument("--B3_count", default=None, type=int)
parser.add_argument("--B4_filters", default=None, type=int)
parser.add_argument("--B4_kernel", default=None, type=int)
parser.add_argument("--B4_dp", default=None, type=float)
parser.add_argument("--B4_subblocks", default=None, type=int)
parser.add_argument("--B4_count", default=None, type=int)
parser.add_argument("--B5_filters", default=None, type=int)
parser.add_argument("--B5_kernel", default=None, type=int)
parser.add_argument("--B5_dp", default=None, type=float)
parser.add_argument("--B5_subblocks", default=None, type=int)
parser.add_argument("--B5_count", default=None, type=int)


args = parser.parse_args()
batch_size = args.batch_size
lr = args.lr
num_epochs = args.num_epochs
num_gpus = args.num_gpus
weight_decay = args.weight_decay
warmup = args.warmup_steps


def construct_name(name, lr, batch_size, num_gpus, num_epochs, wd,
                   batch_per_step):
    return ("{0}-lr_{1}-bs_{2}x{3}-e_{4}-wd_{5}-opt_{6}-bps_{7}-larc_{8}"
            "-luc_{9}".format(name, lr,
                              batch_size,
                              num_gpus,
                              num_epochs, wd,
                              args.optimizer,
                              batch_per_step,
                              False,
                              False))


def generate_toml(cutout_rect_regions=5,
                  cutout_rect_time=60,
                  cutout_rect_freq=25,
                  cutout_x_regions=0,
                  cutout_x_width=10,
                  cutout_y_regions=0,
                  cutout_y_width=10,
                  prolog_filters=256,
                  prolog_kernel=33,
                  epilog_filters1=512,
                  epilog_filters2=1024,
                  epilog_kernel=87,
                  epilog_dilation=2,
                  epilog_dp=0.0,
                  B1_filters=256,
                  B1_kernel=33,
                  B1_dp=0.0,
                  B1_subblocks=5,
                  B1_count=2,
                  B2_filters=512,
                  B2_kernel=39,
                  B2_dp=0.0,
                  B2_subblocks=5,
                  B2_count=2,
                  B3_filters=512,
                  B3_kernel=51,
                  B3_dp=0.0,
                  B3_subblocks=5,
                  B3_count=2,
                  B4_filters=512,
                  B4_kernel=63,
                  B4_dp=0.0,
                  B4_subblocks=5,
                  B4_count=2,
                  B5_filters=512,
                  B5_kernel=75,
                  B5_dp=0.0,
                  B5_subblocks=5,
                  B5_count=2,
                  ):
    config = {'model': 'Jasper',
              'input': {'normalize': 'per_feature',
                        'sample_rate': 16000,
                        'window_size': 0.02,
                        'window_stride': 0.01,
                        'window': 'hann',
                        'features': 64,
                        'n_fft': 512,
                        'frame_splicing': 1,
                        'dither': 1e-05,
                        'feat_type': 'logfbank',
                        'normalize_transcripts': True,
                        'trim_silence': True,
                        'pad_to': 'max',
                        'max_duration': 16.7},
              'input_eval': {'normalize': 'per_feature',
                             'sample_rate': 16000,
                             'window_size': 0.02,
                             'window_stride': 0.01,
                             'window': 'hann',
                             'features': 64,
                             'n_fft': 512,
                             'frame_splicing': 1,
                             'dither': 1e-05,
                             'feat_type': 'logfbank',
                             'normalize_transcripts': True,
                             'trim_silence': True},
              'encoder': {'activation': 'relu'},
              'jasper': []}

    # Spec Augmentation
    if cutout_rect_regions > 0:
        config['input']['cutout_rect_regions'] = cutout_rect_regions
        config['input']['cutout_rect_time'] = cutout_rect_time
        config['input']['cutout_rect_freq'] = cutout_rect_freq

    if cutout_x_regions > 0:
        config['input']['cutout_x_regions'] = cutout_x_regions
        config['input']['cutout_x_width'] = cutout_x_width

    if cutout_y_regions > 0:
        config['input']['cutout_y_regions'] = cutout_y_regions
        config['input']['cutout_y_width'] = cutout_y_width

    prolog = {'filters': prolog_filters,
              'repeat': 1, 'kernel': [prolog_kernel],
              'stride': [2], 'dilation': [1], 'dropout': 0.0,
              'residual': False,
              'separable': True}

    config['jasper'].append(prolog)

    # create blocks
    for ind in range(B1_count):
        block = {'filters': B1_filters,
                 'repeat': B1_subblocks,
                 'kernel': [B1_kernel],
                 'stride': [1],
                 'dilation': [1],
                 'dropout': B1_dp,
                 'residual': True,
                 'separable': True}
        config['jasper'].append(block)

    for ind in range(B2_count):
        block = {'filters': B2_filters,
                 'repeat': B2_subblocks,
                 'kernel': [B2_kernel],
                 'stride': [1],
                 'dilation': [1],
                 'dropout': B2_dp,
                 'residual': True,
                 'separable': True}
        config['jasper'].append(block)

    for ind in range(B3_count):
        block = {'filters': B3_filters,
                 'repeat': B3_subblocks,
                 'kernel': [B3_kernel],
                 'stride': [1],
                 'dilation': [1],
                 'dropout': B3_dp,
                 'residual': True,
                 'separable': True}
        config['jasper'].append(block)

    for ind in range(B4_count):
        block = {'filters': B4_filters,
                 'repeat': B4_subblocks,
                 'kernel': [B4_kernel],
                 'stride': [1],
                 'dilation': [1],
                 'dropout': B4_dp,
                 'residual': True,
                 'separable': True}
        config['jasper'].append(block)

    for ind in range(B5_count):
        block = {'filters': B5_filters,
                 'repeat': B5_subblocks,
                 'kernel': [B5_kernel],
                 'stride': [1],
                 'dilation': [1],
                 'dropout': B5_dp,
                 'residual': True,
                 'separable': True}
        config['jasper'].append(block)

    # create epilog
    epilog1 = {
        'filters': epilog_filters1,
        'repeat': 1,
        'kernel': [epilog_kernel],
        'stride': [1],
        'dilation': [epilog_dilation],
        'dropout': epilog_dp,
        'residual': False,
        'separable': True
    }
    config['jasper'].append(epilog1)
    epilog2 = {
        'filters': epilog_filters2,
        'repeat': 1,
        'kernel': [1],
        'stride': [1],
        'dilation': [1],
        'dropout': epilog_dp,
        'residual': False,
    }
    config['jasper'].append(epilog2)
    config['labels'] = {'labels': [' ',
                                   'a',
                                   'b',
                                   'c',
                                   'd',
                                   'e',
                                   'f',
                                   'g',
                                   'h',
                                   'i',
                                   'j',
                                   'k',
                                   'l',
                                   'm',
                                   'n',
                                   'o',
                                   'p',
                                   'q',
                                   'r',
                                   's',
                                   't',
                                   'u',
                                   'v',
                                   'w',
                                   'x',
                                   'y',
                                   'z',
                                   "'"]}
    return config


jasper_model_definition = generate_toml(
                  cutout_rect_regions=args.cutout_rect_regions,
                  cutout_rect_time=args.cutout_rect_time,
                  cutout_rect_freq=args.cutout_rect_freq,
                  cutout_x_regions=args.cutout_x_regions,
                  cutout_x_width=args.cutout_x_width,
                  cutout_y_regions=args.cutout_y_regions,
                  cutout_y_width=args.cutout_y_width,
                  prolog_filters=256,
                  prolog_kernel=33,
                  epilog_filters1=512,
                  epilog_filters2=1024,
                  epilog_kernel=87,
                  epilog_dilation=2,
                  epilog_dp=0.0,
                  B1_filters=args.B1_filters,
                  B1_kernel=args.B1_kernel,
                  B1_dp=0.0,
                  B1_subblocks=args.B1_subblocks,
                  B1_count=args.B1_count,
                  B2_filters=args.B2_filters,
                  B2_kernel=args.B2_kernel,
                  B2_dp=0.0,
                  B2_subblocks=args.B2_subblocks,
                  B2_count=args.B2_count,
                  B3_filters=args.B3_filters,
                  B3_kernel=args.B3_kernel,
                  B3_dp=0.0,
                  B3_subblocks=args.B3_subblocks,
                  B3_count=args.B3_count,
                  B4_filters=args.B4_filters,
                  B4_kernel=args.B4_kernel,
                  B4_dp=0.0,
                  B4_subblocks=args.B4_subblocks,
                  B4_count=args.B4_count,
                  B5_filters=args.B5_filters,
                  B5_kernel=args.B5_kernel,
                  B5_dp=0.0,
                  B5_subblocks=args.B5_subblocks,
                  B5_count=args.B5_count,
        )
name = construct_name(args.exp_name, lr, batch_size, num_gpus, num_epochs,
                      weight_decay, 1)
print(name)

verbose = (args.local_rank is None or args.local_rank == 0)

tb_writer = None

if args.local_rank is not None:
    device = nemo.core.DeviceType.AllGpu
    if verbose:
        print('Doing ALL GPU')
else:
    device = nemo.core.DeviceType.GPU

jasper_model_definition['placement'] = device
vocab = jasper_model_definition['labels']['labels']

train_manifest = args.train_manifest
val_manifest1 = args.val_manifest1
val_manifest2 = args.val_manifest2


featurizer_config = jasper_model_definition['input']
max_duration = featurizer_config.get("max_duration", 16.7)
pytorch_benchmark = True

# instantiate Neural Factory with supported backend
neural_factory = nemo.core.NeuralModuleFactory(
    backend=nemo.core.Backend.PyTorch,
    local_rank=args.local_rank,
    optimization_level=nemo.core.Optimization.mxprO1,
    placement=device,
    cudnn_benchmark=pytorch_benchmark)

perturb_config = jasper_model_definition.get('perturb', None)
data_layer = neural_factory.get_module(name="AudioToTextDataLayer",
                                       params={
                                           "featurizer_config":
                                               featurizer_config,
                                           "perturb_config": perturb_config,
                                           "manifest_filepath": train_manifest,
                                           "labels": vocab,
                                           "batch_size": batch_size,
                                           "placement": device,
                                           "max_duration": max_duration,
                                           "verbose": verbose
                                           # "normalize_transcripts": False
                                       },
                                       collection="nemo_asr")
N = len(data_layer)
step_per_epoch = int(N / (batch_size * num_gpus))
if verbose:
    print('-----------------')
    print('Have {0} examples to train on.'.format(N))
    print('-----------------')

data_preprocessor = neural_factory.get_module(name="AudioPreprocessing",
                                              collection="nemo_asr",
                                              params=featurizer_config)

data_spectr_augmentation = neural_factory.get_module(
    name="SpectrogramAugmentation",
    collection="nemo_asr",
    params=featurizer_config)

data_layer_eval1 = neural_factory.get_module(name="AudioToTextDataLayer",
                                             params={
                                                 "featurizer_config":
                                                     featurizer_config,
                                                 "manifest_filepath":
                                                     val_manifest1,
                                                 "labels": vocab,
                                                 "batch_size": 8,
                                                 "placement": device,
                                                 "min_duration": None,
                                                 "verbose": verbose
                                             },
                                             collection="nemo_asr")
data_layer_eval2 = neural_factory.get_module(name="AudioToTextDataLayer",
                                             params={
                                                 "featurizer_config":
                                                     featurizer_config,
                                                 "manifest_filepath":
                                                     val_manifest2,
                                                 "labels": vocab,
                                                 "batch_size": 8,
                                                 "placement": device,
                                                 "min_duration": None,
                                                 "verbose": verbose
                                             },
                                             collection="nemo_asr")

jasper_encoder = neural_factory.get_module(name="JasperEncoder",
                                           params=jasper_model_definition,
                                           collection="nemo_asr")
jasper_decoder = neural_factory.get_module(name="JasperDecoderForCTC",
                                           params={
                                               "feat_in": 1024,
                                               "num_classes": len(vocab),
                                               "placement": device
                                           },
                                           collection="nemo_asr")

ctc_loss = neural_factory.get_module(name="CTCLossNM",
                                     params={
                                         "num_classes": len(vocab),
                                         "placement": device
                                     },
                                     collection="nemo_asr")

greedy_decoder = neural_factory.get_module(name="GreedyCTCDecoder",
                                           params={"placement": device},
                                           collection="nemo_asr")

if verbose:
    print('\n\n\n================================')
    print(
        "Number of parameters in encoder: {0}".format(
            jasper_encoder.num_weights))
    print(
        "Number of parameters in decoder: {0}".format(
            jasper_decoder.num_weights))
    print("Total number of parameters in decoder: {0}".format(
        jasper_decoder.num_weights + jasper_encoder.num_weights))
    print('================================\n\n\n')

# Train DAG
audio_signal_t, a_sig_length_t, transcript_t, transcript_len_t = data_layer()
processed_signal_t, p_length_t = data_preprocessor(input_signal=audio_signal_t,
                                                   length=a_sig_length_t)

augmented_spec_t = data_spectr_augmentation(input_spec=processed_signal_t)
encoded_t, encoded_len_t = jasper_encoder(audio_signal=augmented_spec_t,
                                          length=p_length_t)
log_probs_t = jasper_decoder(encoder_output=encoded_t)
predictions_t = greedy_decoder(log_probs=log_probs_t)
loss_t = ctc_loss(log_probs=log_probs_t,
                  targets=transcript_t,
                  input_length=encoded_len_t,
                  target_length=transcript_len_t)

# MULTIPLE EVALUATION DAGs
# Eval DAG1
audio_signal_e1, a_sig_length_e1, transcript_e1, transcript_len_e1 = \
    data_layer_eval1()
processed_signal_e1, p_length_e1 = data_preprocessor(
    input_signal=audio_signal_e1,
    length=a_sig_length_e1)
encoded_e1, encoded_len_e1 = jasper_encoder(audio_signal=processed_signal_e1,
                                            length=p_length_e1)
log_probs_e1 = jasper_decoder(encoder_output=encoded_e1)
predictions_e1 = greedy_decoder(log_probs=log_probs_e1)
loss_e1 = ctc_loss(log_probs=log_probs_e1,
                   targets=transcript_e1,
                   input_length=encoded_len_e1,
                   target_length=transcript_len_e1)

# Eval DAG2
audio_signal_e2, a_sig_length_e2, transcript_e2, transcript_len_e2 = \
    data_layer_eval2()
processed_signal_e2, p_length_e2 = data_preprocessor(
    input_signal=audio_signal_e2,
    length=a_sig_length_e2)
encoded_e2, encoded_len_e2 = jasper_encoder(audio_signal=processed_signal_e2,
                                            length=p_length_e2)
log_probs_e2 = jasper_decoder(encoder_output=encoded_e2)
predictions_e2 = greedy_decoder(log_probs=log_probs_e2)
loss_e2 = ctc_loss(log_probs=log_probs_e2,
                   targets=transcript_e2,
                   input_length=encoded_len_e2,
                   target_length=transcript_len_e2)

# Callbacks needed to print info to console and Tensorboard
train_callback = nemo.core.SimpleLossLoggerCallback(
    tensor_list2string=lambda x: str(x[0].item()),
    tensorboard_writer=tb_writer,
    tensor_list2string_evl=lambda x: monitor_asr_train_progress(x,
                                                               labels=vocab))

saver_callback = nemo.core.ModuleSaverCallback(
    save_modules_list=[jasper_encoder,
                       jasper_decoder],
    folder="/results/",
    step_frequency=-1)

eval_callback1 = nemo.core.EvaluatorCallback(
    eval_tensors=[loss_e1, predictions_e1, transcript_e1, transcript_len_e1],
    user_iter_callback=lambda x, y: process_evaluation_batch(
        x, y, labels=vocab),
    user_epochs_done_callback=lambda x: process_evaluation_epoch(x,
                                                                 tag="DEV-CLEAN"),

    eval_step=args.eval_freq,
    tensorboard_writer=tb_writer)

eval_callback2 = nemo.core.EvaluatorCallback(
    eval_tensors=[loss_e2, predictions_e2, transcript_e2, transcript_len_e2],
    user_iter_callback=lambda x, y: process_evaluation_batch(
        x, y, labels=vocab),
    user_epochs_done_callback=lambda x: process_evaluation_epoch(x,
                                                                 tag="DEV-OTHER"),
    eval_step=args.eval_freq,
    tensorboard_writer=tb_writer)

callbacks = [train_callback, saver_callback, eval_callback1,
             eval_callback2]


def lr_policy(initial_lr, step, N):
 min_lr = 0.00001
 if step < warmup:
   return initial_lr * step / warmup
 res = initial_lr * ((N - warmup - step + 1) / (N - warmup)) ** 2
 return max(res, min_lr)


optimizer = neural_factory.get_trainer(
    params={"optimizer_kind": args.optimizer,
            "optimization_params": {"num_epochs": num_epochs, "lr": lr,
                                    "weight_decay": weight_decay}})

optimizer.train(tensors_to_optimize=[loss_t],
                callbacks=callbacks,
                tensors_to_evaluate=[predictions_t, transcript_t,
                                     transcript_len_t],
                lr_policy=lambda lr, s, e: lr_policy(lr, s,
                                                     num_epochs *
                                                     step_per_epoch),
                batches_per_step=1)

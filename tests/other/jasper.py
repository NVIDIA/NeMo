# Copyright (c) 2019 NVIDIA Corporation
from nemo.backends.pytorch.asr.helpers import monitor_asr_train_progress, \
    process_evaluation_batch, process_evaluation_epoch
import nemo
import argparse
import os
import sys

import toml
from tensorboardX import SummaryWriter

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))

parser = argparse.ArgumentParser(description='Jasper')
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--num_epochs", default=100, type=int)
parser.add_argument("--lr", default=0.02, type=float)
parser.add_argument("--weight_decay", default=0.001, type=float)
parser.add_argument("--train_manifest", type=str)
parser.add_argument("--val_manifest1", type=str)
parser.add_argument("--optimization_level", default=None, type=int)
args = parser.parse_args()
batch_size = args.batch_size
lr = args.lr
num_epochs = args.num_epochs
num_gpus = args.num_gpus
weight_decay = args.weight_decay
if args.optimization_level is None or args.optimization_level == 0:
    opt_level = nemo.core.Optimization.mxprO0
elif args.optimization_level == 1:
    opt_level = nemo.core.Optimization.mxprO1
elif args.optimization_level == 2:
    opt_level = nemo.core.Optimization.mxprO2
elif args.optimization_level == 3:
    opt_level = nemo.core.Optimization.mxprO3
else:
    raise ValueError("Unknown optimization level")


def construct_name(name, lr, batch_size, num_gpus, num_epochs, wd):
    return "{0}-lr_{1}-bs_{2}x{3}-e_{4}-wd_{5}-OPT-{6}".format(name, lr,
                                                               batch_size,
                                                               num_gpus,
                                                               num_epochs, wd,
                                                               opt_level)


name = construct_name('Jasper10x5', lr, batch_size, num_gpus, num_epochs,
                      weight_decay)
tb_writer = SummaryWriter(name)

if args.local_rank is not None:
    device = nemo.core.DeviceType.AllGpu
    print('Doing ALL GPU')
else:
    device = nemo.core.DeviceType.GPU

# instantiate Neural Factory with supported backend
neural_factory = nemo.core.NeuralModuleFactory(
    backend=nemo.core.Backend.PyTorch,
    local_rank=args.local_rank,
    optimization_level=opt_level,
    placement=device)

jasper_model_definition = toml.load("../../examples/nemo_asr/jasper10x5.toml")
jasper_model_definition['placement'] = device
labels = jasper_model_definition['labels']['labels']

train_manifest = "/mnt/D1/Data/librispeech/librivox-train-all.json"
#train_manifest = args.train_manifest
val_manifest1 = "/mnt/D1/Data/librispeech/librivox-dev-clean.json"
# val_manifest2 = "/mnt/D1/Data/librispeech/librivox-dev-other.json"
#val_manifest1 = args.val_manifest1


featurizer_config = jasper_model_definition['input']
data_layer = neural_factory.get_module(name="AudioToTextDataLayer",
                                       params={
                                           "featurizer_config": featurizer_config,
                                           "manifest_filepath": train_manifest,
                                           "labels": labels,
                                           "batch_size": batch_size,
                                           "placement": device,
                                           "max_duration": 16.7
                                       },
                                       collection="nemo_asr")
N = len(data_layer)
print('-----------------')
print('Have {0} examples to train on.'.format(N))
print('-----------------')
step_per_epoch = int(N / (batch_size * num_gpus))

data_preprocessor = neural_factory.get_module(name="AudioPreprocessing",
                                              collection="nemo_asr",
                                              params=featurizer_config)

data_layer_eval1 = neural_factory.get_module(name="AudioToTextDataLayer",
                                             params={
                                                 "featurizer_config": featurizer_config,
                                                 "manifest_filepath": val_manifest1,
                                                 "labels": labels,
                                                 "batch_size": 8,
                                                 "placement": device,
                                             },
                                             collection="nemo_asr")
jasper_encoder = neural_factory.get_module(name="JasperEncoder",
                                           params=jasper_model_definition,
                                           collection="nemo_asr")
jasper_decoder = neural_factory.get_module(name="JasperDecoderForCTC",
                                           params={
                                               "feat_in": 1024,
                                               "num_classes": len(labels),
                                               "placement": device
                                           },
                                           collection="nemo_asr")

ctc_loss = neural_factory.get_module(name="CTCLossNM",
                                     params={
                                         "num_classes": len(labels),
                                         "placement": device
                                     },
                                     collection="nemo_asr")

greedy_decoder = neural_factory.get_module(name="GreedyCTCDecoder",
                                           params={"placement": device},
                                           collection="nemo_asr")
# Train DAG
audio_signal_t, a_sig_length_t, transcript_t, transcript_len_t = data_layer()
processed_signal_t, p_length_t = data_preprocessor(input_signal=audio_signal_t,
                                                   length=a_sig_length_t)
encoded_t, encoded_len_t = jasper_encoder(audio_signal=processed_signal_t,
                                          length=p_length_t)
log_probs_t = jasper_decoder(encoder_output=encoded_t)
predictions_t = greedy_decoder(log_probs=log_probs_t)
loss_t = ctc_loss(log_probs=log_probs_t,
                  targets=transcript_t,
                  input_length=encoded_len_t,
                  target_length=transcript_len_t)

# Eval DAG1
audio_signal_e1, a_sig_length_e1, transcript_e1, transcript_len_e1 = data_layer_eval1()
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


print('\n\n\n================================')
print("Total number of parameters: {0}".format(
    jasper_decoder.num_weights + jasper_encoder.num_weights))
print('================================')

# Callbacks needed to print info to console and Tensorboard
train_callback = nemo.core.SimpleLossLoggerCallback(
    tensor_list2str=lambda x: str(x[0].item()),
    tb_writer=tb_writer,
    tensor_list2str_evl=lambda x: monitor_asr_train_progress(x, labels=labels))

eval_callback1 = nemo.core.EvaluatorCallback(
    eval_tensors=[loss_e1, predictions_e1, transcript_e1, transcript_len_e1],
    user_iter_callback=lambda x, y: process_evaluation_batch(
        x, y, labels=labels),
    user_epochs_done_callback=lambda x: process_evaluation_epoch(x,
                                                                 tag="DEV-CLEAN"),
    eval_step=500,
    tb_writer=tb_writer)


def lr_policy(initial_lr, step, N):
    res = initial_lr * ((N - step + 1) / N) ** 2
    return res


optimizer = neural_factory.get_trainer(
    params={"optimizer_kind": "novograd",
            "optimization_params": {"num_epochs": num_epochs, "lr": lr,
                                    "weight_decay": weight_decay}})
optimizer.train(tensors_to_optimize=[loss_t],
                callbacks=[train_callback, eval_callback1],
                tensors_to_evaluate=[predictions_t, transcript_t,
                                     transcript_len_t],
                lr_policy=lambda lr, s, e: lr_policy(lr, s,
                                                     num_epochs * step_per_epoch))

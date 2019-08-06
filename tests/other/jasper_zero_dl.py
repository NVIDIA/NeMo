# Copyright (c) 2019 NVIDIA Corporation
from nemo.backends.pytorch.asr.helpers import monitor_asr_train_progress
from nemo.core.neural_types import *
import nemo
import argparse
import os
import sys

import toml
import torch
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
    opt_level = nemo.core.Optimization.nothing
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


name = construct_name('ZeroDS-Jasper10x5', lr, batch_size, num_gpus, num_epochs,
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

# train_manifest = "/mnt/D1/Data/librispeech/librivox-train-all.json"
train_manifest = args.train_manifest

featurizer_config = jasper_model_definition['input']
data_preprocessor = neural_factory.get_module(name="AudioPreprocessing",
                                              collection="nemo_asr",
                                              params=featurizer_config)
N = 288000
time = 256
dl = nemo.backends.pytorch.ZerosDataLayer(size=N, dtype=torch.FloatTensor,
                                          batch_size=batch_size,
                                          output_ports={
                                              "processed_signal": NeuralType(
                                                  {0: AxisType(BatchTag),
                                                   1: AxisType(ChannelTag, dim=64),
                                                   2: AxisType(TimeTag, dim=time)}),

                                              "processed_length": NeuralType(
                                                  {0: AxisType(BatchTag)}),

                                              "transcript": NeuralType(
                                                  {0: AxisType(BatchTag),
                                                   1: AxisType(TimeTag, dim=time)}),

                                              "transcript_length": NeuralType(
                                                  {0: AxisType(BatchTag)})
                                          })
print('-----------------')
print('Have {0} examples to train on.'.format(N))
print('-----------------')
step_per_epoch = int(N / (batch_size * num_gpus))

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
processed_signal_t, p_length_t, transcript_t, transcript_len_t = dl()
encoded_t, encoded_len_t = jasper_encoder(audio_signal=processed_signal_t,
                                          length=p_length_t)
log_probs_t = jasper_decoder(encoder_output=encoded_t)
predictions_t = greedy_decoder(log_probs=log_probs_t)
loss_t = ctc_loss(log_probs=log_probs_t,
                  targets=transcript_t,
                  input_length=encoded_len_t,
                  target_length=transcript_len_t)

print('\n\n\n================================')
print("Total number of parameters: {0}".format(
    jasper_decoder.num_weights + jasper_encoder.num_weights))
print('================================')

# Callbacks needed to print info to console and Tensorboard
train_callback = nemo.core.SimpleLossLoggerCallback(
    tensor_list2string=lambda x: str(x[0].item()),
    tensorboard_writer=tb_writer,
    tensor_list2string_evl=lambda x: monitor_asr_train_progress(x, labels=labels))


def lr_policy(initial_lr, step, N):
    res = initial_lr * ((N - step + 1) / N) ** 2
    return res


optimizer = neural_factory.get_trainer(
    params={"optimizer_kind": "novograd",
            "optimization_params": {"num_epochs": num_epochs, "lr": lr,
                                    "weight_decay": weight_decay}})
optimizer.train(tensors_to_optimize=[loss_t],
                callbacks=[train_callback],
                tensors_to_evaluate=[predictions_t, transcript_t,
                                     transcript_len_t],
                lr_policy=lambda lr, s, e: lr_policy(lr, s,
                                                     num_epochs * step_per_epoch))

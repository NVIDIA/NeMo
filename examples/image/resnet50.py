# Copyright (c) 2019 NVIDIA Corporation
from tensorboardX import SummaryWriter
from nemo.utils.lr_policies import SquareAnnealing
from nemo.backends.pytorch.torchvision.helpers import eval_iter_callback, \
    eval_epochs_done_callback, compute_accuracy
import nemo
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))

parser = argparse.ArgumentParser(description='ResNet50 on ImageNet')
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--num_epochs", default=100, type=int)
parser.add_argument("--max_steps", default=None, type=int)
parser.add_argument("--learning_rate", default=0.1, type=float)
parser.add_argument("--weight_decay", default=0.0001, type=float)
parser.add_argument("--momentum", default=0.91, type=float)
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--data_root", default=None, type=str)
parser.add_argument("--tb_folder", default=None, type=str)

args = parser.parse_args()

if args.local_rank is not None:
    device = nemo.core.DeviceType.AllGpu
else:
    device = nemo.core.DeviceType.GPU

batch_size = args.batch_size
num_epochs = args.num_epochs
learning_rate = args.learning_rate
max_steps = args.max_steps
weight_decay = args.weight_decay
momentum = args.momentum
num_gpus = args.num_gpus

if args.tb_folder is None:
    tb_folder = 'resnet50_fp32'
else:
    tb_folder = args.tb_folder

tb_writer = SummaryWriter(tb_folder)

# instantiate Neural Factory with supported backend
neural_factory = nemo.core.NeuralModuleFactory(
    backend=nemo.core.Backend.PyTorch,
    local_rank=args.local_rank,
    # Set this to nemo.core.Optimization.mxprO1
    # if you have Volta or Turing GPU
    optimization_level=nemo.core.Optimization.mxprO0)

resnet = neural_factory.get_module(name="resnet50",
                                   params={"placement": device},
                                   collection="torchvision",
                                   pretrained=False)

dl_train = neural_factory.get_module(
    name="ImageFolderDataLayer", collection="torchvision",
    params={"batch_size": batch_size,
            "input_size": resnet.input_ports["x"].axis2type[2].dim,
            "shuffle": True,
            "path": args.data_root + "train",
            # "path": "/mnt/D1/Data/ImageNet/ImageFolder/train",
            "placement": device
            })

L_train = neural_factory.get_module(
    name="CrossEntropyLoss", collection="toys",
    params={"placement": device})

dl_eval = neural_factory.get_module(
    name="ImageFolderDataLayer", collection="torchvision",
    params={"batch_size": batch_size,
            "input_size": resnet.input_ports["x"].axis2type[2].dim,
            "shuffle": False,
            "is_eval": True,
            "path": args.data_root + "val",
            # "path": "/mnt/D1/Data/ImageNet/ImageFolder/val",
            # "path": "/raid/okuchaiev/Data/ImageNet/ImageFolder/val",
            "placement": device
            })

L_eval = neural_factory.get_module(
    name="CrossEntropyLoss", collection="toys",
    params={"placement": device})

step_per_epoch = int(len(dl_train) / (batch_size * num_gpus))


images, labels = dl_train()
outputs = resnet(x=images)
train_loss = L_train(predictions=outputs, labels=labels)

e_images, e_labels = dl_eval()
e_outputs = resnet(x=e_images)
e_loss = L_eval(predictions=e_outputs, labels=e_labels)

callback = nemo.core.SimpleLossLoggerCallback(
    step_freq=50, tb_writer=tb_writer, tensor_list2str=lambda x: str(
        x[0].item()), tensor_list2str_evl=lambda x: compute_accuracy(x))

callback_eval = nemo.core.EvaluatorCallback(
    eval_tensors=[e_loss, e_outputs, e_labels],
    user_iter_callback=eval_iter_callback,
    user_epochs_done_callback=eval_epochs_done_callback,
    eval_step=10000,
    tb_writer=tb_writer)

# Instantiate an optimizer to perform `train` action
optimizer = neural_factory.get_trainer(
    params={
        "optimization_params": {
            "num_epochs": num_epochs,
            "lr": learning_rate,
            "max_steps": max_steps,
            "weight_decay": weight_decay,
            "momentum": momentum}})

optimizer.train(tensors_to_optimize=[train_loss],
                tensors_to_evaluate=[outputs, labels],
                callbacks=[callback, callback_eval],
                lr_policy=SquareAnnealing(num_epochs * step_per_epoch))

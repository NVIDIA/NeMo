# Copyright (c) 2019 NVIDIA Corporation

import os
import sys

import argparse
import subprocess
import zipfile

import nemo
from nemo import logging
from nemo.backends.pytorch.torchvision.helpers import eval_iter_callback, \
    eval_epochs_done_callback, compute_accuracy

from tensorboardX import SummaryWriter

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))

if not os.path.isdir("hymenoptera_data"):
    logging.info("Datafolder not found. Downloading data from the Web")
    subprocess.run(
        ["wget", "https://download.pytorch.org/tutorial/hymenoptera_data.zip"])
    zip_ref = zipfile.ZipFile('hymenoptera_data.zip', 'r')
    zip_ref.extractall('.')
    zip_ref.close()
else:
    logging.info("Found data folder - hymenoptera_data")

parser = argparse.ArgumentParser(description='Transfer Learning')
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--num_epochs", default=90, type=int)
parser.add_argument("--max_steps", default=None, type=int)
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--weight_decay", default=0.0025, type=float)
parser.add_argument("--momentum", default=0.91, type=float)
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--data_root", default=None, type=str)
parser.add_argument("--tb_folder", default=None, type=str)

args = parser.parse_args()
device = nemo.core.DeviceType.GPU
batch_size = args.batch_size
num_epochs = args.num_epochs
learning_rate = args.learning_rate
max_steps = args.max_steps
weight_decay = args.weight_decay
momentum = args.momentum
num_gpus = args.num_gpus

if args.tb_folder is None:
    tb_folder = 'transfer_learning'
else:
    tb_folder = args.tb_folder

tb_writer = SummaryWriter(tb_folder)

device = nemo.core.DeviceType.GPU

# Instantiate Neural Factory and Neural Modules
neural_factory = nemo.core.NeuralModuleFactory(
    backend=nemo.core.Backend.PyTorch,
    placement=device)

# NOTICE: pretrain=True argument
resnet = neural_factory.get_module(name="resnet18",
                                   params={"num_classes": 2},
                                   collection="torchvision",
                                   pretrained=True)

dl_train = neural_factory.get_module(
    name="ImageFolderDataLayer", collection="torchvision",
    params={"batch_size": batch_size,
            "input_size":
                resnet.inputs["x"].axis2type[2].dim,
            "shuffle": True,
            "path": "hymenoptera_data/train",
            })

dl_eval = neural_factory.get_module(
    name="ImageFolderDataLayer", collection="torchvision",
    params={"batch_size": batch_size,
            "input_size":
                resnet.inputs["x"].axis2type[2].dim,
            "shuffle": False,
            "path": "hymenoptera_data/val",
            })

L_train = neural_factory.get_module(
    name="CrossEntropyLoss", collection="toys",
    params={})

L_eval = neural_factory.get_module(
    name="CrossEntropyLoss", collection="toys",
    params={})

# NOTICE: Freeze all Neural Module's weights
resnet.freeze()
# NOTICE: unfreeze, top classification layer for fine-tuning
resnet.unfreeze(set(["fc.weight", "fc.bias"]))

images, labels = dl_train()
outputs = resnet(x=images)
train_loss = L_train(predictions=outputs, labels=labels)

e_images, e_labels = dl_eval()
e_outputs = resnet(x=e_images)
e_loss = L_eval(predictions=e_outputs, labels=e_labels)

callback = nemo.core.SimpleLossLoggerCallback(
    step_freq=20, tb_writer=tb_writer, tensor_list2str=lambda x: str(
        x[0].item()), tensor_list2str_evl=lambda x: compute_accuracy(x))

callback_eval = nemo.core.EvaluatorCallback(
    eval_tensors=[e_loss, e_outputs, e_labels],
    user_iter_callback=eval_iter_callback,
    user_epochs_done_callback=eval_epochs_done_callback,
    eval_step=30,
    tb_writer=tb_writer)


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
                callbacks=[callback, callback_eval])

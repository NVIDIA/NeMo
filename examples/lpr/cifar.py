import os
from nemo.backends.pytorch.torchvision.helpers import eval_iter_callback, \
    eval_epochs_done_callback, compute_accuracy
import argparse
import nemo

# these are necessary for logging

parser = argparse.ArgumentParser(description='CIFAR10')
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--num_epochs", default=10, type=int)
parser.add_argument("--max_steps", default=None, type=int)
parser.add_argument("--learning_rate", default=0.003, type=float)
parser.add_argument("--weight_decay", default=0.0001, type=float)
parser.add_argument("--momentum", default=0.91, type=float)
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--data_root", default=os.getcwd(), type=str)
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

# instantiate Neural Factory with supported backend
neural_factory = nemo.core.NeuralModuleFactory(
    backend=nemo.core.Backend.PyTorch,
    local_rank=args.local_rank,
    # Set this to nemo.core.Optimization.mxprO1
    # if you have Volta or Turing GPU
    optimization_level=nemo.core.Optimization.mxprO0,
    create_tb_writer=False)


cifar_train = neural_factory.get_module(
    name="CIFAR10DataLayer", collection="nemo_lpr",
    params={"batch_size": batch_size,
            "shuffle": True,
            "train": True,
            "root": args.data_root,
            "placement": device
            })

cifar_eval = neural_factory.get_module(
    name="CIFAR10DataLayer", collection="nemo_lpr",
    params={"batch_size": batch_size,
            "shuffle": True,
            "train": False,
            "root": args.data_root,
            "placement": device
            })

classifier = neural_factory.get_module(
    name="SimpleCNNClassifier",
    collection="nemo_lpr",
    params={})

loss = neural_factory.get_module(name="CrossEntropyLoss",
                                 collection="toys",
                                 params={"placement": device})

t_image, t_label = cifar_train()
t_pred = classifier(image=t_image)
t_loss = loss(predictions=t_pred, labels=t_label)

e_image, e_label = cifar_eval()
e_pred = classifier(image=e_image)
e_loss = loss(predictions=e_pred, labels=e_label)

callback = nemo.core.SimpleLossLoggerCallback(
    tensors=[t_loss, t_pred, t_label],
    step_freq=50,
    print_func=lambda x: compute_accuracy(x))

callback_eval = nemo.core.EvaluatorCallback(
    eval_tensors=[
        e_loss,
        e_pred,
        e_label],
    user_iter_callback=eval_iter_callback,
    user_epochs_done_callback=eval_epochs_done_callback,
    eval_step=1000)

neural_factory.train(
    tensors_to_optimize=[t_loss],
    callbacks=[callback, callback_eval],
    optimizer="sgd",
    optimization_params={
            "num_epochs": num_epochs,
            "lr": learning_rate,
            "max_steps": max_steps,
            "weight_decay": weight_decay,
            "momentum": momentum}
)

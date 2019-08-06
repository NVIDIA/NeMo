import os
from nemo.backends.pytorch.torchvision.helpers import eval_epochs_done_callback
import torch
import statistics
import argparse


# these are necessary for logging


###
# Custom callbacks
###

def compute_accuracy_impl(output, target):
    # output: torch.Size([6, 9, 50, 50])
    # target: torch.Size([1, 9, 50, 50])
    assert(output.dim() == 4)
    assert(target.dim() == 4)

    res = []
    for i0 in range(0, output.size()[0]):
        diff = output[i0, :, :, :] - target[0, :, :, :]
        res.append(torch.sum(torch.sqrt(diff*diff)).item())

    return res


# Generate accuracy information for the user (not used for training)
def compute_dope_accuracy(tensors):
    output = tensors[0]
    target = tensors[1]
    res = compute_accuracy_impl(output, target)
    return "Train Top@1 accuracy  {0} ".format(res)


# Generate accuracy information for the user (not used for training)
def dope_eval_iter_callback(tensors, global_vars):
    if not 'eval_loss' in global_vars.keys():
        global_vars['eval_loss'] = []
    for kv, v in tensors.items():
        if kv.startswith('loss'):
            global_vars['eval_loss'].append(v.item())

    if 'top1' not in global_vars.keys():
        global_vars['top1'] = []

    a_output = None
    a_labels = None
    b_output = None
    b_labels = None

    # Names of input tensors:
    #  belief_label~~~944402f8-23af-4c20-8a8d-fd59efcc4570
    #  affinity_label~~~944402f8-23af-4c20-8a8d-fd59efcc4570
    #  belief_output~~~3fdc8f42-350a-4ff0-8158-fa65769ab809
    #  affinity_output~~~3fdc8f42-350a-4ff0-8158-fa65769ab809
    #  loss~~~679008fc-2c47-4bfe-ae06-0dfcb5bab05b
    for kv, v in tensors.items():
        if kv.startswith('affinity_output'):
            a_output = tensors[kv]
        elif kv.startswith('affinity_label'):
            a_labels = tensors[kv]
        elif kv.startswith('belief_output'):
            b_output = tensors[kv]
        elif kv.startswith('belief_label'):
            b_labels = tensors[kv]

    if a_output is None or b_output is None:
        raise Exception('one of the outputs is None')

    res = compute_accuracy_impl(b_output, b_labels)
    global_vars['top1'].append(statistics.mean(res))


######


parser = argparse.ArgumentParser(description='DOPE')
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--batch_size", default=1, type=int)
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
    optimization_level=nemo.core.Optimization.nothing)

dope_train = neural_factory.get_module(
    name="DopeDataLayer", collection="nemo_lpr",
    params={"batch_size": batch_size,
            "shuffle": True,
            "train": True,
            "root": args.data_root,
            "placement": device
            })

dope_eval = neural_factory.get_module(
    name="DopeDataLayer", collection="nemo_lpr",
    params={"batch_size": batch_size,
            "shuffle": True,
            "train": False,
            "root": args.data_root,
            "placement": device
            })

network = neural_factory.get_module(name="DopeNetwork", collection="nemo_lpr",
                                    params={})

loss = neural_factory.get_module(name="DopeDualLoss", collection="toys",
                                 params={"placement": device})

t_image, t_aff_label, t_bel_label = dope_train()
t_bel_pred, t_aff_pred = network(image=t_image)
t_loss = loss(belief_predictions=t_bel_pred,
              belief_labels=t_bel_label,
              affinity_predictions=t_aff_pred,
              affinity_labels=t_aff_label)


e_image, e_aff_label, e_bel_label = dope_eval()
e_bel_pred, e_aff_pred = network(image=e_image)
e_loss = loss(belief_predictions=e_bel_pred,
              belief_labels=e_bel_label,
              affinity_predictions=e_aff_pred,
              affinity_labels=e_aff_label)


callback_logger = nemo.core.SimpleLossLoggerCallback(
    step_frequency=50,
    tensor_list2string=lambda x: str(x[0].item()),
    tensor_list2string_evl=lambda x: compute_dope_accuracy(x))

callback_eval = nemo.core.EvaluatorCallback(
    eval_tensors=[e_loss, e_bel_pred, e_bel_label, e_aff_pred, e_aff_label],
    user_iter_callback=dope_eval_iter_callback,
    user_epochs_done_callback=eval_epochs_done_callback,
    eval_step=1000)

# Instantiate an optimizer to perform `train` action
optimizer = neural_factory.get_trainer(
    params={"optimization_params": {"num_epochs": num_epochs, "lr": learning_rate,
                                    "max_steps": max_steps,
                                    "weight_decay": weight_decay,
                                    "momentum": momentum}})
# Training
optimizer.train(tensors_to_optimize=[t_loss],
                tensors_to_evaluate=[t_bel_pred,
                                     t_bel_label, t_aff_pred, t_aff_label],
                callbacks=[callback_logger, callback_eval])

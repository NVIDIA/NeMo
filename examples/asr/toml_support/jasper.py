# Copyright (c) 2019 NVIDIA Corporation
import argparse
import os
import copy
from shutil import copyfile
import toml
import nemo
import nemo_asr
from nemo_asr.helpers import monitor_asr_train_progress, \
    process_evaluation_batch, process_evaluation_epoch
from nemo.utils.lr_policies import SquareAnnealing
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Jasper')
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--num_epochs", default=100, type=int)
parser.add_argument("--save_freq", default=15000, type=int)
parser.add_argument("--eval_freq", default=1000, type=int)
parser.add_argument("--lr", default=0.02, type=float)
parser.add_argument("--weight_decay", default=0.001, type=float)
parser.add_argument("--train_manifest", type=str)
parser.add_argument("--model_toml", type=str)
parser.add_argument("--exp_name", default="Jasper", type=str)
parser.add_argument("--val_manifest1", type=str)
parser.add_argument("--val_manifest2", type=str)
parser.add_argument("--val_manifest3", default="", type=str)
parser.add_argument("--batch_per_step", default=1, type=int)
parser.add_argument("--optimizer", default="novograd", type=str)
parser.add_argument("--larc", action='store_true')
parser.add_argument("--larc_eta", default=1e-3, type=float)
parser.add_argument("--luc", action='store_true')
parser.add_argument("--luc_eta", default=1e-3, type=float)
parser.add_argument("--logdir", default=None, type=str)
parser.add_argument("--warmup_steps", default=0, type=float)
parser.add_argument("--mixed_precision", action='store_false')

args = parser.parse_args()
batch_size = args.batch_size
lr = args.lr
num_epochs = args.num_epochs
num_gpus = args.num_gpus
weight_decay = args.weight_decay
bp_step = args.batch_per_step
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
                              args.larc,
                              args.luc))

verbose = (args.local_rank is None or args.local_rank == 0)
name = construct_name(args.exp_name, lr, batch_size, num_gpus, num_epochs,
                      weight_decay, bp_step)
save_folder = args.logdir if args.logdir else "/results/checkpoints/"
if verbose:
    print(name)
    if args.logdir:
        name = os.path.join(args.logdir, name)

        def copy_wo_overwrite(file_to_copy):
            basename = os.path.basename(file_to_copy)
            i = 0
            basename, ending = os.path.splitext(basename)
            basename = basename + "_run{}" + ending
            while(True):
                if os.path.isfile(os.path.join(args.logdir, basename.format(i))):
                    i += 1
                    continue
                else:
                    copyfile(file_to_copy,
                             os.path.join(args.logdir, basename.format(i)))
                    break
        if not os.path.exists(args.logdir):
            # raise ValueError("logdir is not empty")
            os.mkdir(args.logdir)
            copy_wo_overwrite(args.model_toml)
            copy_wo_overwrite(os.path.realpath(__file__))
        else:
            copy_wo_overwrite(args.model_toml)
            copy_wo_overwrite(__file__)

    tb_writer = SummaryWriter(name)
else:
    tb_writer = None

if args.local_rank is not None:
    device = nemo.core.DeviceType.AllGpu
    if verbose:
        print('Doing ALL GPU')
else:
    device = nemo.core.DeviceType.GPU

jasper_params = toml.load(args.model_toml)
vocab = jasper_params['labels']['labels']
sample_rate = jasper_params['sample_rate']

train_manifest = args.train_manifest
val_manifest1 = args.val_manifest1
val_manifest2 = args.val_manifest2
val_manifest3 = args.val_manifest3

# instantiate Neural Factory with supported backend
neural_factory = nemo.core.NeuralModuleFactory(
    backend=nemo.core.Backend.PyTorch,
    local_rank=args.local_rank,
    optimization_level=nemo.core.Optimization.mxprO1 if args.mixed_precision
    else nemo.core.Optimization.nothing,
    placement=device,
    cudnn_benchmark=True)

# Calculate num_workers for dataloader
num_eval_dataloaders = 2
if args.val_manifest3:
    num_eval_dataloaders += 1
total_cpus = os.cpu_count() - num_eval_dataloaders
cpu_per_traindl = int(total_cpus / neural_factory.world_size)

# perturb_config = jasper_params.get('perturb', None)
train_dl_params = copy.deepcopy(jasper_params["AudioToTextDataLayer"])
train_dl_params.update(jasper_params["AudioToTextDataLayer"]["train"])
del train_dl_params["train"]
del train_dl_params["eval"]

data_layer = neural_factory.get_module(
    name="AudioToTextDataLayer",
    params={
        # "perturb_config": perturb_config,
        "manifest_filepath": train_manifest,
        "sample_rate": sample_rate,
        "labels": vocab,
        "batch_size": batch_size,
        "num_workers": cpu_per_traindl,
        "placement": device,
        "verbose": verbose,
        **train_dl_params,
        # "normalize_transcripts": False
    },
    collection="nemo_asr")

N = len(data_layer)
step_per_epoch = int(N / (batch_size * num_gpus))
if verbose:
    print('-----------------')
    print('Have {0} examples to train on.'.format(N))
    print('-----------------')

data_preprocessor = neural_factory.get_module(
    name="AudioPreprocessing",
    collection="nemo_asr",
    params={
        "sample_rate": sample_rate,
        **jasper_params["AudioPreprocessing"]
    })

multiply_batch_config = jasper_params.get('MultiplyBatch', None)
if multiply_batch_config:
    multiply_batch = neural_factory.get_module(
        name="MultiplyBatch",
        collection="nemo_asr",
        params=multiply_batch_config)

spectr_augment_config = jasper_params.get('SpectrogramAugmentation', None)
if spectr_augment_config:
    data_spectr_augmentation = neural_factory.get_module(
        name="SpectrogramAugmentation",
        collection="nemo_asr",
        params=spectr_augment_config)

eval_dl_params = copy.deepcopy(jasper_params["AudioToTextDataLayer"])
eval_dl_params.update(jasper_params["AudioToTextDataLayer"]["eval"])
del eval_dl_params["train"]
del eval_dl_params["eval"]
data_layer_eval1 = neural_factory.get_module(
    name="AudioToTextDataLayer",
    params={
        # "perturb_config": perturb_config,
        "manifest_filepath": val_manifest1,
        "sample_rate": sample_rate,
        "labels": vocab,
        "batch_size": batch_size,
        "num_workers": cpu_per_traindl,
        "placement": device,
        "verbose": verbose,
        **eval_dl_params
        # "normalize_transcripts": False
    },
    collection="nemo_asr")
data_layer_eval2 = neural_factory.get_module(
    name="AudioToTextDataLayer",
    params={
        # "perturb_config": perturb_config,
        "manifest_filepath": val_manifest2,
        "sample_rate": sample_rate,
        "labels": vocab,
        "batch_size": batch_size,
        "num_workers": cpu_per_traindl,
        "placement": device,
        "verbose": verbose,
        **eval_dl_params
        # "normalize_transcripts": False
    },
    collection="nemo_asr")
if val_manifest3:
    data_layer_eval3 = neural_factory.get_module(
        name="AudioToTextDataLayer",
        params={
            # "perturb_config": perturb_config,
            "manifest_filepath": val_manifest3,
            "sample_rate": sample_rate,
            "labels": vocab,
            "batch_size": batch_size,
            "num_workers": cpu_per_traindl,
            "placement": device,
            "verbose": verbose,
            **eval_dl_params
            # "normalize_transcripts": False
        },
        collection="nemo_asr")

jasper_encoder = neural_factory.get_module(
    name="JasperEncoder",
    params={
        "feat_in": jasper_params["AudioPreprocessing"]["features"],
        **jasper_params["JasperEncoder"]
    },
    collection="nemo_asr")

jasper_decoder = neural_factory.get_module(
    name="JasperDecoderForCTC",
    params={
        "feat_in": jasper_params["JasperEncoder"]["jasper"][-1]["filters"],
        "num_classes": len(vocab),
        "placement": device
    },
    collection="nemo_asr")

ctc_loss = neural_factory.get_module(
    name="CTCLossNM",
    params={
        "num_classes": len(vocab),
        "placement": device
    },
    collection="nemo_asr")

greedy_decoder = neural_factory.get_module(
    name="GreedyCTCDecoder",
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
processed_signal_t, p_length_t = data_preprocessor(
    input_signal=audio_signal_t,
    length=a_sig_length_t)

if multiply_batch_config:
    processed_signal_t, p_length_t, transcript_t, transcript_len_t = \
        multiply_batch(
            in_x=processed_signal_t, in_x_len=p_length_t, in_y=transcript_t,
            in_y_len=transcript_len_t)

if spectr_augment_config:
    processed_signal_t = data_spectr_augmentation(
        input_spec=processed_signal_t)

encoded_t, encoded_len_t = jasper_encoder(
    audio_signal=processed_signal_t,
    length=p_length_t)
log_probs_t = jasper_decoder(encoder_output=encoded_t)
predictions_t = greedy_decoder(log_probs=log_probs_t)
loss_t = ctc_loss(
    log_probs=log_probs_t,
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
encoded_e1, encoded_len_e1 = jasper_encoder(
    audio_signal=processed_signal_e1,
    length=p_length_e1)
log_probs_e1 = jasper_decoder(encoder_output=encoded_e1)
predictions_e1 = greedy_decoder(log_probs=log_probs_e1)
loss_e1 = ctc_loss(
    log_probs=log_probs_e1,
    targets=transcript_e1,
    input_length=encoded_len_e1,
    target_length=transcript_len_e1)

# Eval DAG2
audio_signal_e2, a_sig_length_e2, transcript_e2, transcript_len_e2 = \
    data_layer_eval2()
processed_signal_e2, p_length_e2 = data_preprocessor(
    input_signal=audio_signal_e2,
    length=a_sig_length_e2)
encoded_e2, encoded_len_e2 = jasper_encoder(
    audio_signal=processed_signal_e2,
    length=p_length_e2)
log_probs_e2 = jasper_decoder(encoder_output=encoded_e2)
predictions_e2 = greedy_decoder(log_probs=log_probs_e2)
loss_e2 = ctc_loss(
    log_probs=log_probs_e2,
    targets=transcript_e2,
    input_length=encoded_len_e2,
    target_length=transcript_len_e2)

# Eval DAG3
if val_manifest3:
    audio_signal_e3, a_sig_length_e3, transcript_e3, transcript_len_e3 = \
        data_layer_eval3()
    processed_signal_e3, p_length_e3 = data_preprocessor(
        input_signal=audio_signal_e3,
        length=a_sig_length_e3)
    encoded_e3, encoded_len_e3 = jasper_encoder(
        audio_signal=processed_signal_e3,
        length=p_length_e3)
    log_probs_e3 = jasper_decoder(encoder_output=encoded_e3)
    predictions_e3 = greedy_decoder(log_probs=log_probs_e3)
    loss_e3 = ctc_loss(
        log_probs=log_probs_e3,
        targets=transcript_e3,
        input_length=encoded_len_e3,
        target_length=transcript_len_e3)

# Callbacks needed to print info to console and Tensorboard
train_callback = nemo.core.SimpleLossLoggerCallback(
    tensor_list2string=lambda x: str(x[0].item()),
    tensorboard_writer=tb_writer,
    tensor_list2string_evl=lambda x: monitor_asr_train_progress(
        x, labels=vocab))

saver_callback = nemo.core.ModuleSaverCallback(
    save_modules_list=[jasper_encoder,
                       jasper_decoder],
    folder="/results/",
    step_frequency=-1)

chpt_callback = nemo.core.CheckpointCallback(
    folder=save_folder,
    step_freq=args.save_freq)

eval_callback1 = nemo.core.EvaluatorCallback(
    eval_tensors=[loss_e1, predictions_e1, transcript_e1, transcript_len_e1],
    user_iter_callback=lambda x, y: process_evaluation_batch(
        x, y, labels=vocab),
    user_epochs_done_callback=lambda x: process_evaluation_epoch(
        x, tag="DEV-CLEAN"),

    eval_step=args.eval_freq,
    tensorboard_writer=tb_writer)

eval_callback2 = nemo.core.EvaluatorCallback(
    eval_tensors=[loss_e2, predictions_e2, transcript_e2, transcript_len_e2],
    user_iter_callback=lambda x, y: process_evaluation_batch(
        x, y, labels=vocab),
    user_epochs_done_callback=lambda x: process_evaluation_epoch(
        x, tag="DEV-OTHER"),
    eval_step=args.eval_freq,
    tensorboard_writer=tb_writer)

callbacks = [train_callback, saver_callback, eval_callback1,
             eval_callback2, chpt_callback]

if val_manifest3:
    eval_callback3 = nemo.core.EvaluatorCallback(
        eval_tensors=[loss_e3, predictions_e3,
                      transcript_e3, transcript_len_e3],
        user_iter_callback=lambda x, y: process_evaluation_batch(
            x, y, labels=vocab),
        user_epochs_done_callback=lambda x: process_evaluation_epoch(
            x, tag="Drive-Clean"),
        eval_step=args.eval_freq,
        tensorboard_writer=tb_writer)
    callbacks.append(eval_callback3)


optimizer = neural_factory.get_trainer(
    params={"optimizer_kind": args.optimizer,
            "optimization_params": {"num_epochs": num_epochs, "lr": lr,
                                    "weight_decay": weight_decay,
                                    "larc": args.larc,
                                    "larc_eta": args.larc_eta,
                                    "luc": args.luc,
                                    "luc_eta": args.luc_eta}})
optimizer.train(tensors_to_optimize=[loss_t],
                callbacks=callbacks,
                tensors_to_evaluate=[predictions_t, transcript_t,
                                     transcript_len_t],
                lr_policy=SquareAnnealing(num_epochs * step_per_epoch,
                                          warmup_steps=warmup),
                batches_per_step=bp_step)

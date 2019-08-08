# Copyright (c) 2019 NVIDIA Corporation
import argparse
import os
import copy
from shutil import copyfile
from ruamel.yaml import YAML
import nemo
import nemo_asr
from nemo_asr.helpers import monitor_asr_train_progress, \
    process_evaluation_batch, process_evaluation_epoch
from nemo.utils.lr_policies import CosineAnnealing
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
parser.add_argument("--model_config", type=str)
parser.add_argument("--exp_name", default="Jasper", type=str)
parser.add_argument("--val_manifest1", type=str)
parser.add_argument("--val_manifest2", type=str)
parser.add_argument("--val_manifest3", default="", type=str)
parser.add_argument("--batch_per_step", default=1, type=int)
parser.add_argument("--optimizer", default="novograd", type=str)
parser.add_argument("--logdir", default=None, type=str)
parser.add_argument("--warmup_steps", default=0, type=float)
parser.add_argument("--disable_mixed_precision", action='store_true')

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
    return ("{0}-lr_{1}-bs_{2}x{3}-e_{4}-wd_{5}-opt_{6}-bps_{7}".format(
                name, lr,
                batch_size,
                num_gpus,
                num_epochs, wd,
                args.optimizer,
                batch_per_step))


master_process = (args.local_rank is None or args.local_rank == 0)
name = construct_name(args.exp_name, lr, batch_size, num_gpus, num_epochs,
                      weight_decay, bp_step)
save_folder = args.logdir if args.logdir else "/results/checkpoints/"
if master_process:
    print(name)
    if args.logdir:
        name = os.path.join(args.logdir, name)

        def copy_wo_overwrite(file_to_copy):
            basename = os.path.basename(file_to_copy)
            i = 0
            basename, ending = os.path.splitext(basename)
            basename = basename + "_run{}" + ending
            while(True):
                if os.path.isfile(
                        os.path.join(args.logdir, basename.format(i))):
                    i += 1
                    continue
                else:
                    copyfile(file_to_copy,
                             os.path.join(args.logdir, basename.format(i)))
                    break
        if not os.path.exists(args.logdir):
            # raise ValueError("logdir is not empty")
            os.mkdir(args.logdir)
            copy_wo_overwrite(args.model_config)
            copy_wo_overwrite(os.path.realpath(__file__))
        else:
            copy_wo_overwrite(args.model_config)
            copy_wo_overwrite(__file__)

    tb_writer = SummaryWriter(name)
else:
    tb_writer = None

if args.local_rank is not None:
    device = nemo.core.DeviceType.AllGpu
    if master_process:
        print('Doing ALL GPU')
else:
    device = nemo.core.DeviceType.GPU

yaml = YAML(typ="safe")
with open(args.model_config) as f:
    jasper_params = yaml.load(f)
vocab = jasper_params['labels']
sample_rate = jasper_params['sample_rate']

train_manifest = args.train_manifest
val_manifest1 = args.val_manifest1
val_manifest2 = args.val_manifest2
val_manifest3 = args.val_manifest3

# We use mixed precision by default
opt_level = (nemo.core.Optimization.nothing if args.disable_mixed_precision
             else nemo.core.Optimization.mxprO1)

# instantiate Neural Factory with supported backend
neural_factory = nemo.core.NeuralModuleFactory(
    backend=nemo.core.Backend.PyTorch,
    local_rank=args.local_rank,
    optimization_level=opt_level,
    placement=device,
    master_process=master_process,
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

data_layer = nemo_asr.AudioToTextDataLayer(
    manifest_filepath=train_manifest,
    sample_rate=sample_rate,
    labels=vocab,
    batch_size=batch_size,
    num_workers=cpu_per_traindl,
    factory=neural_factory,
    **train_dl_params,
    # normalize_transcripts=False
)

N = len(data_layer)
step_per_epoch = int(N / (batch_size * num_gpus))
if master_process:
    print('-----------------')
    print('Have {0} examples to train on.'.format(N))
    print('-----------------')

data_preprocessor = nemo_asr.AudioPreprocessing(
    sample_rate=sample_rate,
    factory=neural_factory,
    **jasper_params["AudioPreprocessing"])

multiply_batch_config = jasper_params.get('MultiplyBatch', None)
if multiply_batch_config:
    multiply_batch = nemo_asr.MultiplyBatch(
        factory=neural_factory, **multiply_batch_config)

spectr_augment_config = jasper_params.get('SpectrogramAugmentation', None)
if spectr_augment_config:
    data_spectr_augmentation = nemo_asr.SpectrogramAugmentation(
        factory=neural_factory, **spectr_augment_config)

eval_dl_params = copy.deepcopy(jasper_params["AudioToTextDataLayer"])
eval_dl_params.update(jasper_params["AudioToTextDataLayer"]["eval"])
del eval_dl_params["train"]
del eval_dl_params["eval"]
data_layer_eval1 = nemo_asr.AudioToTextDataLayer(
    manifest_filepath=val_manifest1,
    sample_rate=sample_rate,
    labels=vocab,
    batch_size=batch_size,
    num_workers=cpu_per_traindl,
    factory=neural_factory,
    **eval_dl_params,
    # "normalize_transcripts": False
)

data_layer_eval2 = nemo_asr.AudioToTextDataLayer(
    manifest_filepath=val_manifest2,
    sample_rate=sample_rate,
    labels=vocab,
    batch_size=batch_size,
    num_workers=cpu_per_traindl,
    factory=neural_factory,
    **eval_dl_params,
    # "normalize_transcripts": False
)

if val_manifest3:
    data_layer_eval3 = nemo_asr.AudioToTextDataLayer(
        manifest_filepath=val_manifest3,
        sample_rate=sample_rate,
        labels=vocab,
        batch_size=batch_size,
        num_workers=cpu_per_traindl,
        factory=neural_factory,
        **eval_dl_params,
        # "normalize_transcripts": False
    )

jasper_encoder = nemo_asr.JasperEncoder(
    feat_in=jasper_params["AudioPreprocessing"]["features"],
    factory=neural_factory,
    **jasper_params["JasperEncoder"])

jasper_decoder = nemo_asr.JasperDecoderForCTC(
    feat_in=jasper_params["JasperEncoder"]["jasper"][-1]["filters"],
    num_classes=len(vocab),
    factory=neural_factory)

ctc_loss = nemo_asr.CTCLossNM(
    num_classes=len(vocab), factory=neural_factory)

greedy_decoder = nemo_asr.GreedyCTCDecoder(factory=neural_factory)

if master_process:
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

# saver_callback = nemo.core.ModuleSaverCallback(
#     save_modules_list=[jasper_encoder,
#                        jasper_decoder],
#     folder="/results/",
#     step_frequency=-1)

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

callbacks = [train_callback, eval_callback1,
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
                                    "weight_decay": weight_decay}})
optimizer.train(tensors_to_optimize=[loss_t],
                callbacks=callbacks,
                tensors_to_evaluate=[predictions_t, transcript_t,
                                     transcript_len_t],
                lr_policy=CosineAnnealing(num_epochs * step_per_epoch,
                                          warmup_steps=warmup),
                batches_per_step=bp_step)

# Copyright (c) 2019 NVIDIA Corporation
import argparse
import copy
import os
from ruamel.yaml import YAML
import nemo
import nemo_asr
from nemo_asr.helpers import monitor_asr_train_progress, \
    process_evaluation_batch, process_evaluation_epoch, word_error_rate, \
    post_process_predictions, post_process_transcripts

from tensorboardX import SummaryWriter

tb_writer = SummaryWriter('jasper-an4')

parser = argparse.ArgumentParser(description='JasperSmall on AN4 dataset')
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--save_freq", default=200, type=int)
parser.add_argument("--test_after_training", action='store_true')
args = parser.parse_args()

if args.local_rank is not None:
    device = nemo.core.DeviceType.AllGpu
else:
    device = nemo.core.DeviceType.GPU

# instantiate Neural Factory with supported backend
neural_factory = nemo.core.NeuralModuleFactory(
    backend=nemo.core.Backend.PyTorch,
    local_rank=args.local_rank,
    optimization_level=nemo.core.Optimization.mxprO1,
    placement=device,
    random_seed=123)

yaml = YAML(typ="safe")
with open("../../tests/data/jasper_smaller.yaml") as f:
    jasper_params = yaml.load(f)
vocab = jasper_params['labels']
sample_rate = jasper_params['sample_rate']
batch_size = 16

train_manifest = "./an4data/an4_train.json"
val_manifest = "./an4data/an4_val.json"

train_dl_params = copy.deepcopy(jasper_params["AudioToTextDataLayer"])
train_dl_params.update(jasper_params["AudioToTextDataLayer"]["train"])
del train_dl_params["train"]
del train_dl_params["eval"]

data_layer = nemo_asr.AudioToTextDataLayer(
    manifest_filepath=train_manifest,
    sample_rate=sample_rate,
    labels=vocab,
    batch_size=batch_size,
    factory=neural_factory,
    **train_dl_params,
)

data_preprocessor = nemo_asr.AudioPreprocessing(
    sample_rate=sample_rate,
    factory=neural_factory,
    **jasper_params["AudioPreprocessing"])

eval_dl_params = copy.deepcopy(jasper_params["AudioToTextDataLayer"])
eval_dl_params.update(jasper_params["AudioToTextDataLayer"]["eval"])
del eval_dl_params["train"]
del eval_dl_params["eval"]

data_layer_eval = nemo_asr.AudioToTextDataLayer(
    manifest_filepath=val_manifest,
    sample_rate=sample_rate,
    labels=vocab,
    batch_size=batch_size,
    factory=neural_factory,
    **eval_dl_params,
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
# Eval DAG
audio_signal_e, a_sig_length_e, transcript_e, transcript_len_e = \
    data_layer_eval()
processed_signal_e, p_length_e = data_preprocessor(
    input_signal=audio_signal_e,
    length=a_sig_length_e)
encoded_e, encoded_len_e = jasper_encoder(audio_signal=processed_signal_e,
                                          length=p_length_e)
log_probs_e = jasper_decoder(encoder_output=encoded_e)
predictions_e = greedy_decoder(log_probs=log_probs_e)
loss_e = ctc_loss(log_probs=log_probs_e,
                  targets=transcript_e,
                  input_length=encoded_len_e,
                  target_length=transcript_len_e)

print(
    "Number of parameters in encoder: {0}".format(jasper_encoder.num_weights))

# Callbacks needed to print info to console and Tensorboard
train_callback = nemo.core.SimpleLossLoggerCallback(
    tensor_list2string=lambda x: str(x[0].item()),
    tensorboard_writer=tb_writer,
    tensor_list2string_evl=lambda x: monitor_asr_train_progress(x,
                                                                labels=vocab))

checkpointer_callback = nemo.core.CheckpointCallback(folder="jasper-an4",
                                                     step_freq=args.save_freq)

eval_tensors = [loss_e, predictions_e, transcript_e, transcript_len_e]
eval_callback = nemo.core.EvaluatorCallback(
    eval_tensors=eval_tensors,
    user_iter_callback=lambda x, y: process_evaluation_batch(
        x, y, labels=vocab),
    user_epochs_done_callback=process_evaluation_epoch,
    eval_step=200,
    tensorboard_writer=tb_writer)

optimizer = neural_factory.get_trainer(
    params={"optimizer_kind": "novograd",
            "optimization_params": {"num_epochs": 30, "lr": 1e-2,
                                    "weight_decay": 1e-3,
                                    "grad_norm_clip": None}})

optimizer.train(tensors_to_optimize=[loss_t],
                callbacks=[train_callback, eval_callback,
                           checkpointer_callback],
                tensors_to_evaluate=[predictions_t, transcript_t,
                                     transcript_len_t])

if args.test_after_training:
    # Create BeamSearch NM
    beam_search_with_lm = nemo_asr.BeamSearchDecoderWithLM(
        vocab=vocab,
        beam_width=64,
        alpha=2.,
        beta=1.5,
        lm_path="../../tests/data/an4_train-lm.3gram.binary",
        num_cpus=max(os.cpu_count(), 1))
    beam_predictions = beam_search_with_lm(
        log_probs=log_probs_e, log_probs_length=encoded_len_e)
    eval_tensors.append(beam_predictions)

    infer_callback = nemo.core.InferenceCallback(
        eval_tensors=eval_tensors,
    )
    evaluated_tensors = optimizer.infer(
        callback=infer_callback,
    )
    greedy_hypotheses = post_process_predictions(evaluated_tensors[1], vocab)
    references = post_process_transcripts(
        evaluated_tensors[2], evaluated_tensors[3], vocab)
    wer = word_error_rate(hypotheses=greedy_hypotheses, references=references)
    print("Greedy WER: {:.2f}".format(wer*100))
    assert wer <= 0.27, ("Final evaluation greedy WER {:.2f}% was higher than "
                         "the required 27%". format(wer*100))

    beam_hypotheses = []
    # Over mini-batch
    for i in evaluated_tensors[-1]:
        # Over samples
        for j in i:
            beam_hypotheses.append(j[0][1])

    beam_wer = word_error_rate(
        hypotheses=beam_hypotheses, references=references)
    print("Beam WER {:.2f}".format(beam_wer*100))
    assert beam_wer <= 0.2, ("Final evaluation beam WER {:.2f}% was higher "
                             "than the required 20%". format(beam_wer*100))
    assert beam_wer <= wer, ("Final evaluation beam WER was higher than "
                             "the greedy wer.")

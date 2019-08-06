# Copyright (c) 2019 NVIDIA Corporation
import argparse
import nemo
import toml
from nemo_asr.helpers import monitor_asr_train_progress, \
    process_evaluation_batch, process_evaluation_epoch, word_error_rate

from tensorboardX import SummaryWriter

tb_writer = SummaryWriter('jasper-an4')

parser = argparse.ArgumentParser(description='JasperSmall on AN4 dataset')
parser.add_argument("--local_rank", default=None, type=int)
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
    placement=device)

batch_size = 16
jasper_model_definition = toml.load("../../tests/data/jasper_smaller.toml")
jasper_model_definition['placement'] = device
labels = jasper_model_definition['labels']['labels']

train_manifest = "./an4data/an4_train.json"
val_manifest = "./an4data/an4_val.json"
featurizer_config = jasper_model_definition['input']

data_layer = neural_factory.get_module(name="AudioToTextDataLayer",
                                       params={
                                           "featurizer_config": featurizer_config,
                                           "manifest_filepath": train_manifest,
                                           "labels": labels,
                                           "batch_size": batch_size,
                                           "placement": device,
                                       },
                                       collection="nemo_asr")
data_preprocessor = neural_factory.get_module(name="AudioPreprocessing",
                                              collection="nemo_asr",
                                              params=featurizer_config)

data_layer_eval = neural_factory.get_module(name="AudioToTextDataLayer",
                                            params={
                                                "featurizer_config": featurizer_config,
                                                "manifest_filepath": val_manifest,
                                                "labels": labels,
                                                "batch_size": batch_size,
                                                "placement": device
                                            },
                                            collection="nemo_asr")
data_preprocessor_eval = neural_factory.get_module(name="AudioPreprocessing",
                                                   collection="nemo_asr",
                                                   params=featurizer_config)

jasper_encoder = neural_factory.get_module(name="JasperEncoder",
                                           params={
                                               "jasper":jasper_model_definition["jasper"],
                                               "activation": jasper_model_definition["encoder"]["activation"],
                                               "feat_in": jasper_model_definition["input"]["features"],
                                           },
                                           collection="nemo_asr")
jasper_decoder = neural_factory.get_module(name="JasperDecoderForCTC",
                                           params={
                                               "feat_in": 1024,
                                               "num_classes": len(labels),
                                               "placement": device
                                           },
                                           collection="nemo_asr")

ctc_loss_train = neural_factory.get_module(name="CTCLossNM",
                                           params={
                                               "num_classes": len(labels),
                                               "placement": device
                                           },
                                           collection="nemo_asr")
ctc_loss_eval = neural_factory.get_module(name="CTCLossNM",
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
loss_t = ctc_loss_train(log_probs=log_probs_t,
                        targets=transcript_t,
                        input_length=encoded_len_t,
                        target_length=transcript_len_t)
# Eval DAG
audio_signal_e, a_sig_length_e, transcript_e, transcript_len_e = data_layer_eval()
processed_signal_e, p_length_e = data_preprocessor_eval(
    input_signal=audio_signal_e,
    length=a_sig_length_e)
encoded_e, encoded_len_e = jasper_encoder(audio_signal=processed_signal_e,
                                          length=p_length_e)
log_probs_e = jasper_decoder(encoder_output=encoded_e)
predictions_e = greedy_decoder(log_probs=log_probs_e)
loss_e = ctc_loss_eval(log_probs=log_probs_e,
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
                                                                labels=labels))

checkpointer_callback = nemo.core.CheckpointCallback(folder="jasper-an4",
                                                     step_freq=13)

eval_callback = nemo.core.EvaluatorCallback(
    eval_tensors=[loss_e, predictions_e, transcript_e, transcript_len_e],
    user_iter_callback=lambda x, y: process_evaluation_batch(
        x, y, labels=labels),
    user_epochs_done_callback=process_evaluation_epoch,
    eval_step=200,
    tensorboard_writer=tb_writer)

optimizer = neural_factory.get_trainer(
    params={"optimizer_kind": "novograd",
            "optimization_params": {"num_epochs": 30, "lr": 1e-2,
                                    "weight_decay": 1e-3}})

optimizer.train(tensors_to_optimize=[loss_t],
                callbacks=[train_callback, eval_callback,
                           checkpointer_callback],
                tensors_to_evaluate=[predictions_t, transcript_t,
                                     transcript_len_t])

tensor_dict = optimizer.infer(
    callback=eval_callback,
)

hypotheses = tensor_dict["predictions"]
references = tensor_dict["transcripts"]
wer = word_error_rate(hypotheses=hypotheses, references=references)
assert wer <= 0.27, ("Final evaluation WER {:.2f}% was higher than the "
                     "required 27%". format(wer*100))

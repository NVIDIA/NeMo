# Copyright (c) 2019 NVIDIA Corporation
import argparse
import pickle
import toml
import nemo
from nemo_asr.helpers import process_evaluation_batch, word_error_rate


parser = argparse.ArgumentParser(description='Jasper')
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--model_toml", type=str)
parser.add_argument("--val_manifest", type=str)
parser.add_argument("--save_dir", type=str)
parser.add_argument("--save_logits", default=None, type=str)

args = parser.parse_args()
batch_size = args.batch_size
num_gpus = args.num_gpus
save_dir = args.save_dir

if args.local_rank is not None:
    device = nemo.core.DeviceType.AllGpu
    print('Doing ALL GPU')
else:
    device = nemo.core.DeviceType.GPU

jasper_model_definition = toml.load(args.model_toml)
jasper_model_definition['placement'] = device
vocab = jasper_model_definition['labels']['labels']

val_manifest = args.val_manifest

featurizer_config = jasper_model_definition['input']
max_duration = featurizer_config.get("max_duration", 16.7)
pytorch_benchmark = False
featurizer_config["pad_to"] = 16

# instantiate Neural Factory with supported backend
neural_factory = nemo.core.NeuralModuleFactory(
    backend=nemo.core.Backend.PyTorch,
    local_rank=args.local_rank,
    optimization_level=nemo.core.Optimization.mxprO1,
    placement=device)

data_layer = neural_factory.get_module(name="AudioToTextDataLayer",
                                       params={
                                           "featurizer_config": featurizer_config,
                                           "manifest_filepath": val_manifest,
                                           "labels": vocab,
                                           "batch_size": batch_size,
                                           "placement": device,
                                           "shuffle": False,
                                       },
                                       collection="nemo_asr")
N = len(data_layer)
print('-----------------')
print('Evaluating {0} examples'.format(N))
print('-----------------')
data_preprocessor = neural_factory.get_module(name="AudioPreprocessing",
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
                                               "num_classes": len(vocab),
                                               "placement": device
                                           },
                                           collection="nemo_asr")
greedy_decoder = neural_factory.get_module(name="GreedyCTCDecoder",
                                           params={"placement": device},
                                           collection="nemo_asr")

print('\n\n\n================================')
print(
    "Number of parameters in encoder: {0}".format(jasper_encoder.num_weights))
print(
    "Number of parameters in decoder: {0}".format(jasper_decoder.num_weights))
print("Total number of parameters in model: {0}".format(
    jasper_decoder.num_weights + jasper_encoder.num_weights))
print('================================\n\n\n')

audio_signal_e1, a_sig_length_e1, transcript_e1, transcript_len_e1 = data_layer()
processed_signal_e1, p_length_e1 = data_preprocessor(
    input_signal=audio_signal_e1,
    length=a_sig_length_e1)
encoded_e1, encoded_len_e1 = jasper_encoder(audio_signal=processed_signal_e1,
                                            length=p_length_e1)
log_probs_e1 = jasper_decoder(encoder_output=encoded_e1)
predictions_e1 = greedy_decoder(log_probs=log_probs_e1)

infer_callback = nemo.core.EvaluatorCallback(
    eval_tensors=[log_probs_e1, predictions_e1,
                  transcript_e1, transcript_len_e1],
    user_iter_callback=lambda x, y: process_evaluation_batch(
        x, y, labels=vocab),
    user_epochs_done_callback=None,  # Unused
    eval_step=1,  # Unused
    tensorboard_writer=None  # Unused
)

optimizer = neural_factory.get_trainer(params={})
tensor_dict = optimizer.infer(
    callback=infer_callback,
    checkpoint_dir=save_dir,
)

hypotheses = tensor_dict["predictions"]
references = tensor_dict["transcripts"]
wer = word_error_rate(hypotheses=hypotheses, references=references)

print("Greedy WER {:.2f}".format(wer*100))
if args.save_logits:
    # Convert logits to list of numpy arrays
    logits = []
    for batch in tensor_dict["logits"]:
        for i in range(batch.shape[0]):
            logits.append(batch[i].cpu().numpy())
    with open(args.save_logits, 'wb') as f:
        pickle.dump(logits, f, protocol=pickle.HIGHEST_PROTOCOL)

# Copyright (c) 2019 NVIDIA Corporation
import argparse
import copy
import pickle
import os
from ruamel.yaml import YAML
import nemo
import nemo_asr
from nemo_asr.helpers import word_error_rate, post_process_predictions, \
                             post_process_transcripts


parser = argparse.ArgumentParser(description='Jasper')
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--model_config", type=str)
parser.add_argument("--val_manifest", type=str)
parser.add_argument("--load_dir", type=str)
parser.add_argument("--save_logprob", default=None, type=str)
parser.add_argument("--lm_path", default=None, type=str)

args = parser.parse_args()
batch_size = args.batch_size
num_gpus = args.num_gpus
load_dir = args.load_dir

if args.local_rank is not None:
    if args.lm_path:
        raise NotImplementedError(
            "Beam search decoder with LM does not currently support "
            "evaluation.")
    device = nemo.core.DeviceType.AllGpu
    print('Doing ALL GPU')
else:
    device = nemo.core.DeviceType.GPU

yaml = YAML(typ="safe")
with open(args.model_config) as f:
    jasper_params = yaml.load(f)
vocab = jasper_params['labels']
sample_rate = jasper_params['sample_rate']

val_manifest = args.val_manifest

# Instantiate Neural Factory with supported backend
neural_factory = nemo.core.NeuralModuleFactory(
    backend=nemo.core.Backend.PyTorch,
    local_rank=args.local_rank,
    optimization_level=nemo.core.Optimization.mxprO1,
    placement=device)

eval_dl_params = copy.deepcopy(jasper_params["AudioToTextDataLayer"])
eval_dl_params.update(jasper_params["AudioToTextDataLayer"]["eval"])
del eval_dl_params["train"]
del eval_dl_params["eval"]
data_layer = nemo_asr.AudioToTextDataLayer(
    manifest_filepath=val_manifest,
    sample_rate=sample_rate,
    labels=vocab,
    batch_size=batch_size,
    factory=neural_factory,
    **eval_dl_params)

N = len(data_layer)
print('-----------------')
print('Evaluating {0} examples'.format(N))
print('-----------------')

data_preprocessor = nemo_asr.AudioPreprocessing(
    sample_rate=sample_rate,
    factory=neural_factory,
    **jasper_params["AudioPreprocessing"])
jasper_encoder = nemo_asr.JasperEncoder(
    feat_in=jasper_params["AudioPreprocessing"]["features"],
    factory=neural_factory,
    **jasper_params["JasperEncoder"])
jasper_decoder = nemo_asr.JasperDecoderForCTC(
    feat_in=jasper_params["JasperEncoder"]["jasper"][-1]["filters"],
    num_classes=len(vocab),
    factory=neural_factory)
greedy_decoder = nemo_asr.GreedyCTCDecoder(factory=neural_factory)

if args.lm_path:
    beam_width = 128
    alpha = 2.
    beta = 1.5
    beam_search_with_lm = nemo_asr.BeamSearchDecoderWithLM(
        vocab=vocab,
        beam_width=beam_width,
        alpha=alpha,
        beta=beta,
        lm_path=args.lm_path,
        num_cpus=max(os.cpu_count(), 1))

print('\n\n\n================================')
print(
    "Number of parameters in encoder: {0}".format(jasper_encoder.num_weights))
print(
    "Number of parameters in decoder: {0}".format(jasper_decoder.num_weights))
print("Total number of parameters in model: {0}".format(
    jasper_decoder.num_weights + jasper_encoder.num_weights))
print('================================\n\n\n')

audio_signal_e1, a_sig_length_e1, transcript_e1, transcript_len_e1 =\
    data_layer()
processed_signal_e1, p_length_e1 = data_preprocessor(
    input_signal=audio_signal_e1,
    length=a_sig_length_e1)
encoded_e1, encoded_len_e1 = jasper_encoder(audio_signal=processed_signal_e1,
                                            length=p_length_e1)
log_probs_e1 = jasper_decoder(encoder_output=encoded_e1)
predictions_e1 = greedy_decoder(log_probs=log_probs_e1)

eval_tensors = [log_probs_e1, predictions_e1,
                transcript_e1, transcript_len_e1, encoded_len_e1]

if args.lm_path:
    beam_predictions_e1 = beam_search_with_lm(
        log_probs=log_probs_e1, log_probs_length=encoded_len_e1)
    eval_tensors.append(beam_predictions_e1)

infer_callback = nemo.core.InferenceCallback(
    eval_tensors=eval_tensors,
)

optimizer = neural_factory.get_trainer(params={})
evaluated_tensors = optimizer.infer(
    callback=infer_callback,
    checkpoint_dir=load_dir,
)


greedy_hypotheses = post_process_predictions(evaluated_tensors[1], vocab)
references = post_process_transcripts(
    evaluated_tensors[2], evaluated_tensors[3], vocab)
wer = word_error_rate(hypotheses=greedy_hypotheses, references=references)
print("Greedy WER {:.2f}".format(wer*100))

if args.lm_path:
    beam_hypotheses = []
    # Over mini-batch
    for i in evaluated_tensors[-1]:
        # Over samples
        for j in i:
            beam_hypotheses.append(j[0][1])

    wer = word_error_rate(hypotheses=beam_hypotheses, references=references)
    print("Beam WER {:.2f}".format(wer*100))

if args.save_logprob:
    # Convert logits to list of numpy arrays
    logprob = []
    for i, batch in enumerate(evaluated_tensors[0]):
        for j in range(batch.shape[0]):
            logprob.append(
                batch[j][:evaluated_tensors[4][i][j], :].cpu().numpy())
    with open(args.save_logprob, 'wb') as f:
        pickle.dump(logprob, f, protocol=pickle.HIGHEST_PROTOCOL)

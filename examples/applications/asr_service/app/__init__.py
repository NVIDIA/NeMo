import nemo
import nemo_asr
from ruamel.yaml import YAML
import os

from flask import Flask
app = Flask(__name__)
# make sure WORK_DIR exists before calling your service
# in this folder, the service will store received .wav files and constructed
# .json manifests
WORK_DIR = <PATH_TO_YOUR_WORKDIR>
MODEL_YAML = <PATH_TO_YOUR_YAML>
CHECKPOINT_ENCODER = <PATH_TO_ENCODER_CHECKPOINT>
CHECKPOINT_DECODER = <PATH_TO_DECODER_CHECKPOINT>
# Set this to True to enable beam search decoder
ENABLE_NGRAM = False
# This is only necessary if ENABLE_NGRAM = True. Otherwise, set to empty string
LM_PATH = <PATH_TO_KENLM_BINARY>


# Read model YAML
yaml = YAML(typ="safe")
with open(MODEL_YAML) as f:
    jasper_model_definition = yaml.load(f)
labels = jasper_model_definition['labels']

# Instantiate necessary Neural Modules
# Note that data layer is missing from here
neural_factory = nemo.core.NeuralModuleFactory(
    placement=nemo.core.DeviceType.GPU,
    backend=nemo.core.Backend.PyTorch)
data_preprocessor = nemo_asr.AudioPreprocessing(factory=neural_factory)
jasper_encoder = nemo_asr.JasperEncoder(
    factory=neural_factory,
    jasper=jasper_model_definition['JasperEncoder']['jasper'],
    activation=jasper_model_definition['JasperEncoder']['activation'],
    feat_in=jasper_model_definition['AudioPreprocessing']['features'])
jasper_encoder.restore_from(CHECKPOINT_ENCODER, local_rank=0)
jasper_decoder = nemo_asr.JasperDecoderForCTC(
    factory=neural_factory,
    feat_in=1024,
    num_classes=len(labels))
jasper_decoder.restore_from(CHECKPOINT_DECODER, local_rank=0)
greedy_decoder = nemo_asr.GreedyCTCDecoder(factory=neural_factory)

if ENABLE_NGRAM and os.path.isfile(LM_PATH):
    beam_search_with_lm = nemo_asr.BeamSearchDecoderWithLM(
        vocab=labels,
        beam_width=64,
        alpha=2.0,
        beta=1.0,
        lm_path=LM_PATH,
        num_cpus=max(os.cpu_count(), 1))
else:
    print("Beam search is not enabled")

from app import routes

if __name__ == '__main__':
    app.run()
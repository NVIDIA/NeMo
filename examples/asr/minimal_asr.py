# this is nemo's "core" package
from functools import partial

from ruamel.yaml import YAML

import nemo

# this is nemos's ASR collection of speech-recognition related Neural Modules
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.helpers import monitor_asr_train_progress

######### AFTER ###############################
from nemo.core.composite_neural_module import CompositeNeuralModule

train_manifest = "/mnt/D1/Data/an4_dataset/an4_train.json"
val_manifest = "/mnt/D1/Data/an4_dataset/an4_val.json"
model_config_file = "/home/okuchaiev/repos/NeMo/examples/asr/configs/jasper_an4.yaml"

yaml = YAML(typ="safe")
with open(model_config_file) as f:
    jasper_params = yaml.load(f)
# Get vocabulary.
vocab = jasper_params['labels']

nf = nemo.core.NeuralModuleFactory()
data_layer = nemo_asr.AudioToTextDataLayer.import_from_config(
    model_config_file,
    "AudioToTextDataLayer_train",
    overwrite_params={"manifest_filepath": train_manifest, "batch_size": 16},
)

data_preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor.import_from_config(
    model_config_file, "AudioToMelSpectrogramPreprocessor"
)

jasper_encoder = nemo_asr.JasperEncoder.import_from_config(model_config_file, "JasperEncoder")
jasper_decoder = nemo_asr.JasperDecoderForCTC.import_from_config(
    model_config_file, "JasperDecoderForCTC", overwrite_params={"num_classes": len(vocab)}
)
ctc_loss = nemo_asr.CTCLossNM(num_classes=len(vocab))
greedy_decoder = nemo_asr.GreedyCTCDecoder()

######### BEFORE ###############################

# audio_signal, audio_signal_len, transcript, transcript_len = data_layer()
# processed_signal, processed_signal_len = data_preprocessor(input_signal=audio_signal, length=audio_signal_len)
# encoded, encoded_len = jasper_encoder(audio_signal=processed_signal, length=processed_signal_len)
# log_probs = jasper_decoder(encoder_output=encoded)


def pipeline(audio_signal, audio_signal_len):
    processed_signal, processed_signal_len = data_preprocessor(input_signal=audio_signal, length=audio_signal_len)
    encoded, encoded_len = jasper_encoder(audio_signal=processed_signal, length=processed_signal_len)
    log_probs = jasper_decoder(encoder_output=encoded)
    return log_probs, encoded_len


Jasper = CompositeNeuralModule(modules=[data_preprocessor, jasper_encoder, jasper_decoder], pipeline=pipeline)
audio_signal, audio_signal_len, transcript, transcript_len = data_layer()
log_probs, encoded_len = Jasper(audio_signal, audio_signal_len)

predictions = greedy_decoder(log_probs=log_probs)
loss = ctc_loss(log_probs=log_probs, targets=transcript, input_length=encoded_len, target_length=transcript_len)
tensors_to_evaluate = [loss, predictions, transcript, transcript_len]

train_callback = nemo.core.SimpleLossLoggerCallback(
    tensors=tensors_to_evaluate, print_func=partial(monitor_asr_train_progress, labels=vocab)
)

nf.train(
    tensors_to_optimize=[loss],
    optimizer="novograd",
    callbacks=[train_callback],
    optimization_params={"num_epochs": 50, "lr": 0.01},
)

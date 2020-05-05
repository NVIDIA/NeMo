# ! /usr/bin/python
# -*- coding: utf-8 -*-

# =============================================================================
# Copyright (c) 2020 NVIDIA. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from functools import partial
from os.path import expanduser

from ruamel.yaml import YAML

import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.helpers import monitor_asr_train_progress
from nemo.core import NeuralGraph, OperationMode
from nemo.utils import logging

nf = nemo.core.NeuralModuleFactory()

logging.info(
    "This example shows how one can build a Jasper model using the explicit graph."
    F" This approach works for applications containing a single graph."
)

# Set paths to "manifests" and model configuration files.
train_manifest = "~/TestData/an4_dataset/an4_train.json"
val_manifest = "~/TestData/an4_dataset/an4_val.json"
model_config_file = "~/workspace/nemo/examples/asr/configs/jasper_an4.yaml"

yaml = YAML(typ="safe")
with open(expanduser(model_config_file)) as f:
    config = yaml.load(f)
# Get vocabulary.
vocab = config['labels']

# Create neural modules.
data_layer = nemo_asr.AudioToTextDataLayer.deserialize(
    config["AudioToTextDataLayer_train"], overwrite_params={"manifest_filepath": train_manifest, "batch_size": 16},
)

data_preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor.deserialize(config["AudioToMelSpectrogramPreprocessor"])

jasper_encoder = nemo_asr.JasperEncoder.deserialize(config["JasperEncoder"])
jasper_decoder = nemo_asr.JasperDecoderForCTC.deserialize(
    config["JasperDecoderForCTC"], overwrite_params={"num_classes": len(vocab)}
)
ctc_loss = nemo_asr.CTCLossNM(num_classes=len(vocab))
greedy_decoder = nemo_asr.GreedyCTCDecoder()

# Create the Jasper "model".
with NeuralGraph(operation_mode=OperationMode.both) as Jasper:
    # Copy one input port definitions - using "user" port names.
    Jasper.inputs["input"] = data_preprocessor.input_ports["input_signal"]
    # Bind selected inputs - bind other using the default port name.
    i_processed_signal, i_processed_signal_len = data_preprocessor(input_signal=Jasper.inputs["input"], length=Jasper)
    i_encoded, i_encoded_len = jasper_encoder(audio_signal=i_processed_signal, length=i_processed_signal_len)
    i_log_probs = jasper_decoder(encoder_output=i_encoded)
    # Bind selected outputs - using "user" port names.
    Jasper.outputs["log_probs"] = i_log_probs
    Jasper.outputs["encoded_len"] = i_encoded_len

# Print the summary.
logging.info(Jasper.summary())

# Serialize graph
serialized_jasper = Jasper.serialize()
print("Serialized:\n", serialized_jasper)

# Delete everything - aside of jasper encoder, just as a test to show that reusing work! ;)
del Jasper
del data_preprocessor
# del jasper_encoder #
del jasper_decoder

# Deserialize graph - copy of the JASPER "model".
jasper_copy = NeuralGraph.deserialize(serialized_jasper, reuse_existing_modules=True, name="jasper_copy")
serialized_jasper_copy = jasper_copy.serialize()
# print("Deserialized:\n", serialized_jasper_copy)
assert serialized_jasper == serialized_jasper_copy

# Create the "training" graph.
with NeuralGraph(name="training") as training_graph:
    # Create the "implicit" training graph.
    o_audio_signal, o_audio_signal_len, o_transcript, o_transcript_len = data_layer()
    # Use Jasper module as any other neural module.
    o_log_probs, o_encoded_len = jasper_copy(input=o_audio_signal, length=o_audio_signal_len)
    o_predictions = greedy_decoder(log_probs=o_log_probs)
    o_loss = ctc_loss(
        log_probs=o_log_probs, targets=o_transcript, input_length=o_encoded_len, target_length=o_transcript_len
    )
    # Set graph output.
    training_graph.outputs["o_loss"] = o_loss
    # training_graph.outputs["o_predictions"] = o_predictions # DOESN'T WORK?!?

# Print the summary.
logging.info(training_graph.summary())

tensors_to_evaluate = [o_loss, o_predictions, o_transcript, o_transcript_len]
train_callback = nemo.core.SimpleLossLoggerCallback(
    tensors=tensors_to_evaluate, print_func=partial(monitor_asr_train_progress, labels=vocab)
)
# import pdb;pdb.set_trace()
nf.train(
    # tensors_to_optimize=[o_loss, o_predictions], # DOESN'T WORK?!?
    training_graph=training_graph,
    optimizer="novograd",
    callbacks=[train_callback],
    optimization_params={"num_epochs": 50, "lr": 0.01},
)

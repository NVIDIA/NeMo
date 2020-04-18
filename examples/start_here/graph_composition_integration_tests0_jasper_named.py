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
from nemo.core import NeuralGraph
from nemo.utils import logging
from nemo.utils.app_state import AppState

nf = nemo.core.NeuralModuleFactory()
app_state = AppState()

logging.info(
    "This example shows how one can build a Jasper model using the `default` (implicit) graph."
    F" This approach works for applications containing a single graph."
)

# Set paths to "manifests" and model configuration files.
train_manifest = "~/TestData/an4_dataset/an4_train.json"
val_manifest = "~/TestData/an4_dataset/an4_val.json"
model_config_file = "~/workspace/nemo/examples/asr/configs/jasper_an4.yaml"

yaml = YAML(typ="safe")
with open(expanduser(model_config_file)) as f:
    jasper_params = yaml.load(f)
# Get vocabulary.
vocab = jasper_params['labels']

# Create neural modules.
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

# Create the Jasper composite module.
with NeuralGraph() as Jasper:
    i_processed_signal, i_processed_signal_len = data_preprocessor(input_signal=Jasper, length=Jasper)  # Bind inputs.
    i_encoded, i_encoded_len = jasper_encoder(audio_signal=i_processed_signal, length=i_processed_signal_len)
    i_log_probs = jasper_decoder(encoder_output=i_encoded)  # All output ports are bind (for now!)

with NeuralGraph(name="training") as training:
    # Create the "implicit" training graph.
    o_audio_signal, o_audio_signal_len, o_transcript, o_transcript_len = data_layer()
    # Use Jasper module as any other neural module.
    o_processed_signal, o_processed_signal_len, o_encoded, o_encoded_len, o_log_probs = Jasper(
        input_signal=o_audio_signal, length=o_audio_signal_len
    )
    o_predictions = greedy_decoder(log_probs=o_log_probs)
    o_loss = ctc_loss(
        log_probs=o_log_probs, targets=o_transcript, input_length=o_encoded_len, target_length=o_transcript_len
    )
    tensors_to_evaluate = [o_loss, o_predictions, o_transcript, o_transcript_len]

train_callback = nemo.core.SimpleLossLoggerCallback(
    tensors=tensors_to_evaluate, print_func=partial(monitor_asr_train_progress, labels=vocab)
)
# import pdb;pdb.set_trace()
nf.train(
    tensors_to_optimize=[o_loss],
    optimizer="novograd",
    callbacks=[train_callback],
    optimization_params={"num_epochs": 50, "lr": 0.01},
)

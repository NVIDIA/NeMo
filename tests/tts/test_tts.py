# ! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2020 NVIDIA. All Rights Reserved.
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

import os
import tarfile

import nemo
import nemo.collections.asr as nemo_asr
import nemo.collections.tts as nemo_tts
from tests.common_setup import NeMoUnitTest

logging = nemo.logging


class TestTTSPytorch(NeMoUnitTest):
    labels = [
        " ",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "'",
    ]
    manifest_filepath = "tests/data/asr/an4_train.json"

    def setUp(self) -> None:
        super().setUp()
        data_folder = "tests/data/"
        logging.info("Looking up for test speech data")
        if not os.path.exists(data_folder + "asr"):
            logging.info("Extracting speech data to: {0}".format(data_folder + "asr"))
            tar = tarfile.open("tests/data/asr.tar.gz", "r:gz")
            tar.extractall(path=data_folder)
            tar.close()
        else:
            logging.info("speech data found in: {0}".format(data_folder + "asr"))

    def test_tacotron2_training(self):
        data_layer = nemo_asr.AudioToTextDataLayer(
            manifest_filepath=self.manifest_filepath, labels=self.labels, batch_size=4,
        )
        preprocessing = nemo_asr.AudioToMelSpectrogramPreprocessor(
            window_size=None,
            window_stride=None,
            n_window_size=512,
            n_window_stride=128,
            normalize=None,
            preemph=None,
            dither=0,
            mag_power=1.0,
            pad_value=-11.52,
        )
        text_embedding = nemo_tts.TextEmbedding(len(self.labels), 256)
        t2_enc = nemo_tts.Tacotron2Encoder(encoder_n_convolutions=2, encoder_kernel_size=5, encoder_embedding_dim=256,)
        t2_dec = nemo_tts.Tacotron2Decoder(
            n_mel_channels=64,
            n_frames_per_step=1,
            encoder_embedding_dim=256,
            gate_threshold=0.5,
            prenet_dim=128,
            max_decoder_steps=1000,
            decoder_rnn_dim=512,
            p_decoder_dropout=0.1,
            p_attention_dropout=0.1,
            attention_rnn_dim=512,
            attention_dim=64,
            attention_location_n_filters=16,
            attention_location_kernel_size=15,
        )
        t2_postnet = nemo_tts.Tacotron2Postnet(
            n_mel_channels=64, postnet_embedding_dim=256, postnet_kernel_size=5, postnet_n_convolutions=3,
        )
        t2_loss = nemo_tts.Tacotron2Loss()
        makegatetarget = nemo_tts.MakeGate()

        # DAG
        audio, audio_len, transcript, transcript_len = data_layer()
        spec_target, spec_target_len = preprocessing(input_signal=audio, length=audio_len)

        transcript_embedded = text_embedding(char_phone=transcript)
        transcript_encoded = t2_enc(char_phone_embeddings=transcript_embedded, embedding_length=transcript_len,)
        mel_decoder, gate, _ = t2_dec(
            char_phone_encoded=transcript_encoded, encoded_length=transcript_len, mel_target=spec_target,
        )
        mel_postnet = t2_postnet(mel_input=mel_decoder)
        gate_target = makegatetarget(mel_target=spec_target, target_len=spec_target_len)
        loss_t = t2_loss(
            mel_out=mel_decoder,
            mel_out_postnet=mel_postnet,
            gate_out=gate,
            mel_target=spec_target,
            gate_target=gate_target,
            target_len=spec_target_len,
            seq_len=audio_len,
        )

        callback = nemo.core.SimpleLossLoggerCallback(
            tensors=[loss_t], print_func=lambda x: logging.info(f'Train Loss: {str(x[0].item())}'),
        )
        # Instantiate an optimizer to perform `train` action
        neural_factory = nemo.core.NeuralModuleFactory(
            backend=nemo.core.Backend.PyTorch, local_rank=None, create_tb_writer=False,
        )
        optimizer = neural_factory.get_trainer()
        optimizer.train(
            [loss_t], callbacks=[callback], optimizer="sgd", optimization_params={"num_epochs": 10, "lr": 0.0003},
        )

    def test_waveglow_training(self):
        data_layer = nemo_tts.AudioDataLayer(
            manifest_filepath=self.manifest_filepath, n_segments=4000, batch_size=4, sample_rate=16000
        )
        preprocessing = nemo_asr.AudioToMelSpectrogramPreprocessor(
            window_size=None,
            window_stride=None,
            n_window_size=512,
            n_window_stride=128,
            normalize=None,
            preemph=None,
            dither=0,
            mag_power=1.0,
            pad_value=-11.52,
        )
        waveglow = nemo_tts.WaveGlowNM(
            n_mel_channels=64,
            n_flows=6,
            n_group=4,
            n_early_every=4,
            n_early_size=2,
            n_wn_layers=4,
            n_wn_channels=256,
            wn_kernel_size=3,
            sample_rate=16000,
        )
        waveglow_loss = nemo_tts.WaveGlowLoss(sample_rate=16000)

        # DAG
        audio, audio_len, = data_layer()
        spec_target, _ = preprocessing(input_signal=audio, length=audio_len)

        z, log_s_list, log_det_W_list = waveglow(mel_spectrogram=spec_target, audio=audio)
        loss_t = waveglow_loss(z=z, log_s_list=log_s_list, log_det_W_list=log_det_W_list)

        callback = nemo.core.SimpleLossLoggerCallback(
            tensors=[loss_t], print_func=lambda x: logging.info(f'Train Loss: {str(x[0].item())}'),
        )
        # Instantiate an optimizer to perform `train` action
        neural_factory = nemo.core.NeuralModuleFactory(
            backend=nemo.core.Backend.PyTorch, local_rank=None, create_tb_writer=False,
        )
        optimizer = neural_factory.get_trainer()
        optimizer.train(
            [loss_t], callbacks=[callback], optimizer="sgd", optimization_params={"num_epochs": 10, "lr": 0.0003},
        )

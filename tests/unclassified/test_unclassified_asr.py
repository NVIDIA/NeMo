#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
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
import shutil
import tarfile
import unittest
from unittest import TestCase

import pytest
from ruamel.yaml import YAML

import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts import AudioDataset, WaveformFeaturizer, collections, parsers
from nemo.core import DeviceType

logging = nemo.logging


freq = 16000


@pytest.mark.usefixtures("neural_factory")
class TestASRPytorch(TestCase):
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
    manifest_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/asr/an4_train.json"))
    featurizer_config = {
        'window': 'hann',
        'dither': 1e-05,
        'normalize': 'per_feature',
        'frame_splicing': 1,
        'int_values': False,
        'window_stride': 0.01,
        'sample_rate': freq,
        'features': 64,
        'n_fft': 512,
        'window_size': 0.02,
    }
    yaml = YAML(typ="safe")

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/"))
        logging.info("Looking up for test ASR data")
        if not os.path.exists(os.path.join(data_folder, "asr")):
            logging.info("Extracting ASR data to: {0}".format(os.path.join(data_folder, "asr")))
            tar = tarfile.open(os.path.join(data_folder, "asr.tar.gz"), "r:gz")
            tar.extractall(path=data_folder)
            tar.close()
        else:
            logging.info("ASR data found in: {0}".format(os.path.join(data_folder, "asr")))

    @pytest.mark.unclassified
    def test_jasper_training(self):
        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/jasper_smaller.yaml"))) as file:
            jasper_model_definition = self.yaml.load(file)
        dl = nemo_asr.AudioToTextDataLayer(
            # featurizer_config=self.featurizer_config,
            manifest_filepath=self.manifest_filepath,
            labels=self.labels,
            batch_size=4,
        )
        pre_process_params = {
            'frame_splicing': 1,
            'features': 64,
            'window_size': 0.02,
            'n_fft': 512,
            'dither': 1e-05,
            'window': 'hann',
            'sample_rate': 16000,
            'normalize': 'per_feature',
            'window_stride': 0.01,
        }
        preprocessing = nemo_asr.AudioToMelSpectrogramPreprocessor(**pre_process_params)
        jasper_encoder = nemo_asr.JasperEncoder(
            feat_in=jasper_model_definition['AudioToMelSpectrogramPreprocessor']['features'],
            **jasper_model_definition['JasperEncoder'],
        )
        jasper_decoder = nemo_asr.JasperDecoderForCTC(feat_in=1024, num_classes=len(self.labels))
        ctc_loss = nemo_asr.CTCLossNM(num_classes=len(self.labels))

        # DAG
        audio_signal, a_sig_length, transcript, transcript_len = dl()
        processed_signal, p_length = preprocessing(input_signal=audio_signal, length=a_sig_length)

        encoded, encoded_len = jasper_encoder(audio_signal=processed_signal, length=p_length)
        # logging.info(jasper_encoder)
        log_probs = jasper_decoder(encoder_output=encoded)
        loss = ctc_loss(
            log_probs=log_probs, targets=transcript, input_length=encoded_len, target_length=transcript_len,
        )

        callback = nemo.core.SimpleLossLoggerCallback(
            tensors=[loss], print_func=lambda x: logging.info(f'Train Loss: {str(x[0].item())}'),
        )
        # Instantiate an optimizer to perform `train` action
        optimizer = nemo.backends.pytorch.actions.PtActions()
        optimizer.train(
            [loss], callbacks=[callback], optimizer="sgd", optimization_params={"num_epochs": 10, "lr": 0.0003},
        )

    @pytest.mark.unclassified
    def test_double_jasper_training(self):
        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/jasper_smaller.yaml"))) as file:
            jasper_model_definition = self.yaml.load(file)
        dl = nemo_asr.AudioToTextDataLayer(
            # featurizer_config=self.featurizer_config,
            manifest_filepath=self.manifest_filepath,
            labels=self.labels,
            batch_size=4,
        )
        pre_process_params = {
            'frame_splicing': 1,
            'features': 64,
            'window_size': 0.02,
            'n_fft': 512,
            'dither': 1e-05,
            'window': 'hann',
            'sample_rate': 16000,
            'normalize': 'per_feature',
            'window_stride': 0.01,
        }
        preprocessing = nemo_asr.AudioToMelSpectrogramPreprocessor(**pre_process_params)
        jasper_encoder1 = nemo_asr.JasperEncoder(
            feat_in=jasper_model_definition['AudioToMelSpectrogramPreprocessor']['features'],
            **jasper_model_definition['JasperEncoder'],
        )
        jasper_encoder2 = nemo_asr.JasperEncoder(
            feat_in=jasper_model_definition['AudioToMelSpectrogramPreprocessor']['features'],
            **jasper_model_definition['JasperEncoder'],
        )
        # mx_max1 = nemo.backends.pytorch.common.SimpleCombiner(mode="max")
        # mx_max2 = nemo.backends.pytorch.common.SimpleCombiner(mode="max")
        jasper_decoder1 = nemo_asr.JasperDecoderForCTC(feat_in=1024, num_classes=len(self.labels))
        jasper_decoder2 = nemo_asr.JasperDecoderForCTC(feat_in=1024, num_classes=len(self.labels))

        ctc_loss = nemo_asr.CTCLossNM(num_classes=len(self.labels))

        # DAG
        audio_signal, a_sig_length, transcript, transcript_len = dl()
        processed_signal, p_length = preprocessing(input_signal=audio_signal, length=a_sig_length)

        encoded1, encoded_len1 = jasper_encoder1(audio_signal=processed_signal, length=p_length)
        encoded2, encoded_len2 = jasper_encoder2(audio_signal=processed_signal, length=p_length)
        log_probs1 = jasper_decoder1(encoder_output=encoded1)
        log_probs2 = jasper_decoder2(encoder_output=encoded2)
        # log_probs = mx_max1(x1=log_probs1, x2=log_probs2)
        # encoded_len = mx_max2(x1=encoded_len1, x2=encoded_len2)
        log_probs = log_probs1
        encoded_len = encoded_len1
        loss = ctc_loss(
            log_probs=log_probs, targets=transcript, input_length=encoded_len, target_length=transcript_len,
        )

        callback = nemo.core.SimpleLossLoggerCallback(
            tensors=[loss], print_func=lambda x: logging.info(str(x[0].item()))
        )
        # Instantiate an optimizer to perform `train` action
        optimizer = nemo.backends.pytorch.actions.PtActions()
        optimizer.train(
            [loss], callbacks=[callback], optimizer="sgd", optimization_params={"num_epochs": 10, "lr": 0.0003},
        )

    @pytest.mark.unclassified
    def test_quartznet_training(self):
        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quartznet_test.yaml"))) as f:
            quartz_model_definition = self.yaml.load(f)
        dl = nemo_asr.AudioToTextDataLayer(manifest_filepath=self.manifest_filepath, labels=self.labels, batch_size=4,)
        pre_process_params = {
            'frame_splicing': 1,
            'features': 64,
            'window_size': 0.02,
            'n_fft': 512,
            'dither': 1e-05,
            'window': 'hann',
            'sample_rate': 16000,
            'normalize': 'per_feature',
            'window_stride': 0.01,
        }
        preprocessing = nemo_asr.AudioToMelSpectrogramPreprocessor(**pre_process_params)
        jasper_encoder = nemo_asr.JasperEncoder(
            feat_in=quartz_model_definition['AudioToMelSpectrogramPreprocessor']['features'],
            **quartz_model_definition['JasperEncoder'],
        )
        jasper_decoder = nemo_asr.JasperDecoderForCTC(feat_in=1024, num_classes=len(self.labels))
        ctc_loss = nemo_asr.CTCLossNM(num_classes=len(self.labels))

        # DAG
        audio_signal, a_sig_length, transcript, transcript_len = dl()
        processed_signal, p_length = preprocessing(input_signal=audio_signal, length=a_sig_length)

        encoded, encoded_len = jasper_encoder(audio_signal=processed_signal, length=p_length)
        log_probs = jasper_decoder(encoder_output=encoded)
        loss = ctc_loss(
            log_probs=log_probs, targets=transcript, input_length=encoded_len, target_length=transcript_len,
        )

        callback = nemo.core.SimpleLossLoggerCallback(
            tensors=[loss], print_func=lambda x: logging.info(f'Train Loss: {str(x[0].item())}'),
        )
        # Instantiate an optimizer to perform `train` action
        optimizer = nemo.backends.pytorch.actions.PtActions()
        optimizer.train(
            [loss], callbacks=[callback], optimizer="sgd", optimization_params={"num_epochs": 10, "lr": 0.0003},
        )

    @pytest.mark.unclassified
    def test_stft_conv_training(self):
        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/jasper_smaller.yaml"))) as file:
            jasper_model_definition = self.yaml.load(file)
        dl = nemo_asr.AudioToTextDataLayer(manifest_filepath=self.manifest_filepath, labels=self.labels, batch_size=4,)
        pre_process_params = {
            'frame_splicing': 1,
            'features': 64,
            'window_size': 0.02,
            'n_fft': 512,
            'dither': 1e-05,
            'window': 'hann',
            'sample_rate': 16000,
            'normalize': 'per_feature',
            'window_stride': 0.01,
            'stft_conv': True,
        }
        preprocessing = nemo_asr.AudioToMelSpectrogramPreprocessor(**pre_process_params)
        jasper_encoder = nemo_asr.JasperEncoder(
            feat_in=jasper_model_definition['AudioToMelSpectrogramPreprocessor']['features'],
            **jasper_model_definition['JasperEncoder'],
        )
        jasper_decoder = nemo_asr.JasperDecoderForCTC(feat_in=1024, num_classes=len(self.labels))

        ctc_loss = nemo_asr.CTCLossNM(num_classes=len(self.labels))

        # DAG
        audio_signal, a_sig_length, transcript, transcript_len = dl()
        processed_signal, p_length = preprocessing(input_signal=audio_signal, length=a_sig_length)

        encoded, encoded_len = jasper_encoder(audio_signal=processed_signal, length=p_length)
        # logging.info(jasper_encoder)
        log_probs = jasper_decoder(encoder_output=encoded)
        loss = ctc_loss(
            log_probs=log_probs, targets=transcript, input_length=encoded_len, target_length=transcript_len,
        )

        callback = nemo.core.SimpleLossLoggerCallback(
            tensors=[loss], print_func=lambda x: logging.info(str(x[0].item()))
        )
        # Instantiate an optimizer to perform `train` action
        optimizer = nemo.backends.pytorch.actions.PtActions()
        optimizer.train(
            [loss], callbacks=[callback], optimizer="sgd", optimization_params={"num_epochs": 10, "lr": 0.0003},
        )

    @pytest.mark.unclassified
    def test_garnet_training(self):
        with open('examples/asr/experimental/configs/garnet_an4.yaml') as file:
            cfg = self.yaml.load(file)
        dl = nemo_asr.AudioToTextDataLayer(manifest_filepath=self.manifest_filepath, labels=self.labels, batch_size=4,)
        pre_process_params = {
            'frame_splicing': 1,
            'features': 64,
            'window_size': 0.02,
            'n_fft': 512,
            'dither': 1e-05,
            'window': 'hann',
            'sample_rate': 16000,
            'normalize': 'per_feature',
            'window_stride': 0.01,
            'stft_conv': True,
        }
        preprocessing = nemo_asr.AudioToMelSpectrogramPreprocessor(**pre_process_params)
        encoder = nemo_asr.JasperEncoder(
            jasper=cfg['encoder']['jasper'],
            activation=cfg['encoder']['activation'],
            feat_in=cfg['input']['train']['features'],
        )
        connector = nemo_asr.JasperRNNConnector(
            in_channels=cfg['encoder']['jasper'][-1]['filters'], out_channels=cfg['decoder']['hidden_size'],
        )
        decoder = nemo.backends.pytorch.common.DecoderRNN(
            voc_size=len(self.labels),
            bos_id=0,
            hidden_size=cfg['decoder']['hidden_size'],
            attention_method=cfg['decoder']['attention_method'],
            attention_type=cfg['decoder']['attention_type'],
            in_dropout=cfg['decoder']['in_dropout'],
            gru_dropout=cfg['decoder']['gru_dropout'],
            attn_dropout=cfg['decoder']['attn_dropout'],
            teacher_forcing=cfg['decoder']['teacher_forcing'],
            curriculum_learning=cfg['decoder']['curriculum_learning'],
            rnn_type=cfg['decoder']['rnn_type'],
            n_layers=cfg['decoder']['n_layers'],
            tie_emb_out_weights=cfg['decoder']['tie_emb_out_weights'],
        )
        loss = nemo.backends.pytorch.common.SequenceLoss()

        # DAG
        audio_signal, a_sig_length, transcripts, transcript_len = dl()
        processed_signal, p_length = preprocessing(input_signal=audio_signal, length=a_sig_length)
        encoded, encoded_len = encoder(audio_signal=processed_signal, length=p_length)
        encoded = connector(tensor=encoded)
        log_probs, _ = decoder(targets=transcripts, encoder_outputs=encoded)
        loss = loss(log_probs=log_probs, targets=transcripts)

        # Train
        callback = nemo.core.SimpleLossLoggerCallback(
            tensors=[loss], print_func=lambda x: logging.info(str(x[0].item()))
        )
        # Instantiate an optimizer to perform `train` action
        optimizer = nemo.backends.pytorch.actions.PtActions()
        optimizer.train(
            [loss], callbacks=[callback], optimizer="sgd", optimization_params={"num_epochs": 10, "lr": 0.0003},
        )

    @pytest.mark.unclassified
    def test_jasper_evaluation(self):
        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/jasper_smaller.yaml"))) as file:
            jasper_model_definition = self.yaml.load(file)
        dl = nemo_asr.AudioToTextDataLayer(manifest_filepath=self.manifest_filepath, labels=self.labels, batch_size=4,)
        pre_process_params = {
            'frame_splicing': 1,
            'features': 64,
            'window_size': 0.02,
            'n_fft': 512,
            'dither': 1e-05,
            'window': 'hann',
            'sample_rate': 16000,
            'normalize': 'per_feature',
            'window_stride': 0.01,
        }
        preprocessing = nemo_asr.AudioToMelSpectrogramPreprocessor(**pre_process_params)
        jasper_encoder = nemo_asr.JasperEncoder(
            feat_in=jasper_model_definition['AudioToMelSpectrogramPreprocessor']['features'],
            **jasper_model_definition['JasperEncoder'],
        )
        jasper_decoder = nemo_asr.JasperDecoderForCTC(feat_in=1024, num_classes=len(self.labels))
        ctc_loss = nemo_asr.CTCLossNM(num_classes=len(self.labels))
        greedy_decoder = nemo_asr.GreedyCTCDecoder()
        # DAG
        audio_signal, a_sig_length, transcript, transcript_len = dl()
        processed_signal, p_length = preprocessing(input_signal=audio_signal, length=a_sig_length)

        encoded, encoded_len = jasper_encoder(audio_signal=processed_signal, length=p_length)
        # logging.info(jasper_encoder)
        log_probs = jasper_decoder(encoder_output=encoded)
        loss = ctc_loss(
            log_probs=log_probs, targets=transcript, input_length=encoded_len, target_length=transcript_len,
        )
        predictions = greedy_decoder(log_probs=log_probs)

        from nemo.collections.asr.helpers import (
            process_evaluation_batch,
            process_evaluation_epoch,
        )

        eval_callback = nemo.core.EvaluatorCallback(
            eval_tensors=[loss, predictions, transcript, transcript_len],
            user_iter_callback=lambda x, y: process_evaluation_batch(x, y, labels=self.labels),
            user_epochs_done_callback=process_evaluation_epoch,
        )
        # Instantiate an optimizer to perform `train` action
        self.nf.eval(callbacks=[eval_callback])

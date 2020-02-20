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
import shutil
import tarfile
import unittest

from ruamel.yaml import YAML

import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts import AudioDataset, WaveformFeaturizer, collections, parsers
from nemo.core import DeviceType
from tests.common_setup import NeMoUnitTest

logging = nemo.logging


freq = 16000


class TestASRPytorch(NeMoUnitTest):
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

    # @classmethod
    # def tearDownClass(cls) -> None:
    #     super().tearDownClass()
    #     data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/"))
    #     logging.info("Looking up for test ASR data")
    #     if os.path.exists(os.path.join(data_folder, "asr")):
    #         shutil.rmtree(os.path.join(data_folder, "asr"))

    def test_transcript_normalizers(self):
        # Create test json
        test_strings = [
            "TEST CAPITALIZATION",
            '!\\"#$%&\'()*+,-./:;<=>?@[\\\\]^_`{|}~',
            "3+3=10",
            "3 + 3 = 10",
            "why     is \\t whitepsace\\tsuch a problem   why indeed",
            "\\\"Can you handle quotes?,\\\" says the boy",
            "I Jump!!!!With joy?Now.",
            "Maybe I want to learn periods.",
            "$10 10.90 1-800-000-0000",
            "18000000000 one thousand 2020",
            "1 10 100 1000 10000 100000 1000000",
            "Î  ĻƠvɆȩȅĘ ÀÁÃ Ą ÇĊňńŤŧș",
            "‘’“”❛❜❝❞「 」 〈 〉 《 》 【 】 〔 〕 ⦗ ⦘ 😙  👀 🔨",
            "It only costs $1 000 000! Cheap right?",
            "2500, 3000 are separate but 200, 125 is not",
            "1",
            "1 2",
            "1 2 3",
            "10:00pm is 10:00 pm is 22:00 but not 10: 00 pm",
            "10:00 10:01pm 10:10am 10:90pm",
            "Mr. Expand me!",
            "Mr Don't Expand me!",
        ]
        normalized_strings = [
            "test capitalization",
            'percent and \' plus',
            "three plus three ten",
            "three plus three ten",
            "why is whitepsace such a problem why indeed",
            "can you handle quotes says the boy",
            "i jump with joy now",
            "maybe i want to learn periods",
            "ten dollars ten point nine zero one eight hundred zero zero",
            "eighteen billion one thousand two thousand and twenty",
            # Two line string below
            "one ten thousand one hundred one thousand ten thousand one hundred thousand one million",
            "i loveeee aaa a ccnntts",
            "''",
            "it only costs one million dollars cheap right",
            # Two line string below
            "two thousand five hundred three thousand are separate but two "
            "hundred thousand one hundred and twenty five is not",
            "one",
            "one two",
            "one two three",
            "ten pm is ten pm is twenty two but not ten zero pm",
            "ten ten one pm ten ten am ten ninety pm",
            "mister expand me",
            "mr don't expand me",
        ]
        manifest_paths = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/asr/manifest_test.json"))

        def remove_test_json():
            os.remove(manifest_paths)

        self.addCleanup(remove_test_json)

        with open(manifest_paths, "w") as f:
            for s in test_strings:
                f.write('{"audio_filepath": "", "duration": 1.0, "text": ' f'"{s}"}}\n')
        parser = parsers.make_parser(self.labels, 'en')
        manifest = collections.ASRAudioText(manifests_files=[manifest_paths], parser=parser,)

        for i, s in enumerate(normalized_strings):
            self.assertTrue(manifest[i].text_tokens == parser(s))

    def test_pytorch_audio_dataset(self):
        featurizer = WaveformFeaturizer.from_config(self.featurizer_config)
        ds = AudioDataset(manifest_filepath=self.manifest_filepath, labels=self.labels, featurizer=featurizer,)

        for i in range(len(ds)):
            if i == 5:
                logging.info(ds[i])
            # logging.info(ds[i][0].shape)
            # self.assertEqual(freq, ds[i][0].shape[0])

    def test_dataloader(self):
        batch_size = 4
        dl = nemo_asr.AudioToTextDataLayer(
            # featurizer_config=self.featurizer_config,
            manifest_filepath=self.manifest_filepath,
            labels=self.labels,
            batch_size=batch_size,
            # placement=DeviceType.GPU,
            drop_last=True,
        )
        for ind, data in enumerate(dl.data_iterator):
            # With num_workers update, this is no longer true
            # Moving to GPU is handled by AudioPreprocessor
            # data is on GPU
            # self.assertTrue(data[0].is_cuda)
            # self.assertTrue(data[1].is_cuda)
            # self.assertTrue(data[2].is_cuda)
            # self.assertTrue(data[3].is_cuda)
            # first dimension is batch
            self.assertTrue(data[0].size(0) == batch_size)
            self.assertTrue(data[1].size(0) == batch_size)
            self.assertTrue(data[2].size(0) == batch_size)
            self.assertTrue(data[3].size(0) == batch_size)

    def test_preprocessor_errors(self):
        def create_broken_preprocessor_1():
            nemo_asr.AudioToMelSpectrogramPreprocessor(window_size=2, n_window_size=2)

        def create_broken_preprocessor_2():
            nemo_asr.AudioToMelSpectrogramPreprocessor(window_stride=2, n_window_stride=2)

        def create_broken_preprocessor_3():
            nemo_asr.AudioToMelSpectrogramPreprocessor(n_window_stride=2)

        def create_good_preprocessor_1():
            nemo_asr.AudioToMelSpectrogramPreprocessor(window_size=0.02, window_stride=0.01)

        def create_good_preprocessor_2():
            nemo_asr.AudioToMelSpectrogramPreprocessor(
                window_size=None, window_stride=None, n_window_size=256, n_window_stride=32,
            )

        self.assertRaises(ValueError, create_broken_preprocessor_1)
        self.assertRaises(ValueError, create_broken_preprocessor_2)
        self.assertRaises(ValueError, create_broken_preprocessor_3)
        create_good_preprocessor_1()
        create_good_preprocessor_2()

    def test_kaldi_dataloader(self):
        batch_size = 4
        dl = nemo_asr.KaldiFeatureDataLayer(
            kaldi_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/asr/kaldi_an4/')),
            labels=self.labels,
            batch_size=batch_size,
        )
        for data in dl.data_iterator:
            self.assertTrue(data[0].size(0) == batch_size)

        dl_test_min = nemo_asr.KaldiFeatureDataLayer(
            kaldi_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/asr/kaldi_an4/')),
            labels=self.labels,
            batch_size=batch_size,
            min_duration=1.0,
        )
        self.assertTrue(len(dl_test_min) == 18)

        dl_test_max = nemo_asr.KaldiFeatureDataLayer(
            kaldi_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/asr/kaldi_an4/')),
            labels=self.labels,
            batch_size=batch_size,
            max_duration=5.0,
        )
        self.assertTrue(len(dl_test_max) == 19)

    def test_trim_silence(self):
        batch_size = 4
        normal_dl = nemo_asr.AudioToTextDataLayer(
            # featurizer_config=self.featurizer_config,
            manifest_filepath=self.manifest_filepath,
            labels=self.labels,
            batch_size=batch_size,
            # placement=DeviceType.GPU,
            drop_last=True,
            shuffle=False,
        )
        trimmed_dl = nemo_asr.AudioToTextDataLayer(
            # featurizer_config=self.featurizer_config,
            manifest_filepath=self.manifest_filepath,
            trim_silence=True,
            labels=self.labels,
            batch_size=batch_size,
            # placement=DeviceType.GPU,
            drop_last=True,
            shuffle=False,
        )
        for norm, trim in zip(normal_dl.data_iterator, trimmed_dl.data_iterator):
            for point in range(batch_size):
                self.assertTrue(norm[1][point].data >= trim[1][point].data)

    def test_audio_preprocessors(self):
        batch_size = 5
        dl = nemo_asr.AudioToTextDataLayer(
            # featurizer_config=self.featurizer_config,
            manifest_filepath=self.manifest_filepath,
            labels=self.labels,
            batch_size=batch_size,
            # placement=DeviceType.GPU,
            drop_last=True,
            shuffle=False,
        )

        installed_torchaudio = True
        try:
            import torchaudio
        except ModuleNotFoundError:
            installed_torchaudio = False
            with self.assertRaises(ModuleNotFoundError):
                to_spectrogram = nemo_asr.AudioToSpectrogramPreprocessor(n_fft=400, window=None)
            with self.assertRaises(ModuleNotFoundError):
                to_mfcc = nemo_asr.AudioToMFCCPreprocessor(n_mfcc=15)

        if installed_torchaudio:
            to_spectrogram = nemo_asr.AudioToSpectrogramPreprocessor(n_fft=400, window=None)
            to_mfcc = nemo_asr.AudioToMFCCPreprocessor(n_mfcc=15)

        to_melspec = nemo_asr.AudioToMelSpectrogramPreprocessor(features=50)

        for batch in dl.data_iterator:
            input_signals, seq_lengths, _, _ = batch
            input_signals = input_signals.to(to_melspec._device)
            seq_lengths = seq_lengths.to(to_melspec._device)

            melspec = to_melspec.forward(input_signals, seq_lengths)

            if installed_torchaudio:
                spec = to_spectrogram.forward(input_signals, seq_lengths)
                mfcc = to_mfcc.forward(input_signals, seq_lengths)

            # Check that number of features is what we expect
            self.assertTrue(melspec[0].shape[1] == 50)

            if installed_torchaudio:
                self.assertTrue(spec[0].shape[1] == 201)  # n_fft // 2 + 1 bins
                self.assertTrue(mfcc[0].shape[1] == 15)

    # @unittest.skip("Init parameters of nemo_asr.AudioToMelSpectrogramPreprocessor are invalid")
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
        optimizer = self.nf.get_trainer()
        optimizer.train(
            [loss], callbacks=[callback], optimizer="sgd", optimization_params={"num_epochs": 10, "lr": 0.0003},
        )

    # @unittest.skip("Init parameters of nemo_asr.AudioToMelSpectrogramPreprocessor are invalid")
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
        optimizer = self.nf.get_trainer()
        optimizer.train(
            [loss], callbacks=[callback], optimizer="sgd", optimization_params={"num_epochs": 10, "lr": 0.0003},
        )

    # @unittest.skip("Init parameters of nemo_asr.AudioToMelSpectrogramPreprocessor are invalid")
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
        optimizer = self.nf.get_trainer()
        optimizer.train(
            [loss], callbacks=[callback], optimizer="sgd", optimization_params={"num_epochs": 10, "lr": 0.0003},
        )

    def test_stft_conv(self):
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
        optimizer = self.nf.get_trainer()
        optimizer.train(
            [loss], callbacks=[callback], optimizer="sgd", optimization_params={"num_epochs": 10, "lr": 0.0003},
        )

    def test_clas(self):
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
        optimizer = self.nf.get_trainer()
        optimizer.train(
            [loss], callbacks=[callback], optimizer="sgd", optimization_params={"num_epochs": 10, "lr": 0.0003},
        )

    def test_jasper_eval(self):
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

# ! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2019 NVIDIA. All Rights Reserved.
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

import tarfile
import unittest

from nemo.core import *
import nemo.collections.asr as nemo_asr

from ruamel.yaml import YAML

from .common_setup import NeMoUnitTest


class TestNeuralTypes(NeMoUnitTest):
    manifest_filepath = "tests/data/asr/an4_train.json"
    yaml = YAML(typ="safe")

    def setUp(self) -> None:
        super().setUp()
        data_folder = "tests/data/"
        print("Looking up for test ASR data")
        if not os.path.exists(data_folder + "asr"):
            print("Extracting ASR data to: {0}".format(data_folder + "asr"))
            tar = tarfile.open("tests/data/asr.tar.gz", "r:gz")
            tar.extractall(path=data_folder)
            tar.close()
        else:
            print("ASR data found in: {0}".format(data_folder + "asr"))

    def test_same(self):
        btc = NeuralType(axis2type={0: AxisType(BatchTag),
                                    1: AxisType(TimeTag),
                                    2: AxisType(ChannelTag)})
        btc2 = NeuralType(axis2type={0: AxisType(BatchTag),
                                     1: AxisType(TimeTag),
                                     2: AxisType(ChannelTag)})
        self.assertEqual(btc2.compare(btc), NeuralTypeComparisonResult.SAME)

    def test_transpose_same(self):
        btc = NeuralType(axis2type={0: AxisType(BatchTag),
                                    1: AxisType(TimeTag),
                                    2: AxisType(ChannelTag)})
        tbc = NeuralType(axis2type={1: AxisType(BatchTag),
                                    0: AxisType(TimeTag),
                                    2: AxisType(ChannelTag)})

        self.assertEqual(btc.compare(tbc),
                         NeuralTypeComparisonResult.TRANSPOSE_SAME)
        self.assertEqual(tbc.compare(btc),
                         NeuralTypeComparisonResult.TRANSPOSE_SAME)

    def test_dim_incompatible(self):
        nchw1 = NeuralType(axis2type={0: AxisType(BatchTag),
                                      1: AxisType(ChannelTag),
                                      2: AxisType(HeightTag, 224),
                                      3: AxisType(WidthTag, 224)})
        nchw2 = NeuralType(axis2type={0: AxisType(BatchTag),
                                      1: AxisType(ChannelTag),
                                      2: AxisType(HeightTag, 256),
                                      3: AxisType(WidthTag, 256)})
        self.assertEqual(nchw1.compare(nchw2),
                         NeuralTypeComparisonResult.DIM_INCOMPATIBLE)

    def test_rank_incompatible(self):
        btc = NeuralType(axis2type={0: AxisType(BatchTag),
                                    1: AxisType(TimeTag),
                                    2: AxisType(ChannelTag)})
        nchw = NeuralType(axis2type={0: AxisType(BatchTag),
                                     1: AxisType(ChannelTag),
                                     2: AxisType(HeightTag),
                                     3: AxisType(WidthTag)})
        self.assertEqual(nchw.compare(
            btc), NeuralTypeComparisonResult.INCOMPATIBLE)

    def test_axis_type(self):
        ax1 = AxisType(BatchTag)
        ax2 = AxisType(TimeTag)
        ax3 = AxisType(ProcessedTimeTag)
        self.assertEqual(ax1.compare_to(ax2),
                         NeuralTypeComparisonResult.INCOMPATIBLE)
        self.assertEqual(ax3.compare_to(ax2),
                         NeuralTypeComparisonResult.LESS)
        self.assertEqual(ax2.compare_to(ax3),
                         NeuralTypeComparisonResult.GREATER)
        self.assertEqual(ax2.compare_to(AxisType(TimeTag)),
                         NeuralTypeComparisonResult.SAME)

    def test_semantic_incompatible(self):
        nchw = NeuralType(axis2type={0: AxisType(BatchTag),
                                     1: AxisType(ChannelTag),
                                     2: AxisType(HeightTag),
                                     3: AxisType(WidthTag)})
        badd = NeuralType(axis2type={0: AxisType(BatchTag),
                                     1: AxisType(ChannelTag),
                                     2: AxisType(ChannelTag),
                                     3: AxisType(WidthTag)})
        self.assertEqual(nchw.compare(
            badd), NeuralTypeComparisonResult.INCOMPATIBLE)
        self.assertEqual(badd.compare(
            nchw), NeuralTypeComparisonResult.INCOMPATIBLE)

    def test_root(self):
        root = NeuralType({})
        non_tensor = NeuralType(None)
        btc = NeuralType(axis2type={0: AxisType(BatchTag),
                                    1: AxisType(TimeTag),
                                    2: AxisType(ChannelTag)})
        nchw = NeuralType(axis2type={0: AxisType(BatchTag),
                                     1: AxisType(ChannelTag),
                                     2: AxisType(HeightTag),
                                     3: AxisType(WidthTag)})
        self.assertEqual(root.compare(btc),
                         NeuralTypeComparisonResult.SAME)
        self.assertEqual(root.compare(nchw),
                         NeuralTypeComparisonResult.SAME)
        self.assertEqual(root.compare(non_tensor),
                         NeuralTypeComparisonResult.SAME)

        self.assertEqual(non_tensor.compare(root),
                         NeuralTypeComparisonResult.INCOMPATIBLE)
        self.assertEqual(btc.compare(root),
                         NeuralTypeComparisonResult.INCOMPATIBLE)
        self.assertEqual(nchw.compare(root),
                         NeuralTypeComparisonResult.INCOMPATIBLE)

    def test_combiner_type_infer(self):
        combiner = nemo.backends.pytorch.common.SimpleCombiner(mode="add")
        x_tg = nemo.core.NmTensor(producer=None, producer_args=None,
                                  name=None,
                                  ntype=NeuralType(
                                      {
                                          0: AxisType(BatchTag),
                                      }))
        y_tg = nemo.core.NmTensor(producer=None, producer_args=None,
                                  name=None,
                                  ntype=NeuralType(
                                      {
                                          0: AxisType(BatchTag),
                                      }))
        res = combiner(x1=y_tg, x2=x_tg)
        self.assertEqual(res.compare(x_tg),
                         NeuralTypeComparisonResult.SAME)
        self.assertEqual(res.compare(y_tg),
                         NeuralTypeComparisonResult.SAME)
        self.assertEqual(x_tg.compare(res),
                         NeuralTypeComparisonResult.SAME)
        self.assertEqual(y_tg.compare(res),
                         NeuralTypeComparisonResult.SAME)

        combiner1 = nemo.backends.pytorch.common.SimpleCombiner(mode="add")
        x_tg1 = NmTensor(producer=None, producer_args=None,
                         name=None,
                         ntype=NeuralType(
                             {
                                 0: AxisType(BatchTag),
                                 1: AxisType(ChannelTag)
                             }))
        y_tg1 = NmTensor(producer=None, producer_args=None,
                         name=None,
                         ntype=NeuralType(
                             {
                                 0: AxisType(BatchTag),
                                 1: AxisType(ChannelTag)
                             }))
        res1 = combiner1(x1=y_tg1, x2=x_tg1)
        self.assertEqual(res1.compare(x_tg1),
                         NeuralTypeComparisonResult.SAME)
        self.assertEqual(res1.compare(y_tg1),
                         NeuralTypeComparisonResult.SAME)
        self.assertEqual(x_tg1.compare(res1),
                         NeuralTypeComparisonResult.SAME)
        self.assertEqual(y_tg1.compare(res1),
                         NeuralTypeComparisonResult.SAME)

    def test_optional_input_no_input(self):
        data_source = nemo.backends.pytorch.tutorials.RealFunctionDataLayer(
            n=100, batch_size=128)
        trainable_module = nemo.backends.pytorch.tutorials.TaylorNetO(dim=4)
        loss = nemo.backends.pytorch.tutorials.MSELoss()
        x, y = data_source()
        y_pred = trainable_module(x=x)
        loss_tensor = loss(predictions=y_pred, target=y)

        optimizer = nemo.backends.pytorch.actions.PtActions()
        optimizer.train(
            tensors_to_optimize=[loss_tensor],
            optimizer="sgd",
            optimization_params={"lr": 0.0003, "num_epochs": 1})

    def test_optional_input_no_with_input(self):
        data_source = nemo.backends.pytorch.tutorials.RealFunctionDataLayer(
            n=100, batch_size=128)
        trainable_module = nemo.backends.pytorch.tutorials.TaylorNetO(dim=4)
        loss = nemo.backends.pytorch.tutorials.MSELoss()
        x, y = data_source()
        y_pred = trainable_module(x=x, o=x)
        loss_tensor = loss(predictions=y_pred, target=y)
        optimizer = nemo.backends.pytorch.actions.PtActions()
        optimizer.train(
            tensors_to_optimize=[loss_tensor],
            optimizer="sgd",
            optimization_params={"lr": 0.0003, "num_epochs": 1})

    def test_optional_input_no_with_wrong_input(self):

        def wrong_fn():
            data_source = \
                nemo.backends.pytorch.tutorials.RealFunctionDataLayer(
                    n=100, batch_size=128)
            trainable_module = nemo.backends.pytorch.tutorials.TaylorNetO(
                dim=4)
            loss = nemo.backends.pytorch.tutorials.MSELoss()
            x, y = data_source()
            wrong_optional = NmTensor(producer=None, producer_args=None,
                                      name=None,
                                      ntype=NeuralType(
                                          {
                                              0: AxisType(ChannelTag),
                                              1: AxisType(BatchTag)
                                          }))
            y_pred = trainable_module(x=x, o=wrong_optional)
            loss_tensor = loss(predictions=y_pred, target=y)
            optimizer = nemo.backends.pytorch.actions.PtActions()
            optimizer.train(
                tensors_to_optimize=[loss_tensor],
                optimizer="sgd",
                optimization_params={"lr": 0.0003, "num_epochs": 1})

        self.assertRaises(NeuralPortNmTensorMismatchError, wrong_fn)

    def test_simple_dags(self):
        # module instantiation
        with open("tests/data/jasper_smaller.yaml") as file:
            jasper_model_definition = self.yaml.load(file)
        labels = jasper_model_definition['labels']

        data_layer = nemo_asr.AudioToTextDataLayer(
            manifest_filepath=self.manifest_filepath,
            labels=labels, batch_size=4)
        data_preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor(
            **jasper_model_definition['AudioToMelSpectrogramPreprocessor'])
        jasper_encoder = nemo_asr.JasperEncoder(
            feat_in=jasper_model_definition[
                'AudioToMelSpectrogramPreprocessor']['features'],
            **jasper_model_definition['JasperEncoder'])
        jasper_decoder = nemo_asr.JasperDecoderForCTC(feat_in=1024,
                                                      num_classes=len(labels))
        ctc_loss = nemo_asr.CTCLossNM(num_classes=len(labels))
        greedy_decoder = nemo_asr.GreedyCTCDecoder()

        # DAG definition
        audio_signal, audio_signal_len, transcript, transcript_len = \
            data_layer()
        processed_signal, processed_signal_len = data_preprocessor(
            input_signal=audio_signal,
            length=audio_signal_len)

        spec_augment = nemo_asr.SpectrogramAugmentation(rect_masks=5)
        aug_signal = spec_augment(input_spec=processed_signal)

        encoded, encoded_len = jasper_encoder(audio_signal=aug_signal,
                                              length=processed_signal_len)
        log_probs = jasper_decoder(encoder_output=encoded)
        predictions = greedy_decoder(log_probs=log_probs)
        loss = ctc_loss(log_probs=log_probs, targets=transcript,
                        input_length=encoded_len, target_length=transcript_len)

        def wrong():
            with open("tests/data/jasper_smaller.yaml") as file:
                jasper_config = self.yaml.load(file)
            labels = jasper_config['labels']

            data_layer = nemo_asr.AudioToTextDataLayer(
                manifest_filepath=self.manifest_filepath,
                labels=labels, batch_size=4)
            data_preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor(
                **jasper_config['AudioToMelSpectrogramPreprocessor'])
            jasper_encoder = nemo_asr.JasperEncoder(
                feat_in=jasper_config[
                    'AudioToMelSpectrogramPreprocessor']['features'],
                **jasper_config['JasperEncoder']
            )
            jasper_decoder = nemo_asr.JasperDecoderForCTC(feat_in=1024,
                                                          num_classes=len(
                                                              labels))
            # DAG definition
            audio_signal, audio_signal_len, transcript, transcript_len = \
                data_layer()
            processed_signal, processed_signal_len = data_preprocessor(
                input_signal=audio_signal,
                length=audio_signal_len)

            spec_augment = nemo_asr.SpectrogramAugmentation(rect_masks=5)
            aug_signal = spec_augment(input_spec=processed_signal)

            encoded, encoded_len = jasper_encoder(audio_signal=aug_signal,
                                                  length=processed_signal_len)
            log_probs = jasper_decoder(encoder_output=processed_signal)

        self.assertRaises(NeuralPortNmTensorMismatchError, wrong)

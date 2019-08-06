# Copyright (c) 2019 NVIDIA Corporation
import unittest
import numpy as np
import torch
from tests.context import nemo, nemo_asr
from typing import Dict
from nemo.core import WeightShareTransform
from nemo.core.neural_types import *
from ruamel.yaml import YAML


class TestWeightSharing(unittest.TestCase):
    labels = ["'", "a", "b", "c", "d", "e", "f", "g", "h",
              "i", "j", "k", "l", "m", "n", "o", "p", "q",
              "r", "s", "t", "u", "v", "w", "x", "y", "z", " "]
    manifest_filepath = "tests/data/asr/an4_train.json"
    featurizer_config = {'window': 'hann',
                         'dither': 1e-05,
                         'normalize': 'per_feature',
                         'frame_splicing': 1,
                         'int_values': False,
                         'window_stride': 0.01,
                         'sample_rate': 16000,
                         'features': 64,
                         'n_fft': 512,
                         'feat_type': 'logfbank',
                         'window_size': 0.02}
    yaml = YAML(typ="safe")

    def __check_if_weights_are_equal(self, w1: Dict, w2: Dict):
        all_same = (set(w1.keys()) == set(w2.keys()))
        if not all_same:
            return False
        else:
            for key in w1.keys():
                all_same = all_same and np.array_equal(w1[key][0].cpu().detach().numpy(),
                                                       w2[key][0].cpu().detach().numpy())
        return all_same

    def test_TaylorNet_get_weights(self):
        tn1 = nemo.backends.pytorch.tutorials.TaylorNet(dim=4)
        tn2 = nemo.backends.pytorch.tutorials.TaylorNet(dim=4)
        # because of randomness, actual weights should be different
        self.assertFalse(self.__check_if_weights_are_equal(tn1.get_weights(),
                                                           tn2.get_weights()))
        tn3 = nemo.backends.pytorch.tutorials.TaylorNet(dim=4)
        tn3.set_weights(tn1.get_weights())
        # check than weights are the same
        self.assertTrue(self.__check_if_weights_are_equal(tn1.get_weights(),
                                                          tn3.get_weights()))
        # change weights on one module - another module should not change
        tn1.fc1.bias.data = torch.tensor([0.1])
        self.assertFalse(self.__check_if_weights_are_equal(tn1.get_weights(),
                                                           tn3.get_weights()))

    def test_TaylorNet_tie_weights(self):
        tn1 = nemo.backends.pytorch.tutorials.TaylorNet(dim=4)
        tn2 = nemo.backends.pytorch.tutorials.TaylorNet(dim=4)
        # because of randomness, actual weights should be different
        self.assertFalse(self.__check_if_weights_are_equal(tn1.get_weights(),
                                                           tn2.get_weights()))
        tn2.tie_weights_with(tn1, list(tn1.get_weights().keys()))
        # change weights on one module - another module should change too
        tn1.fc1.bias.data = torch.tensor([0.1])
        self.assertTrue(self.__check_if_weights_are_equal(tn1.get_weights(),
                                                          tn2.get_weights()))

    def test_tie_weights2(self):
        voc_size = 3
        dim = 2
        embd = nemo.backends.pytorch.common.SequenceEmbedding(voc_size=voc_size,
                                                                 hidden_size=dim)
        proj = nemo.backends.pytorch.common.SequenceProjection(from_dim=dim,
                                                                  to_dim=voc_size)
        embd.tie_weights_with(proj,
                              weight_names=["embedding.weight"],
                              name2name_and_transform={"embedding.weight":
                                                       ("projection.weight",
                                                        WeightShareTransform.SAME)})
        self.assertTrue(np.array_equal(embd.embedding.weight.detach().numpy(),
                                       proj.projection.weight.detach().numpy()))
        was = embd.embedding.weight.detach().numpy()
        embd.embedding.weight.data = torch.tensor(
            np.random.randint(0, 10, (3, 2))*1.0)
        after = embd.embedding.weight.detach().numpy()
        self.assertTrue(np.array_equal(embd.embedding.weight.detach().numpy(),
                                       proj.projection.weight.detach().numpy()))
        self.assertFalse(np.array_equal(was, after))

    def test_set_weights(self):
        voc_size = 3
        dim = 2
        embd = nemo.backends.pytorch.common.SequenceEmbedding(voc_size=voc_size,
                                                                 hidden_size=dim)
        weights = torch.tensor(np.random.randint(0, 10, (3, 2))*1.0)
        name2weights = {"embedding.weight": (weights, True)}
        embd.set_weights(name2weight=name2weights)
        self.assertTrue(np.array_equal(embd.embedding.weight.detach().numpy(),
                                       weights.detach().numpy()))
        weights = torch.tensor(np.random.randint(0, 10, (3, 2)) * 1.0)
        self.assertFalse(np.array_equal(embd.embedding.weight.detach().numpy(),
                                        weights.detach().numpy()))

    def test_freeze_unfreeze_TrainableNM(self):
        with open("tests/data/jasper_smaller.yaml") as file:
            jasper_model_definition = self.yaml.load(file)
        dl = nemo_asr.AudioToTextDataLayer(
            featurizer_config=self.featurizer_config,
            manifest_filepath=self.manifest_filepath,
            labels=self.labels,
            batch_size=4
        )
        pre_process_params = {'int_values': False, 'frame_splicing': 1, 'features': 64,
                              'window_size': 0.02, 'n_fft': 512, 'dither': 1e-05,
                              'window': 'hann', 'feat_type': 'logfbank', 'sample_rate': 16000,
                              'normalize': 'per_feature', 'window_stride': 0.01}
        preprocessing = nemo_asr.AudioPreprocessing(
            **pre_process_params)
        jasper_encoder = nemo_asr.JasperEncoder(
            feat_in=jasper_model_definition['AudioPreprocessing']['features'],
            **jasper_model_definition['JasperEncoder'])
        jasper_decoder = nemo_asr.JasperDecoderForCTC(
            feat_in=1024,
            num_classes=len(self.labels)
        )
        ctc_loss = nemo_asr.CTCLossNM(num_classes=len(self.labels))
        jasper_encoder.freeze()
        jasper_encoder.unfreeze(set(['encoder.4.conv.1.weight']))
        jasper_decoder.unfreeze()
        # DAG
        audio_signal, a_sig_length, transcript, transcript_len = dl()
        processed_signal, p_length = preprocessing(input_signal=audio_signal,
                                                   length=a_sig_length)

        encoded, encoded_len = jasper_encoder(audio_signal=processed_signal,
                                              length=p_length)
        # print(jasper_encoder)
        log_probs = jasper_decoder(encoder_output=encoded)
        loss = ctc_loss(log_probs=log_probs,
                        targets=transcript,
                        input_length=encoded_len,
                        target_length=transcript_len)

        callback = nemo.core.SimpleLossLoggerCallback(
            tensor_list2string=lambda x: str(x[0].item()))
        # Instantiate an optimizer to perform `train` action
        neural_factory = nemo.core.NeuralModuleFactory(
            backend=nemo.core.Backend.PyTorch, local_rank=None)
        optimizer = neural_factory.get_trainer(
            params={"optimization_params": {"num_epochs": 2, "lr": 0.0003}})
        optimizer.train([loss], callbacks=[callback])

    def test_freeze_unfreeze_Wrapper(self):
        neural_factory = nemo.core.NeuralModuleFactory(
            backend=nemo.core.Backend.PyTorch,
            placement=nemo.core.DeviceType.GPU)

        dl_train = nemo.backends.pytorch.ZerosDataLayer(size=40,
                                                        dtype=[torch.FloatTensor,
                                                              torch.LongTensor],
                                                        batch_size=4,
                                                        output_ports={
                                                            "image": NeuralType({0: AxisType(BatchTag),
                                                                                 1: AxisType(ChannelTag, 3),
                                                                                 2: AxisType(HeightTag, 224),
                                                                                 3: AxisType(WidthTag, 224)}),
                                                            "label": NeuralType({0: AxisType(BatchTag)})
                                                        })

        # NOTICE: pretrain=True argument
        resnet = neural_factory.get_module(name="resnet18",
                                           params={"num_classes": 2},
                                           collection="torchvision",
                                           pretrained=True)

        L_train = neural_factory.get_module(
            name="CrossEntropyLoss", collection="toys",
            params={})

        # NOTICE: Freeze all Neural Module's weights
        resnet.freeze()
        # NOTICE: unfreeze, top classification layer for fine-tuning
        resnet.unfreeze(set(["fc.weight", "fc.bias"]))

        images, labels = dl_train()
        outputs = resnet(x=images)
        train_loss = L_train(predictions=outputs, labels=labels)

        callback = nemo.core.SimpleLossLoggerCallback(
            tensor_list2string=lambda x: str(x[0].item()))
        # Instantiate an optimizer to perform `train` action
        neural_factory = nemo.core.NeuralModuleFactory(
            backend=nemo.core.Backend.PyTorch, local_rank=None)
        optimizer = neural_factory.get_trainer(
            params={"optimization_params": {"num_epochs": 2, "lr": 0.0003}})
        optimizer.train([train_loss], callbacks=[callback])

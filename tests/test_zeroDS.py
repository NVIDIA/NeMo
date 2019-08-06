import unittest
import os
import tarfile
import torch
from ruamel.yaml import YAML

from nemo.core.neural_types import *
from tests.context import nemo,nemo_asr


class TestZeroDL(unittest.TestCase):
    labels = ["'", "a", "b", "c", "d", "e", "f", "g", "h",
              "i", "j", "k", "l", "m", "n", "o", "p", "q",
              "r", "s", "t", "u", "v", "w", "x", "y", "z", " "]
    manifest_filepath = "tests/data/asr/an4_train.json"
    yaml = YAML(typ="safe")

    def setUp(self) -> None:
        data_folder = "tests/data/"
        print("Looking up for test ASR data")
        if not os.path.exists(data_folder + "nemo_asr"):
            print("Extracting ASR data to: {0}".format(data_folder + "nemo_asr"))
            tar = tarfile.open("tests/data/asr.tar.gz", "r:gz")
            tar.extractall(path=data_folder)
            tar.close()
        else:
            print("ASR data found in: {0}".format(data_folder + "asr"))

    def test_simple_train(self):
        print("Simplest train test with ZeroDL")
        neural_factory = nemo.core.neural_factory.NeuralModuleFactory(
            backend=nemo.core.Backend.PyTorch)
        trainable_module = nemo.backends.pytorch.tutorials.TaylorNet(dim=4)
        data_source = nemo.backends.pytorch.common.ZerosDataLayer(size=10000,
                                                           dtype=torch.FloatTensor,
                                                           batch_size=128,
                                                           output_ports={
                                                               "x": NeuralType({
                                                                   0: AxisType(
                                                                       BatchTag),
                                                                   1: AxisType(
                                                                       ChannelTag, dim=1)}),
                                                               "y": NeuralType({
                                                                   0: AxisType(
                                                                       BatchTag),
                                                                   1: AxisType(
                                                                       ChannelTag, dim=1)})})
        loss = nemo.backends.pytorch.tutorials.MSELoss()
        x, y = data_source()
        y_pred = trainable_module(x=x)
        l = loss(predictions=y_pred, target=y)

        callback = nemo.core.SimpleLossLoggerCallback(
            tensor_list2string=lambda x: str(x[0].item()))
        # Instantiate an optimizer to perform `train` action
        optimizer = neural_factory.get_trainer(
            params={"optimization_params": {"num_epochs": 3, "lr": 0.0003}})
        optimizer.train([l], callbacks=[callback])

    def test_asr_with_zero_ds(self):
        print("Testing ASR NMs with ZeroDS and without pre-processing")
        with open("tests/data/jasper_smaller.yaml") as file:
            jasper_model_definition = self.yaml.load(file)

        dl = nemo.backends.pytorch.common.ZerosDataLayer(
            size=100, dtype=torch.FloatTensor,
            batch_size=4,
            output_ports={
                "processed_signal": NeuralType(
                    {0: AxisType(BatchTag),
                     1: AxisType(SpectrogramSignalTag, dim=64),
                     2: AxisType(ProcessedTimeTag, dim=64)}),
                "processed_length": NeuralType(
                    {0: AxisType(BatchTag)}),
                "transcript": NeuralType({0: AxisType(BatchTag),
                                          1: AxisType(TimeTag, dim=64)}),
                "transcript_length": NeuralType({0: AxisType(BatchTag)})
            })

        jasper_encoder = nemo_asr.JasperEncoder(
            feat_in=jasper_model_definition['AudioPreprocessing']['features'],
            **jasper_model_definition["JasperEncoder"])
        jasper_decoder = nemo_asr.JasperDecoderForCTC(
            feat_in=1024,
            num_classes=len(self.labels)
        )
        ctc_loss = nemo_asr.CTCLossNM(num_classes=len(self.labels))

        # DAG
        processed_signal, p_length, transcript, transcript_len = dl()
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

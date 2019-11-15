# Copyright (c) 2019 NVIDIA Corporation
import unittest
import os
import torch
from pathlib import Path
from ruamel.yaml import YAML

from .context import nemo, nemo_asr, nemo_nlp
from .common_setup import NeMoUnitTest


class TestDeployExport(NeMoUnitTest):
    def setUp(self) -> None:
        self.nf = nemo.core.NeuralModuleFactory(
            placement=nemo.core.DeviceType.CPU)

    def __test_export_route(self, module, out_name, mode, input_example=None):
        out = Path(out_name)
        if out.exists():
            os.remove(out)

        self.nf.deployment_export(
            modules=[module],
            outputs=[out_name],
            input_examples=[input_example],
            d_format=mode)

        self.assertTrue(out.exists())
        if out.exists():
            os.remove(out)
            os.remove(out_name + ".json")

    # def test_simple_module_export(self):
    #     simplest_module = \
    #         nemo.backends.pytorch.tutorials.TaylorNet(dim=4, factory=self.nf)
    #     self.__test_export_route(module=simplest_module,
    #                              out_name="simple.pt",
    #                              mode=nemo.core.DeploymentFormat.TORCHSCRIPT,
    #                              input_example=None)
    #
    # def test_simple_module_onnx_export(self):
    #     simplest_module = \
    #         nemo.backends.pytorch.tutorials.TaylorNet(dim=4, factory=self.nf)
    #     self.__test_export_route(module=simplest_module,
    #                              out_name="simple.onnx",
    #                              mode=nemo.core.DeploymentFormat.ONNX,
    #                              input_example=torch.randn(16, 1))
    #
    # def test_TokenClassifier_module_export(self):
    #     t_class = nemo_nlp.TokenClassifier(hidden_size=512, num_classes=16,
    #                                        use_transformer_pretrained=False)
    #     self.__test_export_route(module=t_class,
    #                              out_name="t_class.pt",
    #                              mode=nemo.core.DeploymentFormat.TORCHSCRIPT,
    #                              input_example=torch.randn(16, 16, 512))
    #
    # def test_TokenClassifier_module_onnx_export(self):
    #     t_class = nemo_nlp.TokenClassifier(hidden_size=512, num_classes=16,
    #                                        use_transformer_pretrained=False)
    #     self.__test_export_route(module=t_class,
    #                              out_name="t_class.onnx",
    #                              mode=nemo.core.DeploymentFormat.ONNX,
    #                              input_example=torch.randn(16, 16, 512))
    #
    # def test_jasper_decoder_export(self):
    #     j_decoder = nemo_asr.JasperDecoderForCTC(feat_in=1024,
    #                                              num_classes=33)
    #     self.__test_export_route(module=j_decoder,
    #                              out_name="j_decoder.pt",
    #                              mode=nemo.core.DeploymentFormat.TORCHSCRIPT,
    #                              input_example=None)
    #
    # def test_hf_bert(self):
    #     bert = nemo_nlp.huggingface.BERT(
    #         pretrained_model_name="bert-base-uncased")
    #     input_example = (torch.randint(low=0, high=16, size=(2, 16)),
    #                      torch.randint(low=0, high=1, size=(2, 16)),
    #                      torch.randint(low=0, high=1, size=(2, 16)))
    #     self.__test_export_route(module=bert,
    #                              out_name="bert.pt",
    #                              mode=nemo.core.DeploymentFormat.TORCHSCRIPT,
    #                              input_example=input_example)

    def test_hf_bert_pt(self):
        bert = nemo_nlp.huggingface.BERT(
            pretrained_model_name="bert-base-uncased")
        self.__test_export_route(module=bert,
                                 out_name="bert.pt",
                                 mode=nemo.core.DeploymentFormat.PYTORCH)

    # def test_jasper_encoder_export(self):
    #     out_name = "jasper_encoder.pt"
    #     out = Path(out_name)
    #     if out.exists():
    #         os.remove(out)
    #     with open("tests/data/jasper_smaller.yaml") as file:
    #         yaml = YAML(typ="safe")
    #         jasper_model_definition = yaml.load(file)
    #     nf = nemo.core.NeuralModuleFactory(
    #         placement=nemo.core.DeviceType.CPU)
    #     jasper_encoder = nemo_asr.JasperEncoder(
    #         conv_mask=False,
    #         feat_in=jasper_model_definition['AudioPreprocessing']['features'],
    #         **jasper_model_definition['JasperEncoder']
    #     )
    #     nf.deployment_export(modules=[jasper_encoder],
    #                          output=out_name,
    #                          # input_example=(torch.randn(2, 64, 2), torch.randn(2)),
    #                          d_format=nemo.core.DeploymentFormat.TORCHSCRIPT)
    #     self.assertTrue(out.exists())
    #     if out.exists():
    #         os.remove(out)

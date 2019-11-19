# Copyright (c) 2019 NVIDIA Corporation
import argparse
import os
from pathlib import Path

import torch
from common_setup import NeMoUnitTest
from context import nemo, nemo_asr, nemo_nlp
from ruamel.yaml import YAML


class TestDeployExport(NeMoUnitTest):
    def setUp(self) -> None:
        self.nf = nemo.core.NeuralModuleFactory(
            placement=nemo.core.DeviceType.CPU)

    def __test_export_route(self, module, out_name, mode, input_example=None):
        out = Path(out_name)
        if out.exists():
            os.remove(out)

        nf = nemo.core.NeuralModuleFactory(
            placement=nemo.core.DeviceType.CPU)

        nf.deployment_export(
            modules=[module],
            output=out_name,
            input_example=input_example,
            d_format=mode)

        self.assertTrue(out.exists())
        if out.exists():
            os.remove(out)

    def test_simple_module_export(self):
        simplest_module = \
            nemo.backends.pytorch.tutorials.TaylorNet(dim=4, factory=self.nf)
        self.__test_export_route(module=simplest_module,
                                 out_name="simple.pt",
                                 mode=nemo.core.DeploymentFormat.TORCHSCRIPT,
                                 input_example=None)

    def test_simple_module_onnx_export(self):
        simplest_module = \
            nemo.backends.pytorch.tutorials.TaylorNet(dim=4, factory=self.nf)
        self.__test_export_route(module=simplest_module,
                                 out_name="simple.onnx",
                                 mode=nemo.core.DeploymentFormat.ONNX,
                                 input_example=torch.randn(16, 1))

    def test_TokenClassifier_module_export(self):
        t_class = nemo_nlp.TokenClassifier(hidden_size=512, num_classes=16,
                                           use_transformer_pretrained=False)
        self.__test_export_route(module=t_class,
                                 out_name="t_class.pt",
                                 mode=nemo.core.DeploymentFormat.TORCHSCRIPT,
                                 input_example=torch.randn(16, 16, 512))

    def test_TokenClassifier_module_onnx_export(self):
        t_class = nemo_nlp.TokenClassifier(hidden_size=512, num_classes=16,
                                           use_transformer_pretrained=False)
        self.__test_export_route(module=t_class,
                                 out_name="t_class.onnx",
                                 mode=nemo.core.DeploymentFormat.ONNX,
                                 input_example=torch.randn(16, 16, 512))

    def test_jasper_decoder_export(self):
        j_decoder = nemo_asr.JasperDecoderForCTC(feat_in=1024,
                                                 num_classes=33)
        self.__test_export_route(module=j_decoder,
                                 out_name="j_decoder.onnx",
                                 mode=nemo.core.DeploymentFormat.ONNX,
                                 input_example=None)

    def test_hf_bert(self):
        bert = nemo_nlp.huggingface.BERT(
            pretrained_model_name="bert-base-uncased")
        input_example = (torch.randint(low=0, high=16, size=(2, 16)),
                         torch.randint(low=0, high=1, size=(2, 16)),
                         torch.randint(low=0, high=1, size=(2, 16)))
        self.__test_export_route(module=bert,
                                 out_name="bert.pt",
                                 mode=nemo.core.DeploymentFormat.TORCHSCRIPT,
                                 input_example=input_example)

    def test_jasper_encoder_export(self, out_name,
                                   d_format=nemo.core.DeploymentFormat.ONNX):
        out = Path(out_name)
        if out.exists():
            os.remove(out)
        with open("tests/data/jasper_smaller.yaml") as file:
            yaml = YAML(typ="safe")
            jasper_model_definition = yaml.load(file)
        nf = nemo.core.NeuralModuleFactory(
            placement=nemo.core.DeviceType.CPU)
        jasper_encoder = nemo_asr.JasperEncoder(
            conv_mask=False,
            feat_in=jasper_model_definition['AudioPreprocessing']['features'],
            **jasper_model_definition['JasperEncoder']
        )

        with torch.no_grad():
            nf.deployment_export(modules=[jasper_encoder],
                                 output=out_name,
                                 input_example=(
                                 torch.randn(16, 64, 256), torch.randn(256)),
                                 d_format=d_format)
            self.assertTrue(out.exists())


def main(args):
    td = TestDeployExport()
    print("ONNX. . .")

    if args.pyt_path:
        td.test_jasper_encoder_export(
            d_format=nemo.core.DeploymentFormat.TORCHSCRIPT,
            out_name="jasper_encoder.pt")

    if args.onnx_path:
        td.test_jasper_encoder_export(out_name=args.onnx_path)


def parse_args():
    parser = argparse.ArgumentParser(description='test_deploy')
    parser.add_argument("--onnx_path", default=None, type=str,
                        help="Path to onnx model for engine creation")
    parser.add_argument("--pyt_path", default=None, type=str,
                        help="Path to TS saved engine")
    parser.add_argument("--engine_path", default=None, type=str,
                        help="Path to serialized TRT engine")
    parser.add_argument("--decoder", action="store_true",
                        help="Path to serialized TRT engine")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

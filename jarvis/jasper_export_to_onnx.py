# Copyright (c) 2019 NVIDIA Corporation
import argparse
import nemo
import nemo_asr
import torch
from ruamel.yaml import YAML


def get_parser():
    parser = argparse.ArgumentParser(
        description="Export Jasper NeMo (0.9.x) checkpoint to ONNX")
    parser.add_argument(
        "--config", default=None, type=str, required=True,
        help="NeMo config for Jasper. Should match your checkpoint")
    parser.add_argument(
        "--nn_encoder", default=None, type=str, required=True,
        help="Path to the nn encoder checkpoint.")
    parser.add_argument(
        "--nn_decoder", default=None, type=str, required=True,
        help="Path to the nn decoder checkpoint.")
    parser.add_argument(
        "--onnx_encoder", default=None, type=str, required=True,
        help="Path to the onnx encoder output.")
    parser.add_argument(
        "--onnx_decoder", default=None, type=str, required=True,
        help="Path to the onnx decoder output.")
    return parser


def main(config_file, nn_encoder, nn_decoder, nn_onnx_encoder,
         nn_onnx_decoder):
    # Read configutation
    yaml = YAML(typ="safe")
    with open(config_file) as f:
        jasper_model_definition = yaml.load(f)

    nf = nemo.core.NeuralModuleFactory()
    # Instantiate Neural Modules
    jasper_encoder = nemo_asr.JasperEncoder(
        feat_in=64,
        **jasper_model_definition['JasperEncoder'])
    jasper_decoder = nemo_asr.JasperDecoderForCTC(
        feat_in=1024, num_classes=len(jasper_model_definition['labels']))

    jasper_encoder.restore_from(nn_encoder)
    jasper_decoder.restore_from(nn_decoder)
    # Perform Export
    nf.deployment_export(module=jasper_encoder,
                         output=nn_onnx_encoder,
                         d_format=nemo.core.neural_factory.DeploymentFormat
                         .ONNX,
                         input_example=(
                         torch.zeros(1, 64, 256, dtype=torch.float,
                                     device="cuda:0"),
                         torch.zeros(1, dtype=torch.int,
                                     device="cuda:0")))
    nf.deployment_export(module=jasper_decoder,
                         output=nn_onnx_decoder,
                         d_format=nemo.core.neural_factory.DeploymentFormat
                         .ONNX,
                         input_example=(
                             torch.zeros(1, 1024, 128, dtype=torch.float,
                                         device="cuda:0")))


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args.config, args.nn_encoder, args.nn_decoder, args.onnx_encoder,
         args.onnx_decoder)

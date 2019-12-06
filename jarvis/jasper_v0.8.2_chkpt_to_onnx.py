# Copyright (c) 2019 NVIDIA Corporation
import nemo
import nemo_asr
import torch
from nemo_asr.parts.jasper import MaskedConv1d
from ruamel.yaml import YAML


def get_parser():
    parser = argparse.ArgumentParser(
        description="Convert 0.8.2 jasper checkpoint to onnx")
    parser.add_argument(
        "--config", default=None, type=str, required=True,
        help="Config from nemo")
    parser.add_argument(
        "--nn_encoder", default=None, type=str, required=True,
        help="Path to the nn encoder checkpoint.")
    parser.add_argument(
        "--nn_decoder", default=None, type=str, required=True,
        help="Path to the nn encoder checkpoint.")
    parser.add_argument(
        "--onnx_encoder", default=None, type=str, required=True,
        help="Path to the onnx encoder output.")
    parser.add_argument(
        "--nn_decoder", default=None, type=str, required=True,
        help="Path to the onnx decoder output.")
    return parser


def main(config_file, nn_encoder, nn_decoder, nn_onnx_encoder,
         nn_onnx_decoder):
    yaml = YAML(typ="safe")
    with open(config_file) as f:
        jasper_model_definition = yaml.load(f)

    jasper_encoder = nemo_asr.JasperEncoder(
        feat_in=64,
        **jasper_model_definition['JasperEncoder'])
    jasper_decoder = nemo_asr.JasperDecoderForCTC(
        feat_in=1024, num_classes=len(jasper_model_definition['labels']))

    # hack to map the state dict of previous version
    ckpt = torch.load(nn_encoder)
    new_ckpt = {}
    for k, v in ckpt.items():
        new_k = k.replace('.conv.', '.mconv.')
        if len(v.shape) == 3:
            new_k = new_k.replace('.weight', '.conv.weight')
        new_ckpt[new_k] = v
    torch.save(new_ckpt, nn_encoder + ".new")

    jasper_encoder.restore_from(nn_encoder + ".new")
    jasper_decoder.restore_from(nn_decoder)

    # disable masked convs
    count = 0
    for m in jasper_encoder.modules():
        if isinstance(m, MaskedConv1d):
            m.use_mask = False
            count += 1
    print("Disabled {} masked convolutions".format(count))

    with torch.no_grad():
        nf = nemo.core.NeuralModuleFactory(log_dir="test",
                                           create_tb_writer=False)
        nf.deployment_export(jasper_encoder, nn_onnx_encoder,
                             nemo.core.neural_factory.DeploymentFormat.ONNX,
                             (torch.zeros(1, 64, 256, dtype=torch.float,
                                          device="cuda:0"),
                              torch.zeros(1, dtype=torch.int,
                                          device="cuda:0")))
        nf.deployment_export(jasper_decoder, nn_onnx_decoder,
                             nemo.core.neural_factory.DeploymentFormat.ONNX,
                             (torch.zeros(1, 1024, 128, dtype=torch.float,
                                          device="cuda:0")))


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args.config, args.nn_encoder, args.nn_decoder, args.onnx_encoder,
         args.onnx_decoder)

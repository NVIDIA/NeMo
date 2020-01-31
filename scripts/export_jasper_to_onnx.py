# Copyright (c) 2019 NVIDIA Corporation
import argparse

import torch
from ruamel.yaml import YAML

import nemo
import nemo.collections.asr as nemo_asr


def get_parser():
    parser = argparse.ArgumentParser(description="Convert Jasper NeMo checkpoint to ONNX")
    parser.add_argument(
        "--config", default=None, type=str, required=True, help="Config from nemo",
    )
    parser.add_argument(
        "--nn_encoder", default=None, type=str, required=True, help="Path to the nn encoder checkpoint.",
    )
    parser.add_argument(
        "--nn_decoder", default=None, type=str, required=True, help="Path to the nn encoder checkpoint.",
    )
    parser.add_argument(
        "--onnx_encoder", default=None, type=str, required=True, help="Path to the onnx encoder output.",
    )
    parser.add_argument(
        "--onnx_decoder", default=None, type=str, required=True, help="Path to the onnx decoder output.",
    )
    parser.add_argument(
        "--pre-v09-model", action="store_true", help="Use if checkpoints were generated from NeMo < v0.9",
    )
    return parser


def main(
    config_file,
    nn_encoder,
    nn_decoder,
    nn_onnx_encoder,
    nn_onnx_decoder,
    pre_v09_model=False,
    batch_size=1,
    time_steps=256,
):
    yaml = YAML(typ="safe")

    logging.info("Loading config file...")
    with open(config_file) as f:
        jasper_model_definition = yaml.load(f)

    logging.info("Determining model shape...")
    if 'AudioPreprocessing' in jasper_model_definition:
        num_encoder_input_features = jasper_model_definition['AudioPreprocessing']['features']
    elif 'AudioToMelSpectrogramPreprocessor' in jasper_model_definition:
        num_encoder_input_features = jasper_model_definition['AudioToMelSpectrogramPreprocessor']['features']
    else:
        num_encoder_input_features = 64
    num_decoder_input_features = jasper_model_definition['JasperEncoder']['jasper'][-1]['filters']
    logging.info("  Num encoder input features: {}".format(num_encoder_input_features))
    logging.info("  Num decoder input features: {}".format(num_decoder_input_features))

    logging.info("Initializing models...")
    jasper_encoder = nemo_asr.JasperEncoder(
        feat_in=num_encoder_input_features, **jasper_model_definition['JasperEncoder']
    )
    jasper_decoder = nemo_asr.JasperDecoderForCTC(
        feat_in=num_decoder_input_features, num_classes=len(jasper_model_definition['labels']),
    )

    # This is necessary if you are using checkpoints trained with NeMo
    # version before 0.9
    logging.info("Loading checkpoints...")
    if pre_v09_model:
        logging.info("  Converting pre v0.9 checkpoint...")
        ckpt = torch.load(nn_encoder)
        new_ckpt = {}
        for k, v in ckpt.items():
            new_k = k.replace('.conv.', '.mconv.')
            if len(v.shape) == 3:
                new_k = new_k.replace('.weight', '.conv.weight')
            new_ckpt[new_k] = v
        jasper_encoder.load_state_dict(new_ckpt)
    else:
        jasper_encoder.restore_from(nn_encoder)
    jasper_decoder.restore_from(nn_decoder)

    nf = nemo.core.NeuralModuleFactory(create_tb_writer=False)
    logging.info("Exporting encoder...")
    nf.deployment_export(
        jasper_encoder,
        nn_onnx_encoder,
        nemo.core.neural_factory.DeploymentFormat.ONNX,
        torch.zeros(batch_size, num_encoder_input_features, time_steps, dtype=torch.float, device="cuda:0",),
    )
    logging.info("Exporting decoder...")
    nf.deployment_export(
        jasper_decoder,
        nn_onnx_decoder,
        nemo.core.neural_factory.DeploymentFormat.ONNX,
        (torch.zeros(batch_size, num_decoder_input_features, time_steps // 2, dtype=torch.float, device="cuda:0",)),
    )
    logging.info("Export completed successfully.")


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(
        args.config,
        args.nn_encoder,
        args.nn_decoder,
        args.onnx_encoder,
        args.onnx_decoder,
        pre_v09_model=args.pre_v09_model,
    )

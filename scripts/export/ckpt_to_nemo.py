import argparse

from nemo.collections.asr.models import EncDecClassificationModel, EncDecCTCModel, EncDecSpeakerLabelModel
from nemo.utils import logging


def get_parser():
    parser = argparse.ArgumentParser(description="Convert .ckpt file to .nemo file")
    parser.add_argument(
        "--nemo_file", default=None, type=str, required=True, help="Path to .nemo output",
    )
    parser.add_argument(
        "--ckpt_file", default=None, type=str, required=True, help="Path to the ckpt file",
    )
    parser.add_argument(
        "--model_type",
        default='asr',
        type=str,
        choices=['asr', 'speech_label', 'speaker'],
        help="Type of decoder used by the model.",
    )
    return parser


def main(nemo_file, ckpt_file, model_type='asr'):
    if model_type == 'asr':
        logging.info("Preparing ASR model")
        model = EncDecCTCModel.load_from_checkpoint(ckpt_file)
    elif model_type == 'speech_label':
        logging.info("Preparing Speech Label Classification model")
        model = EncDecClassificationModel.load_from_checkpoint(ckpt_file)
    elif model_type == 'speaker':
        logging.info("Preparing Speaker Recognition model")
        model = EncDecSpeakerLabelModel.load_from_checkpoint(ckpt_file)
    else:
        raise NameError("Available model names are asr, speech_label and speaker")

    logging.info("Writing nemo file")
    model.save_to(nemo_file)
    logging.info("Successfully ported nemo file")


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args.nemo_file, args.ckpt_file, model_type=args.model_type)

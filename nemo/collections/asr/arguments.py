from argparse import ArgumentParser

def add_asr_args(parent_parser: ArgumentParser) -> ArgumentParser:
    """Extends existing argparse with default ASR collection args.

    Args:
        parent_parser (ArgumentParser): Custom CLI parser that will be extended.

    Returns:
        ArgumentParser: Parser extended by NeMo ASR Collection arguments.
    """
    parser = ArgumentParser(parents=[parent_parser], add_help=False, )
    parser.add_argument("--asr_model", type=str, required=True, default="bad_quartznet15x5.yaml", help="")
    parser.add_argument("--train_dataset", type=str, required=True, default=None, help="training dataset path")
    parser.add_argument("--eval_dataset", type=str, required=True, help="evaluation dataset path")
    return parser
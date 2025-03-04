import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from nemo.collections import llm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo-path", type=str, default="/root/.cache/nemo/models/models/llama_31_toy")
    parser.add_argument("--original-hf-path", type=str, default="models/llama_31_toy")
    parser.add_argument("--output-path", type=str, default="/tmp/output_hf")
    parser.add_argument("--add-model-name", action="store_true", default=False)
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    kwargs = {}
    if args.add_model_name:
        kwargs = {
            'target_model_name': args.original_hf_path,
        }
    llm.export_ckpt(
        path=Path(args.nemo_path),
        target='hf',
        output_path=Path(args.output_path),
        overwrite=True,
        **kwargs,
    )

    original_hf = AutoModelForCausalLM.from_pretrained(args.original_hf_path, trust_remote_code=True)
    converted_hf = AutoModelForCausalLM.from_pretrained(args.output_path, trust_remote_code=True)

    for (name1, parameter1), (name2, parameter2) in zip(
        converted_hf.named_parameters(), original_hf.named_parameters()
    ):
        try:
            assert name1 == name2, f'Parameter names do not match: {name1} != {name2}'
            assert torch.equal(parameter1, parameter2), f'Parameter weight do not match for {name1}'
        except Exception as e:
            breakpoint()
            print('ri')

    print('All weights matched.')

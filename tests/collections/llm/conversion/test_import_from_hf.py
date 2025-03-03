import argparse

from nemo.collections import llm
import nemo.lightning as nl
import json
import torch
from pathlib import Path
from nemo.lightning import io
from nemo.lightning.ckpt_utils import ckpt_to_context_subdir
from nemo.collections.llm.inference.base import _setup_trainer_and_restore_model
from prettytable import PrettyTable

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-path", type=str, default="models/llama_31_toy")
    parser.add_argument("--model-type", type=str, help="Name of the model", default="LlamaModel")
    parser.add_argument("--model-config", type=str, help="Model config", default="Llama31Config8B")
    parser.add_argument("--output-path", type=str, help="Output NeMo2 model")
    return parser

def count_parameters(model):
    table = PrettyTable(["NeMo Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"NeMo Total Trainable Params: {total_params}")
    return total_params

if __name__ == '__main__':
    args = get_parser().parse_args()

    config = getattr(llm, args.model_config)()
    model = getattr(llm, args.model_type)(config=config)
    output_path = llm.import_ckpt(
        model=model,
        source=f"hf://{args.hf_path}",
        output_path=args.output_path,
        overwrite=True,
    )

    trainer = nl.Trainer(
        accelerator="cpu",
        devices=1,
        num_nodes=1,
        strategy=nl.MegatronStrategy(ckpt_load_strictness=False),
    )
    path = Path(output_path)
    model_nemo: io.TrainerContext = io.load_context(path=ckpt_to_context_subdir(path), subpath="model")
    _setup_trainer_and_restore_model(path=path, trainer=trainer, model=model_nemo)

    # Load HF Stats
    with open(f'{args.hf_path}/stats.json') as f:
        hf_stats = json.load(f)

    table = PrettyTable(["HuggingFace Modules", "Parameters"])
    for key, value in hf_stats.items():
        table.add_row([key, value])
    print(table)

    nemo_param_cnt = count_parameters(model_nemo)
    hf_param_cnt = hf_stats['total_params']

    assert hf_param_cnt == nemo_param_cnt, f'Total converted params count does not match: NeMo model has {nemo_param_cnt} and HF model has {hf_param_cnt}.'
    print('HuggingFace -> NeMo conversion completed successfully.')




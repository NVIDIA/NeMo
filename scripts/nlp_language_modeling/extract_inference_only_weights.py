#!/usr/bin/env

"""A script to extract the final p-tuning representations used for inference.
"""
import argparse
import os
import torch

parser = argparse.ArgumentParser()
parser.add_argument("nemo", help="path to nemo file", type=str)
parser.add_argument("taskname", help="taskname for the nemo model", type=str, default="taskname", required=False)
args = parser.parse_args()

os.system(f"tar xvf {args.nemo}")

for p in '', 'mp_rank_00/', 'tp_rank_00_pp_rank_000/':
    try:
        a = torch.load(f'{p}model_weights.ckpt')
        break
    except FileNotFoundError:
        pass
inf_weights = a['prompt_table'][f'prompt_table.{args.taskname}.prompt_embeddings.weight']
torch.save(inf_weights, "p_tuned.inf_only.ckpt")

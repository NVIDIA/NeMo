import argparse

import modelopt.torch.export as mtex
import torch
from megatron.core.models.gpt import GPTModel
from megatron.training.utils import unwrap_model

from nemo.collections.llm.modelopt import setup_trainer_and_restore_model_with_modelopt_spec
from nemo.utils import logging


def get_args():
    """Parse command line arguments for exporting a NeMo model to HuggingFace format.

    Returns:
        argparse.Namespace: The parsed command line arguments containing:
            model_path (str): Path to the NeMo model checkpoint to export
            pretrained_model_name (str): Name or path of the HuggingFace model to use as reference
                for architecture configuration
            export_dir (str): Directory where the exported HuggingFace model will be saved
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--pretrained_model_name", type=str, required=True)
    parser.add_argument("--export_dir", type=str, required=True)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--num_layers_in_first_pipeline_stage", type=int, default=None)
    parser.add_argument("--num_layers_in_last_pipeline_stage", type=int, default=None)
    parser.add_argument("--expert_parallel_size", type=int, default=1)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--legacy_ckpt", type=bool, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    model, trainer = setup_trainer_and_restore_model_with_modelopt_spec(
        model_path=args.model_path,
        pipeline_model_parallel_size=args.pipeline_parallel_size,
        expert_model_parallel_size=args.expert_parallel_size,
        num_layers_in_first_pipeline_stage=args.num_layers_in_first_pipeline_stage,
        num_layers_in_last_pipeline_stage=args.num_layers_in_last_pipeline_stage,
        devices=args.devices,
        num_nodes=args.nodes,
        inference_only=True,
        legacy_ckpt=args.legacy_ckpt,
    )
    model.configure_model()
    mcore_model = model.module
    if type(mcore_model) != GPTModel:
        mcore_model = mcore_model.module
    assert type(mcore_model) == GPTModel, f"Module is not a GPTModel, found {type(mcore_model)} instead"
    unwrapped_model = unwrap_model(mcore_model)
    logging.info("Unwrapped Megatron-Core model. Exporting to HuggingFace format...")
    mtex.export_mcore_gpt_to_hf(
        unwrapped_model,
        args.pretrained_model_name,
        dtype=torch.bfloat16,
        export_dir=args.export_dir,
    )

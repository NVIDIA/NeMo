from nemo import lightning as nl
from nemo.collections import llm
from megatron.core.optimizer import OptimizerConfig
import torch
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from nemo.lightning.megatron_parallel import MegatronParallel
from nemo.lightning.io.pl import TrainerContext
from nemo.utils.get_rank import is_global_rank_zero
from nemo.lightning.ckpt_utils import ckpt_to_context_subdir, ckpt_to_weights_subdir
from pathlib import Path
import os
from argparse import ArgumentParser
from nemo.utils import logging




def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        required=True,
        help="Path to NeMo 1.0 checkpoints",
    )
    parser.add_argument("--output_path", type=str, default=None, required=True, help="Path to output NeMo 2.0 directory.")
    parser.add_argument("--precision", type=str, default="16", help="Model precision")
    args = parser.parse_args()
    return args




def main() -> None:    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    model = llm.LlamaModel(llm.Llama3Config8B(), tokenizer=tokenizer)
    tp_size=1


    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=tp_size,
        setup_optimizers=False,
        init_model_parallel=False
    )

    trainer = nl.Trainer(
        devices=tp_size,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
    )
    trainer.strategy.connect(model)
    trainer.strategy.setup_environment()
    trainer.strategy._setup_optimizers = False
    trainer.strategy._init_model_parallel = False
    trainer.strategy.setup(trainer)
    trainer.model.configure_model()


    logging.info(f"loading checkpoint {args.input_path}")

    sharded_state_dict = {"state_dict": trainer.model.sharded_state_dict()}
    for k in list(sharded_state_dict['state_dict'].keys()):
        new_key = k.replace('module', 'model', 1)
        sharded_state_dict['state_dict'][new_key] = sharded_state_dict['state_dict'].pop(k)
        sharded_state_dict['state_dict'][new_key].key = sharded_state_dict['state_dict'][new_key].key.replace('module', 'model', 1)
    model_ckpt = trainer.strategy.checkpoint_io.load_checkpoint(args.input_path, sharded_state_dict, None)
    model_ckpt['state_dict'] = {k.replace('model', 'module', 1): v for k, v in model_ckpt['state_dict'].items()}
    trainer.model.module.load_state_dict(model_ckpt['state_dict'])
    trainer.save_checkpoint(ckpt_to_weights_subdir(args.output_path))

    if is_global_rank_zero():
        #Corresponding to Connector: on_import_ckpt
        import pdb; 
        if hasattr(trainer.model, "__io__") and hasattr(trainer.model.tokenizer, '__io__'):
            trainer.model.__io__.tokenizer = trainer.model.tokenizer.__io__
        TrainerContext.from_trainer(trainer).io_dump(ckpt_to_context_subdir(args.output_path))

if __name__ == '__main__':
    args = get_args()
    main() 
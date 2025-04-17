# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import ArgumentParser

from nemo.collections.llm.modelopt import setup_trainer_and_restore_model_with_modelopt_spec
from nemo.collections.llm.modelopt.speculative import SpeculativeTransform
from nemo.collections.llm.utils import barrier
from nemo.lightning.ckpt_utils import ckpt_to_context_subdir
from nemo.lightning.io.pl import TrainerContext
from nemo.utils.get_rank import is_global_rank_zero


def get_args():
    """Parse the command line arguments."""
    parser = ArgumentParser(description="""Enable Speculative Decoding on a NeMo 2.0 checkpoint.""")

    parser.add_argument("--model_path", type=str, required=True, help="""Path to NeMo 2 checkpoint""")
    parser.add_argument("--num_eagle_layers", type=int, default=1, help="""Number of Eagle layers to use""")
    parser.add_argument("--num_medusa_heads", type=int, default=0, help="""Number of Medusa heads to use""")
    parser.add_argument("--export_dir", type=str, required=True, help="""Path to export checkpoint""")
    parser.add_argument("--tp_size", type=int, default=1, help="""Tensor parallel size""")
    parser.add_argument("--pp_size", type=int, default=1, help="""Pipeline parallel size""")
    parser.add_argument("--devices", type=int, default=1, help="""Number of GPUs to use per node""")
    parser.add_argument("--num_nodes", type=int, default=1, help="""Number of nodes to use""")
    parser.add_argument("--tokenizer", type=str, default=None, help="""Name of tokenizer model to override default""")
    parser.add_argument("--legacy_ckpt", action="store_true", help="""Load ckpt saved with TE < 1.14""")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Set up dummy trainer to restore model
    model, trainer = setup_trainer_and_restore_model_with_modelopt_spec(
        model_path=args.model_path,
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        devices=args.devices,
        num_nodes=args.num_nodes,
        tokenizer_path=args.tokenizer,
        legacy_ckpt=args.legacy_ckpt,
        inference_only=True,
        strategy_kwargs={"sequence_parallel": False, "replace_progress_bar": False},
        trainer_kwargs={},
        model_config_overrides={
            "sequence_parallel": False,
            "moe_grouped_gemm": False,
            "gradient_accumulation_fusion": False,
            "make_vocab_size_divisible_by": 1,
        },
        # WAR: We force vocab size to be divisible by 1 as it performs inconsistent padding upon restore
    )

    model_transform = SpeculativeTransform(
        num_eagle_layers=args.num_eagle_layers, num_medusa_heads=args.num_medusa_heads
    )
    model_transform(model)

    # Save to disk
    trainer.save_checkpoint(args.export_dir)
    barrier()
    if is_global_rank_zero():
        TrainerContext.from_trainer(trainer).io_dump(ckpt_to_context_subdir(args.export_dir), yaml_attrs=["model"])

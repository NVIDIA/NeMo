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

import os
from argparse import ArgumentParser

from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer.enums import AttnBackend

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.gpt.data.packed_sequence import PackedSequenceSpecs
from nemo.lightning.ckpt_utils import ckpt_to_context_subdir
from nemo.lightning.pytorch.callbacks import ModelCheckpoint, NsysCallback
from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from datetime import datetime
from nemo.utils.exp_manager import TimingCallback
from nemo.utils import logging

# Suppress lengthy HF warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_args():
    """Parse the command line arguments."""
    parser = ArgumentParser(description="""Run Supervised Finetune from a pretrained checkpoint.""")

    parser.add_argument("--name", type=str, required=True, help="""Experiment name""")
    parser.add_argument("--model_path", type=str, required=True, help="""Path to NeMo 2 checkpoint""")

    parser.add_argument("--tp_size", type=int, default=1, help="""Tensor parallel size""")
    parser.add_argument("--cp_size", type=int, default=1, help="""Context parallel size""")
    parser.add_argument("--pp_size", type=int, default=1, help="""Pipeline parallel size""")
    parser.add_argument("--pp_layout", type=str, default=None, help="""Virtual Pipeline parallel size""")
    parser.add_argument("--vpp_size", type=int, default=None, help="""Virtual Pipeline parallel size""")
    parser.add_argument("--ep_size", type=int, default=1, help="""Expert parallel size""")
    parser.add_argument("--etp_size", type=int, default=None, help="""Expert tensor parallel size""")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="""Datatype for models and optimizer""")
    parser.add_argument("--devices", type=int, default=1, help="""Number of GPUs to use per node""")
    parser.add_argument("--num_nodes", type=int, default=1, help="""Number of nodes to use""")
    parser.add_argument("--log_dir", type=str, required=True, help="""Folder for logging and checkpoint saving""")
    parser.add_argument("--max_steps", type=int, required=True, help="""Number of global batches to process""")
    parser.add_argument("--gbs", type=int, required=True, help="""Global Batch Size""")
    parser.add_argument("--mbs", type=int, required=True, help="""Micro-batch Size""")
    parser.add_argument("--data_path", type=str, required=True, help="""Data paths containing train/valid/test.jsonl""")

    parser.add_argument("--seq_length", type=int, required=True, help="""Number of tokens per input sample""")
    parser.add_argument("--lr", type=float, default=3e-5, help="""Base LR for Cosine-Annealing scheduler""")
    parser.add_argument("--min_lr", type=float, default=2e-7, help="""Minimum LR for Cosine-Annealing scheduler""")
    parser.add_argument("--warmup_steps", type=int, default=50, help="""Number of scheduler warmup steps""")
    parser.add_argument("--val_check_interval", type=int, default=500, help="""Validate + checkpoint every _ steps""")
    parser.add_argument("--limit_val_batches", type=int, default=32, help="""Number of batches per validation stage""")
    parser.add_argument("--log_interval", type=int, default=10, help="""Write to log every _ steps""")
    parser.add_argument("--nsys-profile", action='store_true', dest='nsys_profile', help="""Enable nsys profiling""")
    parser.add_argument("--packing_length", type=int, default=None, help="""Sequence packing length""")
    parser.add_argument("--use-hf-chat-template", action='store_true', dest='use_hf_chat_template', help="""Use hf chat template""")
    parser.add_argument("--use-flash-attn", action='store_true', dest='use_flash_attn', help="""Use flash attn""")
    parser.add_argument("--recompute_type", type=str, default=None, help="""Recompute type""")

    return parser.parse_args()


def _extract_tokenizer_model_name(tokenizer):
    name = tokenizer.tokenizer.name_or_path
    if name.endswith("context/nemo_tokenizer"):
        # NEMO_HOME/hf_org/hf_model/context/nemo_tokenizer => hf_org--hf_model
        tokenizer_model_name = '--'.join(name.split("/")[-4:-2])
    elif name.endswith("nemo_tokenizer"):
        # NEMO_HOME/hf_org/hf_model/nemo_tokenizer => hf_org--hf_model
        tokenizer_model_name = '--'.join(name.split("/")[-3:-1])
    else:
        # hf_org/hf_model => hf_org--hf_model
        tokenizer_model_name = name.replace("/", "--")
    return tokenizer_model_name

def synthesize_pack_data_path(data_path, tokenizer, packing_length, cp=1):
    #Judge if the packed data already exists!
    tokenizer_model_name = _extract_tokenizer_model_name(tokenizer)
    packed_data_path = os.path.join(data_path, 'packed', tokenizer_model_name, f"CP{cp}")

    packed_train_data_path = os.path.join(packed_data_path, f"training_{packing_length}.npy")
    packed_val_data_path = os.path.join(packed_data_path, f"validation_{packing_length}.npy")
    packed_metadata_path = os.path.join(packed_data_path, f"{packing_length}_metadata.jsonl")

    # If the packed data already exists, we pass this path so that the data won't be packed to a new
    # path if we resume the model from the checkpointed path.
    if (os.path.isfile(packed_train_data_path) and \
        os.path.isfile(packed_val_data_path) and \
        os.path.isfile(packed_metadata_path)) or cp > 1:
        return packed_train_data_path, packed_val_data_path, packed_metadata_path
    else:
        return None, None, None


if __name__ == "__main__":
    args = get_args()
    if args.pp_layout:
        logging.info(f"Using pp_layout: {args.pp_layout}.")

    ## Calculate microbatch_group_size_per_vp_stage for megatron strategy
    DP = args.num_nodes * args.devices // (args.tp_size * args.cp_size * args.pp_size)
    microbatch_group_size_per_pp_stage = args.gbs // args.mbs // DP
    microbatch_group_size_per_vp_stage = microbatch_group_size_per_pp_stage // args.vpp_size if args.vpp_size else None

    ## Initialize the strategy and trainer
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        microbatch_group_size_per_vp_stage=microbatch_group_size_per_vp_stage,
        expert_model_parallel_size=args.ep_size,
        expert_tensor_parallel_size=args.etp_size,
        context_parallel_size=args.cp_size,
        virtual_pipeline_model_parallel_size=args.vpp_size,
        # pipeline_model_parallel_layout = args.pp_layout,
        account_for_embedding_in_pipeline_split=True,  # add
        account_for_loss_in_pipeline_split=True,  # add
        sequence_parallel=(args.tp_size > 1),  # change
        ckpt_load_strictness=True,
        ckpt_assume_constant_structure=True,
        ckpt_parallel_save_within_dp=False,  # change
        ckpt_async_save=True,  # change
        #use_te_rng_tracker=True,
    )
    callbacks = [TimingCallback()]
    #Add tp overlap callbck
    #tp overlap seems to lower efficiency???
    #callbacks.append(
    #    MegatronCommOverlapCallback(tp_comm_overlap=True)
    #)
    if args.nsys_profile:
        logging.info("Appending nsys callback to trainer.")
        nsys_callback = NsysCallback(
            start_step=10,
            end_step=12,
            ranks=[0,1],
            gen_shape=True,
            nvtx_ranges=False
        )
        callbacks.append(nsys_callback)
    trainer = nl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        max_steps=args.max_steps,
        log_every_n_steps=args.log_interval,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        strategy=strategy,
        accelerator="gpu",
        plugins=nl.MegatronMixedPrecision(precision=args.precision),
        callbacks=callbacks,
        num_sanity_val_steps=0
    )
    ## Fix the sft CP grad-non issue
    setattr(trainer.strategy.ddp_config, 'average_in_collective', False)

    ## Load pretrained models in nemo2 format
    model = nl.io.load_context(path=ckpt_to_context_subdir(args.model_path), subpath="model")

    ## Set tokenizer for dateset building
    tokenizer = getattr(model, "tokenizer", None)
    assert tokenizer is not None, "Please provide a model checkpoint with tokenizer included."
    
    ## Qwen3-235B-A22B and Qwen2.5-32B tokenizer has different eos_id
    logging.info(f"tokenizer.eos_id: {tokenizer.eos_id}")

    ## For sft CP
    setattr(model.config, 'calculate_per_token_loss', True)

    ## set model config
    # Use flash AttnBackend. Default use AttnBackend.auto: cudnn > flash > pytorch
    if args.use_flash_attn:
        setattr(model.config, 'attention_backend', AttnBackend.flash)

    setattr(model.config, "gradient_accumulation_fusion", True)

    ## set fusion config, in case not configured in the model config or json
    setattr(model.config, 'apply_rope_fusion', True)
    setattr(model.config, 'bias_activation_fusion', True)

    ## This seems to have no influence on training but we'd better keep it in align?
    setattr(model.config, 'seq_length', args.seq_length) 
    setattr(model.config, 'max_position_embeddings', args.seq_length)

    logging.info(f"Using recompute_type: {args.recompute_type}")
    if args.recompute_type is None:
        pass
    elif args.recompute_type == "full":
        setattr(model.config, 'recompute_granularity', "full")
        setattr(model.config, 'recompute_method', "uniform")
        setattr(model.config, 'recompute_num_layers', 1)
    elif args.recompute_type == "selective":
        ## You may define your own selective recompute modules by modifying recompute_modules
        setattr(model.config, 'recompute_granularity', "selective")
        setattr(model.config, 'recompute_modules', ['core_attn','moe_act','layernorm', 'moe'])
        logging.info(f"Recompute_modules: {model.config.recompute_modules}")
    else:
        raise ValueError(f"Unrecognized recompute_type {args.recompute_type}")

    #TODO: cpu_offloading is not working for memory saving
    #setattr(model.config, 'cpu_offloading', True)
    #setattr(model.config, 'cpu_offloading_num_layers', 10)

    #TODO: external_cuda_graph is not ready.
    #setattr(model.config, 'use_te_rng_tracker', True)
    #setattr(model.config, 'external_cuda_graph', True)
    #setattr(model.config, 'cuda_graph_scope', 'attn')
    #setattr(model.config, 'enable_cuda_graph', True)

    #Deal with the checkpoint saving timeout
    setattr(model.config, 'distributed_timeout_minutes', 100)

    logging.info(f"Using model config: {model.config}")

    # Set up dataset
    # special_tokens_dict does not work when use_hf_tokenizer_chat_template
    special_tokens_dict = {
        'system_turn_start': '<|im_start|>',
        'turn_start': '<|im_start|>',
        'label_start': '<extra_id_1>',
        'end_of_turn': '<|im_end|>\n',
        'end_of_name': '\n'
    }
    dataset_kwargs = {
        'chat': True,
        'use_hf_tokenizer_chat_template': args.use_hf_chat_template,
        'pad_to_max_length': True,
        'special_tokens': tuple(special_tokens_dict.items())
    }
    if args.packing_length:
        ptrain, pvalid, pmeta = synthesize_pack_data_path(
            args.data_path,
            tokenizer,
            args.packing_length,
            args.cp_size
        )
        packed_sequence_specs = PackedSequenceSpecs(
            packed_sequence_size = args.packing_length,
            pad_cu_seqlens=True,
            packed_train_data_path=ptrain,
            packed_val_data_path=pvalid,
            packed_metadata_path=pmeta,
        )
    else:
        packed_sequence_specs = None

    data = llm.FineTuningDataModule(
        dataset_root=args.data_path,
        seq_length=args.seq_length,
        global_batch_size=args.gbs,
        micro_batch_size=args.mbs,
        tokenizer=tokenizer,
        packed_sequence_specs=packed_sequence_specs,
        dataset_kwargs=dataset_kwargs,
        num_workers=0,
    )

    ## Set up optimizer
    opt_config = OptimizerConfig(
        optimizer="adam",
        lr=args.lr,
        bf16=("bf16" in args.precision),
        use_distributed_optimizer=True,
    )
    sched = CosineAnnealingScheduler(
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        constant_steps=0,
        min_lr=args.min_lr,
    )
    opt = nl.MegatronOptimizerModule(opt_config, sched)

    # Set up checkpointing and logging
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        every_n_train_steps=args.val_check_interval,
    )
    logger = nl.NeMoLogger(
        name=args.name,
        log_dir=args.log_dir,
        ckpt=checkpoint_callback,
        tensorboard=TensorBoardLogger(os.path.join(args.log_dir, args.name)),
        wandb=WandbLogger(project=args.name,name=datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        update_logger_directory=False,
    )

    # Set up resume and/or restore functionality
    resume = nl.AutoResume(
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
        restore_config=nl.RestoreConfig(path=args.model_path),
    )

    # Run
    llm.finetune(
        model=model,
        data=data,
        optim=opt,
        tokenizer="model",
        trainer=trainer,
        log=logger,
        resume=resume,
    )
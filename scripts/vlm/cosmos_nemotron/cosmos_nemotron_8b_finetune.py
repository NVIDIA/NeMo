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

"""
Mock Data Example:
  torchrun --nproc_per_node=8 scripts/vlm/cosmos_nemotron/cosmos_nemotron_8b_finetune.py \
  --devices=8 --tp=4 --data_type=mock

  torchrun --nproc_per_node=8 scripts/vlm/cosmos_nemotron/cosmos_nemotron_8b_finetune.py \
  --devices=8 --tp=4 --data_type=mock --peft lora

Llava Data Example:
   torchrun --nproc_per_node=8 scripts/vlm/cosmos_nemotron/cosmos_nemotron_8b_finetune.py  \
     --data_path "/path/to/dataset/llava_v1_5_mix665k.json" \
     --image_folder "/path/to/dataset/images" \
     --data_type llava \
     --num_nodes 1 \
     --log_dir "/path/to/experiments/cosmos_nemotron_finetune" \
     --devices=8 \
     --projector_type=mcore_mlp \
     --tp_size 2 --pp_size 1 \
     --gbs 128 --mbs 4 \
     --wandb_project=cosmos_nemotron_demo \
     --name=cosmos_nemotron_finetune \
     --restore_path "/path/to/experiments/cosmos_nemotron_pretrain_checkpoint"
"""

import argparse

import torch
from lightning.pytorch.loggers import WandbLogger
from megatron.core.optimizer import OptimizerConfig

from nemo import lightning as nl
from nemo.collections import llm, vlm
from nemo.collections.multimodal.data.energon.conversation import LLama3TemplateConfig
from nemo.collections.multimodal.data.energon.task_encoder import MultiModalTaskEncoder
from nemo.collections.vlm import ImageDataConfig
from nemo.collections.vlm.vision.vision_transform import VisualProcessor
from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.utils.exp_manager import TimingCallback


def main(args):
    # pylint: disable=C0115,C0116

    # Global and micro batch sizes
    gbs = args.gbs
    mbs = args.mbs
    max_steps = args.max_steps

    decoder_seq_length = 16384
    recompute_num_layers = 0 if args.peft == "lora" else 20

    # Submodules configurations
    language_transformer_config = llm.Llama31Config8B(
        make_vocab_size_divisible_by=512,
        seq_length=decoder_seq_length,
        recompute_granularity="full",
        recompute_method="block",
        recompute_num_layers=recompute_num_layers,
    )

    vision_transformer_config = vlm.RADIO_25_h_Config(
        img_w=512, img_h=512, patch_dim=16,
    )
    vision_projection_config = vlm.MultimodalProjectorConfig(
        input_size=5120, hidden_size=4096, ffn_hidden_size=4096,
        normalization='LayerNorm', projector_type="mcore_mlp",
    )

    # NEVA model configuration
    neva_config = vlm.CosmosNemotronConfig(
        language_transformer_config=language_transformer_config,
        vision_transformer_config=vision_transformer_config,
        vision_projection_config=vision_projection_config,
        language_model_from_pretrained=args.language_model_path,
        freeze_language_model=False,
        freeze_vision_model=False,
        freeze_vision_projection=False,
    )
    num_image_embeddings_per_tile = (
            vision_transformer_config.num_image_embeddings_per_tile
            - vision_transformer_config.class_token_len * neva_config.drop_vision_class_token
    )

    from nemo.collections.common.tokenizers import AutoTokenizer
    tokenizer = AutoTokenizer("meta-llama/Llama-3.1-8B-Instruct")
    new_special_tokens = {
        "additional_special_tokens": [
            "<image>", "<img>", "</img>",
            "<quad>", "</quad>",
            "<ref>", "</ref>",
            "<box>", "</box>"
        ]
    }
    tokenizer.tokenizer.add_special_tokens(new_special_tokens)

    image_processor = VisualProcessor(
        crop_height=512,
        crop_width=512,
        use_tiling=True,
        max_num_tiles=12,
        use_thumbnail=True,
        augment=False,
        vision_model_type="radio",
    )

    if args.data_type == "llava":
        # Data configuration
        data_config = ImageDataConfig(
            image_folder=args.image_folder,
            conv_template="cosmos_nemotron",
        )

        # Data module setup
        data = vlm.NevaPreloadedDataModule(
            paths=args.data_path,
            data_config=data_config,
            seq_length=decoder_seq_length,
            decoder_seq_length=None,
            global_batch_size=gbs,
            micro_batch_size=mbs,
            tokenizer=tokenizer,
            image_processor=image_processor,
            num_workers=4,
            packed_sequence=args.use_packed_sequence,
            pixel_shuffle_ratio=0.5,
            num_image_embeddings_per_tile=num_image_embeddings_per_tile,
            image_tag_type="internvl"
        )
    elif args.data_type == "energon":

        from nemo.collections.multimodal.data.energon import (
            EnergonMultiModalDataModule,
            ImageToken,
            MultiModalSampleConfig,
        )

        # Configure multimodal samples
        config = MultiModalSampleConfig(
            image_token=ImageToken(token_str="<image>", token_id=-200),
            ignore_place_holder=-100,
            conversation_template_config=LLama3TemplateConfig(),
        )

        # Initialize the data module
        data = EnergonMultiModalDataModule(
            path=args.data_path,
            tokenizer=tokenizer,
            image_processor=image_processor,
            seq_length=decoder_seq_length,
            micro_batch_size=mbs,
            global_batch_size=gbs,
            num_workers=4,
            multimodal_sample_config=config,
            task_encoder=MultiModalTaskEncoder(
                tokenizer=tokenizer,
                image_processor=image_processor,
                multimodal_sample_config=config,
                packed_sequence=args.use_packed_sequence,
                # leave some space for perf padding, otherwise after packing and padding,
                # it will go beyond max seq len, then it will need a truncation.
                packed_sequence_size=int(decoder_seq_length * 0.9),
                pixel_shuffle_ratio=0.5,
                num_image_embeddings_per_tile=num_image_embeddings_per_tile,
                image_tag_type="internvl"
            ),
            packing_buffer_size=200 if args.use_packed_sequence else None,
            image_decode="pil",
        )

    elif args.data_type == "mock":

        data = vlm.NevaMockDataModule(
            seq_length=decoder_seq_length,
            global_batch_size=gbs,
            micro_batch_size=mbs,
            tokenizer=tokenizer,
            image_processor=image_processor,
            num_workers=4,
            packed_sequence=args.use_packed_sequence,
            pixel_shuffle_ratio=0.5,
            num_image_embeddings_per_tile=num_image_embeddings_per_tile,
            num_tiles_per_image=5,
        )
    else:
        raise ValueError(f"Data type {args.data_type} not supported")

    from megatron.core.distributed import DistributedDataParallelConfig

    # Training strategy setup
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        encoder_pipeline_model_parallel_size=args.encoder_pp_size,
        context_parallel_size=args.cp_size,
        pipeline_dtype=torch.bfloat16,
        sequence_parallel=True,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            average_in_collective=True,
        ),
        ckpt_load_strictness="log_all",
    )

    model = vlm.NevaModel(neva_config, tokenizer=data.tokenizer)

    # Checkpoint callback setup
    checkpoint_callback = nl.ModelCheckpoint(
        save_last=True,
        monitor="reduced_train_loss",
        save_top_k=2,
        every_n_train_steps=1000,
        dirpath=args.log_dir,
    )

    # Trainer setup
    trainer = nl.Trainer(
        num_nodes=args.num_nodes,
        devices=args.devices,
        max_steps=max_steps,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        callbacks=[
            checkpoint_callback,
            TimingCallback(),
            MegatronCommOverlapCallback(
                tp_comm_overlap=False,
                overlap_grad_reduce=False,
                overlap_param_gather=False,
            ),
        ],
        val_check_interval=500,
        limit_val_batches=gbs,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )

    # Logger setup
    nemo_logger = nl.NeMoLogger(
        log_dir=args.log_dir,
        name=args.name,
        wandb=WandbLogger(project=args.wandb_project, name=args.name) if args.wandb_project is not None else None,
    )

    # Auto resume setup
    resume = nl.AutoResume(
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
        resume_from_directory=args.log_dir,
        restore_config=nl.RestoreConfig(path=args.restore_path) if args.restore_path is not None else None,
    )

    # Optimizer and scheduler setup
    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=args.lr,
        adam_beta1=0.9,
        adam_beta2=0.95,
        use_distributed_optimizer=True,
        bf16=True,
    )
    sched = CosineAnnealingScheduler(
        max_steps=trainer.max_steps,
        warmup_steps=150,
        constant_steps=0,
        min_lr=1.0e-07,
    )
    opt = MegatronOptimizerModule(opt_config, sched)

    # PEFT setup
    if args.peft == 'lora':
        peft = vlm.peft.LoRA(
            target_modules=[
                "*.language_model.*.linear_qkv",
                "*.language_model.*.linear_proj",
                "*.language_model.*.linear_fc1",
                "*.language_model.*.linear_fc2",
            ],
            freeze_language_model=True,
            freeze_vision_model=False,
            freeze_vision_projection=False,
            dim=32,
        )
    else:
        peft = None

    llm.finetune(
        model=model,
        data=data,
        trainer=trainer,
        peft=peft,
        log=nemo_logger,
        optim=opt,
        resume=resume,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEVA Model Training Script")

    # Argument parsing
    parser.add_argument("--data_type", type=str, required=False, default="mock", help="mock | llava")
    parser.add_argument("--data_path", type=str, required=False, default=None, help="Path to the dataset JSON file")
    parser.add_argument("--image_folder", type=str, required=False, default=None, help="Path to the image folder")
    parser.add_argument(
        "--log_dir", type=str, required=False, default="/results", help="Directory for logging and checkpoints"
    )
    parser.add_argument(
        "--language_model_path", type=str, required=False, default=None, help="Path to the pretrained language model"
    )
    parser.add_argument(
        "--restore_path", type=str, required=False, default=None, help="Path to restore model from checkpoint"
    )
    parser.add_argument("--devices", type=int, required=False, default=1)
    parser.add_argument("--num_nodes", type=int, required=False, default=1)
    parser.add_argument("--max_steps", type=int, required=False, default=5190)
    parser.add_argument("--tp_size", type=int, required=False, default=1)
    parser.add_argument("--pp_size", type=int, required=False, default=1)
    parser.add_argument("--cp_size", type=int, required=False, default=1)
    parser.add_argument("--encoder_pp_size", type=int, required=False, default=0)
    parser.add_argument("--projector_type", type=str, required=False, default="mcore_mlp")
    parser.add_argument("--name", type=str, required=False, default="neva_pretrain")
    parser.add_argument("--peft", type=str, default='none', help="none | lora")
    parser.add_argument("--wandb_project", type=str, required=False, default=None)
    parser.add_argument("--gbs", type=int, required=False, default=128, help="Global batch size")
    parser.add_argument("--mbs", type=int, required=False, default=1, help="Micro batch size")
    parser.add_argument("--lr", type=float, required=False, default=2.0e-06, help="Learning rate")
    parser.add_argument(
        "--use_packed_sequence",
        action="store_true",
    )
    args = parser.parse_args()
    main(args)

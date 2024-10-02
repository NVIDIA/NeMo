# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import torch
from megatron.core.optimizer import OptimizerConfig
from transformers import AutoProcessor
from nemo import lightning as nl
from nemo.collections.multimodal.data.energon import SimpleMultiModalDataModule
from nemo.collections.multimodal.data.energon.config import MultiModalSampleConfig
from nemo.collections.vlm.llama.data.task_encoder import LlamaTaskEncoder
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.utils.exp_manager import TimingCallback


def get_data_module(data_path, micro_batch_size, global_batch_size):
    """
    Initializes and returns a data module configured for multimodal training.

    This function sets up the data paths, tokenizers, image processors,
    and other configurations necessary for multimodal training.

    Args:
        data_path (str): Path to the dataset.
        micro_batch_size (int): Micro batch size.
        global_batch_size (int): Global batch size.

    Returns:
        tuple: Contains the data module and tokenizer.
    """
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    image_processor = processor.image_processor
    tokenizer = processor.tokenizer

    multimodal_sample_config = MultiModalSampleConfig()
    multimodal_sample_config.conversation_template_config.system = ""
    multimodal_sample_config.conversation_template_config.chat_template = None
    multimodal_sample_config.image_token.token_id = 128256
    multimodal_sample_config.conversation_template_config.stop_string = '<|eot_id|>'

    task_encoder = LlamaTaskEncoder(
        tokenizer=tokenizer, image_processor=image_processor, multimodal_sample_config=multimodal_sample_config
    )
    data_module = SimpleMultiModalDataModule(
        path=data_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_workers=8,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        multimodal_sample_config=multimodal_sample_config,
        task_encoder=task_encoder,
    )
    return data_module, tokenizer


def main(args):
    """
    Main function for setting up and training the MLLama model.

    This function prepares the data module, model, training strategy,
    logger, checkpointing, and optimizer configuration. It then starts
    the training loop using PyTorch Lightning's trainer.

    Args:
        args (argparse.Namespace): The command-line arguments passed to the script.
    """
    gbs = 128
    mbs = 2
    seq_length = 512
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    processor = AutoProcessor.from_pretrained(model_id)
    image_processor = processor.image_processor
    tokenizer = processor.tokenizer

    from nemo.collections.vlm.llama.data.mock import MockDataModule
    data = MockDataModule(
        seq_length=seq_length,
        global_batch_size=gbs,
        micro_batch_size=mbs,
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_workers=0,
    )

    from nemo.collections.vlm.llama.model.base import MLlamaModel, CrossAttentionVisionConfig, \
        MLlamaModelConfig, CrossAttentionTextConfig

    # Model configuration
    vision_config = CrossAttentionVisionConfig(
        num_layers=32, hidden_size=1280, num_attention_heads=16, vision_chunk_size=560, vision_max_num_chunks=4,
    )
    text_config = CrossAttentionTextConfig(
        num_layers=32,
    )

    llama_config = MLlamaModelConfig(
        language_model_config=text_config,
        vision_model_config=vision_config,
    )
    model = MLlamaModel(llama_config, tokenizer=tokenizer)

    # Training strategy setup
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        encoder_pipeline_model_parallel_size=args.encoder_pp_size,
        pipeline_dtype=torch.bfloat16,
    )

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
        devices=args.devices,
        max_steps=5190,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        callbacks=[checkpoint_callback, TimingCallback()],
        val_check_interval=1000,
        limit_val_batches=gbs,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )

    # Logger setup
    from pytorch_lightning.loggers import WandbLogger

    nemo_logger = nl.NeMoLogger(
        dir=args.log_dir,
        name=args.name,
        wandb=WandbLogger(project=args.wandb_project, name=args.name) if args.wandb_project is not None else None,
    )
    nemo_logger.setup(trainer, resume_if_exists=True)

    # Auto resume setup
    from nemo.lightning.pytorch.strategies.utils import RestoreConfig
    resume = nl.AutoResume(
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
        resume_from_directory=args.log_dir,
        restore_config=RestoreConfig(path=args.restore_path) if args.restore_path is not None else None,
    )
    resume.setup(trainer, model)

    # Optimizer and scheduler setup
    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=2.0e-05,
        adam_beta1=0.9,
        adam_beta2=0.95,
        use_distributed_optimizer=False,
        bf16=True,
    )
    sched = CosineAnnealingScheduler(
        max_steps=trainer.max_steps,
        warmup_steps=150,
        constant_steps=0,
        min_lr=2.0e-07,
    )
    opt = MegatronOptimizerModule(opt_config, sched)
    opt.connect(model)

    # Start training
    trainer.fit(model, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mllama Model Training Script")

    parser.add_argument("--restore_path", type=str, required=False, default=None,
                        help="Path to restore model from checkpoint")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--log_dir", type=str, required=False, default="./nemo_experiments",
                        help="Directory for logging and checkpoints")
    parser.add_argument("--language_model_path", type=str, required=False, default=None,
                        help="Path to the pretrained language model")
    parser.add_argument("--devices", type=int, required=False, default=1)
    parser.add_argument("--tp_size", type=int, required=False, default=1)
    parser.add_argument("--pp_size", type=int, required=False, default=1)
    parser.add_argument("--encoder_pp_size", type=int, required=False, default=0)
    parser.add_argument("--name", type=str, required=False, default="neva_pretrain")
    parser.add_argument("--wandb_project", type=str, required=False, default=None)

    args = parser.parse_args()
    main(args)

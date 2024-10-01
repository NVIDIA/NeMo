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

from nemo import lightning as nl
from nemo.collections import llm, vlm
from nemo.collections.vlm.llama.data.mock import MockDataModule
from nemo.collections.vlm.neva.model.base import MultimodalProjectorConfig
from nemo.collections.vlm import ImageDataConfig
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.utils.exp_manager import TimingCallback
from nemo.collections.multimodal.data.energon import SimpleMultiModalDataModule
from nemo.collections.multimodal.data.energon.config import MultiModalSampleConfig
from nemo.collections.vlm.llama.data.task_encoder import LlamaTaskEncoder

from transformers import MllamaForConditionalGeneration, AutoProcessor
from transformers.models.mllama.image_processing_mllama import get_all_supported_aspect_ratios
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS


def get_aspect_ratio(aspect_ratio_ids):
    max_image_tiles = 4
    mapping = get_all_supported_aspect_ratios(max_image_tiles)
    aspect_ratio = [mapping[i.item() - 1] for i in aspect_ratio_ids]
    return torch.tensor(aspect_ratio)


def get_data_module():
    data_path = '/lustre/fsw/coreai_dlalgo_genai/datasets/energon_datasets/LLaVA-Instruct-150K/'
    # model_directory = "/lustre/fsw/coreai_dlalgo_llm/aot/checkpoints/evian3/evian3-11b-vision-instruct-final-hf_vv1/"
    # model_id = "evian3-11b-vision-instruct-final-hf_vv1"
    model_directory = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    processor = AutoProcessor.from_pretrained(model_directory)
    image_processor = processor.image_processor
    image_processor.size = {'height': 448, 'width': 448}

    tokenizer = processor.tokenizer

    multimodal_sample_config = MultiModalSampleConfig()
    multimodal_sample_config.conversation_template_config.system = "You are a helpful assistant"
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
        num_workers=0,
        micro_batch_size=2,
        global_batch_size=8,
        multimodal_sample_config=multimodal_sample_config,
        task_encoder=task_encoder,
    )
    return data_module


def main(args):
    # Global and micro batch sizes
    gbs = 8
    mbs = 2
    seq_length = 512

    # Data configuration
    # data_config = ImageDataConfig(
    #     image_folder=args.image_folder,
    #     conv_template="plain",
    # )

    # Data module setup
    # data = vlm.NevaLazyDataModule(
    #     paths=args.data_path,
    #     data_config=data_config,
    #     seq_length=seq_length,
    #     global_batch_size=gbs,
    #     micro_batch_size=mbs,
    #     tokenizer=None,
    #     image_processor=None,
    #     num_workers=8,
    # )
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")
    # data = MockDataModule(
    #     seq_length=seq_length,
    #     global_batch_size=gbs,
    #     micro_batch_size=mbs,
    #     tokenizer=tokenizer,
    #     image_processor=None,
    #     num_workers=0,
    # )
    data = get_data_module()

    from nemo.collections.vlm.llama.model.base import MLlamaModel, CrossAttentionVisionModelConfig, \
        MLlamaModelConfig, CrossAttentionTextModelConfig, CrossAttentionTextModelConfig8B

    vision_config = CrossAttentionVisionModelConfig(
        num_layers=32, hidden_size=1280, num_attention_heads=16, vision_chunk_size=448, vision_max_num_chunks=4,
    )
    text_config = CrossAttentionTextModelConfig8B(
        num_layers=2,
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
        max_steps=2170,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        callbacks=[checkpoint_callback, TimingCallback()],
        val_check_interval=100,
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
    nemo_logger.setup(
        trainer,
        resume_if_exists=True,
    )

    # Auto resume setup
    resume = nl.AutoResume(
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
        resume_from_directory=args.log_dir,
    )
    resume.setup(trainer, model)

    # Optimizer and scheduler setup
    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=0.001,
        adam_beta1=0.9,
        adam_beta2=0.95,
        use_distributed_optimizer=False,
        bf16=True,
    )
    sched = CosineAnnealingScheduler(
        max_steps=trainer.max_steps,
        warmup_steps=70,
        constant_steps=0,
        min_lr=2.0e-05,
    )
    opt = MegatronOptimizerModule(opt_config, sched)
    opt.connect(model)

    # Start training
    trainer.fit(model, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEVA Model Training Script")

    # Argument parsing
    parser.add_argument("--data_path", type=str, required=False, help="Path to the dataset JSON file")
    parser.add_argument("--image_folder", type=str, required=False, help="Path to the image folder")
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

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
from megatron.core.optimizer import OptimizerConfig
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoProcessor

from nemo import lightning as nl
from nemo.collections import llm, vlm
from nemo.collections.multimodal.data import SimpleMultiModalDataModule
from nemo.collections.multimodal.data.energon import MultiModalSampleConfig
from nemo.collections.vlm.llama.data.task_encoder import LlamaTaskEncoder


def get_args():
    parser = argparse.ArgumentParser(description='Finetune a small GPT model using NeMo 2.0')
    parser.add_argument('--restore_path', type=str, help="Path to model to be finetuned", default="meta-llama/Llama-3.2-11B-Vision")
    parser.add_argument('--experiment_dir', type=str, help="directory to write results and checkpoints to")
    parser.add_argument('--peft', type=str, default='none', help="none | lora")
    parser.add_argument('--devices', type=int, default=1, help="number of devices")
    parser.add_argument('--max_steps', type=int, default=4, help="number of steps")
    parser.add_argument('--mbs', type=int, default=1, help="micro batch size")
    parser.add_argument('--tp_size', type=int, default=1, help="tensor parallel size")
    parser.add_argument('--pp_size', type=int, default=1, help="pipeline parallel size")

    return parser.parse_args()


def get_data_module(args, vision_chunk_size) -> pl.LightningDataModule:
    data_path = '/lustre/fsw/coreai_dlalgo_genai/datasets/energon_datasets/LLaVA-Pretrain-LCS-558K'
    model_directory = "/lustre/fsw/coreai_dlalgo_llm/aot/checkpoints/evian3/evian3-11b-vision-instruct-final-hf_vv1/"
    # model_id = "evian3-11b-vision-instruct-final-hf_vv1"
    processor = AutoProcessor.from_pretrained(model_directory)
    image_processor = processor.image_processor
    image_processor.size = {'height': vision_chunk_size, 'width': vision_chunk_size}

    tokenizer = processor.tokenizer

    multimodal_sample_config = MultiModalSampleConfig()
    multimodal_sample_config.conversation_template_config.system = "You are a helpful assistant"
    multimodal_sample_config.conversation_template_config.chat_template = None
    multimodal_sample_config.image_token.token_id = 128256
    multimodal_sample_config.conversation_template_config.stop_string = None

    task_encoder = LlamaTaskEncoder(
        tokenizer=tokenizer, image_processor=image_processor, multimodal_sample_config=multimodal_sample_config
    )
    data_module = SimpleMultiModalDataModule(
        path=data_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_workers=0,
        micro_batch_size=args.mbs,
        global_batch_size=16,
        multimodal_sample_config=multimodal_sample_config,
        task_encoder=task_encoder,
    )

    return data_module


if __name__ == '__main__':
    args = get_args()

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        pipeline_parallel_size=args.pp_size,
    )

    trainer = nl.Trainer(
        devices=args.devices,
        max_steps=args.max_steps,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        log_every_n_steps=1,
        limit_val_batches=2,
        val_check_interval=10,
        num_sanity_val_steps=0,
    )

    ckpt = nl.ModelCheckpoint(
        save_last=True,
        monitor="reduced_train_loss",
        save_top_k=1,
        save_on_train_epoch_end=True,
        save_optim_on_train_end=True,
    )

    logger = nl.NeMoLogger(
        dir=args.experiment_dir,
        use_datetime_version=False,  # must be false if using auto resume
        ckpt=ckpt,
    )

    adam = nl.MegatronOptimizerModule(
        config=OptimizerConfig(
            optimizer="adam",
            lr=0.0001,
            adam_beta2=0.98,
            use_distributed_optimizer=True,
            clip_grad=1.0,
            bf16=True,
        ),
    )

    if args.peft == 'lora':
        peft = vlm.peft.LoRA(
            target_modules=[
                "*.language_model.*.linear_qkv",
                "*.language_model.*.linear_q",
                "*.language_model.*.linear_kv",
                "*.language_model.*.linear_proj",
                "*.language_model.*.linear_fc1",
                "*.language_model.*.linear_fc2",
            ]
        )
    else:
        peft = None

    vision_chunk_size = 448 if args.restore_path == "meta-llama/Llama-3.2-11B-Vision" else 560
    mm_datamodule = get_data_module(args, vision_chunk_size)

    tokenizer = AutoTokenizer.from_pretrained(args.restore_path)
    model = vlm.MLlamaModel(
        vlm.MLlamaModelConfig(
            language_model_config=vlm.CrossAttentionTextModelConfig8B(rotary_interleaved=False, apply_rope_fusion=False),
            vision_model_config=vlm.CrossAttentionVisionModelConfig(num_layers=32, hidden_size=1280,
                                                                    num_attention_heads=16, vision_chunk_size=vision_chunk_size,
                                                                    vision_max_num_chunks=4, ),
        ),
        tokenizer=tokenizer)
    resume = nl.AutoResume(
        restore_config=nl.RestoreConfig(path=f"hf://{args.restore_path}"),
        resume_if_exists=False,
    )

    llm.finetune(
        model=model,
        data=mm_datamodule,
        trainer=trainer,
        peft=peft,
        log=logger,
        optim=adam,
        resume=resume,
    )
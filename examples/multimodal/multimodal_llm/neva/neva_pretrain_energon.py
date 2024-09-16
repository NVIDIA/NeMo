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

from transformers import AutoProcessor

from nemo import lightning as nl
from nemo.collections import llm, vlm
from nemo.collections.multimodal.data.energon import SimpleMultiModalDataModule

if __name__ == "__main__":
    gbs = 4
    mbs = 2
    seq_length = 256
    data_path = '/home/ykarnati/Downloads/LLaVA-Pretrain/wds'  # VQA
    # data_path ='/home/ykarnati/Downloads/datasets/energon_datasets/mmc4/stage2/core_faces' # similarity interleaved
    # data_path = '/home/ykarnati/Downloads/datasets/energon_datasets/obelics/stage4/no-partial/' # interleaved
    # data module
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    tokenizer = processor.tokenizer
    image_processor = processor.image_processor
    data = SimpleMultiModalDataModule(
        path=data_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_workers=0,
        micro_batch_size=mbs,
        global_batch_size=gbs,
    )
    #     model

    language_transformer_config = llm.Llama3Config8B(num_layers=2)
    language_transformer_config.apply_query_key_layer_scaling = False
    vision_transformer_config = vlm.CLIPViTConfig(num_layers=2, hidden_size=64, num_attention_heads=4)
    vision_projection_config = vlm.MultimodalProjectorConfig(
        input_size=64, num_layers=2, hidden_size=4096, num_attention_heads=4
    )

    neva_config = vlm.NevaConfig(
        language_transformer_config=language_transformer_config,
        vision_transformer_config=vision_transformer_config,
        vision_projection_config=vision_projection_config,
        num_layers=1,
        num_attention_heads=4,
    )

    model = vlm.NevaModel(neva_config, tokenizer=data.tokenizer)

    log_dir = './test_logs_neva_train'

    #   strategy

    strategy = nl.MegatronStrategy(tensor_model_parallel_size=1)

    checkpoint_callback = nl.ModelCheckpoint(
        save_best_model=True,
        save_last=True,
        monitor="reduced_train_loss",
        save_top_k=2,
        every_n_train_steps=10,
        enable_nemo_ckpt_io=False,
        dirpath=log_dir,
    )

    #     Trainer
    trainer = nl.Trainer(
        devices=1,  ## you can change the numebr of devices to suit your setup
        max_steps=50,
        accelerator="gpu",
        strategy=strategy,
        # plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        callbacks=[checkpoint_callback],
    )

    nemo_logger = nl.NeMoLogger(dir=log_dir, name='energon_neva_test_v2')

    nemo_logger.setup(
        trainer,
        resume_if_exists=True,
    )

    resume = nl.AutoResume(resume_if_exists=True, resume_ignore_no_checkpoint=True, dirpath=log_dir)
    resume.setup(trainer, model)
    trainer.fit(model, data)

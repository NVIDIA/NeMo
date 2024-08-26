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

from nemo import lightning as nl
from nemo.collections import llm, vlm
from transformers import AutoProcessor

if __name__ == "__main__":
    gbs = 4
    mbs = 2
    seq_length = 256

    # data module
    data = vlm.MockDataModule(
        seq_length=seq_length,
        global_batch_size=gbs,
        micro_batch_size=mbs,
        tokenizer=None,
        image_processor=None,
        num_workers=0,
    )

    #     model

    language_transformer_config = llm.Llama2Config7B()
    from nemo.collections.vlm.neva.model.base import HFCLIPVisionConfig
    vision_transformer_config = HFCLIPVisionConfig(pretrained_model_name_or_path="openai/clip-vit-large-patch14-336")
    vision_projection_config = vlm.MultimodalProjectorConfig(input_size=1024, hidden_size=4096)

    neva_config = vlm.NevaConfig(
        language_transformer_config=language_transformer_config,
        vision_transformer_config=vision_transformer_config,
        vision_projection_config=vision_projection_config,
    )

    model = vlm.NevaModel(neva_config, tokenizer=data.tokenizer)

    log_dir = './test_logs_v4'

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
        max_steps=100,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
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

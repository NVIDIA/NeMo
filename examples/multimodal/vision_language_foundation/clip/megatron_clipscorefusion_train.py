# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import clip
from omegaconf.omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler

from nemo.collections.multimodal.data.clip.clip_dataset import get_preprocess_fns
from nemo.collections.multimodal.data.clip.mbeir_dataset import (
    MBEIRCandidatePoolCollator,
    MBEIRCandidatePoolDataset,
    MBEIRMainCollator,
    MBEIRMainDataset,
    Mode,
)
from nemo.collections.multimodal.models.vision_language_foundation.clip.megatron_clip_models import MegatronCLIPModel
from nemo.collections.multimodal.models.vision_language_foundation.clip.megatron_clip_scorefusion_models import (
    MegatronCLIPScoreFusionModel,
)
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


def get_tokenizer(tokenizer):
    def tokenizer_wrapper(txt):
        txt_tensor = tokenizer(txt, context_length=77, truncate=True)
        return txt_tensor

    return tokenizer_wrapper


@hydra_runner(config_path="conf", config_name="megatron_clipscorefusion_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    assert (
        cfg.trainer.devices * cfg.trainer.num_nodes
    ) * cfg.model.micro_batch_size == cfg.model.global_batch_size, (
        "Gradient accumulation is not supported in CLIP yet."
    )

    trainer = MegatronTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)

    model = MegatronCLIPScoreFusionModel(cfg.model, trainer)
    model.train()
    val_image_transform, text_transform = get_preprocess_fns(model.cfg, model.tokenizer, is_train=True,)

    # data loaders
    train_dataset = MBEIRMainDataset(
        mbeir_data_dir=cfg.data_config.mbeir_data_dir,
        query_data_path=cfg.data_config.train_query_data_path,
        cand_pool_path=cfg.data_config.train_cand_pool_path,
        query_instruct_path=cfg.data_config.query_instruct_path,
        img_preprocess_fn=val_image_transform,
        mode=Mode.TRAIN,
        enable_query_instruct=cfg.data_config.enable_query_instruct,
        shuffle_cand=cfg.data_config.shuffle_cand,
        hard_neg_num=0,  # TODO
        returns=cfg.data_config.returns,
    )

    clip_model, img_preprocess_fn = clip.load("ViT-B/32", "cuda", jit=False, download_root=None)
    clip_tokenizer = clip.tokenize

    train_collector = MBEIRMainCollator(
        tokenizer=get_tokenizer(clip_tokenizer),
        image_size=tuple(map(int, cfg.data_config.image_size.split(','))),
        mode=Mode.TRAIN,
    )
    train_sampler = DistributedSampler(dataset=train_dataset, num_replicas=1, rank=0, shuffle=True,)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.dataloader_config.train_batch_size,
        num_workers=cfg.dataloader_config.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        shuffle=False,  # Note: since we use sampler, shuffle should be False
        collate_fn=train_collector,
        drop_last=True,
    )

    trainer.fit(model, train_dataloader)


if __name__ == '__main__':
    main()

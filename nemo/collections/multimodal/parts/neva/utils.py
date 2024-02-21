# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import tempfile
from typing import Any, Callable, Tuple

import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from transformers import CLIPImageProcessor

from nemo.collections.multimodal.models.multimodal_llm.neva.neva_model import MegatronNevaModel
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.collections.nlp.parts.peft_config import PEFT_CONFIG_MAP
from nemo.utils import AppState
from nemo.utils.model_utils import inject_model_parallel_rank


def create_neva_model_and_processor(cfg):

    plugins = []
    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())
    # trainer required for restoring model parallel models
    trainer = Trainer(plugins=plugins, strategy=NLPDDPStrategy(), **cfg.trainer)

    if (
        cfg.tensor_model_parallel_size < 0
        or cfg.pipeline_model_parallel_size < 0
        or cfg.get('pipeline_model_parallel_split_rank', -1) < 0
    ):
        model_config = MegatronNevaModel.restore_from(
            restore_path=cfg.neva_model_file, trainer=trainer, return_config=True,
        )

        with open_dict(cfg):
            cfg.tensor_model_parallel_size = model_config.get('tensor_model_parallel_size', 1)
            cfg.pipeline_model_parallel_size = model_config.get('pipeline_model_parallel_size', 1)
            cfg.pipeline_model_parallel_split_rank = model_config.get('pipeline_model_parallel_split_rank', 0)

    assert (
        cfg.trainer.devices * cfg.trainer.num_nodes
        == cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
    ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"

    if cfg.neva_model_file:
        save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(cfg.neva_model_file):
            save_restore_connector.model_extracted_dir = cfg.neva_model_file

        neva_cfg = MegatronNevaModel.restore_from(
            restore_path=cfg.neva_model_file,
            trainer=trainer,
            return_config=True,
            save_restore_connector=save_restore_connector,
        )
        OmegaConf.set_struct(neva_cfg, True)
        with open_dict(neva_cfg):
            neva_cfg.sequence_parallel = False
            neva_cfg.activations_checkpoint_granularity = None
            neva_cfg.activations_checkpoint_method = None
            neva_cfg.precision = trainer.precision
            neva_cfg.mm_cfg.llm.from_pretrained = cfg.get('base_model_file', None)
            neva_cfg.apply_rope_fusion = False
        #    neva_cfg.mm_cfg.vision_encoder.from_pretrained = None

        model = MegatronNevaModel.restore_from(
            restore_path=cfg.neva_model_file,
            trainer=trainer,
            override_config_path=neva_cfg,
            save_restore_connector=save_restore_connector,
        )
        if neva_cfg.get('peft') is not None:
            peft_cfg_cls = PEFT_CONFIG_MAP[neva_cfg.peft.peft_scheme]
            if peft_cfg_cls is not None:
                model.load_adapters(cfg.neva_model_file, peft_cfg_cls(neva_cfg))

    elif cfg.checkpoint_dir:
        app_state = AppState()
        if cfg.tensor_model_parallel_size > 1 or cfg.pipeline_model_parallel_size > 1:
            app_state.model_parallel_size = cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
            app_state.tensor_model_parallel_size = cfg.tensor_model_parallel_size
            app_state.pipeline_model_parallel_size = cfg.pipeline_model_parallel_size
            (
                app_state.tensor_model_parallel_rank,
                app_state.pipeline_model_parallel_rank,
                app_state.model_parallel_size,
                app_state.data_parallel_size,
                app_state.pipeline_model_parallel_split_rank,
                app_state.virtual_pipeline_model_parallel_rank,
            ) = fake_initialize_model_parallel(
                world_size=app_state.model_parallel_size,
                rank=trainer.global_rank,
                tensor_model_parallel_size_=cfg.tensor_model_parallel_size,
                pipeline_model_parallel_size_=cfg.pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank_=cfg.pipeline_model_parallel_split_rank,
            )
        checkpoint_path = inject_model_parallel_rank(os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name))
        # TODO: This wont work properly (We need to set model.llm.from_pretrained model.vision.from_pretrained to nul)
        model = MegatronNevaModel.load_from_checkpoint(checkpoint_path, hparams_file=cfg.hparams_file, trainer=trainer)
    else:
        raise ValueError("need at least a nemo file or checkpoint dir")

    model.freeze()

    # Have to turn off activations_checkpoint_method for inference
    try:
        model.model.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass
    try:
        model.model.module.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass

    def image_processor(maybe_image_path):
        if isinstance(maybe_image_path, str):
            image = Image.open(maybe_image_path).convert('RGB')
        else:
            image = maybe_image_path

        if neva_cfg.mm_cfg.vision_encoder.from_hf:
            processor = CLIPImageProcessor.from_pretrained(
                neva_cfg.mm_cfg.vision_encoder.from_pretrained, torch_dtype=torch.bfloat16
            )
        else:
            processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.bfloat16)

        if neva_cfg.data.image_aspect_ratio == 'keep':
            max_hw, min_hw = max(image.size), min(image.size)
            aspect_ratio = max_hw / min_hw
            max_len, min_len = 448, 224
            shortest_edge = int(min(max_len / aspect_ratio, min_len))
            image = processor.preprocess(
                image, return_tensors='pt', do_center_crop=False, size={"shortest_edge": shortest_edge}
            )['pixel_values'][0]
        elif neva_cfg.data.image_aspect_ratio == 'pad':

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        else:
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        if neva_cfg.precision in [16, '16', '16-mixed']:
            media = image.type(torch.float16)
        elif neva_cfg.precision in [32, '32', '32-true']:
            media = image.type(torch.float32)
        else:
            media = image.type(torch.bfloat16)

        return media.unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0)

    return model, image_processor

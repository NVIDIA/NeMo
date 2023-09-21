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

import os
from typing import Dict, List, Optional

import torch
from omegaconf.omegaconf import OmegaConf, open_dict
from PIL import Image

from nemo.collections.multimodal.data.clip.clip_dataset import get_preprocess_fns
from nemo.collections.multimodal.models.clip.megatron_clip_models import MegatronCLIPModel
from nemo.collections.multimodal.parts.utils import setup_trainer_and_model_for_inference
from nemo.core.classes.exportable import Exportable
from nemo.core.config import hydra_runner
from nemo.core.neural_types import ChannelType, NeuralType
from nemo.utils import logging
from nemo.utils.trt_utils import build_engine


class CLIPWrapper(torch.nn.Module, Exportable):
    def __init__(self, vision_encoder, text_encoder, text_transform):
        super(CLIPWrapper, self).__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.text_transform = text_transform

    def forward(self, image, texts):
        image_features = self.vision_encoder(image)
        text_features = self.text_encoder(texts)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        return text_probs

    # For onnx export
    def input_example(self, max_batch=8, max_dim=224, max_text=64):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        sample = next(self.parameters())
        images = torch.randn(max_batch, 3, max_dim, max_dim, device=sample.device)
        texts = self.text_transform(["a girl"] * max_text).to(sample.device)
        return (images, texts)

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "images": NeuralType(('B', 'C', 'H', 'W'), ChannelType()),
            "texts": NeuralType(('H', 'D'), ChannelType()),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"text_probs": NeuralType(('B', 'H'), ChannelType())}

    @property
    def input_names(self) -> List[str]:
        return ['images', 'texts']

    @property
    def output_names(self) -> List[str]:
        return ['text_probs']


@hydra_runner(config_path="conf", config_name="megatron_clip_export")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    output_dir = cfg.infer.out_path
    max_batch_size = cfg.infer.max_batch_size
    max_dim = cfg.infer.max_dim
    max_text = cfg.infer.max_text
    trt_precision = cfg.trainer.precision
    cfg.trainer.precision = 32

    # These configs are required to be off during inference.
    def model_cfg_modifier(model_cfg):
        model_cfg.precision = cfg.trainer.precision
        model_cfg.vision.precision = cfg.trainer.precision
        model_cfg.text.precision = cfg.trainer.precision
        if cfg.trainer.precision != "bf16":
            model_cfg.megatron_amp_O2 = False
        model_cfg.sequence_parallel = False
        model_cfg.activations_checkpoint_granularity = None
        model_cfg.activations_checkpoint_method = None

    trainer, model = setup_trainer_and_model_for_inference(
        model_provider=MegatronCLIPModel, cfg=cfg, model_cfg_modifier=model_cfg_modifier,
    )

    if model.cfg.get("megatron_amp_O2", False):
        vision_encoder = model.model.module.vision_encoder
        text_encoder = model.model.module.text_encoder
    else:
        vision_encoder = model.model.vision_encoder
        text_encoder = model.model.text_encoder

    val_image_transform, text_transform = get_preprocess_fns(model.cfg, model.tokenizer, is_train=False,)

    os.makedirs(f"{output_dir}/onnx/", exist_ok=True)
    os.makedirs(f"{output_dir}/plan/", exist_ok=True)

    clip_model = CLIPWrapper(vision_encoder, text_encoder, text_transform)
    dynamic_axes = {'images': {0: 'B'}, 'texts_input': {0, 'H'}}
    clip_model.export(f"{output_dir}/onnx/clip.onnx", dynamic_axes=None)

    input_profile = {}
    bs1_example = clip_model.input_example(max_batch=1, max_dim=max_dim, max_text=1)
    bsmax_example = clip_model.input_example(max_batch=max_batch_size, max_dim=max_dim, max_text=max_text)
    input_profile['images'] = [
        tuple(bs1_example[0].shape),
        tuple(bsmax_example[0].shape),
        tuple(bsmax_example[0].shape),
    ]
    input_profile['texts'] = [
        tuple(bs1_example[1].shape),
        tuple(bsmax_example[1].shape),
        tuple(bsmax_example[1].shape),
    ]
    build_engine(
        f"{output_dir}/onnx/clip.onnx",
        f"{output_dir}/plan/clip.plan",
        fp16=(trt_precision in [16, '16', '16-mixed']),
        input_profile=input_profile,
        timing_cache=None,
        workspace_size=0,
    )


if __name__ == '__main__':
    main()

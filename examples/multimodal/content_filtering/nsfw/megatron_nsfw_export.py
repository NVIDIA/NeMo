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
from omegaconf.omegaconf import OmegaConf

from nemo.collections.multimodal.models.content_filter.megatron_nsfw_clip_models import MegatronContentFilteringModel
from nemo.collections.multimodal.parts.utils import setup_trainer_and_model_for_inference
from nemo.core.config import hydra_runner
from nemo.core.neural_types import ChannelType, NeuralType
from nemo.utils import logging
from nemo.utils.trt_utils import build_engine


class ContentFilteringWrapper(MegatronContentFilteringModel):
    def __init__(self, cfg, trainer):
        super(ContentFilteringWrapper, self).__init__(cfg, trainer)

    def forward(self, image: torch.Tensor):
        return super().forward(image, mlp_factor=1.0, emb_factor=1.0).sigmoid()

    def input_example(self, max_batch: int = 64, max_dim: int = 224):
        device = next(self.parameters()).device
        return (torch.randn(max_batch, 3, max_dim, max_dim, device=device),)

    @property
    def input_names(self) -> List[str]:
        return ["images"]

    @property
    def output_names(self) -> List[str]:
        return ["nsfw_probs"]

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"images": NeuralType(("B", "C", "H", "W"), ChannelType())}

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"nsfw_probs": NeuralType(("B",), ChannelType())}


def set_envvar():
    os.environ["LOCAL_RANK"] = os.environ.get("LOCAL_RANK", "0")
    os.environ["RANK"] = os.environ.get("RANK", "0")
    os.environ["LOCAL_SIZE"] = os.environ.get("LOCAL_SIZE", "1")
    os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", "1")


@hydra_runner(config_path="conf", config_name="megatron_nsfw_export")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    set_envvar()

    output_dir = cfg.infer.out_path
    max_batch_size = cfg.infer.max_batch_size
    trt_precision = cfg.trainer.precision
    cfg.trainer.precision = 32

    # These configs are required to be off during inference.
    def model_cfg_modifier(model_cfg):
        model_cfg.precision = cfg.trainer.precision
        model_cfg.vision.precision = cfg.trainer.precision
        if cfg.trainer.precision != "bf16":
            model_cfg.megatron_amp_O2 = False
        model_cfg.sequence_parallel = False
        model_cfg.activations_checkpoint_granularity = None
        model_cfg.activations_checkpoint_method = None

    trainer, model = setup_trainer_and_model_for_inference(
        model_provider=ContentFilteringWrapper, cfg=cfg, model_cfg_modifier=model_cfg_modifier
    )

    bs1_example = model.input_example(max_batch=1, max_dim=cfg.infer.max_dim)
    bsmax_example = model.input_example(max_batch=max_batch_size, max_dim=cfg.infer.max_dim)

    os.makedirs(f"{output_dir}/onnx", exist_ok=True)
    model.export(f"{output_dir}/onnx/nsfw.onnx", dynamic_axes={"images": {0: "B"}}, input_example=bsmax_example)

    input_profile = {
        "images": [tuple(bs1_example[0].shape), tuple(bsmax_example[0].shape), tuple(bsmax_example[0].shape),]
    }

    build_engine(
        f"{output_dir}/onnx/nsfw.onnx",
        f"{output_dir}/plan/nsfw.plan",
        fp16=(trt_precision == 16),
        input_profile=input_profile,
        timing_cache=None,
        workspace_size=0,
    )


if __name__ == '__main__':
    main()

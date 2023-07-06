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
import gc
import math
import os
import random
import sys
from argparse import ArgumentParser
from typing import Dict, List, Optional

import einops
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from omegaconf import OmegaConf, open_dict
from PIL import Image, ImageOps

from nemo.collections.multimodal.models.instruct_pix2pix.ldm.ddpm_edit import MegatronLatentDiffusionEdit
from nemo.collections.multimodal.models.stable_diffusion.samplers.k_diffusion import DiscreteEpsDDPMDenoiser
from nemo.collections.multimodal.modules.stable_diffusion.encoders.modules import FrozenCLIPEmbedder
from nemo.collections.multimodal.parts.utils import setup_trainer_and_model_for_inference
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.core.classes.exportable import Exportable
from nemo.core.config import hydra_runner
from nemo.core.neural_types import ChannelType, NeuralType
from nemo.utils import logging
from nemo.utils.trt_utils import build_engine


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "b ... -> (n b) ...", n=3)
        cfg_sigma = einops.repeat(sigma, "b ... -> (n b) ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        print(cfg_z.shape, cfg_sigma.shape)
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        out = out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)
        return out


@hydra_runner(config_path='conf', config_name='sd_export')
def main(cfg):
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')
    fp16 = 16 == cfg.trainer.get("precision", 32)
    if cfg.trainer.get("precision", 32) == "bf16":
        print("BF16 not supported for export, will use fp32")
    with open_dict(cfg):
        edit_cfg = cfg.pop("edit")

    def model_cfg_modifier(model_cfg):
        model_cfg.precision = cfg.trainer.precision
        model_cfg.ckpt_path = None
        model_cfg.inductor = False

    trainer, megatron_diffusion_model = setup_trainer_and_model_for_inference(
        model_provider=MegatronLatentDiffusionEdit, cfg=cfg, model_cfg_modifier=model_cfg_modifier,
    )

    # inference use the latent diffusion part of megatron wrapper
    model = megatron_diffusion_model.model
    model.eval()
    model_wrap = DiscreteEpsDDPMDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])

    # input_image = Image.open(edit_cfg.input).convert("RGB")
    # width, height = input_image.size
    # factor = edit_cfg.resolution / max(width, height)
    # factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    # width = int((width * factor) // 64) * 64
    # height = int((height * factor) // 64) * 64
    # input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)
    input_image = np.random.rand(edit_cfg.resolution, edit_cfg.resolution, 3) * 255
    input_image = Image.fromarray(input_image.astype('uint8')).convert('RGB')
    batch_size = edit_cfg.get("num_images_per_prompt", 1)
    height = edit_cfg.resolution
    width = edit_cfg.resolution

    output_dir = edit_cfg.out_path

    os.makedirs(f"{output_dir}/onnx/unet/", exist_ok=True)
    os.makedirs(f"{output_dir}/onnx/clip/", exist_ok=True)
    os.makedirs(f"{output_dir}/onnx/vae/", exist_ok=True)
    os.makedirs(f"{output_dir}/plan/", exist_ok=True)
    deployment_conf = OmegaConf.create(
        {
            'clip': OmegaConf.create({}),
            'unet': OmegaConf.create({}),
            'vaee': OmegaConf.create({}),
            'vaed': OmegaConf.create({}),
            'sampler': OmegaConf.create({}),
            'batch_size': batch_size,
            'height': height,
            'width': width,
            'resolution': edit_cfg.resolution,
            'steps': edit_cfg.steps,
            'text_cfg_scale': edit_cfg.cfg_text,
            'image_cfg_scale': edit_cfg.cfg_image,
        }
    )

    fake_text = [""]
    out_cond = model.cond_stage_model(fake_text)

    ### VAE Encode Export
    class VAEEncodeWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            h = self.model.encoder(x)
            moments = self.model.quant_conv(h)
            return moments

    input_names = ["x"]
    output_names = ["logits"]
    x = torch.randn(1, 3, width, height, device="cuda")
    # z = torch.randn(1, *shape_of_internal, device="cuda")
    torch.onnx.export(
        VAEEncodeWrapper(model.first_stage_model),
        (x,),
        f"{output_dir}/onnx/vae/vae_encode.onnx",
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={"x": {0: 'B'}, "logits": {0: 'B'}},
        opset_version=17,
    )
    input_profile_vaee = {}
    input_profile_vaee["x"] = [(1, *(x.shape[1:]))] * 3
    with torch.no_grad():
        out_vaee = VAEEncodeWrapper(model.first_stage_model)(x)
    deployment_conf.vaee.x = input_profile_vaee["x"][0]
    deployment_conf.vaee.logits = tuple(out_vaee.shape)

    x = torch.randn(3, *(out_vaee.shape[1:]), device="cuda")
    t = torch.randn(3, device="cuda")
    cc = torch.randn(3, out_cond.shape[1], out_cond.shape[2], device="cuda")
    # x, t = torch.randn(2, *shape_of_internal, device="cuda"), torch.randint(high=10, size=(2,), device="cuda")
    # cc = torch.randn(2, out.shape[1], out.shape[2], device="cuda")
    input_names = ["x", "t", "context"]
    output_names = ["logits"]
    torch.onnx.export(
        model.model.diffusion_model,
        (x, t, cc),
        f"{output_dir}/onnx/unet/unet.onnx",
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={"x": {0: 'B'}, "t": {0: 'B'}, "context": {0: 'B'}},
        opset_version=17,
    )

    input_profile_unet = {}
    input_profile_unet["x"] = [(3 * batch_size, *(x.shape[1:]))] * 3
    input_profile_unet["t"] = [(3 * batch_size, *(t.shape[1:]))] * 3
    input_profile_unet["context"] = [(3 * batch_size, *(cc.shape[1:]))] * 3
    with torch.no_grad():
        out_unet = model.model.diffusion_model(x, t, context=cc)
    deployment_conf.unet.x = input_profile_unet["x"][0]
    deployment_conf.unet.t = input_profile_unet["t"][0]
    deployment_conf.unet.context = input_profile_unet["context"][0]
    deployment_conf.unet.logits = (3 * batch_size, *(out_unet.shape[1:]))

    ### VAE Decode Export
    class VAEDecodeWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, z):
            h = self.model.post_quant_conv(z)
            dec = self.model.decoder(h)
            return dec

    input_names = ["z"]
    output_names = ["logits"]
    z = torch.randn(1, *(out_unet.shape[1:]), device="cuda")
    # z = torch.randn(1, *shape_of_internal, device="cuda")
    torch.onnx.export(
        VAEDecodeWrapper(model.first_stage_model),
        (z,),
        f"{output_dir}/onnx/vae/vae_decode.onnx",
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={"z": {0: 'B'}, "logits": {0: 'B'}},
        opset_version=17,
    )
    input_profile_vaed = {}
    input_profile_vaed["z"] = [(batch_size, *(z.shape[1:]))] * 3
    deployment_conf.vaed.z = input_profile_vaed["z"][0]
    deployment_conf.vaed.logits = (batch_size, 3, height, width)

    ### CLIP Export
    class CLIPWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids):
            outputs = self.model(input_ids=input_ids)
            return outputs.last_hidden_state

    class OpenCLIPWrapper(nn.Module, Exportable):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids):
            outputs = self.model.encode_with_transformer(input_ids)
            return outputs

        def input_example(self, max_text=64):
            sample = next(self.parameters())
            tokens = torch.randint(high=10, size=(1, self.model.max_length)).to(sample.device)
            return (tokens,)

        @property
        def input_types(self) -> Optional[Dict[str, NeuralType]]:
            return {
                "tokens": NeuralType(('H', 'D'), ChannelType()),
            }

        @property
        def output_types(self) -> Optional[Dict[str, NeuralType]]:
            return {"logits": NeuralType(('B', 'H'), ChannelType())}

        @property
        def input_names(self) -> List[str]:
            return ['tokens']

        @property
        def output_names(self) -> List[str]:
            return ['logits']

    openai_clip = isinstance(model.cond_stage_model, FrozenCLIPEmbedder)
    tokens = torch.randint(high=10, size=(1, model.cond_stage_model.max_length), device="cuda")

    if openai_clip:
        input_names = ["tokens"]
        output_names = ["logits"]
        torch.onnx.export(
            CLIPWrapper(model.cond_stage_model.transformer),
            (tokens,),
            f"{output_dir}/onnx/clip/clip.onnx",
            verbose=False,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={"tokens": {0: 'B'}, "logits": {0: 'B'}},
            opset_version=17,
            do_constant_folding=True,
            export_params=True,
        )
    else:
        clip_model = OpenCLIPWrapper(model.cond_stage_model)
        clip_model.export(f"{output_dir}/onnx/clip/clip.onnx")

    input_profile_clip = {}
    input_profile_clip["tokens"] = [(1, *(tokens.shape[1:]))] * 3
    deployment_conf.clip.tokens = input_profile_clip["tokens"][0]
    deployment_conf.clip.logits = (1, model.cond_stage_model.max_length, out_cond.shape[2])
    deployment_conf.clip.max_length = model.cond_stage_model.max_length
    with open(f"{output_dir}/plan/conf.yaml", "wb") as f:
        OmegaConf.save(config=deployment_conf, f=f.name)
    del model, trainer, megatron_diffusion_model, x, t, cc, z, tokens, out_cond, out_vaee, out_unet
    torch.cuda.empty_cache()
    gc.collect()
    build_engine(
        f"{output_dir}/onnx/unet/unet.onnx",
        f"{output_dir}/plan/unet.plan",
        fp16=fp16,
        input_profile=input_profile_unet,
        timing_cache=None,
        workspace_size=0,
    )
    build_engine(
        f"{output_dir}/onnx/vae/vae_decode.onnx",
        f"{output_dir}/plan/vae_decode.plan",
        fp16=fp16,
        input_profile=input_profile_vaed,
        timing_cache=None,
        workspace_size=0,
    )
    build_engine(
        f"{output_dir}/onnx/vae/vae_encode.onnx",
        f"{output_dir}/plan/vae_encode.plan",
        fp16=fp16,
        input_profile=input_profile_vaee,
        timing_cache=None,
        workspace_size=0,
    )
    build_engine(
        f"{output_dir}/onnx/clip/clip.onnx",
        f"{output_dir}/plan/clip.plan",
        fp16=fp16,
        input_profile=input_profile_clip,
        timing_cache=None,
        workspace_size=0,
    )


if __name__ == "__main__":
    main()

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
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from omegaconf.omegaconf import OmegaConf

from nemo.collections.multimodal.models.controlnet.controlnet import MegatronControlNet
from nemo.collections.multimodal.modules.stable_diffusion.encoders.modules import FrozenCLIPEmbedder
from nemo.collections.multimodal.parts.utils import setup_trainer_and_model_for_inference
from nemo.core.classes.exportable import Exportable
from nemo.core.config import hydra_runner
from nemo.core.neural_types import ChannelType, NeuralType
from nemo.utils.trt_utils import build_engine


@hydra_runner(config_path='conf', config_name='controlnet_export')
def main(cfg):
    # setup default values for inference configs

    batch_size = cfg.infer.get('num_images_per_prompt', 1)
    height = cfg.infer.get('height', 512)
    width = cfg.infer.get('width', 512)
    hint_image_size = cfg.infer.get('hint_image_size', 512)
    downsampling_factor = cfg.infer.get('down_factor', 8)
    fp16 = 16 == cfg.trainer.get("precision", 32)
    if cfg.trainer.get("precision", 32) in ['bf16', 'bf16-mixed']:
        print("BF16 not supported for export, will use fp32")

    def model_cfg_modifier(model_cfg):
        model_cfg.precision = cfg.trainer.precision
        model_cfg.ckpt_path = None
        model_cfg.inductor = False
        model_cfg.unet_config.from_pretrained = None
        model_cfg.first_stage_config.from_pretrained = None
        model_cfg.control_stage_config.from_pretrained_unet = None
        model_cfg.channels_last = True
        model_cfg.capture_cudagraph_iters = -1

    trainer, megatron_diffusion_model = setup_trainer_and_model_for_inference(
        model_provider=MegatronControlNet, cfg=cfg, model_cfg_modifier=model_cfg_modifier
    )
    model = megatron_diffusion_model.model
    model.cuda().eval()

    in_channels = model.model.diffusion_model.in_channels
    shape_of_internal = [in_channels, height // downsampling_factor, width // downsampling_factor]
    fake_text = [""]
    out = model.cond_stage_model(fake_text)

    output_dir = cfg.infer.out_path
    os.makedirs(f"{output_dir}/onnx/controlnet/", exist_ok=True)
    os.makedirs(f"{output_dir}/onnx/unet/", exist_ok=True)
    os.makedirs(f"{output_dir}/onnx/clip/", exist_ok=True)
    os.makedirs(f"{output_dir}/onnx/vae/", exist_ok=True)
    os.makedirs(f"{output_dir}/plan/", exist_ok=True)
    deployment_conf = OmegaConf.create(
        {
            'controlnet': OmegaConf.create({}),
            'clip': OmegaConf.create({}),
            'unet': OmegaConf.create({}),
            'vae': OmegaConf.create({}),
            'sampler': OmegaConf.create({}),
            'batch_size': batch_size,
            'downsampling_factor': downsampling_factor,
            'in_channels': in_channels,
            'height': height,
            'width': width,
            'hint_image_size': hint_image_size,
        }
    )
    deployment_conf.sampler.eta = cfg.infer.get('eta', 0)
    deployment_conf.sampler.inference_steps = cfg.infer.get('inference_steps', 50)
    deployment_conf.sampler.sampler_type = cfg.infer.get('sampler_type', "ddim")

    ### Controlnet Export
    x = torch.randn(1, *shape_of_internal, device="cuda")
    t = torch.randint(high=10, size=(1,), device="cuda")
    cc = torch.randn(1, out.shape[1], out.shape[2], device="cuda")
    hint = torch.randn(1, 3, hint_image_size, hint_image_size, device="cuda")  # b c h w

    controlnet_inputs = (x, hint, t, cc)
    control_outs = model.control_model(*controlnet_inputs)
    control_names = [f"control_{i}" for i in range(len(control_outs))]

    input_names = ["x", "hint", "t", "context"]
    output_names = control_names

    print('Running Controlnet onnx export')
    torch.onnx.export(
        model.control_model,
        controlnet_inputs,
        f"{output_dir}/onnx/controlnet/controlnet.onnx",
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={"x": {0: 'B'}, "hint": {0: 'B'}, "t": {0: 'B'}, "context": {0: 'B'}},
        opset_version=17,
    )

    input_profile_controlnet = {}
    input_profile_controlnet["x"] = [(batch_size, *(x.shape[1:]))] * 3
    input_profile_controlnet["hint"] = [(batch_size, *(hint.shape[1:]))] * 3
    input_profile_controlnet["t"] = [(batch_size, *(t.shape[1:]))] * 3
    input_profile_controlnet["context"] = [(batch_size, *(cc.shape[1:]))] * 3

    deployment_conf.controlnet.x = input_profile_controlnet["x"][0]
    deployment_conf.controlnet.hint = input_profile_controlnet["hint"][0]
    deployment_conf.controlnet.t = input_profile_controlnet["t"][0]
    deployment_conf.controlnet.context = input_profile_controlnet["context"][0]
    deployment_conf.controlnet.control = OmegaConf.create({})

    for control_name, control_out in zip(control_names, control_outs):
        deployment_conf.controlnet.control.update({control_name: (batch_size, *(control_out.shape[1:]))})

    ### UNet Export
    input_names = ["x", "t", "context"] + control_names
    output_names = ["logits"]

    class UNETControlWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x, t, cc, *control):
            if any(part_control is None for part_control in control):
                control = None
            else:
                control = list(control)

            return self.model(x=x, timesteps=t, context=cc, control=control)

    print('Running UNET onnx export')
    torch.onnx.export(
        UNETControlWrapper(model.model.diffusion_model),
        (x, t, cc, *control_outs),
        f"{output_dir}/onnx/unet/unet.onnx",
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            **{"x": {0: 'B'}, "t": {0: 'B'}, "context": {0: 'B'}},
            **{control_name: {0: 'B'} for control_name in control_names},
        },
        opset_version=17,
    )

    input_profile_unet = {}
    input_profile_unet["x"] = [(batch_size, *(x.shape[1:]))] * 3
    input_profile_unet["t"] = [(batch_size, *(t.shape[1:]))] * 3
    input_profile_unet["context"] = [(batch_size, *(cc.shape[1:]))] * 3

    deployment_conf.unet.x = input_profile_unet["x"][0]
    deployment_conf.unet.t = input_profile_unet["t"][0]
    deployment_conf.unet.context = input_profile_unet["context"][0]
    deployment_conf.unet.logits = input_profile_unet["x"][0]
    deployment_conf.unet.control = OmegaConf.create({})

    for control_name, control_out in zip(control_names, control_outs):
        input_profile_unet[control_name] = [(batch_size, *(control_out.shape[1:]))] * 3
        deployment_conf.unet.control.update({control_name: input_profile_unet[control_name][0]})

    ### VAE Export
    class VAEWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, z):
            z = self.model.post_quant_conv(z)
            return self.model.decoder(z)

    input_names = ["z"]
    output_names = ["logits"]
    z = torch.randn(1, *shape_of_internal, device="cuda")

    print('Running VAE onnx export')
    torch.onnx.export(
        VAEWrapper(model.first_stage_model),
        (z,),
        f"{output_dir}/onnx/vae/vae.onnx",
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={"z": {0: 'B'}, "logits": {0: 'B'}},
        opset_version=17,
    )

    input_profile_vae = {}
    input_profile_vae["z"] = [(batch_size, *(z.shape[1:]))] * 3
    deployment_conf.vae.z = input_profile_vae["z"][0]

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

    print('Running CLIP onnx export')
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
        clip_model.export("stable-diffusion/onnx/clip/clip.onnx")

    input_profile_clip = {}
    input_profile_clip["tokens"] = [(batch_size, *(tokens.shape[1:]))] * 3
    deployment_conf.clip.tokens = input_profile_clip["tokens"][0]
    deployment_conf.clip.logits = (batch_size, model.cond_stage_model.max_length, out.shape[2])
    deployment_conf.clip.unconditional_guidance_scale = cfg.infer.get("unconditional_guidance_scale", 7.5)
    deployment_conf.clip.max_length = model.cond_stage_model.max_length
    deployment_conf.clip.openai_clip = openai_clip
    with open(f"{output_dir}/plan/conf.yaml", "wb") as f:
        OmegaConf.save(config=deployment_conf, f=f.name)

    del model, trainer, megatron_diffusion_model, x, t, cc, z, tokens, out, hint, control_outs
    torch.cuda.empty_cache()
    gc.collect()

    print('Running Controlnet TRT conversion')
    build_engine(
        f"{output_dir}/onnx/controlnet/controlnet.onnx",
        f"{output_dir}/plan/controlnet.plan",
        fp16=fp16,
        input_profile=input_profile_controlnet,
        timing_cache=None,
        workspace_size=0,
    )

    print('Running UNET TRT conversion')
    build_engine(
        f"{output_dir}/onnx/unet/unet.onnx",
        f"{output_dir}/plan/unet.plan",
        fp16=fp16,
        input_profile=input_profile_unet,
        timing_cache=None,
        workspace_size=0,
    )

    print('Running VAE TRT conversion')
    build_engine(
        f"{output_dir}/onnx/vae/vae.onnx",
        f"{output_dir}/plan/vae.plan",
        fp16=fp16,
        input_profile=input_profile_vae,
        timing_cache=None,
        workspace_size=0,
    )

    print('Running CLIP TRT conversion')
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

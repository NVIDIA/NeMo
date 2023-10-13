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

import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from nemo.collections.multimodal.modules.imagen.diffusionmodules import attention_alt
from nemo.core.config import hydra_runner
from nemo.utils.trt_utils import build_engine


@hydra_runner(config_path='conf', config_name='export')
def main(inference_config):
    if inference_config.get('infer'):
        # invoking from launcher
        trainer = Trainer(inference_config.trainer)
        inference_config = inference_config.infer
    else:
        trainer = Trainer()

    # Set up variable to use alternative attention
    attention_alt.USE_ALT = True
    from nemo.collections.multimodal.models.imagen.imagen_pipeline import ImagenPipeline, ImagenPipelineConfig

    inference_config: ImagenPipelineConfig = OmegaConf.merge(ImagenPipelineConfig(), inference_config)
    fp16 = 16 == int(inference_config.get("inference_precision", 32))
    # Set model to FP32 for ONNX export
    inference_config.inference_precision = 32

    pipeline = ImagenPipeline.from_pretrained(cfg=inference_config, trainer=trainer)
    batch_size = inference_config.get('num_images_per_promt', 1)
    thresholding_method = inference_config.get('thresholding_method', 'dynamic')
    fake_text = [""]
    out_embed, out_mask = pipeline.get_text_encodings(fake_text, repeat=batch_size)
    output_dir = inference_config.output_path
    deployment_conf = OmegaConf.create(
        {
            't5': OmegaConf.create({}),
            'models': OmegaConf.create([]),
            'batch_size': batch_size,
            'thresholding_method': thresholding_method,
        }
    )

    ### T5 Export
    class T5Wrapper(torch.nn.Module):
        def __init__(self, t5_encoder):
            super(T5Wrapper, self).__init__()
            self.t5_encoder = t5_encoder

        def forward(self, input_ids, attn_mask):
            t5_encoder = self.t5_encoder

            with torch.no_grad():
                output = t5_encoder.model(input_ids=input_ids, attention_mask=attn_mask)
                encoded_text = output.last_hidden_state

            encoded_text = encoded_text[:, 0 : t5_encoder.max_seq_len]
            attn_mask = attn_mask[:, 0 : t5_encoder.max_seq_len]

            return encoded_text, attn_mask

    t5_wrapper = T5Wrapper(pipeline.text_encoder)
    # Exporting T5Encoder in CPU
    t5_wrapper.to('cpu')

    input_names = ['input_ids', 'attn_mask']
    output_names = ['encoded_text', 'text_mask']
    input_ids = torch.randint(high=10, size=(1, pipeline.text_encoder.model_seq_len), dtype=torch.int)
    attn_mask = torch.zeros(1, pipeline.text_encoder.model_seq_len, dtype=torch.int)

    os.makedirs(f"{output_dir}/onnx/t5/", exist_ok=True)
    torch.onnx.export(
        t5_wrapper,
        (input_ids, attn_mask),
        f"{output_dir}/onnx/t5/t5.onnx",
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={"input_ids": {0: 'B'}, "attn_mask": {0: 'B'},},
        opset_version=17,
    )

    input_profile_t5 = {}
    input_profile_t5["input_ids"] = [input_ids.shape] * 3
    input_profile_t5["attn_mask"] = [attn_mask.shape] * 3
    deployment_conf.t5.model_seq_len = pipeline.text_encoder.model_seq_len
    del pipeline.text_encoder, input_ids, attn_mask

    ### UNet Export
    os.makedirs(f"{output_dir}/onnx/unet/", exist_ok=True)

    low_res_size = None
    cfgs = [each.cfg for each in inference_config.samplings]
    cfgs = cfgs[: len(pipeline.models)]
    steps = [each.step for each in inference_config.samplings]
    steps = steps[: len(pipeline.models)]
    input_profile_unets = []

    for i, model in enumerate(pipeline.models):
        unet_model = model.unet

        ### UNet Export
        x = torch.randn(batch_size, 3, unet_model.image_size, unet_model.image_size, device="cuda")
        time = torch.randn(batch_size, device='cuda')
        text_embed = torch.randn(batch_size, out_embed.shape[1], out_embed.shape[2], device='cuda')
        text_mask = torch.zeros((batch_size, out_mask.shape[1]), dtype=torch.int, device='cuda')
        input_names = ["x", "time", "text_embed", "text_mask"]
        output_names = ["logits"]
        dynamic_axes = {
            "x": {0: 'B'},
            "time": {0: 'B'},
            "text_embed": {0: 'B'},
            "text_mask": {0: 'B'},
        }
        inputs = [x, time, text_embed, text_mask]

        if low_res_size is not None:
            input_names.append("x_low_res")
            dynamic_axes['x_low_res'] = {0: 'batch'}
            x_low_res = torch.randn(batch_size, 3, low_res_size, low_res_size, device="cuda")
            inputs.append(x_low_res)

            if model.noise_cond_aug:
                input_names.append("time_low_res")
                dynamic_axes['time_low_res'] = {0: 'batch'}
                time_low_res = torch.ones(batch_size, device="cuda")
                inputs.append(time_low_res)

        torch.onnx.export(
            unet_model,
            tuple(inputs),
            f"{output_dir}/onnx/unet/unet{i}.onnx",
            verbose=False,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
        )

        input_profile_unet = {}
        input_profile_unet["x"] = [(batch_size, *(x.shape[1:]))] * 3
        input_profile_unet["time"] = [(batch_size,)] * 3
        input_profile_unet["text_embed"] = [(batch_size, *(text_embed.shape[1:]))] * 3
        input_profile_unet["text_mask"] = [(batch_size, *(text_mask.shape[1:]))] * 3

        config = OmegaConf.create({})
        config.preconditioning_type = model.preconditioning_type
        config.preconditioning = model.cfg.preconditioning
        config.noise_cond_aug = model.noise_cond_aug
        config.cond_scale = cfgs[i]
        config.step = steps[i]
        config.x = input_profile_unet["x"][0]

        if i == 0:
            config.text_embed = input_profile_unet["text_embed"][0]
            config.text_mask = input_profile_unet["text_mask"][0]

        if low_res_size is not None:
            input_profile_unet["x_low_res"] = [(batch_size, *(x_low_res.shape[1:]))] * 3

            if model.noise_cond_aug:
                input_profile_unet["time_low_res"] = [(batch_size,)] * 3

        for key in input_profile_unet:
            # set up min and max batch to 1 and 2 * batch_size
            input_profile_unet[key][0] = (1, *input_profile_unet[key][0][1:])
            input_profile_unet[key][2] = (2 * batch_size, *input_profile_unet[key][2][1:])

        deployment_conf.models.append(config)
        input_profile_unets.append(input_profile_unet)

        low_res_size = unet_model.image_size

    os.makedirs(f"{output_dir}/plan", exist_ok=True)
    with open(f"{output_dir}/plan/conf.yaml", "wb") as f:
        OmegaConf.save(config=deployment_conf, f=f.name)

    del pipeline, x, time, text_embed, text_mask
    torch.cuda.empty_cache()
    gc.collect()

    build_engine(
        f"{output_dir}/onnx/t5/t5.onnx",
        f"{output_dir}/plan/t5.plan",
        fp16=False,
        input_profile=input_profile_t5,
        timing_cache=None,
        workspace_size=0,
    )

    for i, input_profile in enumerate(input_profile_unets):
        build_engine(
            f"{output_dir}/onnx/unet/unet{i}.onnx",
            f"{output_dir}/plan/unet{i}.plan",
            fp16=fp16,
            input_profile=input_profile,
            timing_cache=None,
            workspace_size=0,
        )


if __name__ == "__main__":
    main()

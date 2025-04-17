# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=C0115,C0116,C0301

import gc
import os
from argparse import ArgumentParser
from typing import List

import imageio
import numpy as np
import torch
from cosmos1.models.autoregressive.nemo.utils import run_diffusion_decoder_model
from cosmos1.models.autoregressive.tokenizer.discrete_video import DiscreteVideoFSQJITTokenizer
from cosmos1.models.autoregressive.utils.inference import load_vision_input
from cosmos1.models.guardrail.common import presets as guardrail_presets
from cosmos1.utils import log
from einops import rearrange
from huggingface_hub import snapshot_download
from megatron.core.inference.common_inference_params import CommonInferenceParams
from megatron.core.inference.engines.mcore_engine import MCoreEngine
from megatron.core.inference.text_generation_controllers.simple_text_generation_controller import (
    SimpleTextGenerationController,
)

import nemo.lightning as nl
from nemo.collections.llm.inference.base import _setup_trainer_and_restore_model
from nemo.lightning import io
from nemo.lightning.ckpt_utils import ckpt_to_context_subdir

torch._C._jit_set_texpr_fuser_enabled(False)

LATENT_SHAPE = [5, 40, 64]
NUM_INPUT_FRAMES_VIDEO = 9


class MockMCoreTokenizer:
    """
    A small dummy wrapper to pass into the text generation controller.
    """

    def __init__(self, vocab_size: int = 64000):
        self.tokenizer = None
        self.eod = -1
        self.vocab_size = vocab_size

    def detokenize(self, tokens: List[int], remove_special_tokens: bool = False):
        return tokens

    def tokenize(self, prompt: List[int]):
        return prompt


def main(args):
    num_input_frames = 1 if args.input_type == "image" else NUM_INPUT_FRAMES_VIDEO
    torch.distributed.init_process_group(backend="nccl")
    TOKENIZER_COMPRESSION_FACTOR = list(map(int, args.tokenizer_compression_factor.split(",")))
    NUM_CONTEXT_FRAMES = args.num_context_frames

    vision_input_dict = load_vision_input(
        input_type=args.input_type,
        batch_input_path=None,
        input_image_or_video_path=args.input_image_or_video_path,
        data_resolution=[args.height, args.width],
        num_input_frames=num_input_frames,
        num_total_frames=args.num_context_frames,
    )

    vision_input = list(vision_input_dict.values())[0].cuda()
    vision_input = vision_input[:, :, :NUM_CONTEXT_FRAMES, :, :]

    T, H, W = LATENT_SHAPE
    latent_context_t_size = 1 if args.input_type == "image" else 2
    num_tokens_to_generate = int(np.prod([T - latent_context_t_size, H, W]))

    # Encode and Tokenize
    if args.encoder_path.startswith("nvidia/"):
        args.encoder_path = os.path.join(snapshot_download(args.encoder_path), "encoder.jit")
    if args.decoder_path.startswith("nvidia/"):
        args.decoder_path = os.path.join(snapshot_download(args.decoder_path), "decoder.jit")

    video_tokenizer = DiscreteVideoFSQJITTokenizer(
        enc_fp=args.encoder_path,
        dec_fp=args.decoder_path,
        name="discrete_video_fsq",
        pixel_chunk_duration=NUM_CONTEXT_FRAMES,
        latent_chunk_duration=T,
        compression_ratio=TOKENIZER_COMPRESSION_FACTOR,
    ).cuda()

    quantized_out, _ = video_tokenizer.encode(vision_input, pixel_chunk_duration=None)
    indices = video_tokenizer.fsq_quantizer.codes_to_indices(quantized_out.permute(0, 2, 3, 4, 1))
    indices = rearrange(indices, "B T H W -> B (T H W)")
    video_tokens = [indices[0][0:-num_tokens_to_generate].tolist()]

    # Load the nemo model
    if args.ar_model_dir in ["nvidia/Cosmos-1.0-Autoregressive-4B", "nvidia/Cosmos-1.0-Autoregressive-12B"]:
        args.ar_model_dir = os.path.join(snapshot_download(args.ar_model_dir, allow_patterns=["nemo/*"]), "nemo")
    model: io.TrainerContext = io.load_context(path=ckpt_to_context_subdir(args.ar_model_dir), subpath="model")

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        setup_optimizers=False,
        store_optimizer_states=False,
    )

    trainer = nl.Trainer(
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        num_nodes=1,
        strategy=strategy,
        num_sanity_val_steps=0,
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_enabled=False,
            grad_reduce_in_fp32=False,
        ),
    )

    _setup_trainer_and_restore_model(path=args.ar_model_dir, trainer=trainer, model=model)

    inference_wrapped_model = model.get_inference_wrapper(torch.bfloat16, inference_batch_times_seqlen_threshold=1000)

    # Generate tokens
    text_generation_controller = SimpleTextGenerationController(
        inference_wrapped_model=inference_wrapped_model, tokenizer=MockMCoreTokenizer()
    )

    mcore_engine = MCoreEngine(text_generation_controller=text_generation_controller, max_batch_size=1)

    common_inference_params = CommonInferenceParams(
        temperature=args.temperature, top_p=args.top_p, num_tokens_to_generate=num_tokens_to_generate
    )

    log.info(f"Running Inference to generate {num_tokens_to_generate} tokens. This will take some time. ")
    results = mcore_engine.generate(
        prompts=video_tokens,
        add_BOS=False,
        encoder_prompts=None,
        common_inference_params=common_inference_params,
    )

    result = list(results)[0]

    prompt_tokens = torch.tensor(result.prompt_tokens).cuda()
    prompt_tokens = torch.cat((prompt_tokens, result.generated_tokens))

    indices_tensor = prompt_tokens.unsqueeze(dim=0)
    indices_tensor = rearrange(
        indices_tensor,
        "B (T H W) -> B T H W",
        T=LATENT_SHAPE[0],
        H=LATENT_SHAPE[1],
        W=LATENT_SHAPE[2],
    )

    if torch.cuda.current_device() == 0:
        # Decode the generated tokens
        log.info("Running diffusion model on the generated result")
        video_decoded = video_tokenizer.decode(indices_tensor.cuda())
        out_video = (video_decoded * 0.5 + 0.5).clamp_(0, 1)

        if not args.disable_diffusion_decoder:
            del model
            del inference_wrapped_model
            del video_tokenizer
            model = None
            inference_wrapped_model = None
            video_tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()

            out_video = run_diffusion_decoder_model(
                indices_tensor_cur_batch=[indices_tensor.squeeze()], out_videos_cur_batch=out_video
            )

        out_video = out_video[0].detach().clone()
        output_video = (out_video * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()

        if args.guardrail_dir:
            log.info("Running guardrails on the generated video")
            if args.guardrail_dir == "nvidia/Cosmos-1.0-Guardrail":
                args.guardrail_dir = snapshot_download(args.guardrail_dir)
            video_guardrail = guardrail_presets.create_video_guardrail_runner(checkpoint_dir=args.guardrail_dir)
            output_video = guardrail_presets.run_video_guardrail(output_video, video_guardrail)
            if output_video is None:
                raise ValueError("Guardrail blocked world generation.")

        # Write the video to disk
        imageio.mimsave(
            args.video_save_name,
            output_video,
            fps=25,  # We use a fps of 25 just for visualization.
        )

        log.info(f"Saved to {args.video_save_name}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--width", required=False, default=1024, type=int, help="Width of the input videos")
    parser.add_argument("--height", required=False, default=640, type=int, help="Height of the input videos")
    parser.add_argument("--num_context_frames", required=False, default=33, type=int, help="Number of context frames")
    parser.add_argument(
        "--tokenizer_compression_factor",
        required=False,
        default="8,16,16",
        type=str,
        help="Tokenizer compression factor",
    )
    parser.add_argument("--input_type", type=str, default="video", help="Type of input", choices=["image", "video"])
    parser.add_argument(
        "--input_image_or_video_path", required=True, type=str, help="The path to the input video to run inference"
    )
    parser.add_argument(
        "--video_save_name", default="./nemo_generated_video.mp4", type=str, help="The path to generated video"
    )
    parser.add_argument("--num_input_frames", default=9, type=int, help="The number of input frames")
    parser.add_argument(
        "--ar_model_dir",
        default="nvidia/Cosmos-1.0-Autoregressive-4B",
        type=str,
        help="The path to the nemo autoregressive model",
    )
    parser.add_argument(
        "--encoder_path", default="nvidia/Cosmos-1.0-Tokenizer-DV8x16x16", type=str, help="The path to encoder"
    )
    parser.add_argument(
        "--decoder_path", default="nvidia/Cosmos-1.0-Tokenizer-DV8x16x16", type=str, help="The path to the decoder"
    )
    parser.add_argument(
        "--guardrail_dir", default="nvidia/Cosmos-1.0-Guardrail", type=str, help="The path to the guardrails"
    )
    parser.add_argument("--top_p", default=0.8, type=float, help="The top_p inference parameter ")
    parser.add_argument("--temperature", default=1, type=int, help="Sampling temperature")
    parser.add_argument("--disable_diffusion_decoder", action="store_true", help="Disable diffusion decoder")
    parser.add_argument(
        "--tensor_model_parallel_size", default=torch.cuda.device_count(), type=int, help="The number of GPUs to use"
    )

    args = parser.parse_args()

    main(args)

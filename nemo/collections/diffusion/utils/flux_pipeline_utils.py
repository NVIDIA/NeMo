from nemo.collections.diffusion.flux.model import FluxParams
from nemo.collections.diffusion.vae.autoencoder import AutoEncoderParams
import torch
from dataclasses import dataclass
from megatron.core.transformer.utils import openai_gelu


@dataclass
class FluxModelParams:
    flux_params: FluxParams
    vae_params: AutoEncoderParams
    clip_params: dict | None
    t5_params: dict | None
    scheduler_params: dict | None
    device: str | torch.device

configs = {
    "flux": FluxModelParams(
        flux_params=FluxParams(
            num_joint_layers=19,
            num_single_layers=38,
            hidden_size=3072,
            num_attention_heads=24,
            activation_func=openai_gelu,
            add_qkv_bias=True,
            ffn_hidden_size=16384,
            in_channels=64,
            context_dim=4096,
            model_channels=256,
            patch_size=1,
            guidance_embed=True,
            vec_in_dim=768,
        ),
        vae_params=AutoEncoderParams(
            ch_mult=[1,2,4,4],
            attn_resolutions=[],
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
        clip_params={
            'version':"openai/clip-vit-large-patch14",
            'max_length': 77,
            'always_return_pooled': True,
        },
        t5_params={
            'version': "google/t5-v1_1-xxl",
            'max_length':  512,
        },
        scheduler_params={
            'num_train_timesteps':1000,
        },
        device='cpu',

    )

}

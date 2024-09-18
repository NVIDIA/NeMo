import torch

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.dit.dit_layer_spec import (
    get_mm_dit_block_with_transformer_engine_spec,
    get_flux_single_transformer_engine_spec,
    MMDiTLayer,
    FluxSingleTransformerBlock,
    AdaLNContinous)
from megatron.core.transformer.utils import openai_gelu

from nemo.collections.diffusion.flux.layer import EmbedND, MLPEmbedder, TimeStepEmbedder
from torch import nn

class FluxConfig(TransformerConfig):
    num_joint_layers: int = 19
    num_single_layers: int = 38
    hidden_size: int = 3072
    num_attention_heads: int = 24
    activation_func: Callable = openai_gelu
    add_qkv_bias: bool = True
    ffn_hidden_size: int = 16384
    in_channels: int = 64
    context_dim: int = 4096
    model_channels: int = 256
    patch_size: int = 1
    guidance_embed: bool = False
    vec_in_dim: int = 768






class Flux(nn.Module):
    def __init__(self, config: FluxConfig):
        super().__init__()
        self.out_channels = config.in_channels
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.patch_size = config.patch_size
        self.pos_embed = EmbedND(dim = self.hidden_size, theta=10000, axes_dim=[16, 56, 56])
        self.img_embed = nn.Linear(config.in_channels, self.hidden_size)
        self.txt_embed = nn.Linear(config.context_dim, self.hidden_size)
        self.timestep_embedding = TimeStepEmbedder(config.model_channels, self.hidden_size)
        self.guidance_embedding = MLPEmbedder(in_dim=config.model_channels, hidden_dim=self.hidden_size) if config.guidance_embed else nn.Identity()
        self.vector_embedding = MLPEmbedder(in_dim=config.vec_in_dim, hidden_dim=self.hidden_size)

        transformer_config = TransformerConfig(num_layers=1, hidden_size=self.hidden_size, num_attention_heads=self.num_attention_heads,
                                               use_cpu_initialization=True, activation_func=config.activation_func,
                                               hidden_dropout=0, attention_dropout=0, layernorm_epsilon=1e-6,
                                               add_qkv_bias=True, ffn_hidden_size=int(self.hidden_size*4))


        self.transformer_blocks = nn.ModuleList([
            MMDiTLayer(
                config=transformer_config,
                submodules=get_mm_dit_block_with_transformer_engine_spec().submodules,
                layer_number=i,
                context_pre_only=False)
            for i in range(config.num_joint_layers)
        ])

        self.single_transformer_blocks = nn.ModuleList([
            FluxSingleTransformerBlock(
                config=transformer_config,
                submodules=get_flux_single_transformer_engine_spec().submodules,
                layer_number=i
            )
            for i in range(config.num_single_layers)
        ])

        self.norm_out = AdaLNContinuous(config=TransformerConfig, conditioning_embedding_dim=self.hidden_size)
        self.proj_out = nn.Linear(self.hidden_size, self.patch_size * self.patch_size * self.out_channels, bias=True)
    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor = None,
        y: torch.Tensor = None,
        timesteps: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
    ):
        hidden_states = self.img_embed(img)
        encoder_hidden_states = self.txt_embed(txt)

        timesteps = timesteps.to(img.dtype) * 1000
        vec_emb = self.timestep_embedding(timesteps)
        if guidance is not None:
            vec_emb = vec_emb + self.guidance_embedding(guidance * 1000)
        vec_emb = vec_emb + self.vector_embedding(y)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        rotary_pos_emb = self.pos_embed(ids)


        for id_block, block in enumerate(self.transformer_blocks):
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                rotary_pos_emb = rotary_pos_emb,
                emb = vec_emb,
            )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for id_block, block in enumerate(self.single_transformer_blocks):
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                rotary_pos_emb = rotary_pos_emb,
                emb = vec_emb,
            )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.norm_out(hidden_states, vec_emb)
        output = self.proj_out(hidden_states)

        return output


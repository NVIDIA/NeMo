import copy
from typing import Dict, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from megatron.core import InferenceParams, mpu, parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_sharded_tensor_for_checkpoint
from torch import Tensor

from .stdit_embeddings import (
    CaptionEmbedder,
    PatchEmbed3D,
    PositionEmbedding2D,
    SizeEmbedder,
    T2IFinalLayer,
    TblockEmbedder,
    TimestepEmbedder,
    approx_gelu,
)
from .stdit_layer_spec import get_stdit_analn_block_with_transformer_engine_spec as STDiTLayerWithAdaLNspec


class STDiTModel(VisionModule):
    """STDiT model with a Transformer backbone.

    Args:
        config (TransformerConfig): transformer config

        transformer_decoder_layer_spec (ModuleSpec): transformer layer customization specs for decoder

        pre_process (bool): Include embedding layer (used with pipeline parallelism)
        post_process (bool): Include an output layer (used with pipeline parallelism)

        fp16_lm_cross_entropy (bool, optional): Defaults to False

        parallel_output (bool): Do not gather the outputs, keep them split across tensor parallel ranks

        share_embeddings_and_output_weights (bool): When True, input embeddings and output logit weights are
            shared. Defaults to False.

        position_embedding_type (string): Position embedding type. Options ['learned_absolute', 'rope'].
            Defaults is 'learned_absolute'.

        rotary_percent (float): Percent of rotary dimension to use for rotary position embeddings.
            Defaults to 1.0 (100%). Ignored unless position_embedding_type is 'rope'.

        seq_len_interpolation_factor (float): scale of linearly interpolating RoPE for longer sequences.
            The value must be a float larger than 1.0. Defaults to None.

        /need to add some config for other part of stditmodel

    """

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec = STDiTLayerWithAdaLNspec,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        position_embedding_type: Literal[
            "learned_absolute", "rope"
        ] = "learned_absolute",  # Todo : need to change in open-sora stditv3
        rotary_percent: float = 1.0,
        seq_len_interpolation_factor: Optional[float] = None,
        # some config kwargs need to add for stdit model
        max_img_h: int = 26,
        max_img_w: int = 36,
        max_frames: int = 15,
        in_channels: int = 4,
        patch_spatial: int = 2,
        patch_temporal: int = 1,
        crossattn_emb_size: int = 1152,
        input_sq_size: int = 512,  # Todo: need to check its number
        class_dropout_prob: float = 0.1,
        pred_sigma: bool = True,
        drop_path: float = 0.0,
        caption_channels: int = 4096,
        model_max_length: int = 300,
        skip_y_embedder: bool = True,
        t_embed_seed=None,
        dynamic_sequence_parallel: bool = False,
        **kwargs,
    ):
        super(STDiTModel, self).__init__(config=config)

        # model transformer config set
        self.config: TransformerConfig = config
        self.transformer_decoder_layer_spec = transformer_layer_spec()
        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = True
        self.add_decoder = True
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.position_embedding_type = position_embedding_type
        self.share_embeddings_and_output_weights = False

        # opensora input size config set
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.input_sq_size = input_sq_size
        self.config.crossattn_emb_size = crossattn_emb_size

        # ============================
        # config change in decoder part
        # =============================
        T, H, W = self.get_patchify_size()
        decoder_config = copy.deepcopy(self.config)
        stdit_block_T = T
        stdit_block_S = H * W
        # dsp & cp : split input in h dim
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size > 1:
            assert H % cp_size == 0, "h must divisibly by cp_size"
            stdit_block_S = (H // cp_size) * W
        # if sequence parallel is true and need to fit it
        tp_size = mpu.get_tensor_model_parallel_world_size()
        if tp_size > 1 and self.config.sequence_parallel == True:
            assert stdit_block_T % tp_size == 0, "stdit_block_T must divisibly by tp_size due to sequence_parallel"
            stdit_block_T = stdit_block_T // tp_size

        # set stdit_dim_T & stdit_dim_S just for stdit model due to spatial_attn & temporal_attn op
        setattr(decoder_config, "stdit_dim_T", stdit_block_T)
        setattr(decoder_config, "stdit_dim_S", stdit_block_S)
        setattr(decoder_config, "dynamic_sequence_parallel", dynamic_sequence_parallel)

        self.drop_path = drop_path
        self.skip_y_embedder = skip_y_embedder

        ## Decoder part
        ## Todo : Drop_path need to add to TransformerBlock
        drop_path = [x.item() for x in torch.linspace(0, self.drop_path, self.config.num_layers)]
        self.decoder_spatial_temporal = TransformerBlock(
            config=decoder_config,
            spec=self.transformer_decoder_layer_spec,
            pre_process=self.pre_process,
            post_process=False,
            post_layer_norm=False,
        )

        # replicate in every pp rank - vpp rank
        # t-embedder part : timestp & fps embedding && t_block embedding
        self.t_embedder = TimestepEmbedder(hidden_size=self.config.hidden_size, seed=t_embed_seed)
        self.fps_embedder = SizeEmbedder(hidden_size=self.config.hidden_size, seed=t_embed_seed)
        self.t_block = TblockEmbedder(hidden_size=self.config.hidden_size, chunk_size=6, seed=t_embed_seed)

        # y-embedder part : caption embedding
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=self.config.hidden_size,
            uncond_prob=class_dropout_prob,
            act_layer=approx_gelu,
            token_num=model_max_length,
            seed=t_embed_seed,
        )

        # rotary-embedder part ： RotaryEmbedding
        if self.position_embedding_type == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
            )

        # preprocess embedding follow implemention of open-sora
        if self.pre_process:
            # pos-embedder part : pos2d embedding
            self.pos_embedder = PositionEmbedding2D(self.config.hidden_size)

            # x-embedder part : position embedding
            self.x_embedder = PatchEmbed3D(
                (self.patch_temporal, self.patch_spatial, self.patch_spatial), in_channels, self.config.hidden_size
            )

        # final layer_part
        if self.post_process:
            self.final_layer = T2IFinalLayer(
                self.config.hidden_size,
                np.prod((self.patch_temporal, self.patch_spatial, self.patch_spatial)),
                self.out_channels,
            )

        ## init weight for all layer
        # self.initialize_weights()

    # Todo need to add some params
    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        context_embedding: Tensor,
        context_mask: Tensor = None,
        x_mask: Tensor = None,
        fps: Tensor = None,
        height: Tensor = None,
        width: Tensor = None,
        pos_ids: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        **kwargs,
    ) -> Tensor:
        '''
        Args:
            x : after vae noise_latent
            timesteps : timesteps
            context_embedding : context embedding
            context_mask : for crossattn context_mask with context

        Returns:
            fps, height, weight : for resolution size as some embedding input
        '''

        # ===============
        # get latent & patch size
        # ===============
        T_latent, H_latent, W_latent = self.get_latent_size()  # latent size
        Origin_T, Origin_H, Origin_W = self.get_patchify_size()  # before split in cp
        T = Origin_T  # T temporal block
        H = Origin_H
        W = Origin_W
        Origin_S = Origin_H * Origin_W  # S spatial block

        # ================
        # calculate batch, Origin-thw, cp_split-thw, scale & resolution
        # ================
        # dsp & cp : split in h dim
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size > 1:
            assert H_latent % cp_size == 0, "latent_size must divisibly by cp_size"
            assert H % cp_size == 0, "patchify_size must divisibly by cp_size"
            H_latent = H_latent // cp_size
            H = H // cp_size
        S = H * W

        # =================
        # hidden_state part
        # =================
        if self.pre_process:
            # position embedding calculate
            base_size = round(Origin_S**0.5)
            resolution_sq = (height[0].item() * width[0].item()) ** 0.5  # Todo: maybe need to check it
            scale = resolution_sq / self.input_sq_size  # Todo: maybe need to check it

            pos_emb = self.pos_embedder(x, Origin_H, Origin_W, scale=scale, base_size=base_size)
            x = self.x_embedder(x)  # after emb [Batch, Sequence, Dim]
            x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
            x = x + pos_emb
            x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)
            x_S_B_D = rearrange(x, "B S D -> S B D").contiguous()
        else:
            x_S_B_D = None

        Batch = timesteps.shape[0]
        # ============
        # context part
        # ============
        if self.skip_y_embedder:
            context_embedding_S_B_D = rearrange(context_embedding, "B S D -> S B D").contiguous()
            packed_seq_params = None
            context_mask = None
        else:
            qkv_format = 'sbhd'
            context_embedding = self.encode_text(context_embedding.unsqueeze(1), context_mask, qkv_format)
            context_embedding_S_B_D = rearrange(context_embedding.squeeze(1), "B S D -> S B D")
            packed_seq_params = self.gen_packed_seq_params(Batch, S * T, context_mask, qkv_format)
            context_mask = None

        # =============
        # timestep part
        # =============
        timestep_emb, t = self.encode_timesteps(
            timesteps, fps, Batch, torch.bfloat16
        )  # timestep_emb shape [Batch, Dim]

        # rotary_embedding
        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        if self.position_embedding_type == 'rope':
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params, self.decoder_spatial_temporal, x_S_B_D, self.config
            )
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

        if self.config.sequence_parallel:
            if self.pre_process:
                x_S_B_D = tensor_parallel.scatter_to_sequence_parallel_region(x_S_B_D)
            context_embedding_S_B_D = tensor_parallel.scatter_to_sequence_parallel_region(context_embedding_S_B_D)

            if self.config.clone_scatter_output_in_embedding:
                if self.pre_process:
                    x_S_B_D = x_S_B_D.clone()
                context_embedding_S_B_D = context_embedding_S_B_D.clone()

        # decoder part
        x_S_B_D = self.decoder_spatial_temporal(
            hidden_states=x_S_B_D,
            attention_mask=timestep_emb,
            context=context_embedding_S_B_D,
            context_mask=context_mask,
            rotary_pos_emb=rotary_pos_emb,  # Todo : stditv3 rotary_pos_embedding just in temporal block
            packed_seq_params=packed_seq_params,
        )

        if not self.post_process:
            return x_S_B_D

        if self.config.sequence_parallel:
            x_S_B_D = tensor_parallel.gather_from_sequence_parallel_region(x_S_B_D)

        x_B_S_D = rearrange(x_S_B_D, 'S B D -> B S D')

        # final layer & unpatchify
        x_B_S_D = self.final_layer(x_B_S_D, t, x_mask, t0=None, T=T, S=S)
        x_B_S_D = self.unpatchify(x_B_S_D, T, H, W, T_latent, H_latent, W_latent)

        # Todo : in opensora it cast to float32 for better accuracy
        # x_B_S_D = x_B_S_D.to(torch.float32)

        return x_B_S_D

    def get_latent_size(self, x=None):
        """
        calculate patchify-THW from latent space
        """
        if x == None:
            T, H, W = (self.max_frames, self.max_img_h, self.max_img_w)
        else:
            _, _, T, H, W = x.shape
        return (T, H, W)

    def get_patchify_size(self, x=None):
        """
        calculate patchify-THW from latent space
        """
        if x == None:
            T, H, W = (self.max_frames, self.max_img_h, self.max_img_w)
        else:
            _, _, T, H, W = x.shape
        # patchify padding
        if T % self.patch_temporal != 0:
            T += self.patch_temporal - T % self.patch_temporal
        if H % self.patch_spatial != 0:
            H += self.patch_spatial - H % self.patch_spatial
        if W % self.patch_spatial != 0:
            W += self.patch_spatial - W % self.patch_spatial
        T = T // self.patch_temporal
        H = H // self.patch_spatial
        W = W // self.patch_spatial
        return (T, H, W)

    def encode_timesteps(self, timesteps, fps, batch, type):
        """
        Args:
            timesteps(torch.Tensor) : input tensor shape [B, 1]
            fps(torch.Tensor) : input tensor shape [B, 1]
            batch(int) : batch size
            type : dtype
        Return:
            timestep_emb(torch.Tensor) : output_tensor for decoder

        """
        t = self.t_embedder(timesteps.squeeze(1), dtype=type)  # t shape : [Batch, Dim]
        fps = self.fps_embedder(fps, batch)  # fps shape : [Batch, Dim]
        t = t + fps  # t shape : [Batch, Dim]
        timestep_emb = self.t_block(t)  # timestep_emb shape : [Batch, chunk_size * Dim]
        return timestep_emb, t

    def encode_text(self, y, mask=None, qkv_format='sbhd'):
        """
        encode_text if skip_y_embedder is false
        """
        y = self.y_embedder(y, self.training)  # [B, N_token, C]
        if qkv_format == 'thd':
            if mask is not None:
                if mask.shape[0] != y.shape[0]:
                    mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
                mask = mask.squeeze(1).squeeze(1)
                y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, self.config.hidden_size)
            else:
                y = y.squeeze(1).view(1, -1, self.config.hidden_size)
        return y

    def unpatchify(self, x, Sequence_t, Sequence_h, Sequence_w, Origin_t, Origin_h, Origin_w):
        """
        unpatchify x from final layer output [B, N, C] to video output[B, C_out, T, H, W]
        Args:
            x (torch.Tensor): of shape [B, N, C] with (B (N_t N_h N_w) (T_p H_p W_p C_out))
        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """
        x = rearrange(
            x,
            "B (S_t S_h S_w) (P_t P_h P_w C_out) -> B C_out (S_t P_t) (S_h P_h) (S_w P_w)",
            S_t=Sequence_t,
            S_h=Sequence_h,
            S_w=Sequence_w,
            P_t=self.patch_temporal,
            P_h=self.patch_spatial,
            P_w=self.patch_spatial,
            C_out=self.out_channels,
        )
        x = x[:, :, :Origin_t, :Origin_h, :Origin_w]
        return x

    def gen_packed_seq_params(self, batch_size, q_fiexed_seqlen, context_mask, qkv_format):
        """
        using context_mask generate packed_seqlen_params for cross attention
        using context_mask to solve the kv_format : thd_format
        """
        # increments_list
        cu_seqlens_q_list = torch.arange(0, q_fiexed_seqlen * (batch_size + 1), q_fiexed_seqlen).tolist()
        cu_seqlens_kv_list = torch.cumsum(
            torch.cat((torch.tensor([0]).cuda(), context_mask.sum(dim=1))), dim=0
        ).tolist()

        # cu_seqlens
        cu_seqlens_q = torch.IntTensor(cu_seqlens_q_list).cuda()
        cu_seqlens_kv = torch.IntTensor(cu_seqlens_kv_list).cuda()

        # max_seqlen_set
        seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        max_seqlen_q, _ = seqlens_q.max(dim=0, keepdim=True)
        seqlens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
        max_seqlen_kv, _ = seqlens_kv.max(dim=0, keepdim=True)

        # qkv_format set
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            qkv_format=qkv_format,
        )
        return packed_seq_params

    # open-sora stditv3 init_weight part
    # def initialize_weights(self):
    #     if self.pre_process:
    #         # Initialize x_part weight conv3d
    #         x_embedder_proj = self.x_embedder.proj
    #         nn.init.xavier_uniform_(x_embedder_proj.weight.data.view([x_embedder_proj.weight.data.shape[0], -1]))
    #         nn.init.constant_(x_embedder_proj.bias, 0)

    #     # Initialzie y_part weight
    #     y_embedder_proj = self.y_embedder.y_proj
    #     nn.init.xavier_uniform_(y_embedder_proj[0].weight.data.view(y_embedder_proj[0].weight.data.shape[0], -1))
    #     nn.init.constant_(y_embedder_proj[0].bias, 0)
    #     nn.init.xavier_uniform_(y_embedder_proj[2].weight.data.view(y_embedder_proj[2].weight.data.shape[0], -1))
    #     nn.init.constant_(y_embedder_proj[2].bias, 0)

    #     # Initialize t_part
    #     # Initialize timestep_embedder
    #     timestep_mlp = self.t_embedder.mlp
    #     nn.init.xavier_uniform_(timestep_mlp[0].weight.data.view(timestep_mlp[0].weight.data.shape[0], -1))
    #     nn.init.constant_(timestep_mlp[0].bias, 0)
    #     nn.init.xavier_uniform_(timestep_mlp[2].weight.data.view(timestep_mlp[2].weight.data.shape[0], -1))
    #     nn.init.constant_(timestep_mlp[2].bias, 0)

    #     # Initailize fps_embedder
    #     nn.init.normal_(self.fps_embedder.mlp[0].weight, std=0.02)
    #     nn.init.constant_(self.fps_embedder.mlp[0].bias, 0)
    #     nn.init.constant_(self.fps_embedder.mlp[2].weight, 0)
    #     nn.init.constant_(self.fps_embedder.mlp[2].bias, 0)

    #     # Initialize t_block
    #     nn.init.xavier_uniform_(self.t_block[1].weight.data.view(self.t_block[1].weight.data.shape[0], -1))
    #     nn.init.constant_(self.t_block[1].bias, 0)

    #     if self.post_process:
    #         # Initial final_layer weight & bias
    #         final_linear = self.final_layer.linear
    #         nn.init.xavier_uniform_(final_linear.weight.data.view(final_linear.weight.data.shape[0], -1))
    #         nn.init.constant_(final_linear.bias, 0)

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt/bert'
        self.decoder_spatial_temporal.set_input_tensor(input_tensor[0])

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from typing import List, Optional

from omegaconf import DictConfig, open_dict

from nemo.collections.asr.modules import conformer_encoder, conv_asr
from nemo.collections.asr.parts.submodules import conformer_modules, jasper, multi_head_attention
from nemo.utils import logging


def change_conv_asr_se_context_window(model: 'ASRModel', context_window: int, update_config: bool = True):
    """
    Update the context window of the SqueezeExcitation module if the provided model contains an
    `encoder` which is an instance of `ConvASREncoder`.

    Args:
        model: A subclass of `ASRModel`, itself a subclass of `ModelPT`.
        context_window:  An integer representing the number of input timeframes that will be used
            to compute the context. Each timeframe corresponds to a single window stride of the
            STFT features.

            Say the window_stride = 0.01s, then a context window of 128 represents 128 * 0.01 s
            of context to compute the Squeeze step.
        update_config: Whether to update the config or not with the new context window.
    """
    if update_config and not hasattr(model.cfg, 'encoder'):
        logging.info(
            "Could not change the context window in SqueezeExcite module "
            "since the model provided does not contain an `encoder` module in its config."
        )
        return

    if not isinstance(model.encoder, conv_asr.ConvASREncoder):
        logging.info(
            f"Could not change the context window in SqueezeExcite module "
            f"since the `encoder` module is not an instance of `ConvASREncoder`.\n"
            f"Provided encoder class = {model.encoder.__class__.__name__}"
        )
        return

    enc_cfg = model.cfg.encoder if update_config else None

    if enc_cfg is not None:
        with open_dict(enc_cfg):
            _update_se_context_window(model, context_window, cfg=enc_cfg)
    else:
        _update_se_context_window(model, context_window)

    # Update model config
    if update_config:
        model.cfg.encoder = enc_cfg


def _update_se_context_window(model: 'ASRModel', context_window: int, cfg: Optional[DictConfig] = None):
    jasper_block_counter = -1
    for name, m in model.named_modules():
        if type(m) == jasper.JasperBlock:
            jasper_block_counter += 1

        if type(m) == jasper.MaskedConv1d:
            if m.conv.stride[0] > 1 and 'mconv' in name:
                context_window = context_window // m.conv.stride[0]

        if type(m) == jasper.SqueezeExcite:
            m.change_context_window(context_window=context_window)

            # update config
            if cfg is not None:
                cfg.jasper[jasper_block_counter].se_context_size = context_window


def change_conformer_attention_model(
    model: 'ASRModel', self_attention_model: str, att_context_size: List[int] = None, update_config: bool = True
):
    """
    Update the self_attention_model in a Conformer Encoder,
    which changes the positional encoding and attention layers.

    Args:
        model: A subclass of `ASRModel`, itself a subclass of `ModelPT`.
        self_attention_model (str): type of the attention layer and positional encoding
            'rel_pos': relative positional embedding and Transformer-XL
            'rel_pos_local_attn': relative positional embedding and Transformer-XL with local attention using
                overlapping windows. Attention context is determined by att_context_size parameter.
            'abs_pos': absolute positional embedding and Transformer
        att_context_size (List[int]): List of 2 ints corresponding to left and right attention context sizes,
            or None for full context.
        update_config: Whether to update the config or not with the new attention model
    """

    if update_config and not hasattr(model.cfg, 'encoder'):
        logging.info(
            "Could not change the self_attention_model in Conformer Encoder "
            "since the model provided does not contain an `encoder` module in its config."
        )
        return

    if not isinstance(model.encoder, conformer_encoder.ConformerEncoder):
        logging.info(
            f"Could not change the self_attention_model in Conformer Encoder "
            f"since the `encoder` module is not an instance of `ConformerEncoder`.\n"
            f"Provided encoder class = {model.encoder.__class__.__name__}"
        )
        return

    if att_context_size:
        att_context_size = list(att_context_size)
    else:
        att_context_size = [-1, -1]

    if self_attention_model == 'rel_pos_local_attn':
        model.encoder.att_mask = None

    if self_attention_model == "rel_pos":
        new_pos_enc = multi_head_attention.RelPositionalEncoding(
            d_model=model.cfg.encoder.d_model,
            dropout_rate=model.cfg.encoder.dropout,
            max_len=model.cfg.encoder.pos_emb_max_len,
            xscale=model.encoder.xscale,
            dropout_rate_emb=model.cfg.encoder.dropout_emb,
        )
    elif self_attention_model == 'rel_pos_local_attn':
        new_pos_enc = multi_head_attention.ChunkedRelPositionalEncoding(
            att_context_size=att_context_size,
            d_model=model.cfg.encoder.d_model,
            dropout_rate=model.cfg.encoder.dropout,
            max_len=model.cfg.encoder.pos_emb_max_len,
            xscale=model.encoder.xscale,
            dropout_rate_emb=model.cfg.encoder.dropout_emb,
        )
    elif self_attention_model == "abs_pos":
        new_pos_enc = multi_head_attention.PositionalEncoding(
            d_model=model.cfg.encoder.d_model,
            dropout_rate=model.cfg.encoder.dropout,
            max_len=model.cfg.encoder.pos_emb_max_len,
            xscale=model.encoder.xscale,
        )
    else:
        raise ValueError(f"Not valid self_attention_model: '{self_attention_model}'!")

    new_pos_enc = new_pos_enc.to(device=model.device)
    new_pos_enc.load_state_dict(state_dict=model.encoder.pos_enc.state_dict(), strict=False)
    del model.encoder.pos_enc
    model.encoder.pos_enc = new_pos_enc

    model.encoder.self_attention_model = self_attention_model
    model.encoder.set_max_audio_length(model.encoder.pos_emb_max_len)

    for name, m in model.named_modules():
        if type(m) == conformer_modules.ConformerLayer:

            if self_attention_model == 'rel_pos':
                new_attn = multi_head_attention.RelPositionMultiHeadAttention(
                    n_head=model.cfg.encoder.n_heads,
                    n_feat=model.cfg.encoder.d_model,
                    dropout_rate=model.cfg.encoder.dropout_att,
                    max_cache_len=att_context_size[0],
                    pos_bias_u=None,
                    pos_bias_v=None,
                )
            elif self_attention_model == 'rel_pos_local_attn':
                new_attn = multi_head_attention.RelPositionMultiHeadAttentionLongformer(
                    n_head=model.cfg.encoder.n_heads,
                    n_feat=model.cfg.encoder.d_model,
                    dropout_rate=model.cfg.encoder.dropout_att,
                    max_cache_len=att_context_size[0],
                    att_context_size=att_context_size,
                    pos_bias_u=None,
                    pos_bias_v=None,
                )
            elif self_attention_model == 'abs_pos':
                new_attn = multi_head_attention.MultiHeadAttention(
                    n_head=model.cfg.encoder.n_heads,
                    n_feat=model.cfg.encoder.d_model,
                    dropout_rate=model.cfg.encoder.dropout_att,
                    max_cache_len=att_context_size[0],
                )
            else:
                raise ValueError(
                    f"'{self_attention_model}' is not not a valid value for 'self_attention_model', "
                    f"valid values can be from ['rel_pos', 'rel_pos_local_attn', 'abs_pos']"
                )
            new_attn = new_attn.to(device=model.device)
            new_attn.load_state_dict(m.self_attn.state_dict(), strict=False)
            del m.self_attn
            m.self_attn = new_attn

    if update_config:
        model.cfg.encoder.self_attention_model = self_attention_model
        model.cfg.encoder.att_context_size = att_context_size

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

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed
import torch.nn as nn
from apex.transformer.enums import AttnMaskType, AttnType
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.collections.common.parts.multi_layer_perceptron import MultiLayerPerceptron as MLP
from nemo.collections.nlp.modules.common.megatron.attention import ParallelAttention
from nemo.collections.nlp.modules.common.megatron.utils import (
    build_attention_mask_3d,
    init_method_normal,
    scaled_init_method_normal,
)
from nemo.core import adapter_mixins
from nemo.core.classes import Exportable, NeuralModule
from nemo.core.classes.common import typecheck
from nemo.core.classes.mixins import AccessMixin
from nemo.core.neural_types import AcousticEncodedRepresentation, AudioSignal, LengthsType, NeuralType, SpectrogramType
from nemo.utils import logging

__all__ = ["AudioPerceptionModel"]


class AudioPerceptionModel(NeuralModule, Exportable):
    """Audio perception model with basic modality_adapter (some fc layers)."""

    def input_example(self, max_batch: int = 8, max_dim: int = 32000, min_length: int = 200):
        batch_size = torch.randint(low=1, high=max_batch, size=[1]).item()
        max_length = torch.randint(low=min_length, high=max_dim, size=[1]).item()
        signals = torch.rand(size=[batch_size, max_length]) * 2 - 1
        lengths = torch.randint(low=min_length, high=max_dim, size=[batch_size])
        lengths[0] = max_length
        return signals, lengths, None, None

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        return OrderedDict(
            {
                "input_signal": NeuralType(("B", "T"), AudioSignal(freq=self.preprocessor._sample_rate)),
                "input_signal_length": NeuralType(
                    tuple("B"), LengthsType()
                ),  # Please note that length should be in samples not seconds.
                "processed_signal": NeuralType(("B", "D", "T"), SpectrogramType()),
                "processed_signal_length": NeuralType(tuple("B"), LengthsType()),
            }
        )

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return OrderedDict(
            {
                "encoded": NeuralType(("B", "T", "D"), AcousticEncodedRepresentation()),
                "encoded_len": NeuralType(tuple("B"), LengthsType()),
            }
        )

    def setup_adapter(self, cfg: DictConfig, model):
        if 'adapter' not in cfg:
            return
        # Setup adapters
        with open_dict(cfg.adapter):
            # Extract the name of the adapter (must be give for training)
            adapter_name = 'adapter'
            adapter_type = cfg.adapter.pop("adapter_type")

            # Resolve the config of the specified `adapter_type`
            if adapter_type not in cfg.adapter.keys():
                raise ValueError(
                    f"Adapter type ({adapter_type}) config could not be found. Adapter setup config - \n"
                    f"{OmegaConf.to_yaml(cfg.adapter)}"
                )

            adapter_type_cfg = cfg.adapter[adapter_type]
            print(f"Found `{adapter_type}` config :\n" f"{OmegaConf.to_yaml(adapter_type_cfg)}")

        model.add_adapter(adapter_name, cfg=adapter_type_cfg)
        assert model.is_adapter_available()

        # Disable all other adapters, enable just the current adapter.
        model.set_enabled_adapters(enabled=False)  # disable all adapters prior to training
        model.set_enabled_adapters(adapter_name, enabled=True)  # enable just one adapter by name

        # First, Freeze all the weights of the model (not just encoder, everything)
        model.freeze()
        # Then, Unfreeze just the adapter weights that were enabled above (no part of encoder/decoder/joint/etc)
        model.unfreeze_enabled_adapters()

    def __init__(self, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        if 'adapter' in cfg:
            # Update encoder adapter compatible config
            adapter_metadata = adapter_mixins.get_registered_adapter(cfg.encoder._target_)
            if adapter_metadata is not None:
                cfg.encoder._target_ = adapter_metadata.adapter_class_path
        # Initialize components
        self.preprocessor = self.from_config_dict(cfg.preprocessor)
        overwrite_cfgs = cfg.get("overwrite_cfgs", None)
        if overwrite_cfgs is not None:
            for k, v in overwrite_cfgs.items():
                setattr(cfg.encoder, k, v)
        encoder = self.from_config_dict(cfg.encoder)
        if cfg.get("use_multi_layer_feat", False) and cfg.get("multi_layer_feat", None):
            self.encoder = ConformerMultiLayerFeatureExtractor(cfg=cfg.multi_layer_feat, encoder=encoder)
            if cfg.multi_layer_feat.aggregator.mode == "cat":
                with open_dict(cfg.modality_adapter):
                    cfg.modality_adapter.input_dim = cfg.modality_adapter.input_dim * len(
                        cfg.multi_layer_feat.layer_idx_list
                    )
        else:
            self.encoder = encoder
        if 'spec_augment' in cfg and cfg.spec_augment is not None:
            self.spec_augmentation = self.from_config_dict(cfg.spec_augment)
        else:
            self.spec_augmentation = None
        self.modality_adapter = self.from_config_dict(cfg.modality_adapter)
        if 'output_dim' not in cfg.modality_adapter and "d_model" in cfg.modality_adapter:  # e.g., conformer encoder
            self.proj = nn.Linear(cfg.modality_adapter.d_model, cfg.output_dim)
        else:
            self.proj = nn.Identity()
        self.setup_adapter(cfg, self.encoder)

    def maybe_preprocess_audio(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None,
    ):
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) is False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )
        return processed_signal, processed_signal_length

    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        *args,
        **kwargs,
    ):
        processed_signal, processed_signal_length = self.maybe_preprocess_audio(
            input_signal, input_signal_length, processed_signal, processed_signal_length
        )

        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        encoded, encoded_len = self.modality_adapter(audio_signal=encoded, length=encoded_len)

        # b, c, t -> b, t, c
        encoded = self.proj(encoded.transpose(1, 2))

        return encoded, encoded_len, {}


class Aggregator(nn.Module):
    def __init__(self, cfg: DictConfig, channel_dim: int = 1):
        super().__init__()
        self.mode = cfg.get("mode", "cat")
        self.channel_dim = channel_dim
        self.pooling = cfg.get("pooling", "avg")
        self.rounding = cfg.get("rounding", "floor")

    def _have_same_length(self, encoded_len: List[torch.Tensor]) -> bool:
        sample_len = encoded_len[0]
        for x in encoded_len:
            if torch.sum(x - sample_len) != 0:
                return False
        return True

    def forward(
        self, encoded: List[torch.Tensor], encoded_len: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._have_same_length(encoded_len):
            return self.merge_different_features(encoded, encoded_len)

        if self.mode == "cat":
            return torch.cat(encoded, dim=self.channel_dim), encoded_len[0]
        elif self.mode == "sum":
            return torch([x.unsqueeze(-1) for x in encoded], dim=-1).sum(dim=-1), encoded_len[0]
        elif self.mode == "mean":
            return torch([x.unsqueeze(-1) for x in encoded], dim=-1).mean(dim=-1), encoded_len[0]
        elif self.mode == "max":
            return torch([x.unsqueeze(-1) for x in encoded], dim=-1).max(dim=-1), encoded_len[0]
        elif self.mode == "min":
            return torch([x.unsqueeze(-1) for x in encoded], dim=-1).min(dim=-1), encoded_len[0]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

    def merge_different_features(self, encoded, encoded_len):
        raise NotImplementedError


class ConformerMultiLayerFeatureExtractor(NeuralModule, Exportable, AccessMixin):
    def __init__(self, cfg: DictConfig, encoder: ConformerEncoder):
        super().__init__()
        self.encoder = encoder
        self.layer_idx_list = [int(l) for l in cfg.layer_idx_list]
        for x in self.layer_idx_list:
            if x < 0 or x >= len(encoder.layers):
                raise ValueError(f"layer index {x} out of range [0, {len(encoder.layers)})")
        access_cfg = {
            "interctc": {"capture_layers": self.layer_idx_list,},
            "detach": cfg.get("detach", False),
            "convert_to_cpu": cfg.get("convert_to_cpu", False),
        }
        self.update_access_cfg(access_cfg)
        self.aggregator = Aggregator(cfg.aggregator, channel_dim=1)

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        old_access_flag = self.is_access_enabled()
        self.set_access_enabled(access_enabled=True)

        _ = self.encoder(*args, **kwargs)

        total_registry = {}
        for module_registry in self.get_module_registry(self.encoder).values():
            for key in module_registry:
                if key.startswith("interctc/") and key in total_registry:
                    raise RuntimeError(f"layer {key} has been logged multiple times!")
            total_registry.update(module_registry)

        encoded_list = []
        encoded_len_list = []
        for layer_idx in self.layer_idx_list:
            try:
                layer_outputs = total_registry[f"interctc/layer_output_{layer_idx}"]
                layer_lengths = total_registry[f"interctc/layer_length_{layer_idx}"]
            except KeyError:
                raise RuntimeError(
                    f"Intermediate layer {layer_idx} was not captured! Check the layer index and the number of ConformerEncoder layers."
                )
            if len(layer_outputs) > 1 or len(layer_lengths) > 1:
                raise RuntimeError("Make sure encoder.forward is called exactly one time")
            encoded_list.append(layer_outputs[0])  # [B, D, T]
            encoded_len_list.append(layer_lengths[0])  # [B]

        self.encoder.reset_registry()
        self.set_access_enabled(access_enabled=old_access_flag)

        return self.aggregator(encoded_list, encoded_len_list)


class AmQueryAudioPerceptionModel(AudioPerceptionModel):
    """Audio perception model with extra attention to match LM."""

    def _concat_features(self, embs1, emb1_lens, embs2, emb2_lens):
        concat_emb = []
        concat_len = []
        for emb1, emb1_len, emb2, emb2_len in zip(embs1, emb1_lens, embs2, emb2_lens):
            new_len = emb1_len + emb2_len
            new_emb = torch.concat([emb1[:emb1_len], emb2[:emb2_len]], axis=0)
            padded_new_emb = torch.zeros(emb1.shape[0] + emb2.shape[0], emb1.shape[-1], device=emb1.device)
            padded_new_emb[:new_len, ...] = new_emb
            concat_emb.append(padded_new_emb)
            concat_len.append(new_len)
        concat_emb = torch.stack(concat_emb, dim=0)
        concat_len = torch.stack(concat_len, dim=0)
        return concat_emb, concat_len

    def __init__(self, cfg: DictConfig, pretrained_audio_model: str, llm_tokenizer):
        super(AudioPerceptionModel, self).__init__()
        if pretrained_audio_model.endswith('.nemo'):
            logging.info(f'Loading pretrained audio model from local file: {pretrained_audio_model}')
            self.asr_model = ASRModel.restore_from(pretrained_audio_model, map_location='cpu')
        else:
            logging.info(f'Loading pretrained audio model from NGC: {pretrained_audio_model}')
            self.asr_model = ASRModel.from_pretrained(pretrained_audio_model, map_location='cpu')
        if 'spec_augment' in cfg and cfg.spec_augment is not None:
            self.asr_model.spec_augmentation = self.from_config_dict(cfg.spec_augment)
        else:
            self.asr_model.spec_augmentation = None

        self.preprocessor = self.asr_model.preprocessor
        self.encoder = self.asr_model.encoder
        if cfg.get("use_multi_layer_feat", False) and cfg.get("multi_layer_feat", None):
            self.encoder = ConformerMultiLayerFeatureExtractor(cfg=cfg.multi_layer_feat, encoder=self.encoder)
            if cfg.multi_layer_feat.aggregator.mode == "cat":
                with open_dict(cfg.modality_adapter):
                    cfg.modality_adapter.feat_in = cfg.modality_adapter.feat_in * len(
                        cfg.multi_layer_feat.layer_idx_list
                    )
        else:
            self.encoder = self.encoder
        self.spec_augmentation = self.asr_model.spec_augmentation

        self.modality_adapter = self.from_config_dict(cfg.modality_adapter)
        self.proj = nn.Linear(cfg.modality_adapter.d_model, cfg.output_dim)

        num_layers = 1
        init_method_std = 0.02
        num_attention_heads = 8
        scaled_init_method = scaled_init_method_normal(init_method_std, num_layers)
        init_method = init_method_normal(init_method_std)
        scaled_init_method = scaled_init_method_normal(init_method_std, num_layers)
        self.lm_attention = ParallelAttention(
            init_method=init_method,
            output_layer_init_method=scaled_init_method,
            layer_number=num_layers,
            num_attention_heads=num_attention_heads,
            hidden_size=cfg.output_dim,
            attention_type=AttnType.cross_attn,
            precision=32,
        )
        self.llm_tokenizer = llm_tokenizer
        self.cfg = cfg
        if self.cfg.get('learnable_combine', False):
            self.lm_attention_ratio = nn.Parameter(torch.tensor(0.5))
        if self.cfg.get('consistency_loss_weight', 0.0) > 0.0:
            self.reconstruction_layer = MLP(cfg.output_dim, cfg.output_dim, num_layers, "relu", log_softmax=False)

    def get_am_text_output(self, encoded, logits_len):
        with torch.no_grad():
            is_ctc = self.cfg.get('is_ctc', True)
            if is_ctc:
                decoder_instance = (
                    self.asr_model.ctc_decoder if hasattr(self.asr_model, 'ctc_decoder') else self.asr_model.decoder
                )
                decoding_instance = (
                    self.asr_model.ctc_decoding if hasattr(self.asr_model, 'ctc_decoding') else self.asr_model.decoding
                )
                logits = decoder_instance(encoder_output=encoded)

                current_hypotheses, _ = decoding_instance.ctc_decoder_predictions_tensor(
                    logits, decoder_lengths=logits_len, return_hypotheses=False,
                )
            else:
                decoding_instance = self.asr_model.decoding
                current_hypotheses, _ = decoding_instance.rnnt_decoder_predictions_tensor(
                    encoder_output=encoded, encoded_lengths=logits_len, return_hypotheses=True
                )
                current_hypotheses = [i.text for i in current_hypotheses]
                logits = None
            # TODO: add hypotheses/logits logging
            # logging.info(f"CTC/RNNT hyps: {current_hypotheses[0]}")
            return current_hypotheses, logits

    def get_text_embed(self, inputs, lm_embedding, pad_id=0):
        with torch.no_grad():
            input_ids = self.llm_tokenizer.text_to_ids(inputs)
            input_length = torch.LongTensor([len(x) for x in input_ids]).to(lm_embedding.weight.device)
            max_length = max(input_length)
            input_ids = torch.LongTensor([x + [pad_id] * (max_length - len(x)) for x in input_ids]).to(
                lm_embedding.weight.device
            )
            input_embeds = lm_embedding(input_ids)
            return input_embeds, input_length

    def cross_attend(self, encoded, encoded_len, llm_encoded, llm_encoded_len):
        # TODO(zhehuai): explore causal-ish attention mask
        max_len = encoded.size(1)
        b = encoded.size(0)
        attention_mask = torch.ones(b, 1, max_len, llm_encoded.shape[1], device=encoded.device) < 0.5
        # AM output as query
        attended_encoded, _ = self.lm_attention(
            encoded.transpose(0, 1).contiguous(),
            attention_mask,
            encoder_output=llm_encoded.transpose(0, 1).contiguous(),
        )
        attended_encoded = attended_encoded.transpose(0, 1)
        aux_loss = {}
        loss_func = torch.nn.MSELoss()
        # TODO: consider pad_id
        consistency_loss_weight = self.cfg.get('consistency_loss_weight', 0.0)
        if consistency_loss_weight > 0.0:
            reconstructed_emb = self.reconstruction_layer(attended_encoded)
            aux_loss['consistency_loss'] = loss_func(reconstructed_emb, encoded.detach()) * consistency_loss_weight
        if self.cfg.get('learnable_combine', False):
            attended_encoded = attended_encoded * self.lm_attention_ratio + encoded * (1 - self.lm_attention_ratio)

        return attended_encoded, encoded_len, aux_loss

    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        lm_embedding=None,
        labels=None,
        labels_len=None,
        pad_id=0,
    ):
        processed_signal, processed_signal_length = self.maybe_preprocess_audio(
            input_signal, input_signal_length, processed_signal, processed_signal_length
        )

        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        am_encoded, am_encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        encoded, encoded_len = self.modality_adapter(audio_signal=am_encoded, length=am_encoded_len)
        # b, t, c
        encoded = self.proj(encoded.transpose(1, 2))

        if self.cfg.get("use_multi_layer_feat", False) and self.cfg.get("multi_layer_feat", None):
            am_encoded_asr, am_encoded_len = self.asr_model.encoder(
                audio_signal=processed_signal, length=processed_signal_length
            )
        else:
            am_encoded_asr = am_encoded
        am_hyps_text, log_probs = self.get_am_text_output(am_encoded_asr, am_encoded_len)
        llm_encoded, llm_encoded_len = self.get_text_embed(am_hyps_text, lm_embedding, pad_id=pad_id)
        attend_encoded, encoded_len, aux_loss = self.cross_attend(encoded, encoded_len, llm_encoded, llm_encoded_len)

        asr_loss_weight = self.cfg.get('asr_loss_weight', 0.0)
        if labels is not None and asr_loss_weight > 0.0:
            assert labels_len is not None
            text = self.llm_tokenizer.ids_to_text(labels.tolist())
            text = [x[:-2] for x in text]  # remove end string
            transcript = self.asr_model.tokenizer.text_to_ids(text)
            transcript_len = torch.LongTensor([len(x) for x in transcript]).to(lm_embedding.weight.device)
            max_length = max(transcript_len)
            transcript = torch.LongTensor([x + [pad_id] * (max_length - len(x)) for x in transcript]).to(
                lm_embedding.weight.device
            )
            asr_loss = self.asr_model.loss(
                log_probs=log_probs, targets=transcript, input_lengths=am_encoded_len, target_lengths=transcript_len
            )
            aux_loss['asr_loss'] = asr_loss * asr_loss_weight
        return attend_encoded, encoded_len, aux_loss


class LmQueryAudioPerceptionModel(AmQueryAudioPerceptionModel):
    """Audio perception model with extra attention to match LM."""

    def cross_attend(self, encoded, encoded_len, llm_encoded, llm_encoded_len):
        return super().cross_attend(llm_encoded, llm_encoded_len, encoded, encoded_len)


class CascadedAudioPerceptionModel(AmQueryAudioPerceptionModel):
    """Audio perception model with extra attention to match LM."""

    def cross_attend(self, encoded, encoded_len, llm_encoded, llm_encoded_len):
        return llm_encoded, llm_encoded_len, {}


class ConcatCascadedAudioPerceptionModel(AmQueryAudioPerceptionModel):
    """Audio perception model with extra attention to match LM."""

    def cross_attend(self, encoded, encoded_len, llm_encoded, llm_encoded_len):
        concat_encoded, concat_encoded_len = self._concat_features(encoded, encoded_len, llm_encoded, llm_encoded_len)
        return concat_encoded, concat_encoded_len, {}


class AmAdaptQueryAudioPerceptionModel(AmQueryAudioPerceptionModel):
    """Audio perception model with extra attention to match LM."""

    def __init__(self, cfg: DictConfig, pretrained_audio_model: str, llm_tokenizer):
        super(AudioPerceptionModel, self).__init__()
        if 'adapter' in cfg:
            # Update encoder adapter compatible config
            model_cfg = ASRModel.from_pretrained(pretrained_audio_model, return_config=True)
            adapter_metadata = adapter_mixins.get_registered_adapter(model_cfg.encoder._target_)
            if adapter_metadata is not None:
                model_cfg.encoder._target_ = adapter_metadata.adapter_class_path
            override_config_path = model_cfg
            if 'adapters' not in model_cfg:
                model_cfg.adapters = OmegaConf.create({})
        else:
            override_config_path = None
        if pretrained_audio_model.endswith('.nemo'):
            logging.info(f'Loading pretrained audio model from local file: {pretrained_audio_model}')
            self.asr_model = ASRModel.restore_from(
                pretrained_audio_model, map_location='cpu', override_config_path=override_config_path
            )
        else:
            logging.info(f'Loading pretrained audio model from NGC: {pretrained_audio_model}')
            self.asr_model = ASRModel.from_pretrained(
                pretrained_audio_model, map_location='cpu', override_config_path=override_config_path
            )
        if 'spec_augment' in cfg and cfg.spec_augment is not None:
            self.asr_model.spec_augmentation = self.from_config_dict(cfg.spec_augment)
        else:
            self.asr_model.spec_augmentation = None
        self.preprocessor = self.asr_model.preprocessor
        self.encoder = self.asr_model.encoder
        self.spec_augmentation = self.asr_model.spec_augmentation

        self.modality_adapter = self.from_config_dict(cfg.modality_adapter)
        self.proj = nn.Linear(cfg.modality_adapter.d_model, cfg.output_dim)

        num_layers = 1
        init_method_std = 0.02
        num_attention_heads = 8
        scaled_init_method = scaled_init_method_normal(init_method_std, num_layers)
        init_method = init_method_normal(init_method_std)
        scaled_init_method = scaled_init_method_normal(init_method_std, num_layers)
        self.lm_attention = ParallelAttention(
            init_method=init_method,
            output_layer_init_method=scaled_init_method,
            layer_number=num_layers,
            num_attention_heads=num_attention_heads,
            hidden_size=cfg.output_dim,
            attention_type=AttnType.cross_attn,
            precision=32,
        )
        self.llm_tokenizer = llm_tokenizer
        self.cfg = cfg
        if self.cfg.get('learnable_combine', False):
            self.lm_attention_ratio = nn.Parameter(torch.tensor(0.5))
        if self.cfg.get('consistency_loss_weight', 0.0) > 0.0:
            self.reconstruction_layer = MLP(cfg.output_dim, cfg.output_dim, num_layers, "relu", log_softmax=False)
        self.setup_adapter(cfg, self.encoder)


class ConcatAmQueryAudioPerceptionModel(AmQueryAudioPerceptionModel):
    """Audio perception model with extra attention to match LM."""

    def cross_attend(self, encoded, encoded_len, llm_encoded, llm_encoded_len):

        attended_encoded, encoded_len, aux_los = super().cross_attend(
            encoded, encoded_len, llm_encoded, llm_encoded_len
        )
        concat_encoded, concat_encoded_len = self._concat_features(
            attended_encoded, encoded_len, llm_encoded, llm_encoded_len
        )
        return concat_encoded, concat_encoded_len, aux_los


class ConcatLmQueryAudioPerceptionModel(LmQueryAudioPerceptionModel):
    """Audio perception model with extra attention to match LM."""

    def cross_attend(self, encoded, encoded_len, llm_encoded, llm_encoded_len):

        attended_encoded, encoded_len, aux_los = super().cross_attend(
            encoded, encoded_len, llm_encoded, llm_encoded_len
        )
        concat_encoded, concat_encoded_len = self._concat_features(
            attended_encoded, encoded_len, llm_encoded, llm_encoded_len
        )
        return concat_encoded, concat_encoded_len, aux_los


class AmFixQueryAudioPerceptionModel(AmQueryAudioPerceptionModel):
    """Audio perception model with extra attention to match LM."""

    def __init__(self, cfg: DictConfig, pretrained_audio_model: str, llm_tokenizer):
        super().__init__(cfg, pretrained_audio_model, llm_tokenizer)
        self.encoder = self.from_config_dict(cfg.encoder)
        self.asr_model.freeze()

    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        lm_embedding=None,
        labels=None,
        labels_len=None,
        pad_id=0,
    ):
        processed_signal, processed_signal_length = self.maybe_preprocess_audio(
            input_signal, input_signal_length, processed_signal, processed_signal_length
        )

        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        with torch.no_grad():
            am_encoded_original, am_encoded_len = self.asr_model.encoder(
                audio_signal=processed_signal, length=processed_signal_length
            )
            am_hyps_text, log_probs = self.get_am_text_output(am_encoded_original, am_encoded_len)
        am_encoded, am_encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        encoded, encoded_len = self.modality_adapter(audio_signal=am_encoded, length=am_encoded_len)
        # b, t, c
        encoded = self.proj(encoded.transpose(1, 2))

        llm_encoded, llm_encoded_len = self.get_text_embed(am_hyps_text, lm_embedding, pad_id=pad_id)
        attend_encoded, encoded_len, aux_loss = self.cross_attend(encoded, encoded_len, llm_encoded, llm_encoded_len)

        asr_loss_weight = self.cfg.get('asr_loss_weight', 0.0)
        if labels is not None and asr_loss_weight > 0.0:
            assert labels_len is not None
            text = self.llm_tokenizer.ids_to_text(labels.tolist())
            text = [x[:-2] for x in text]  # remove end string
            transcript = self.asr_model.tokenizer.text_to_ids(text)
            transcript_len = torch.LongTensor([len(x) for x in transcript]).to(lm_embedding.weight.device)
            max_length = max(transcript_len)
            transcript = torch.LongTensor([x + [pad_id] * (max_length - len(x)) for x in transcript]).to(
                lm_embedding.weight.device
            )
            asr_loss = self.asr_model.loss(
                log_probs=log_probs, targets=transcript, input_lengths=am_encoded_len, target_lengths=transcript_len
            )
            aux_loss['asr_loss'] = asr_loss * asr_loss_weight
        return attend_encoded, encoded_len, aux_loss

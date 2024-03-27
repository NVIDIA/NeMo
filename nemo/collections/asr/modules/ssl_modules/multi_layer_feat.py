# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from typing import List, Optional, Tuple

import torch
import torch.distributed
import torch.nn as nn
from omegaconf import DictConfig, open_dict

from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.core.classes import Exportable, NeuralModule
from nemo.core.classes.common import typecheck
from nemo.core.classes.mixins import AccessMixin
from nemo.core.neural_types import AcousticEncodedRepresentation, AudioSignal, LengthsType, NeuralType, SpectrogramType


class Aggregator(nn.Module):
    def __init__(self, cfg: DictConfig, channel_dim: int = 1):
        super().__init__()
        self.mode = cfg.get("mode", "cat")
        self.channel_dim = channel_dim
        self.pooling = cfg.get("pooling", "mean")
        self.weights = cfg.get("weights", None)

    def _forward_for_weighted_pooling(
        self, encoded: List[torch.Tensor], encoded_len: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded_weighted = [encoded[i] * self.weights[i] for i in range(len(encoded))]
        encoded_weighted = torch.sum(torch.stack(encoded_weighted, dim=-1), dim=-1)
        return encoded_weighted, encoded_len[0]

    def forward(
        self, encoded: List[torch.Tensor], encoded_len: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.mode == "cat":
            return torch.cat(encoded, dim=self.channel_dim), encoded_len[0]
        elif self.mode == "sum":
            return torch([x.unsqueeze(-1) for x in encoded], dim=-1).sum(dim=-1), encoded_len[0]
        elif self.mode == "mean" or self.mode == "avg":
            return torch([x.unsqueeze(-1) for x in encoded], dim=-1).mean(dim=-1), encoded_len[0]
        elif self.mode == "max":
            return torch([x.unsqueeze(-1) for x in encoded], dim=-1).max(dim=-1), encoded_len[0]
        elif self.mode == "min":
            return torch([x.unsqueeze(-1) for x in encoded], dim=-1).min(dim=-1), encoded_len[0]
        elif self.mode == "none":
            return encoded, encoded_len
        else:
            raise ValueError(f"Unknown mode {self.mode}")


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
        self.update_access_cfg(access_cfg, guid=getattr(self, "model_guid", None))
        self.aggregator = Aggregator(cfg.aggregator, channel_dim=1)

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        old_access_flag = self.is_access_enabled(guid=getattr(self, "model_guid", None))
        self.set_access_enabled(access_enabled=True, guid=getattr(self, "model_guid", None))

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
        self.set_access_enabled(access_enabled=old_access_flag, guid=getattr(self, "model_guid", None))

        return self.aggregator(encoded_list, encoded_len_list)

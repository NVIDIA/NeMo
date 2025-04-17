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

import copy
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from cosmos1.models.diffusion.diffusion.functional.batch_ops import batch_mul
from cosmos1.utils import log
from cosmos1.utils.lazy_config import instantiate


class BaseConditionEntry(nn.Module):
    def __init__(self):
        super().__init__()

        self._dropout_rate = None
        self._input_key = None
        self._return_dict = False

    @property
    def dropout_rate(self) -> Union[float, torch.Tensor]:
        return self._dropout_rate

    @property
    def input_key(self) -> str:
        return self._input_key

    @property
    def is_return_dict(self) -> bool:
        return self._return_dict

    @dropout_rate.setter
    def dropout_rate(self, value: Union[float, torch.Tensor]):
        self._dropout_rate = value

    @input_key.setter
    def input_key(self, value: str):
        self._input_key = value

    @is_return_dict.setter
    def is_return_dict(self, value: bool):
        self._return_dict = value

    @dropout_rate.deleter
    def dropout_rate(self):
        del self._dropout_rate

    @input_key.deleter
    def input_key(self):
        del self._input_key

    @is_return_dict.deleter
    def is_return_dict(self):
        del self._return_dict

    def random_dropout_input(
        self, in_tensor: torch.Tensor, dropout_rate: Optional[float] = None, key: Optional[str] = None
    ) -> torch.Tensor:
        del key
        dropout_rate = dropout_rate if dropout_rate is not None else self.dropout_rate
        return batch_mul(
            torch.bernoulli((1.0 - dropout_rate) * torch.ones(in_tensor.shape[0])).type_as(in_tensor),
            in_tensor,
        )

    def summary(self) -> str:
        pass


class DataType(Enum):
    IMAGE = "image"
    VIDEO = "video"


class TextAttr(BaseConditionEntry):
    def __init__(self):
        super().__init__()

    def forward(self, token: torch.Tensor, mask: torch.Tensor):
        return {"crossattn_emb": token, "crossattn_mask": mask}

    def random_dropout_input(
        self, in_tensor: torch.Tensor, dropout_rate: Optional[float] = None, key: Optional[str] = None
    ) -> torch.Tensor:
        if key is not None and "mask" in key:
            return in_tensor
        return super().random_dropout_input(in_tensor, dropout_rate, key)


@dataclass
class BaseVideoCondition:
    crossattn_emb: torch.Tensor
    crossattn_mask: torch.Tensor
    data_type: DataType = DataType.VIDEO
    padding_mask: Optional[torch.Tensor] = None
    fps: Optional[torch.Tensor] = None
    num_frames: Optional[torch.Tensor] = None
    image_size: Optional[torch.Tensor] = None
    scalar_feature: Optional[torch.Tensor] = None
    frame_repeat: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, Optional[torch.Tensor]]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class VideoExtendCondition(BaseVideoCondition):
    video_cond_bool: Optional[torch.Tensor] = None  # whether or not it conditioned on video
    gt_latent: Optional[torch.Tensor] = None
    condition_video_indicator: Optional[torch.Tensor] = None  # 1 for condition region

    # condition_video_input_mask will concat to the input of network, along channel dim;
    # Will be concat with the input tensor
    condition_video_input_mask: Optional[torch.Tensor] = None
    # condition_video_augment_sigma: (B, T) tensor of sigma value for the conditional input augmentation, only valid when apply_corruption_to_condition_region is "noise_with_sigma" or "noise_with_sigma_fixed"
    condition_video_augment_sigma: Optional[torch.Tensor] = None


class GeneralConditioner(nn.Module, ABC):
    """
    An abstract module designed to handle various embedding models with conditional and
    unconditional configurations. This abstract base class initializes and manages a collection
    of embedders that can dynamically adjust their dropout rates based on conditioning.

    Attributes:
        KEY2DIM (dict): A mapping from output keys to dimensions used for concatenation.
        embedders (nn.ModuleDict): A dictionary containing all embedded models initialized and
            configured based on the provided configurations.

    Parameters:
        emb_models (Union[List, Any]): A dictionary where keys are embedder names and values
            are configurations for initializing the embedders.

    """

    KEY2DIM = {"crossattn_emb": 1, "crossattn_mask": 1}

    def __init__(self, **emb_models: Union[List, Any]):
        super().__init__()
        self.embedders = nn.ModuleDict()
        for n, (emb_name, embconfig) in enumerate(emb_models.items()):
            embedder = instantiate(embconfig.obj)
            assert isinstance(
                embedder, BaseConditionEntry
            ), f"embedder model {embedder.__class__.__name__} has to inherit from AbstractEmbModel"
            embedder.dropout_rate = getattr(embconfig, "dropout_rate", 0.0)

            if hasattr(embconfig, "input_key"):
                embedder.input_key = embconfig.input_key
            elif hasattr(embconfig, "input_keys"):
                embedder.input_keys = embconfig.input_keys
            else:
                raise KeyError(f"need either 'input_key' or 'input_keys' for embedder {embedder.__class__.__name__}")

            log.debug(f"Initialized embedder #{n}-{emb_name}: \n {embedder.summary()}")
            self.embedders[emb_name] = embedder

    @abstractmethod
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> Any:
        """Should be implemented in subclasses to handle conditon datatype"""
        raise NotImplementedError

    def _forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Processes the input batch through all configured embedders, applying conditional dropout rates if specified.
        Output tensors for each key are concatenated along the dimensions specified in KEY2DIM.

        Parameters:
            batch (Dict): The input data batch to process.
            override_dropout_rate (Optional[Dict[str, float]]): Optional dictionary to override default dropout rates
                                                                per embedder key.

        Returns:
            Dict: A dictionary of output tensors concatenated by specified dimensions.

        Note:
            In case the network code is sensitive to the order of concatenation, you can either control the order via \
            config file or make sure the embedders return a unique key for each output.
        """
        output = defaultdict(list)
        if override_dropout_rate is None:
            override_dropout_rate = {}

        # make sure emb_name in override_dropout_rate is valid
        for emb_name in override_dropout_rate.keys():
            assert emb_name in self.embedders, f"invalid name found {emb_name}"
        for emb_name, embedder in self.embedders.items():
            with torch.no_grad():
                if hasattr(embedder, "input_key") and (embedder.input_key is not None):
                    emb_out = embedder(
                        embedder.random_dropout_input(
                            batch[embedder.input_key], override_dropout_rate.get(emb_name, None)
                        )
                    )
                elif hasattr(embedder, "input_keys"):
                    emb_out = embedder(
                        *[
                            embedder.random_dropout_input(batch[k], override_dropout_rate.get(emb_name, None), k)
                            for k in embedder.input_keys
                        ]
                    )
            for k, v in emb_out.items():
                output[k].append(v)
        # Concatenate the outputs
        return {k: torch.cat(v, dim=self.KEY2DIM.get(k, -1)) for k, v in output.items()}

    def get_condition_uncondition(
        self,
        data_batch: Dict,
    ) -> Tuple[Any, Any]:
        """
        Processes the provided data batch to generate conditioned and unconditioned outputs.

        This method manipulates dropout rates to simulate two scenarios:
        1. All conditions applied (conditioned)
        2. Conditions removed/reduced to minimum (unconditioned)

        This method sets dropout rates to zero for the conditioned scenario to fully apply
        embedders' effects. For unconditioned, it sets rates to 1 (or 0 if initial rate is
        insignificant) to minimize embedder influences.

        Parameters:
            data_batch (Dict): Input data batch containing all necessary information for
                              embedding processing.

        Returns:
            Tuple[Any, Any]: A tuple containing:
                - Outputs with all embedders fully applied (conditioned)
                - Outputs with embedders minimized/not applied (unconditioned)
        """
        cond_dropout_rates, dropout_rates = {}, {}
        for emb_name, embedder in self.embedders.items():
            cond_dropout_rates[emb_name] = 0.0
            dropout_rates[emb_name] = 1.0 if embedder.dropout_rate > 1e-4 else 0.0

        condition: Any = self(data_batch, override_dropout_rate=cond_dropout_rates)
        un_condition: Any = self(data_batch, override_dropout_rate=dropout_rates)
        return condition, un_condition

    def get_condition_with_negative_prompt(
        self,
        data_batch: Dict,
    ) -> Tuple[Any, Any]:
        """
        Similar functionality as get_condition_uncondition
        But use negative prompts for unconditon
        """
        cond_dropout_rates, uncond_dropout_rates = {}, {}
        for emb_name, embedder in self.embedders.items():
            cond_dropout_rates[emb_name] = 0.0
            if isinstance(embedder, TextAttr):
                uncond_dropout_rates[emb_name] = 0.0
            else:
                uncond_dropout_rates[emb_name] = 1.0 if embedder.dropout_rate > 1e-4 else 0.0

        data_batch_neg_prompt = copy.deepcopy(data_batch)
        if "neg_t5_text_embeddings" in data_batch_neg_prompt:
            if isinstance(data_batch_neg_prompt["neg_t5_text_embeddings"], torch.Tensor):
                data_batch_neg_prompt["t5_text_embeddings"] = data_batch_neg_prompt["neg_t5_text_embeddings"]
                data_batch_neg_prompt["t5_text_mask"] = data_batch_neg_prompt["neg_t5_text_mask"]

        condition: Any = self(data_batch, override_dropout_rate=cond_dropout_rates)
        un_condition: Any = self(data_batch_neg_prompt, override_dropout_rate=uncond_dropout_rates)

        return condition, un_condition


@dataclass
class CosmosCondition:
    crossattn_emb: torch.Tensor
    crossattn_mask: torch.Tensor
    padding_mask: Optional[torch.Tensor] = None
    scalar_feature: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, Optional[torch.Tensor]]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


class VideoConditioner(GeneralConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> BaseVideoCondition:
        output = super()._forward(batch, override_dropout_rate)
        return BaseVideoCondition(**output)


class VideoExtendConditioner(GeneralConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> VideoExtendCondition:
        output = super()._forward(batch, override_dropout_rate)
        return VideoExtendCondition(**output)

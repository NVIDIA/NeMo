# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.

# pylint: disable=C0115,C0116,C0301

import copy
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from einops import reduce
from einops.layers.torch import Rearrange
from torch import nn

from nemo.collections.diffusion.sampler.batch_ops import batch_mul

# Utils


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def disabled_train(self: Any, mode: bool = True) -> Any:
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


# TODO: Implement in MCore later
class FourierFeatures(nn.Module):
    """
    Implements a layer that generates Fourier features from input tensors, based on randomly sampled
    frequencies and phases. This can help in learning high-frequency functions in low-dimensional problems.

    [B] -> [B, D]

    Parameters:
        num_channels (int): The number of Fourier features to generate.
        bandwidth (float, optional): The scaling factor for the frequency of the Fourier features. Defaults to 1.
        normalize (bool, optional): If set to True, the outputs are scaled by sqrt(2), usually to normalize
                                    the variance of the features. Defaults to False.

    Example:
        >>> layer = FourierFeatures(num_channels=256, bandwidth=0.5, normalize=True)
        >>> x = torch.randn(10, 256)  # Example input tensor
        >>> output = layer(x)
        >>> print(output.shape)  # Expected shape: (10, 256)
    """

    def __init__(self, num_channels, bandwidth=1, normalize=False):
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * bandwidth * torch.randn(num_channels), persistent=True)
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels), persistent=True)
        self.gain = np.sqrt(2) if normalize else 1

    def forward(self, x, gain: float = 1.0):
        """
        Apply the Fourier feature transformation to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.
            gain (float, optional): An additional gain factor applied during the forward pass. Defaults to 1.

        Returns:
            torch.Tensor: The transformed tensor, with Fourier features applied.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32).ger(self.freqs.to(torch.float32)).add(self.phases.to(torch.float32))
        x = x.cos().mul(self.gain * gain).to(in_dtype)
        return x


# TODO: Switch to MCore implementation later

# ---------------------- Feed Forward Network -----------------------


class FeedForward(nn.Module):
    """
    Transformer FFN with optional gating

    Parameters:
        d_model (int): Dimensionality of input features.
        d_ff (int): Dimensionality of the hidden layer.
        dropout (float, optional): Dropout rate applied after the activation function. Defaults to 0.1.
        activation (callable, optional): The activation function applied after the first linear layer.
                                         Defaults to nn.ReLU().
        is_gated (bool, optional): If set to True, incorporates gating mechanism to the feed-forward layer.
                                   Defaults to False.
        bias (bool, optional): If set to True, adds a bias to the linear layers. Defaults to True.

    Example:
        >>> ff = FeedForward(d_model=512, d_ff=2048)
        >>> x = torch.randn(64, 10, 512)  # Example input tensor
        >>> output = ff(x)
        >>> print(output.shape)  # Expected shape: (64, 10, 512)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation=nn.ReLU(),
        is_gated: bool = False,
        bias: bool = False,
    ):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff, bias=bias)
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.is_gated = is_gated
        if is_gated:
            self.linear_gate = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor):
        g = self.activation(self.layer1(x))
        if self.is_gated:
            x = g * self.linear_gate(x)
        else:
            x = g
        # Apply dropout
        x = self.dropout(x)
        return self.layer2(x)


class SwiGLUFeedForward(FeedForward):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=nn.SiLU(),
            is_gated=True,
            bias=False,
        )


class AbstractEmbModel(nn.Module):
    def __init__(self):
        super().__init__()

        self._is_trainable = None
        self._dropout_rate = None
        self._input_key = None
        # TODO: (qsh 2024-02-14) a cleaner define or we use return dict by default?
        self._return_dict = False

    @property
    def is_trainable(self) -> bool:
        return self._is_trainable

    @property
    def dropout_rate(self) -> Union[float, torch.Tensor]:
        return self._dropout_rate

    @property
    def input_key(self) -> str:
        return self._input_key

    @property
    def is_return_dict(self) -> bool:
        return self._return_dict

    @is_trainable.setter
    def is_trainable(self, value: bool):
        self._is_trainable = value

    @dropout_rate.setter
    def dropout_rate(self, value: Union[float, torch.Tensor]):
        self._dropout_rate = value

    @input_key.setter
    def input_key(self, value: str):
        self._input_key = value

    @is_return_dict.setter
    def is_return_dict(self, value: bool):
        self._return_dict = value

    @is_trainable.deleter
    def is_trainable(self):
        del self._is_trainable

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

    def details(self) -> str:
        return ""

    def summary(self) -> str:
        input_key = self.input_key if self.input_key is not None else getattr(self, "input_keys", None)
        return (
            f"{self.__class__.__name__} \n\tinput key: {input_key}"
            f"\n\tParam count: {count_params(self, False)} \n\tTrainable: {self.is_trainable}"
            f"\n\tDropout rate: {self.dropout_rate}"
            f"\n\t{self.details()}"
        )


class TrainingOnlyEmbModel(AbstractEmbModel):
    """
    Class to denote special case embedding that is
    only used for training, and is dropped out at inference
    """

    def __init__(self):
        super().__init__()


class ReMapkey(AbstractEmbModel):
    def __init__(self, output_key: Optional[str] = None, dtype: Optional[str] = None):
        super().__init__()
        self.output_key = output_key
        self.dtype = {
            None: None,
            "float": torch.float32,
            "bfloat16": torch.bfloat16,
            "half": torch.float16,
            "float16": torch.float16,
            "int": torch.int32,
            "long": torch.int64,
        }[dtype]

    def forward(self, element: torch.Tensor) -> Dict[str, torch.Tensor]:
        key = self.output_key if self.output_key else self.input_key
        if isinstance(element, torch.Tensor):
            element = element.to(dtype=self.dtype)
        return {key: element}

    def details(self) -> str:
        key = self.output_key if self.output_key else self.input_key
        return f"Output key: {key} \n\tDtype: {self.dtype}"


class ScalarEmb(AbstractEmbModel):
    def __init__(
        self,
        num_channels: int,
        num_token_per_scalar_condition: int = 2,
        num_scalar_condition: int = 4,
        output_key: Optional[str] = None,
    ):
        super().__init__()
        self.model_channels = num_channels
        self.num_token_per_scalar_condition = num_token_per_scalar_condition
        self.num_scalar_condition = num_scalar_condition
        self.output_key = output_key
        self.feature_proj = nn.Sequential(
            FourierFeatures(num_channels * num_token_per_scalar_condition),
            Rearrange("b (l c) -> b l c", l=num_token_per_scalar_condition),
            nn.LayerNorm(num_channels),
            SwiGLUFeedForward(num_channels, num_channels, 0.0),
            Rearrange("(b n) l c -> b (n l) c", n=num_scalar_condition),
        )

    def forward(self, element: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = element.shape[0]
        assert (
            torch.numel(element) == batch_size * self.num_scalar_condition
        ), f"element shape {element.shape} does not match with {batch_size}x{self.num_scalar_condition}"
        key = self.output_key if self.output_key else self.input_key
        return {key: self.feature_proj(element.flatten())}

    def details(self) -> str:
        key = self.output_key if self.output_key else self.input_key
        return "\n\t".join(
            [
                f"Output key: {key}",
                f"num_token_per_scalar_condition: {self.num_token_per_scalar_condition}",
                f"num_scalar_condition: {self.num_scalar_condition}",
                f"model_channels: {self.model_channels}",
            ]
        )


class CameraAttr(AbstractEmbModel):
    def __init__(self, context_dim: int, num_pitch: int, num_shot_type: int, num_depth_of_field: int):
        super().__init__()
        self.num_pitch = num_pitch
        self.num_shot_type = num_shot_type
        self.num_depth_of_field = num_depth_of_field

        self.pitch_projection = nn.Embedding(self.num_pitch, context_dim)
        self.shot_type_projection = nn.Embedding(self.num_shot_type, context_dim)
        self.depth_of_field_projection = nn.Embedding(self.num_depth_of_field, context_dim)

    def forward(self, camera_attributes: torch.Tensor) -> Dict[str, torch.Tensor]:
        pitch_emb = self.pitch_projection(camera_attributes[:, 0].unsqueeze(-1).long())
        shot_type_emb = self.shot_type_projection(camera_attributes[:, 1].unsqueeze(-1).long())
        depth_of_field_emb = self.depth_of_field_projection(camera_attributes[:, 2].unsqueeze(-1).long())

        tokens = torch.cat([pitch_emb, shot_type_emb, depth_of_field_emb], dim=1)

        return {
            "crossattn_emb": tokens,
            "crossattn_mask": torch.ones(tokens.shape[0], tokens.shape[1], device=tokens.device, dtype=torch.bool),
        }

    def details(self) -> str:
        return f"Num pitch: {self.num_pitch} \n\tNum shot type: {self.num_shot_type} \n\tNum depth of field: {self.num_depth_of_field} \n\tOutput key: [crossattn_emb, crossattn_mask]"


class HumanAttr(AbstractEmbModel):
    def __init__(self, context_dim, num_race, num_gender, num_age, max_num_humans):
        super().__init__()
        self.num_race = num_race
        self.num_gender = num_gender
        self.num_age = num_age
        self.max_num_humans = max_num_humans

        self.race_projection = nn.Embedding(self.num_race, context_dim)
        self.gender_projection = nn.Embedding(self.num_gender, context_dim)
        self.age_projection = nn.Embedding(self.num_age, context_dim)
        self.num_human_projection = nn.Embedding(self.max_num_humans + 1, context_dim)

        self.human_projection = nn.Linear(3 * context_dim, context_dim, bias=True)

        self.human_bias = nn.Parameter(torch.zeros(5, context_dim))

    def forward(self, human_attributes: torch.Tensor) -> Dict[str, torch.Tensor]:
        num_humans = human_attributes.max(dim=2)[0].bool().sum(dim=1)
        num_human_emb = self.num_human_projection(num_humans.unsqueeze(-1).long())

        race_emb = self.race_projection(human_attributes[:, :, 0].long())
        gender_emb = self.gender_projection(human_attributes[:, :, 1].long())
        age_emb = self.age_projection(human_attributes[:, :, 2].long())

        # TODO: (qsh 2024-02-14) I do not understand the purpose of additional linear layer instead of just concatenation
        human_emb = self.human_projection(
            torch.cat([race_emb, gender_emb, age_emb], dim=-1)
        ) + self.human_bias.unsqueeze(0)

        token = torch.cat([human_emb, num_human_emb], dim=1)

        return {
            "crossattn_emb": token,
            "crossattn_mask": torch.ones(token.shape[0], token.shape[1], device=token.device, dtype=torch.bool),
        }

    def details(self) -> str:
        return f"NumRace : {self.num_race} \n\tNumGender : {self.num_gender} \n\tNumAge : {self.num_age} \n\tMaxNumHumans : {self.max_num_humans} \n\tOutput key: [crossattn_emb, crossattn_mask]"


class PaddingMask(AbstractEmbModel):
    def __init__(self, spatial_reduction: int = 16):
        super().__init__()
        self.spatial_reduction = spatial_reduction

    def forward(self, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "padding_mask": reduce(
                mask, "... (h n) (w m) -> ... h w", "mean", n=self.spatial_reduction, m=self.spatial_reduction
            )
        }

    def details(self) -> str:
        return f"Spatial reduction: {self.spatial_reduction} \n\tOutput key: padding_mask"


class SingleAttr(AbstractEmbModel):
    def __init__(self, context_dim: int, num_label: int):
        super().__init__()
        self.num_label = num_label
        self.emb = nn.Embedding(num_label, context_dim)

    def forward(self, attr: torch.Tensor) -> Dict[str, torch.Tensor]:
        token = self.emb(attr.long())
        return {
            "crossattn_emb": token,
            "crossattn_mask": torch.ones(attr.shape[0], 1, device=attr.device, dtype=torch.bool),
        }

    def details(self) -> str:
        return f"NumLabel : {self.num_label} \n\tOutput key: [crossattn_emb, crossattn_mask]"


class TextAttr(AbstractEmbModel):
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

    def details(self) -> str:
        return "Output key: [crossattn_emb, crossattn_mask]"


class BooleanFlag(AbstractEmbModel):
    def __init__(self, output_key: Optional[str] = None):
        super().__init__()
        self.output_key = output_key

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        del args, kwargs
        key = self.output_key if self.output_key else self.input_key
        return {key: self.flag}

    def random_dropout_input(
        self, in_tensor: torch.Tensor, dropout_rate: Optional[float] = None, key: Optional[str] = None
    ) -> torch.Tensor:
        del key
        dropout_rate = dropout_rate if dropout_rate is not None else self.dropout_rate
        self.flag = torch.bernoulli((1.0 - dropout_rate) * torch.ones(1)).bool().to(device=in_tensor.device)
        return in_tensor

    def details(self) -> str:
        key = self.output_key if self.output_key else self.input_key
        return f"Output key: {key} \n\t This is a boolean flag"


class GeneralConditioner(nn.Module, ABC):
    """
    An abstract module designed to handle various embedding models with conditional and unconditional configurations.
    This abstract base class initializes and manages a collection of embedders that can dynamically adjust
    their dropout rates based on conditioning.

    Attributes:
        KEY2DIM (dict): A mapping from output keys to dimensions used for concatenation.
        embedders (nn.ModuleDict): A dictionary containing all embedded models initialized and configured
                                   based on the provided configurations.

    Parameters:
        emb_models (Union[List, Any]): A dictionary where keys are embedder names and values are configurations
                                       for initializing the embedders.

    Example:
        See Edify4ConditionerConfig
    """

    KEY2DIM = {"crossattn_emb": 1, "crossattn_mask": 1}

    def __init__(self, **emb_models: Union[List, Any]):
        super().__init__()
        self.embedders = nn.ModuleDict()
        for n, (emb_name, embconfig) in enumerate(emb_models.items()):
            embedder = embconfig.obj
            assert isinstance(
                embedder, AbstractEmbModel
            ), f"embedder model {embedder.__class__.__name__} has to inherit from AbstractEmbModel"
            embedder.is_trainable = getattr(embconfig, "is_trainable", True)
            embedder.dropout_rate = getattr(embconfig, "dropout_rate", 0.0)
            if not embedder.is_trainable:
                embedder.train = disabled_train
                for param in embedder.parameters():
                    param.requires_grad = False
                embedder.eval()

            if hasattr(embconfig, "input_key"):
                embedder.input_key = embconfig.input_key
            elif hasattr(embconfig, "input_keys"):
                embedder.input_keys = embconfig.input_keys
            else:
                raise KeyError(f"need either 'input_key' or 'input_keys' for embedder {embedder.__class__.__name__}")

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
            embedding_context = nullcontext if embedder.is_trainable else torch.no_grad
            with embedding_context():
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
        Processes the provided data batch to generate two sets of outputs: conditioned and unconditioned. This method
        manipulates the dropout rates of embedders to simulate two scenarios — one where all conditions are applied
        (conditioned), and one where they are removed or reduced to the minimum (unconditioned).

        This method first sets the dropout rates to zero for the conditioned scenario to fully apply the embedders' effects.
        For the unconditioned scenario, it sets the dropout rates to 1 (or to 0 if the initial unconditional dropout rate
        is insignificant) to minimize the embedders' influences, simulating an unconditioned generation.

        Parameters:
            data_batch (Dict): The input data batch that contains all necessary information for embedding processing. The
                            data is expected to match the required format and keys expected by the embedders.

        Returns:
            Tuple[Any, Any]: A tuple containing two condition:
                - The first one contains the outputs with all embedders fully applied (conditioned outputs).
                - The second one contains the outputs with embedders minimized or not applied (unconditioned outputs).
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
class Edify4Condition:
    crossattn_emb: torch.Tensor
    crossattn_mask: torch.Tensor
    padding_mask: Optional[torch.Tensor] = None
    scalar_feature: Optional[torch.Tensor] = None
    pos_ids: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, Optional[torch.Tensor]]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


class Edify4Conditioner(GeneralConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> Edify4Condition:
        output = super()._forward(batch, override_dropout_rate)
        return Edify4Condition(**output)


class DataType(Enum):
    IMAGE = "image"
    VIDEO = "video"
    MIX = "mix"


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
    trajectory: Optional[torch.Tensor] = None
    frame_repeat: Optional[torch.Tensor] = None

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


@dataclass
class VideoExtendCondition(BaseVideoCondition):
    video_cond_bool: Optional[torch.Tensor] = None  # whether or not it conditioned on video
    gt_latent: Optional[torch.Tensor] = None
    condition_video_indicator: Optional[torch.Tensor] = None  # 1 for condition region
    action_control_condition: Optional[torch.Tensor] = (
        None  # Optional action control embedding input to the V2W model for fine-tuning.
    )

    # condition_video_input_mask will concat to the input of network, along channel dim;
    # Will be concat with the input tensor
    condition_video_input_mask: Optional[torch.Tensor] = None
    # condition_video_augment_sigma: (B, T) tensor of sigma value for the conditional input augmentation, only valid when apply_corruption_to_condition_region is "noise_with_sigma" or "noise_with_sigma_fixed"
    condition_video_augment_sigma: Optional[torch.Tensor] = None
    # pose conditional input, will be concat with the input tensor
    condition_video_pose: Optional[torch.Tensor] = None

    # NOTE(jjennings): All members below can be wrapped into a separate "Config" class

    dropout_rate: float = 0.2
    input_key: str = "fps"  # This is a placeholder, we never use this value
    # Config below are for long video generation only
    compute_loss_for_condition_region: bool = False  # Compute loss for condition region

    # How to sample condition region during training. "first_random_n" set the first n frames to be condition region, n is random, "random" set the condition region to be random,
    condition_location: str = "first_n"
    random_conditon_rate: float = 0.5  # The rate to sample the condition region randomly
    first_random_n_num_condition_t_max: int = (
        4  # The maximum number of frames to sample as condition region, used when condition_location is "first_random_n"
    )
    first_random_n_num_condition_t_min: int = (
        0  # The minimum number of frames to sample as condition region, used when condition_location is "first_random_n"
    )

    # How to dropout value of the conditional input frames
    cfg_unconditional_type: str = (
        "zero_condition_region_condition_mask"  # Unconditional type. "zero_condition_region_condition_mask" set the input to zero for condition region, "noise_x_condition_region" set the input to x_t, same as the base model
    )

    # How to corrupt the condition region
    apply_corruption_to_condition_region: str = (
        "noise_with_sigma_fixed"  # Apply corruption to condition region, option: "gaussian_blur", "noise_with_sigma", "clean" (inference), "noise_with_sigma_fixed" (inference)
    )
    # Inference only option: list of sigma value for the corruption at different chunk id, used when apply_corruption_to_condition_region is "noise_with_sigma" or "noise_with_sigma_fixed"
    # apply_corruption_to_condition_region_sigma_value: [float] = [0.001, 0.2] + [
    #    0.5
    # ] * 10  # Sigma value for the corruption, used when apply_corruption_to_condition_region is "noise_with_sigma_fixed"

    # Add augment_sigma condition to the network
    condition_on_augment_sigma: bool = False
    # The following arguments is to match with previous implementation where we use train sde to sample augment sigma (with adjust video noise turn on)
    augment_sigma_sample_p_mean: float = 0.0  # Mean of the augment sigma
    augment_sigma_sample_p_std: float = 1.0  # Std of the augment sigma
    augment_sigma_sample_multiplier: float = 4.0  # Multipler of augment sigma

    # Add pose condition to the network
    add_pose_condition: bool = False

    # Sample PPP... from IPPP... sequence
    sample_tokens_start_from_p_or_i: bool = False

    # Normalize the input condition latent
    normalize_condition_latent: bool = False


class VideoExtendConditioner(GeneralConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> VideoExtendCondition:
        output = super()._forward(batch, override_dropout_rate)
        return VideoExtendCondition(**output)


@dataclass
class VideoExtendConditionControl(VideoExtendCondition):
    control_input_canny: Optional[torch.Tensor] = None
    control_input_blur: Optional[torch.Tensor] = None
    control_input_canny_blur: Optional[torch.Tensor] = None
    control_input_depth: Optional[torch.Tensor] = None
    control_input_segmentation: Optional[torch.Tensor] = None
    control_input_depth_segmentation: Optional[torch.Tensor] = None
    control_input_mask: Optional[torch.Tensor] = None
    control_input_human_kpts: Optional[torch.Tensor] = None
    control_input_upscale: Optional[torch.Tensor] = None
    control_input_identity: Optional[torch.Tensor] = None
    base_model: Optional[torch.nn.Module] = None
    hint_key: Optional[str] = None
    control_weight: Optional[float] = 1.0


class VideoExtendConditionerControl(GeneralConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> VideoExtendCondition:
        output = super()._forward(batch, override_dropout_rate)
        return VideoExtendConditionControl(**output)

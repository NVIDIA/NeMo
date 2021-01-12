# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class Wav2VecActivationType(Enum):
    relu = 'relu'
    gelu = 'gelu'


class Wav2VecMaskType(Enum):
    """
    Used to select configuration to compute mask lengths
        static = fixed size
        uniform = sample from uniform distribution [mask_other, mask_length*2]
        normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
        poisson = sample from possion distribution with lambda = mask length
    """

    static = auto()
    uniform = auto()
    normal = auto()
    poisson = auto()


class Wav2VecConvExtractorMode(Enum):
    """
    Mode for feature extractor. default has a single group norm with d groups in the first conv block,
    whereas layer_norm has layer norms in every block.
    """

    default = auto()
    layer_norm = auto()


@dataclass
class ConvConfig:
    conv_pos: int = field(default=128, metadata={'help': 'Number of filters for convolutional positional embeddings'})
    conv_pos_groups: int = field(
        default=16, metadata={'help': 'Number of groups for convolutional positional embeddings'}
    )


@dataclass
class Wav2VecTransformerEncoderConfig:
    encoder_layers: int = field(default=12, metadata={'help': 'Number of encoder layers in transformer model'})
    encoder_layerdrop: float = field(default=0.05, metadata={'help': 'Probability of dropping transformer layers'})
    embedding_dim: int = field(default=768, metadata={'help': 'Encoder embedding dim'})
    ffn_embedding_dim: int = field(default=3072, metadata={'help': 'Encoder embedding dim for feed forward'})
    num_attention_heads: int = field(default=8, metadata={'help': 'Number of encoder attention heads'})
    dropout: float = field(default=0.1, metadata={'help': 'Dropout probability for transformer encoder'})
    activation_fn: Wav2VecActivationType = field(
        default=Wav2VecActivationType.gelu, metadata={'help': 'Activation for transformer'}
    )
    layer_norm_first: bool = field(default=False, metadata={'help': 'Apply layer norm first within the transformer'})


@dataclass
class Wav2VecTransformerConfig:
    dropout: float = field(default=0.1, metadata={'help': 'Dropout probability for the transformer'})
    conv: ConvConfig = ConvConfig()
    encoder: Wav2VecTransformerEncoderConfig = Wav2VecTransformerEncoderConfig()


@dataclass
class QuantizerConfig:
    quantize_targets: bool = field(default=True, metadata={'help': 'Use quantized targets'})
    quantize_input: bool = field(default=False, metadata={'help': 'Use quantized inputs'})
    same_quantizer: bool = field(default=False, metadata={'help': 'Use the same quantizer for inputs and targets'})
    latent_vars: int = field(
        default=320, metadata={'help': 'Number of latent variables in each group of the codebook'}
    )
    latent_groups: int = field(default=2, metadata={'help': 'Number of groups within the codebook'})
    latent_dim: int = field(
        default=0,
        metadata={
            'help': 'If greater than 0, use dim for latent variables, else infered by final_dim / latent_groups'
        },
    )
    latent_temp: tuple = field(
        default=(2, 0.5, 0.999995), metadata={'help': 'Quantize temperature (start, stop, decay factor)'}
    )


@dataclass
class ConvFeatureEncoderConfig:
    extractor_mode: Wav2VecConvExtractorMode = field(default=Wav2VecConvExtractorMode.default)
    conv_bias: bool = field(default=False, metadata={'help': 'Include bias in convolution feature extractor model'})
    conv_feature_layers: List = field(
        default_factory=lambda: [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] + [(512, 2, 2)],
        metadata={'help': 'convolutional feature extraction layers [(dim, kernel_size, stride), ...'},
    )


@dataclass
class LossConfig:
    prob_ppl_weight: float = field(default=0.1, metadata={'help': 'Weight applied to quantized prob perplexity loss'})
    feature_loss_weight: float = field(default=0, metadata={'help': 'Weight applied to feature L2 Norm'})


@dataclass
class Wav2VecMaskingConfig:
    mask_prob: float = field(default=0.65, metadata={'help': 'Probability of replacing token with mask'})
    mask_type: Wav2VecMaskType = field(default=Wav2VecMaskType.static,)
    mask_other: int = field(
        default=0,
        metadata={'help': 'Secondary mask used for complex distributions (see help in compute_mask_indices)'},
    )
    mask_length: int = field(default=10, metadata={'help': 'Length of mask when masking time steps'})
    no_mask_overlap: bool = field(default=False, metadata={'help': 'Whether to allow masks to overlap'})
    mask_min_space: int = field(
        default=1, metadata={'help': 'Minimum space beetween spans (if no overlap is enabled)'}
    )
    mask_channel_prob: float = field(default=0, metadata={'help': 'Probability of replacing a feature with 0'})
    mask_channel_type: Wav2VecMaskType = field(default=Wav2VecMaskType.static,)
    mask_channel_other: int = field(
        default=0,
        metadata={
            'help': 'Secondary mask argument (used for more complex distributions (see help in compute_mask_indices)'
        },
    )
    mask_channel_length: int = field(default=10, metadata={'help': 'Length of masks for features (channels)'})
    no_mask_channel_overlap: bool = field(
        default=False, metadata={'help': 'Whether to allow channel masks to overlap'}
    )
    mask_channel_min_space: int = field(
        default=1, metadata={'help': 'Minimum space between spans (if no overlap is enabled)'}
    )


@dataclass
class Wav2VecEncoderModelConfig:
    loss: LossConfig = LossConfig()
    quantizer: QuantizerConfig = QuantizerConfig()
    conv_feature_encoder: ConvFeatureEncoderConfig = ConvFeatureEncoderConfig()
    transformer_encoder: Wav2VecTransformerConfig = Wav2VecTransformerConfig()
    masking: Wav2VecMaskingConfig = Wav2VecMaskingConfig()

    dropout_input: float = field(default=0.1, metadata={'help': 'Dropout applied to input raw features'})
    dropout_features: float = field(
        default=0.1, metadata={'help': 'Dropout applied to the features generator by convolutions'}
    )
    final_dim: int = field(default=0, metadata={'help': 'Project final representations and targets to this dimension'})
    n_negatives: int = field(
        default=100, metadata={'help': 'Number of negatives to sample from the same audio sample'}
    )
    cross_sample_negatives: int = field(
        default=0, metadata={'help': 'Number of negatives to sample from any sample in the batch'}
    )
    codebook_negatives: int = field(default=0, metadata={'help': 'Number of negative examples in codebook'})
    negatives_from_everywhere: bool = field(
        default=False, metadata={'help': 'Sample negatives from everywhere, not just masked states'}
    )
    logit_temp: float = field(default=0.1, metadata={'help': 'Temperature to divide logits by'})
    target_glu: bool = field(default=False, metadata={'help': 'Adds project and applies GLU to targets'})
    feature_grad_mult: float = field(default=0.1, metadata={'help': 'Multiply extracted feature gradients'})

    train_ds: Optional[Dict[Any, Any]] = None
    validation_ds: Optional[Dict[Any, Any]] = None
    test_ds: Optional[Dict[Any, Any]] = None
    optim: Optional[Dict[Any, Any]] = None

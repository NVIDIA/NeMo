from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
from megatron.core import parallel_state as ps
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.inference_params import InferenceParams
from megatron.core.models.multimodal.llava_model import LLaVAModel as MCoreLLaVAModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType as MCoreAttnMaskType
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.transformer_config import TransformerConfig

from nemo.collections.llm.gpt.model import transformer_engine_layer_spec

# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import enum
from typing import Any, Dict, List, Mapping, Optional

import torch
from torch import nn
from torch.nn import init

from nemo.collections.nlp.modules.common.megatron.fused_bias_gelu import fused_bias_gelu
from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults, init_method_normal
from nemo.core.classes import Exportable, NeuralModule
from nemo.core.classes.common import typecheck
from nemo.core.neural_types import ChannelType, NeuralType

try:
    from apex.transformer import tensor_parallel, parallel_state

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

    # fake missing classes with None attributes
    ModelType = AttnMaskType = AttnType = LayerType = ApexGuardDefaults()


__all__ = [
    "PromptEncoder",
    "BIGLSTMPromptEncoder",
    "PromptEncoderType",
    "PromptEncoderMLP",
    "PromptEncoderLinearCombination",
]


class PromptEncoderType(enum.Enum):
    BIGLSTM = 'biglstm'  # LSTM model that works with large language model
    TPMLP = 'tpmlp'  # mlp model that support tensor parallel, better work together with a large language model
    MLP = 'mlp'
    LSTM = 'lstm'
    SIMPLE_LSTM = 'simple_lstm'
    SIMPLE_MLP = 'simple_mlp'
    FROZEN_MLP = 'frozen_mlp'
    FROZEN_EMBEDDING_MLP = 'frozen_embedding_mlp'
    BOTTLENECK_MLP = 'bottleneck_mlp'
    EYE_MLP = 'eye_mlp'
    SIMPLE_EMBEDDING = "simple_embedding"
    SCALED_EMBEDDING = "scaled_embedding"
    LINEAR_COMBINATION = 'linear_combination'
    LINEAR_COMBINATION_BASELINE = 'linear_combination_baseline'


class BIGLSTMPromptEncoder(NeuralModule, Exportable):
    """
    The LSTM prompt encoder network that is used to generate the virtual 
    token embeddings for p-tuning.  It is specially used to work with large language model. 

    To handle large language model, the LSTM only uses hidden_size as its hidden internal dimension, which is independent of LM hidden dimension.
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "taskname_embeddings": NeuralType(('B', 'T', 'C'), ChannelType(), optional=False),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"output_embeds": NeuralType(('B', 'T', 'C'), ChannelType())}

    def __init__(
        self, total_virtual_tokens: int, hidden_size: int, output_size: int, lstm_dropout: float, num_layers: int
    ):
        """
        Initializes the LSTM PromptEncoder module that works with large language model.
        Args:
            total_virtual_tokens: the total number of vitural tokens
            hidden_size: the lstm hidden dimension
            output_size:  the output dimension
            lstm_dropout: lstm dropout rate
            num_layers: number of layers used in the LSTM
        """
        super().__init__()
        self.token_dim = token_dim
        self.input_size = token_dim
        self.output_size = token_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.total_virtual_tokens = total_virtual_tokens
        self.encoder_type = encoder_type

        # Set fixed indicies for forward pass
        self.register_buffer('indices', torch.LongTensor(list(range(self.total_virtual_tokens))))

        # embedding
        self.embedding = torch.nn.Embedding(self.total_virtual_tokens, hidden_size)

        # LSTM
        self.lstm_head = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=self.hidden_size // 2,
            num_layers=num_layers,
            dropout=lstm_dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.mlp_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, output_size)
        )

    @typecheck()
    def forward(self, taskname_embeddings) -> torch.Tensor:
        input_embeds = self.embedding(self.indices).unsqueeze(0)
        batch_size, task_seq_length, _ = taskname_embeddings.shape
        input_embeds = input_embeds.expand(batch_size, self.total_virtual_tokens, self.token_dim).clone()
        length = min(task_seq_length, self.total_virtual_tokens)
        # need to adapt taskname embedding hidden to the same size as hidden_size
        taskname_embeddings = torch.matmul(taskname_embeddings, self.mlp_head[2].weight)
        # Replace general input with task specific embeddings to specify the correct task
        input_embeds[:, 0:length, :] = taskname_embeddings[:, 0:length, :]
        output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0])
        return output_embeds


class PromptEncoderMLP(NeuralModule, Exportable):
    """
    The Tensor Parallel MLP prompt encoder network that is used to generate the virtual 
    token embeddings for p-tuning. It only have two layers.
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "taskname_embeddings": NeuralType(('B', 'T', 'C'), ChannelType(), optional=False),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"output_embeds": NeuralType(('B', 'T', 'C'), ChannelType())}

    def __init__(self, total_virtual_tokens: int, hidden_size: int, output_size: int, init_std: float):
        """
        Initializes the Tensor Model parallel MLP PromptEncoderMLP module.
        Args:
            total_virtual_tokens: the total number of vitural tokens
            hidden_size: hidden dimension
            output_size:  the output dimension
            init_std: the MLP init std value 
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.total_virtual_tokens = total_virtual_tokens
        self.activation = 'gelu'
        sequence_parallel = False
        gradient_accumulation_fusion = False
        # Set fixed indicies for forward pass
        self.register_buffer('indices', torch.LongTensor(list(range(self.total_virtual_tokens))))

        # embedding
        self.embedding = torch.nn.Embedding(self.total_virtual_tokens, output_size)

        no_async_tensor_model_parallel_allreduce = (
            parallel_state.get_tensor_model_parallel_world_size() == 1 or sequence_parallel
        )
        self.first = tensor_parallel.ColumnParallelLinear(
            output_size,
            self.hidden_size,
            gather_output=False,
            init_method=init_method_normal(init_std),
            skip_bias_add=True,
            use_cpu_initialization=False,
            bias=True,
            sequence_parallel_enabled=sequence_parallel,
            no_async_tensor_model_parallel_allreduce=no_async_tensor_model_parallel_allreduce,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
        )
        self.second = tensor_parallel.RowParallelLinear(
            self.hidden_size,
            output_size,
            input_is_parallel=True,
            init_method=init_method_normal(init_std),
            skip_bias_add=True,
            use_cpu_initialization=False,
            bias=True,
            sequence_parallel_enabled=sequence_parallel,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
        )

    @typecheck()
    def forward(self, taskname_embeddings) -> torch.Tensor:
        input_embeds = self.embedding(self.indices).unsqueeze(0)
        batch_size, task_seq_length, _ = taskname_embeddings.shape
        input_embeds = input_embeds.expand(batch_size, self.total_virtual_tokens, self.output_size).clone()
        length = min(task_seq_length, self.total_virtual_tokens)
        # Replace general input with task specific embeddings to specify the correct task
        input_embeds[:, 0:length, :] = taskname_embeddings[:, 0:length, :]
        intermediate_parallel, bias_parallel = self.first(input_embeds)
        intermediate_parallel = fused_bias_gelu(intermediate_parallel, bias_parallel)
        output_embeds, bias_parallel = self.second(intermediate_parallel)
        output_embeds = output_embeds + bias_parallel
        return output_embeds


class PromptEncoder(NeuralModule, Exportable):
    """
    The prompt encoder network that is used to generate the virtual 
    token embeddings for p-tuning.
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "taskname_embeddings": NeuralType(('B', 'T', 'C'), ChannelType(), optional=False),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"output_embeds": NeuralType(('B', 'T', 'C'), ChannelType())}

    def __init__(
        self,
        encoder_type: enum,
        total_virtual_tokens: int,
        token_dim: int,
        hidden_size,
        dropout: float,
        num_layers: int,
        cs_scale: float,
        insert_tasknames: bool,
        max_embedding_norm: Optional[float],
        max_prompt_norm: Optional[float],
        final_layer_norm: bool,
    ):
        """
        Initializes the PromptEncoder module.
        Args:
            total_virtual_tokens: the total number of vitural tokens
            hidden_size: hidden dimension
            lstm_dropout: the dropout used for the LSTM
            num_layers: number of layers used in the LSTM
        """
        super().__init__()
        self.token_dim = token_dim
        self.input_size = token_dim
        self.output_size = token_dim
        self.hidden_size = hidden_size
        self.total_virtual_tokens = total_virtual_tokens
        self.encoder_type = encoder_type
        self.l1_scale = 0.0
        self.l2_scale = 0.0
        self.cs_scale = cs_scale
        self.dropout = torch.nn.Dropout(dropout)
        self.insert_taskname_embeddings = insert_tasknames
        self.final_layer_norm = final_layer_norm

        if max_prompt_norm == "None":
            self.max_prompt_norm = None
        else:
            self.max_prompt_norm = max_prompt_norm

        if max_embedding_norm == "None":
            self.max_embedding_norm = None
        else:
            self.max_embedding_norm = max_embedding_norm

        # Set fixed indicies for forward pass
        self.register_buffer('indices', torch.LongTensor(list(range(self.total_virtual_tokens))))

        # embedding
        self.embedding = torch.nn.Embedding(
            self.total_virtual_tokens, self.token_dim, max_norm=self.max_embedding_norm
        )

        if self.final_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(self.token_dim)

        if self.encoder_type == PromptEncoderType.SIMPLE_EMBEDDING:
            init.xavier_normal(self.embedding.weight.data)
        elif self.encoder_type == PromptEncoderType.SCALED_EMBEDDING:
            init.xavier_normal(self.embedding.weight.data)
            self.scale =  torch.nn.parameter.Parameter(data=20 * (torch.abs(torch.randn(self.total_virtual_tokens, 1))))
        elif self.encoder_type == PromptEncoderType.LSTM:
            # LSTM
            self.lstm_head = torch.nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=True,
                batch_first=True,
            )

            self.mlp_head = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
                nn.ReLU(),
                nn.Linear(self.hidden_size * 2, self.output_size),
            )

        elif self.encoder_type == PromptEncoderType.SIMPLE_LSTM:
            # LSTM
            assert self.hidden_size % 2 == 0
            self.lstm_head = torch.nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size // 2,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=True,
                batch_first=True,
            )
        elif self.encoder_type == PromptEncoderType.SIMPLE_MLP:
            self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.output_size),)
        elif self.encoder_type == PromptEncoderType.FROZEN_MLP:
            self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.output_size, bias=False),)
            self.mlp_head[0].weight.requires_grad = False
        elif self.encoder_type == PromptEncoderType.FROZEN_EMBEDDING_MLP:
            self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.output_size),)
            self.embedding.weight.requires_grad = False

        elif self.encoder_type == PromptEncoderType.BOTTLENECK_MLP:
            self.mlp_head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2), nn.Linear(self.hidden_size // 2, self.output_size),
            )
        elif self.encoder_type == PromptEncoderType.EYE_MLP:
            assert self.hidden_size == self.output_size
            l = nn.Linear(self.hidden_size, self.output_size, bias=True)
            l.weight.data.copy_(torch.eye(self.hidden_size))
            l.weight.requires_grad = False
            l.bias.requires_grad = True
            self.mlp_head = nn.Sequential(l)
        elif self.encoder_type == PromptEncoderType.MLP:
            if num_layers <= 1:
                raise ValueError(
                    "The MLP prompt encoder must have at least 2 layers, and exactly 2 layers is recommended."
                )

            layers = [nn.Linear(self.input_size, self.hidden_size), nn.ReLU()]
            for _ in range(num_layers - 2):
                layers.extend([nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU()])

            layers.append(nn.Linear(self.hidden_size, self.output_size))
            self.mlp_head = nn.Sequential(*layers)

        else:
            raise ValueError("Prompt encoder type not recognized. Please use one of MLP (recommended) or LSTM.")

    @typecheck()
    def forward(self, taskname_embeddings) -> torch.Tensor:
        input_embeds = self.embedding(self.indices).unsqueeze(0)
        batch_size, task_seq_length, _ = taskname_embeddings.shape

        # Replace general input with task specific embeddings to specify the correct task
        if self.insert_taskname_embeddings:
            length = min(task_seq_length, self.total_virtual_tokens)
            input_embeds[:, 0:length, :] = taskname_embeddings[:, 0:length, :]

        if self.encoder_type == PromptEncoderType.LSTM:
            output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0])
        elif self.encoder_type == PromptEncoderType.SIMPLE_LSTM:
            output_embeds = self.lstm_head(input_embeds)[0]
        elif self.encoder_type in [
            PromptEncoderType.MLP,
            PromptEncoderType.SIMPLE_MLP,
            PromptEncoderType.FROZEN_MLP,
            PromptEncoderType.FROZEN_EMBEDDING_MLP,
            PromptEncoderType.BOTTLENECK_MLP,
            PromptEncoderType.EYE_MLP,
        ]:
            output_embeds = self.mlp_head(input_embeds)
        elif self.encoder_type == PromptEncoderType.SIMPLE_EMBEDDING:
            output_embeds = input_embeds
        elif self.encoder_type == PromptEncoderType.SCALED_EMBEDDING:
            output_embeds = input_embeds * self.scale
        else:
            raise ValueError("Prompt encoder type not recognized. Please use one of MLP (recommended) or LSTM.")

        output_embeds = self.apply_max_norm(output_embeds, self.max_prompt_norm)
        if self.final_layer_norm:
            output_embeds = self.layer_norm(output_embeds)
        output_embeds = self.dropout(output_embeds)
        output_embeds = output_embeds.expand(batch_size, -1, -1)
        return output_embeds

    def apply_max_norm(self, x, m):
        if m:
            n = x.norm(dim=2, keepdim=True) * (1.0 / m)
            x = x / n
        return x

    def encoder_reg(self,):
        l1 = torch.zeros(1)
        input_embeds = self.embedding(self.indices).unsqueeze(0)
        if self.encoder_type == PromptEncoderType.LSTM:
            output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0])
        elif self.encoder_type == PromptEncoderType.SIMPLE_LSTM:
            output_embeds = self.lstm_head(input_embeds)[0]
        elif self.encoder_type == PromptEncoderType.MLP:
            output_embeds = self.mlp_head(input_embeds)
        elif self.encoder_type == PromptEncoderType.SIMPLE_MLP:
            output_embeds = self.mlp_head(input_embeds)
        elif self.encoder_type == PromptEncoderType.FROZEN_MLP:
            output_embeds = self.mlp_head(input_embeds)
        elif self.encoder_type == PromptEncoderType.FROZEN_EMBEDDING_MLP:
            output_embeds = self.mlp_head(input_embeds)
        elif self.encoder_type == PromptEncoderType.BOTTLENECK_MLP:
            output_embeds = self.mlp_head(input_embeds)
        elif self.encoder_type == PromptEncoderType.EYE_MLP:
            output_embeds = self.mlp_head(input_embeds)
        elif self.encoder_type == PromptEncoderType.SIMPLE_EMBEDDING:
            output_embeds = input_embeds
        elif self.encoder_type == PromptEncoderType.SCALED_EMBEDDING:
            output_embeds = input_embeds * self.scale

        output_embeds = self.apply_max_norm(output_embeds, 1.0)
        # (10, 2048)
        output_embeds = output_embeds.squeeze(0)
        cs = output_embeds @ output_embeds.transpose(0, 1)
        cs.fill_diagonal_(0.0)
        cs = torch.abs(cs).mean().unsqueeze(0)
        return l1, l1, cs


class PromptEncoderLinearCombination(NeuralModule, Exportable):
    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "taskname_embeddings": NeuralType(('B', 'T', 'C'), ChannelType(), optional=False),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"output_embeds": NeuralType(('B', 'T', 'C'), ChannelType())}

    def __init__(
        self,
        total_virtual_tokens: int,
        original_embeddings: torch.Tensor,
        l1_scale: float,
        l2_scale: float,
        cs_scale: float,
        normalize: bool,
        use_relu: bool,
        normalize_original_embeddings: bool,
        init_val: str,
        spaced_init: str,
        mask_restrict: bool,
        noise_std: float,
    ):
        """
        Initializes the PromptEncoder module.
        Args:
            total_virtual_tokens: the total number of vitural tokens
            hidden_size: hidden dimension
            lstm_dropout: the dropout used for the LSTM
            num_layers: number of layers used in the LSTM
        """
        super().__init__()
        self.total_virtual_tokens = total_virtual_tokens
        self.original_embeddings = original_embeddings

        self.l1_scale = l1_scale
        self.l2_scale = l2_scale
        self.cs_scale = cs_scale
        self.normalize = normalize
        self.normalize_original_embeddings = normalize_original_embeddings
        self.use_relu = use_relu
        self.spaced_init = spaced_init
        self.mask_restrict = mask_restrict
        self.noise_std = noise_std

        assert self.original_embeddings.requires_grad == False
        if self.normalize_original_embeddings:
            self.original_embeddings = self.original_embeddings / self.original_embeddings.norm(dim=1).unsqueeze(1)
        vocab_size, _ = self.original_embeddings.size()

        t = torch.zeros((vocab_size, self.total_virtual_tokens))
        if self.use_relu:
            init_val = 'avg'
        if init_val == 'one':
            self.init_val = 1.0
        elif init_val == 'zero':
            self.init_val = 0.0
        else:
            self.init_val = 1.0 / vocab_size

        t = t + self.init_val
        self.linear_combination = torch.nn.parameter.Parameter(data=t)
        self.scale =  torch.nn.parameter.Parameter(data=20 * (1+torch.abs(torch.randn(self.total_virtual_tokens, 1))))
        self.cos = torch.nn.CosineSimilarity(dim=0)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        for k, v in state_dict.items():
            if k == 'linear_combination':
                self.linear_combination = torch.nn.parameter.Parameter(data=v)
            else:
                raise RuntimeError(f"Linear Combination cant parse key {k} in state_dict")
        super().load_state_dict(state_dict, strict)
        return True

    def encoder_reg(self,):
        w = self.linear_combination
        if self.use_relu:
            w = torch.nn.functional.gelu(w)
        if self.normalize:
            w = w / w.sum(dim=0)

        l2 = (w ** 2).sum(dim=0)
        l1 = torch.abs(w).sum(dim=0)

        selected_original_embeddings = self.original_embeddings

        output_embeds = selected_original_embeddings.transpose(0, 1) @ w
        output_embeds = output_embeds.transpose(0, 1)  # (num_virtual_tokens, embedding_size)
        output_embeds_norm = output_embeds.norm(dim=1).unsqueeze(1) + 1e-4
        output_embeds = output_embeds / output_embeds_norm
        cs = output_embeds @ output_embeds.transpose(0, 1)
        cs.fill_diagonal_(0.0)
        cs = torch.abs(cs).mean()
        return l1.mean().unsqueeze(0), l2.mean().unsqueeze(0), cs.unsqueeze(0)

    @typecheck()
    def forward(self, taskname_embeddings) -> torch.Tensor:
        batch_size, _, _ = taskname_embeddings.shape
        w = self.linear_combination
        if self.use_relu:
            w = torch.nn.functional.gelu(w)

        if self.normalize:
            w = w / w.sum(dim=0)

        selected_original_embeddings = self.original_embeddings

        if self.noise_std == 0.0:
            output_embeds = selected_original_embeddings.transpose(0, 1) @ w
        else:
            _n = (torch.randn_like(selected_original_embeddings) * self.noise_std) + selected_original_embeddings
            output_embeds = _n.transpose(0, 1) @ w

        output_embeds = output_embeds.transpose(0, 1)  # (num_virtual_tokens, embedding_size)
        output_embeds *= self.scale
        output_embeds = output_embeds.expand(
            batch_size, output_embeds.size(0), output_embeds.size(1)
        )  # (batch, num_virtual_tokens, embed_size)
        return output_embeds


class PromptEncoderLinearCombinationBaseline(NeuralModule, Exportable):
    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "taskname_embeddings": NeuralType(('B', 'T', 'C'), ChannelType(), optional=False),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"output_embeds": NeuralType(('B', 'T', 'C'), ChannelType())}

    def __init__(
        self, total_virtual_tokens: int, original_embeddings: torch.Tensor, cs_scale: float, top_tokens: List[int]
    ):
        """
        Initializes the PromptEncoder module.
        Args:
            total_virtual_tokens: the total number of vitural tokens
            hidden_size: hidden dimension
            lstm_dropout: the dropout used for the LSTM
            num_layers: number of layers used in the LSTM
        """
        super().__init__()
        self.total_virtual_tokens = total_virtual_tokens
        self.original_embeddings = original_embeddings
        vocab_size, embedding_dim = self.original_embeddings.size()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.l1_scale = 0.0
        self.l2_scale = 0.0
        self.cs_scale = cs_scale
        if top_tokens is not None:
            self.original_embeddings = original_embeddings[top_tokens, :]
        else:
            self.original_embeddings = original_embeddings

        assert self.original_embeddings.requires_grad == False
        # self.linear_combination = torch.nn.Linear(1, self.total_virtual_tokens * embedding_dim)
        self.linear_combination = torch.nn.Embedding(self.total_virtual_tokens, embedding_dim)
        t = self.original_embeddings.data[: self.total_virtual_tokens, :]
        # self.linear_combination.weight.data = torch.ones((self.linear_combination.weight.shape))
        self.linear_combination.weight.data = t.clone().detach().float()
        self.idx = torch.nn.parameter.Parameter(data=torch.arange(self.total_virtual_tokens), requires_grad=False)

    def encoder_reg(self,):
        l1 = torch.zeros(1).type_as(self.linear_combination.weight)
        output_embeds = self.linear_combination(self.idx)
        output_embeds_norm = output_embeds.norm(dim=1).unsqueeze(1) + 1e-8
        output_embeds = output_embeds / output_embeds_norm
        cs = output_embeds @ output_embeds.transpose(0, 1)
        cs.fill_diagonal_(0.0)
        cs = torch.abs(cs).mean().unsqueeze(0)
        return l1, l1, cs

    @typecheck()
    def forward(self, taskname_embeddings) -> torch.Tensor:
        batch_size, _, _ = taskname_embeddings.shape
        output_embeds = self.linear_combination(self.idx)
        output_embeds = output_embeds.unsqueeze(0)
        output_embeds = output_embeds.expand(
            batch_size, output_embeds.size(1), output_embeds.size(2)
        )  # (batch, num_virtual_tokens, embed_size)
        return output_embeds

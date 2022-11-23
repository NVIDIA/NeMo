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
from typing import Dict, Optional

import torch
from torch import nn

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
        lstm_dropout: float,
        num_layers: int,
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

        # Set fixed indicies for forward pass
        self.register_buffer('indices', torch.LongTensor(list(range(self.total_virtual_tokens))))

        # embedding
        self.embedding = torch.nn.Embedding(self.total_virtual_tokens, self.token_dim)

        if self.encoder_type == PromptEncoderType.LSTM:
            # LSTM
            self.lstm_head = torch.nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=num_layers,
                dropout=lstm_dropout,
                bidirectional=True,
                batch_first=True,
            )

            self.mlp_head = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
                nn.ReLU(),
                nn.Linear(self.hidden_size * 2, self.output_size),
            )

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
        input_embeds = input_embeds.expand(batch_size, self.total_virtual_tokens, self.token_dim).clone()
        length = min(task_seq_length, self.total_virtual_tokens)

        # Replace general input with task specific embeddings to specify the correct task
        input_embeds[:, 0:length, :] = taskname_embeddings[:, 0:length, :]

        if self.encoder_type == PromptEncoderType.LSTM:
            output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0])
        elif self.encoder_type == PromptEncoderType.MLP:
            output_embeds = self.mlp_head(input_embeds)
        else:
            raise ValueError("Prompt encoder type not recognized. Please use one of MLP (recommended) or LSTM.")

        return output_embeds


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
        limit_vocab: int,
        normalize: bool,
        use_relu: bool,
        init_val: str,
        spaced_init: str,
        mask_restrict: bool,
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
        if limit_vocab > -1:
            self.original_embeddings = original_embeddings[:limit_vocab, :]
        else:
            self.original_embeddings = original_embeddings
        self.l1_scale = l1_scale
        self.l2_scale = l2_scale
        self.normalize = normalize
        self.use_relu = use_relu
        self.spaced_init = spaced_init
        self.mask_restrict = mask_restrict

        assert self.original_embeddings.requires_grad == False
        vocab_size, _ = self.original_embeddings.size()
        group_size = vocab_size // self.total_virtual_tokens
        t = torch.zeros((vocab_size, self.total_virtual_tokens))
        
        if init_val == 'group':
            self.init_val = 1.0 / group_size
        elif init_val == 'one':
            self.init_val = 1.0
        else:
            self.init_val = 0.0
        

        for i in range(self.total_virtual_tokens):
            if self.spaced_init:
                t[torch.arange(i, vocab_size, group_size), i] = self.init_val
            else:
                t[i * group_size : (i + 1) * group_size, i] = self.init_val

        m = torch.ones_like(t)
        if self.mask_restrict:
            m = (t > 0.0).int()
        self.linear_combination_mask = torch.nn.parameter.Parameter(data=m, requires_grad=False)
        self.linear_combination = torch.nn.parameter.Parameter(data=t)

    def encoder_reg(self,):
        w = self.linear_combination * self.linear_combination_mask
        if self.use_relu:
            w = torch.nn.functional.relu(w)
        l2 = (w ** 2).sum(dim=0)
        l1 = torch.abs(w).sum(dim=0)
        return l1.mean().unsqueeze(0), l2.mean().unsqueeze(0)

    @typecheck()
    def forward(self, taskname_embeddings) -> torch.Tensor:
        batch_size, _, _ = taskname_embeddings.shape
        w = self.linear_combination * self.linear_combination_mask
        if self.use_relu:
            w = torch.nn.functional.relu(w)

        if self.normalize:
            w = w / w.sum(dim=0)

        output_embeds = self.original_embeddings.transpose(0, 1) @ w
        output_embeds = output_embeds.transpose(0, 1)  # (num_virtual_tokens, embedding_size)
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
        self, total_virtual_tokens: int, original_embeddings: torch.Tensor,
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

        assert self.original_embeddings.requires_grad == False
        self.linear_combination = torch.nn.Linear(1, self.total_virtual_tokens * embedding_dim)
        # torch.nn.Embedding(self.total_virtual_tokens, embedding_dim)
        t = self.original_embeddings.data[: self.total_virtual_tokens, :].reshape(
            self.total_virtual_tokens * embedding_dim, 1
        )
        # self.linear_combination.weight.data = torch.randn((self.linear_combination.weight.shape), dtype=torch.float16)
        self.linear_combination.weight.data = t.clone().detach().float()
        self.idx = torch.nn.parameter.Parameter(data=torch.ones(1), requires_grad=False)

    def encoder_reg(self,):
        l1 = torch.zeros(1).type_as(self.linear_combination.weight)
        l2 = torch.zeros(1).type_as(self.linear_combination.weight)
        return l1, l2

    @typecheck()
    def forward(self, taskname_embeddings) -> torch.Tensor:
        batch_size, _, _ = taskname_embeddings.shape
        output_embeds = self.linear_combination(self.idx).reshape(1, self.total_virtual_tokens, self.embedding_dim)
        output_embeds = output_embeds.expand(
            batch_size, output_embeds.size(1), output_embeds.size(2)
        )  # (batch, num_virtual_tokens, embed_size)
        return output_embeds

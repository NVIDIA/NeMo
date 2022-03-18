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

import math
import torch
import torch.nn as nn
import torch.nn.init as init

from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.utils import init_method_normal
from nemo.core.classes import Exportable, NeuralModule #TODO: See if I need to add these
ModelPT

try:
    from apex.transformer import tensor_parallel
    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

__all__ = ['PromptTable']

class PromptTable(ModelPT, Exportable):
    def __init__(
        self, total_soft_tokens, hidden_size,
    ):
        super().__init__()

        self.total_soft_tokens = total_soft_tokens
        self.hidden_size = hidden_size
        self.prompt_table = torch.nn.ModuleDict()
        self.taskname_id_to_name = {}

    def forward(self, taskname_id):
        taskname_id = taskname_id.item()
        taskname = self.taskname_id_to_name[taskname_id]
        return self.prompt_table[taskname]()

    def remove_prompt(self, taskname):
        if taskname not in prompt_table:
            return

        # find the taskname_id assocaited with the tag to delete
        taskname_id = None
        for key, value in taskname_id_to_name.items():
            if value == taskname:
                taskname_id = key
                break

        del self.taskname_id_to_name[taskname_id]
        del self.prompt_table[taskname]

    def init_prompt_from_random(self, taskname):
        """Add new soft prompt to be tuned.
           Intialize prompt weights using pytorch init method

        """
        # Initalize prompt embeddings from a pytorch random init method
        self.prompt_table[taskname] = PromptEmbedding(
            init_from_prompt_text=False, hidden_size=self.hidden_size, total_soft_tokens=self.total_soft_tokens,
        )

        self.taskname_id_to_name[taskname_id] = taskname

    def init_prompt_from_text(self, taskname, init_token_ids, word_embeddings):
        """Add new soft prompt to be tuned.
           Intialize prompt weights from existing embeddings from specific vocab tokens.

        """
        # Trim or iterate until num_text_tokens matches total_soft_tokens
        num_text_tokens = len(init_token_ids)
        total_soft_tokens = self.total_soft_tokens

        if num_text_tokens > total_soft_tokens:
            init_token_ids = init_token_ids[:total_soft_tokens]
        elif num_text_tokens < total_soft_tokens:
            num_reps = math.ceil(total_soft_tokens / num_text_tokens)
            init_token_ids = init_token_ids * num_reps

        # Set dictionary item keys and datatypes for broadcasting
        keys = ['text']
        datatype = torch.int64

        # Broadcast int ids across gpus for tensor parallel
        init_token_ids = init_token_ids[:total_soft_tokens]
        init_token_ids = {'text': torch.tensor(init_token_ids, dtype=torch.int64)}
        init_token_ids_b = tensor_parallel.broadcast_data(keys, init_token_ids, datatype)
        init_token_ids = init_token_ids_b['text'].long()
        init_position_ids = torch.arange(self.total_soft_tokens, dtype=torch.long, device=init_token_ids.device)

        # Use a copy of token embedding weights to initalize the prompt embeddings
        word_embedding_weights = word_embeddings(init_token_ids).detach().clone()

        self.prompt_table[taskname] = PromptEmbedding(
            init_from_prompt_text=True,
            hidden_size=self.hidden_size,
            total_soft_tokens=self.total_soft_tokens,
            word_embedding_weights=word_embedding_weights,
        )

        self.taskname_id_to_name[taskname_id] = taskname

    def init_prompt_from_p_tuning_encoder(self):
        pass

class PromptEmbedding(MegatronModule):
    """Prompt embeddings

    Arugments:
        init_from_prompt_text: Whether to intialize prompt embeddings
                               from from certain lm embeddings
                               corresponding to a prompt string
        hidden_size: hidden size should match lm embedding size
        total_soft_tokens: length of prompt initalized from torch init method
        word_embedding_weights: token embedding vectors for text init option
        init_method: pytorch init method
        prompt_embedding_dropout_prob: dropout probablity
    """

    def __init__(
        self,
        init_from_prompt_text,
        hidden_size,
        total_soft_tokens,
        word_embedding_weights=None,
        init_method=init.xavier_normal_,
        prompt_embedding_dropout_prob=0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.total_soft_tokens = total_soft_tokens

        # Randomly init token and position embeddings
        self.prompt_embeddings = torch.nn.Embedding(self.total_soft_tokens, self.hidden_size)
        init_method(self.prompt_embeddings.weight)

        # Set embedding weights to be embeddings from prompt tokens
        if init_from_prompt_text:
            self.prompt_embeddings.weight = nn.Parameter(word_embedding_weights)

        # Set fixed indicies for forward pass
        self.register_buffer('indices', torch.LongTensor(list(range(self.total_soft_tokens))))
        self.embedding_dropout = torch.nn.Dropout(prompt_embedding_dropout_prob)

    def forward(self):
        # Just get embeddings and dropout
        prompt_embeddings = self.prompt_embeddings(seq_indices)
        prompt_embeddings = self.embedding_dropout(prompt_embeddings)

        return prompt_embeddings

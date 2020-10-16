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

from dataclasses import dataclass, field
from typing import Any, List

import hydra
import torch

from nemo.collections.cv.losses import NLLLoss
from nemo.collections.cv.models.model import Model
from nemo.collections.vis.modules import SentenceEmbeddingsConfig
from nemo.collections.vis.transforms import Compose
from nemo.core.neural_types import *


@dataclass
class QTypeConfig:
    """
    Structured config for the QType model.

    For more details please refer to:
    https://cs.stanford.edu/people/jcjohns/clevr/

    Args:
        _target_: Specification of target class
    """

    question_transforms: List[Any] = field(default_factory=list)
    answer_transforms: List[Any] = field(default_factory=list)
    embeddings: SentenceEmbeddingsConfig = SentenceEmbeddingsConfig(
        word_mappings_filepath="", embeddings_size=50,
    )
    hidden_size: int = 100
    num_layers: int = 1
    batch_first: bool = True
    bidirectional: bool = True
    # Target class name.
    _target_: str = "nemo.collections.vis.models.QType"


class QType(Model):
    """
    The VQA model providing answers by relying only questions.
    """

    def __init__(self, cfg: QTypeConfig):
        super().__init__(cfg=cfg)

        # Instantiate question transforms.
        if cfg.question_transforms is not None:
            self._q_transforms = Compose([hydra.utils.instantiate(trans) for trans in cfg.question_transforms])
        else:
            self._q_transforms = Compose([])

        # Instantiate answer transforms.
        if cfg.question_transforms is not None:
            self._a_transforms = Compose([hydra.utils.instantiate(trans) for trans in cfg.answer_transforms])
        else:
            self._a_transforms = Compose([])

        # Sentence embeddings.
        self._se = hydra.utils.instantiate(cfg.embeddings)
        # Question encoder.
        self.eq = torch.nn.LSTM(
            input_size=cfg.embeddings.embeddings_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=cfg.batch_first,
            bidirectional=cfg.bidirectional,
        )
        # Initialize c0 and h0.
        num_directions = 2 if cfg.bidirectional else 1
        self.h0 = torch.zeros(cfg.num_layers * num_directions, 1, cfg.hidden_size)  # 1 is batch size.
        self.c0 = torch.zeros(cfg.num_layers * num_directions, 1, cfg.hidden_size)

        # Activation applied to output.
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

        # Loss function.
        self.loss = NLLLoss()

    def forward(self, questions: List[List[str]]):
        """ Propagates data throught the model. """
        # Get batch size.
        batch_size = len(questions)
        dim0 = self.h0.shape[0]
        dim2 = self.h0.shape[2]
        init_hidden = self.h0.expand(dim0, batch_size, dim2).contiguous()
        init_memory = self.c0.expand(dim0, batch_size, dim2).contiguous()

        # Process sentences using provided transformations.
        questions = self._q_transforms(questions)

        # Embed words.
        embs = self._se(questions)

        # Encode question.
        activations, (_, _) = self.eq(embs, (init_hidden, init_memory))

        # Return the last output.
        return activations.contiguous()[:, -1, :].squeeze(1)

    def training_step(self, batch, what_is_this_input):
        """ Training step, calculate loss. """
        # "Unpack" the batch.
        indices, img_ids, imgs, questions, answers, question_type_ids, question_type_names = batch

        # Get predictions.
        predictions = self(questions=questions)

        log_preds = self.log_softmax(predictions)

        # Get targets.
        targets = self._a_transforms(answers)

        # Calculate loss.
        loss = self.loss(predictions=log_preds, targets=targets)

        # Return it.
        return {"loss": loss}

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

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.asr.losses.lattice_losses import LatticeLoss
from nemo.collections.asr.modules.graph_decoder import ViterbiDecoderWithGraph
from nemo.core.classes.common import typecheck
from nemo.utils import logging

# use k2 import guard
# fmt: off
from nemo.core.utils.k2_utils import k2_import_guard # isort:skip
k2_import_guard()
import k2 # isort:skip
# fmt: on


@dataclass
class GraphIntersectDenseConfig:
    """Graph dense intersection config.
    """

    search_beam: float = 20.0
    output_beam: float = 10.0
    min_active_states: int = 30
    max_active_states: int = 10000


@dataclass
class GraphModuleConfig:
    """Config for graph modules.
    Typically used with graph losses and decoders.
    """

    topo_type: str = "default"
    topo_with_self_loops: bool = True
    graph_type: str = "topo"
    loss_type: str = "mmi"
    token_lm: Optional[Any] = None
    intersect_pruned: bool = False
    intersect_conf: GraphIntersectDenseConfig = GraphIntersectDenseConfig()
    boost_coeff: float = 0.0


class ASRK2Mixin(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # use k2 import guard
        k2_import_guard()

        super().__init__(cfg=cfg, trainer=trainer)

        self.graph_module_cfg = self._cfg.graph_module_cfg

        # register token_lm for MAPLoss
        criterion_type = self.graph_module_cfg.get("criterion_type", "ml")
        self.use_graph_lm = criterion_type == "map"
        if self.use_graph_lm:
            token_lm_path = self.graph_module_cfg.background_cfg.get("token_lm", None)
            if token_lm_path == None:
                raise ValueError(
                    f"graph_module_cfg.background_cfg.token_lm is empty. It must be set for criterion_type == `{criterion_type}`"
                )
            token_lm_path = self.register_artifact('graph_module_cfg.background_cfg.token_lm', token_lm_path)
            self.graph_module_cfg.background_cfg["token_lm"] = token_lm_path

        self.update_k2_modules(self.graph_module_cfg)

    def update_k2_modules(self, input_cfg):
        """
        Helper function to initialize or update k2 loss and transcribe_decoder.
        """
        del self.loss
        if hasattr(self, "transcribe_decoder"):
            del self.transcribe_decoder

        self.loss = LatticeLoss(
            num_classes=self.decoder.num_classes_with_blank - 1,
            reduction=self._cfg.get("ctc_reduction", "mean_batch"),
            backend="k2",
            criterion_type=input_cfg.get("criterion_type", "ml"),
            split_batch_size=input_cfg.get("split_batch_size", 0),
            graph_module_cfg=input_cfg.background_cfg,
        )
        remove_consecutive = input_cfg.background_cfg.get(
            "topo_with_self_loops", True
        ) and input_cfg.background_cfg.get("topo_type", "default") not in ["forced_blank", "identity",]
        self._wer.remove_consecutive = remove_consecutive

        criterion_type = self.loss.criterion_type
        self.use_graph_lm = criterion_type == "map"
        transcribe_training = input_cfg.get("transcribe_training", False)
        if transcribe_training and criterion_type == "ml":
            logging.warning(
                f"""You do not need to use transcribe_training=`{transcribe_training}` 
                            with criterion_type=`{criterion_type}`. transcribe_training will be set to False."""
            )
            transcribe_training = False
        self.transcribe_training = transcribe_training
        if self.use_graph_lm:
            self.transcribe_decoder = ViterbiDecoderWithGraph(
                num_classes=self.decoder.num_classes_with_blank - 1,
                backend="k2",
                dec_type="tokenlm",
                return_type="1best",
                return_ilabels=True,
                output_aligned=True,
                split_batch_size=input_cfg.get("split_batch_size", 0),
                graph_module_cfg=input_cfg.background_cfg,
            )

    @typecheck()
    def forward(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None,
    ):
        """
        Forward pass of the model.

        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            processed_signal: Tensor that represents a batch of processed audio signals,
                of shape (B, D, T) that has undergone processing via some DALI preprocessor.
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.

        Returns:
            A tuple of 3 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
            3) The greedy token predictions of the model of shape [B, T] (via argmax)
        """
        log_probs, encoded_len, greedy_predictions = super().forward(
            input_signal=input_signal,
            input_signal_length=input_signal_length,
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
        )

        # greedy_predictions from .forward() are incorrect for criterion_type=`map`
        # getting correct greedy_predictions, if needed
        if self.use_graph_lm and (not self.training or self.transcribe_training):
            greedy_predictions, encoded_len, _ = self.transcribe_decoder.forward(
                log_probs=log_probs, log_probs_length=encoded_len
            )
        return log_probs, encoded_len, greedy_predictions

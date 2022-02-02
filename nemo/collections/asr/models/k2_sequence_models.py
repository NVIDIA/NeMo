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

from typing import List, Optional

from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.asr.losses.lattice_losses import LatticeLoss
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.modules.graph_decoder import ViterbiDecoderWithGraph
from nemo.collections.asr.parts.k2.utils import load_graph
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.utils import logging

# use k2 import guard
# fmt: off
from nemo.core.utils.k2_utils import k2_import_guard # isort:skip
k2_import_guard()
import k2 # isort:skip
# fmt: on


class EncDecK2SeqModel(EncDecCTCModel):
    """Encoder decoder models with various lattice losses."""

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        pass

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # use k2 import guard
        k2_import_guard()

        super().__init__(cfg=cfg, trainer=trainer)

        loss_kwargs = self._cfg.get("loss", {})

        # collecting prior knowledge for MAPLoss
        self.use_graph_lm = loss_kwargs.get("criterion_type", "ml") == "map"
        if self.use_graph_lm:
            self.token_lm = None
            self.token_lm_cache_dict = None
            self.token_lm_path = self._cfg.get("token_lm", None)
            token_lm_overwrite = self._cfg.get("token_lm_overwrite", False)
            if token_lm_overwrite:
                logging.info(
                    f"""Overwriting token_lm with `{self.token_lm_path}`. 
                             Previously saved token_lm, if it exists, will be ignored."""
                )
                self.token_lm = load_graph(self.token_lm_path)
                loss_kwargs["token_lm"] = self.token_lm

        self._update_k2_modules(loss_kwargs)

    def _update_k2_modules(self, loss_kwargs):
        """
        Helper function to initialize or update k2 loss and transcribe_decoder.
        """
        del self.loss
        if hasattr(self, "transcribe_decoder"):
            del self.transcribe_decoder

        self.loss = LatticeLoss(
            num_classes=self.decoder.num_classes_with_blank - 1,
            reduction=self._cfg.get("ctc_reduction", "mean_batch"),
            **loss_kwargs,
        )
        remove_consecutive = loss_kwargs.get("topo_with_selfloops", True) and loss_kwargs.get(
            "topo_type", "default"
        ) not in ["forced_blank", "identity",]
        self._wer.remove_consecutive = remove_consecutive

        criterion_type = self.loss.criterion_type
        transcribe_training = self._cfg.get("transcribe_training", False)
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
                dec_type="tokenlm",
                return_type="1best",
                **loss_kwargs,
            )

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """
        Custom state_dict method to save token_lm graph.
        """
        state_dict = super().state_dict(destination, prefix, keep_vars)
        # fail if k2.Fsa ever supports .state_dict()
        assert "token_lm" not in state_dict
        if hasattr(self, "token_lm") and self.token_lm is not None:
            state_dict["wfst_graph.token_lm"] = self.token_lm.as_dict()
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        """
        Custom load_state_dict method to load token_lm dict.
        The graph itself will be initialized at the first .forward() call.
        """
        if "wfst_graph.token_lm" in state_dict:
            # loading only if self.token_lm is not initialized in __init__
            if self.token_lm is None:
                # we cannot load self.token_lm directly here
                # because of a weird error at runtime
                # TypeError: _broadcast_coalesced(): incompatible function arguments.
                self.token_lm_cache_dict = state_dict.pop("wfst_graph.token_lm")
        super().load_state_dict(state_dict, strict=strict)

    def change_vocabulary(self, new_vocabulary: List[str]):
        """
        Changes vocabulary used during CTC decoding process. Use this method when fine-tuning on from pre-trained model.
        This method changes only decoder and leaves encoder and pre-processing modules unchanged. For example, you would
        use it if you want to use pretrained encoder when fine-tuning on a data in another language, or when you'd need
        model to learn capitalization, punctuation and/or special characters.

        If new_vocabulary == self.decoder.vocabulary then nothing will be changed.

        Args:
            new_vocabulary: list with new vocabulary. Must contain at least 2 elements. Typically, \
            this is target alphabet.

        Returns: None

        """
        super().change_vocabulary(new_vocabulary)

        loss_kwargs = self._cfg.get("loss", {})

        if self.use_graph_lm:
            self.token_lm = None
            self.token_lm_cache_dict = None
            self.token_lm_path = None
            logging.warning(
                f"""With .change_vocabulary() call for a model with criterion_type=`{self.loss.criterion_type}`,
                            either a new token_lm or a token_lm_path has to be set manually."""
            )

        self._update_k2_modules(loss_kwargs)

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
        # trying to load token_lm from token_lm_cache_dict or token_lm_path if it hasn't been loaded yet
        if self.use_graph_lm and self.token_lm is None:
            if self.token_lm_cache_dict is not None:
                logging.info(f"""Loading token_lm from the dict cache at the first .forward() call.""")
                self.token_lm = k2.Fsa.from_dict(self.token_lm_cache_dict)
                self.token_lm_cache_dict = None
            elif self.token_lm_path is not None:
                logging.warning(f"""Loading token_lm from `{self.token_lm_path}` at the first .forward() call.""")
                self.token_lm = load_graph(self.token_lm_path)
                if self.token_lm is None:
                    raise ValueError(f"""Failed to load token_lm""")
            else:
                raise ValueError(f"""Failed to load token_lm""")
            self.loss.update_graph(self.token_lm)
            if self.use_graph_lm:
                self.transcribe_decoder.update_graph(self.token_lm)

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


class EncDecK2SeqModelBPE(EncDecCTCModelBPE):
    """Encoder decoder models with Byte Pair Encoding and various lattice losses."""

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        pass

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # use k2 import guard
        k2_import_guard()

        super().__init__(cfg=cfg, trainer=trainer)

        loss_kwargs = self._cfg.get("loss", {})

        # collecting prior knowledge for MAPLoss
        self.use_graph_lm = loss_kwargs.get("criterion_type", "ml") == "map"
        if self.use_graph_lm:
            self.token_lm = None
            self.token_lm_cache_dict = None
            self.token_lm_path = self._cfg.get("token_lm", None)
            token_lm_overwrite = self._cfg.get("token_lm_overwrite", False)
            if token_lm_overwrite:
                logging.info(
                    f"""Overwriting token_lm with `{self.token_lm_path}`. 
                             Previously saved token_lm, if it exists, will be ignored."""
                )
                self.token_lm = load_graph(self.token_lm_path)
                loss_kwargs["token_lm"] = self.token_lm

        self._update_k2_modules(loss_kwargs)

    def _update_k2_modules(self, loss_kwargs):
        """
        Helper function to initialize or update k2 loss and transcribe_decoder.
        """
        del self.loss
        if hasattr(self, "transcribe_decoder"):
            del self.transcribe_decoder

        self.loss = LatticeLoss(
            num_classes=self.decoder.num_classes_with_blank - 1,
            reduction=self._cfg.get("ctc_reduction", "mean_batch"),
            **loss_kwargs,
        )
        remove_consecutive = loss_kwargs.get("topo_with_selfloops", True) and loss_kwargs.get(
            "topo_type", "default"
        ) not in ["forced_blank", "identity",]
        self._wer.remove_consecutive = remove_consecutive

        criterion_type = self.loss.criterion_type
        transcribe_training = self._cfg.get("transcribe_training", False)
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
                dec_type="tokenlm",
                return_type="1best",
                **loss_kwargs,
            )

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """
        Custom state_dict method to save token_lm graph.
        """
        state_dict = super().state_dict(destination, prefix, keep_vars)
        # fail if k2.Fsa ever supports .state_dict()
        assert "token_lm" not in state_dict
        if hasattr(self, "token_lm") and self.token_lm is not None:
            state_dict["wfst_graph.token_lm"] = self.token_lm.as_dict()
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        """
        Custom load_state_dict method to load token_lm dict.
        The graph itself will be initialized at the first .forward() call.
        """
        if "wfst_graph.token_lm" in state_dict:
            # loading only if self.token_lm is not initialized in __init__
            if self.token_lm is None:
                # we cannot load self.token_lm directly here
                # because of a weird error at runtime
                # TypeError: _broadcast_coalesced(): incompatible function arguments.
                self.token_lm_cache_dict = state_dict.pop("wfst_graph.token_lm")
        super().load_state_dict(state_dict, strict=strict)

    def change_vocabulary(self, new_tokenizer_dir: str, new_tokenizer_type: str):
        """
        Changes vocabulary of the tokenizer used during CTC decoding process.
        Use this method when fine-tuning on from pre-trained model.
        This method changes only decoder and leaves encoder and pre-processing modules unchanged. For example, you would
        use it if you want to use pretrained encoder when fine-tuning on a data in another language, or when you'd need
        model to learn capitalization, punctuation and/or special characters.

        Args:
            new_tokenizer_dir: Path to the new tokenizer directory.
            new_tokenizer_type: Either `bpe` or `wpe`. `bpe` is used for SentencePiece tokenizers,
                whereas `wpe` is used for `BertTokenizer`.

        Returns: None

        """
        super().change_vocabulary(new_tokenizer_dir, new_tokenizer_type)

        loss_kwargs = self._cfg.get("loss", {})

        if self.use_graph_lm:
            self.token_lm = None
            self.token_lm_cache_dict = None
            self.token_lm_path = None
            logging.warning(
                f"""With .change_vocabulary() call for a model with criterion_type=`{self.loss.criterion_type}`,
                            either a new token_lm or a token_lm_path has to be set manually."""
            )

        self._update_k2_modules(loss_kwargs)

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
        # trying to load token_lm from token_lm_cache_dict or token_lm_path if it hasn't been loaded yet
        if self.use_graph_lm and self.token_lm is None:
            if self.token_lm_cache_dict is not None:
                logging.info(f"""Loading token_lm from the dict cache at the first .forward() call.""")
                self.token_lm = k2.Fsa.from_dict(self.token_lm_cache_dict)
                self.token_lm_cache_dict = None
            elif self.token_lm_path is not None:
                logging.warning(f"""Loading token_lm from `{self.token_lm_path}` at the first .forward() call.""")
                self.token_lm = load_graph(self.token_lm_path)
                if self.token_lm is None:
                    raise ValueError(f"""Failed to load token_lm""")
            else:
                raise ValueError(f"""Failed to load token_lm""")
            self.loss.update_graph(self.token_lm)
            if self.use_graph_lm:
                self.transcribe_decoder.update_graph(self.token_lm)

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

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import editdistance
import torch

from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.losses.lattice_losses import LatticeLoss
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.modules.graph_decoder import ViterbiDecoderWithGraph
from nemo.collections.asr.parts.k2.utils import load_graph
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.utils import logging


class GradCheck(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor_with_grad: torch.Tensor):
        assert not torch.isnan(tensor_with_grad).any()
        assert not torch.isinf(tensor_with_grad).any()
        return tensor_with_grad

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        assert not torch.isnan(grad_output).any()
        assert not torch.isinf(grad_output).any()
        return grad_output


class EncDecK2SeqModel(EncDecCTCModel):
    """Encoder decoder models with various lattice losses."""

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        raise NotImplementedError

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

        del self.loss
        loss_kwargs=self._cfg.get("loss", {})
        self.loss = LatticeLoss(
            num_classes=self.decoder.num_classes_with_blank - 1,
            reduction=self._cfg.get("ctc_reduction", "mean_batch"),
            **loss_kwargs
        )

        transcribe_decode = self._cfg.get("transcribe_decode", False)
        loss_type = loss_kwargs.get("loss_type", "ctc")
        if transcribe_decode and loss_type == "ctc":
            logging.warning(f"""You do not need to use transcribe_decode={transcribe_decode} 
                            with loss_type={loss_type}. transcribe_decode will be set to False.""")
            transcribe_decode = False
        self.transcribe_decode = transcribe_decode
        if self.transcribe_decode:
            decode_kwargs=self._cfg.get("graph_decode", {})
            self.transcribe_decoder = ViterbiDecoderWithGraph(num_classes=self.decoder.num_classes_with_blank - 1, **decode_kwargs)

    def _calc_cer_loc(self, transcript, transcript_len, predictions, pred_lengths):
        return torch.tensor([editdistance.eval(tran[:tran_len].tolist(), pred[:pred_len].tolist()) / tran_len for tran, tran_len, pred, pred_len in zip(transcript, transcript_len, predictions, pred_lengths)], dtype=torch.float).mean()

    def _make_fake_aligned_predictions(self, predictions):
        # we assume that self._blank == self.decoder.num_classes_with_blank - 1
        fake_lengths = torch.tensor([2 * len(pred) + 1 for pred in predictions], device=predictions[0].device)
        fake_predictions = torch.full((len(predictions), fake_lengths.max()), self.decoder.num_classes_with_blank - 1).to(device=predictions[0].device)
        for i, pred in enumerate(predictions):
            fake_predictions[i,1:fake_lengths[i]:2] = pred
        return fake_predictions, fake_lengths

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
        raise NotImplementedError
        super().change_vocabulary(new_vocabulary)

        del self.loss
        loss_kwargs = self._cfg.get("loss_kwargs", {})
        self.loss = LatticeLoss(
            num_classes=self.decoder.num_classes_with_blank - 1,
            reduction=self._cfg.get("ctc_reduction", "mean_batch"),
            **loss_kwargs
        )

    @typecheck()
    def forward(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None
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
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        log_probs = self.decoder(encoder_output=encoded.float())
        greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)

        return log_probs, encoded_len, greedy_predictions

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        if self.transcribe_decode:
            signal, signal_len, transcript, transcript_len = batch
            if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
                log_probs, encoded_len, greedy_predictions = self.forward(
                    processed_signal=signal, processed_signal_length=signal_len
                )
            else:
                log_probs, encoded_len, greedy_predictions = self.forward(input_signal=signal, input_signal_length=signal_len)

            loss_value = self.loss(
                log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
            )

            tensorboard_logs = {'train_loss': loss_value, 'learning_rate': self._optimizer.param_groups[0]['lr']}

            if hasattr(self, '_trainer') and self._trainer is not None:
                log_every_n_steps = self._trainer.log_every_n_steps
            else:
                log_every_n_steps = 1

            if (batch_nb + 1) % log_every_n_steps == 0:
                self._wer.update(
                    predictions=greedy_predictions,
                    targets=transcript,
                    target_lengths=transcript_len,
                    predictions_lengths=encoded_len,
                )
                wer_greedy, _, _ = self._wer.compute()
                tensorboard_logs.update({'training_batch_wer_greedy': wer_greedy})

                predictions, pred_lengths, _ = self.transcribe_decoder.forward(log_probs=log_probs, log_probs_length=encoded_len)
                # cer = self._calc_cer_loc(transcript, transcript_len, predictions, pred_lengths)
                # tensorboard_logs.update({'training_batch_cer': cer})

                # fake_predictions, fake_lengths = self._make_fake_aligned_predictions(predictions)
                self._wer.update(
                    predictions=predictions,
                    targets=transcript,
                    target_lengths=transcript_len,
                    predictions_lengths=pred_lengths,
                )
                wer, _, _ = self._wer.compute()
                tensorboard_logs.update({'training_batch_wer': wer})

            return {'loss': loss_value, 'log': tensorboard_logs}
        else:
            return super().training_step(batch, batch_nb)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.transcribe_decode:
            signal, signal_len, transcript, transcript_len = batch
            if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
                log_probs, encoded_len, greedy_predictions = self.forward(
                    processed_signal=signal, processed_signal_length=signal_len
                )
            else:
                log_probs, encoded_len, greedy_predictions = self.forward(input_signal=signal, input_signal_length=signal_len)

            loss_value = self.loss(
                log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
            )
            self._wer.update(
                predictions=greedy_predictions, targets=transcript, target_lengths=transcript_len, predictions_lengths=encoded_len
            )
            wer_greedy, wer_num_greedy, wer_denom_greedy = self._wer.compute()

            predictions, pred_lengths, _ = self.transcribe_decoder.forward(log_probs=log_probs, log_probs_length=encoded_len)
            # cer = self._calc_cer_loc(transcript, transcript_len, predictions, pred_lengths)

            # fake_predictions, fake_lengths = self._make_fake_aligned_predictions(predictions)
            self._wer.update(predictions=predictions, targets=transcript, target_lengths=transcript_len, predictions_lengths=pred_lengths)
            wer, wer_num, wer_denom = self._wer.compute()
            return {
                'val_loss': loss_value,
                'val_wer_num_greedy': wer_num_greedy,
                'val_wer_denom_greedy': wer_denom_greedy,
                'val_wer_greedy': wer_greedy,
                'val_wer_num': wer_num,
                'val_wer_denom': wer_denom,
                'val_wer': wer,
            }
        else:
            return super().validation_step(batch, batch_idx, dataloader_idx)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if self.transcribe_decode:
            logs = self.validation_step(batch, batch_idx, dataloader_idx=dataloader_idx)
            test_logs = {
                'test_loss': logs['val_loss'],
                'test_wer_num_greedy': logs['val_wer_num_greedy'],
                'test_wer_denom_greedy': logs['val_wer_denom_greedy'],
                'test_wer_greedy': logs['val_wer_greedy'],
                'test_wer_num': logs['val_wer_num'],
                'test_wer_denom': logs['val_wer_denom'],
                'test_wer': logs['val_wer'],
            }
            return test_logs
        else:
            return super().test_step(batch, batch_idx, dataloader_idx)


class EncDecK2SeqModelBPE(EncDecCTCModelBPE):
    """Encoder decoder models with Byte Pair Encoding and various lattice losses."""

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        raise NotImplementedError

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

        del self.loss
        loss_kwargs = self._cfg.get("loss", {})
        self.loss = LatticeLoss(
            num_classes=self.decoder.num_classes_with_blank - 1,
            reduction=self._cfg.get("ctc_reduction", "mean_batch"),
            **loss_kwargs
        )
        remove_consecutive = loss_kwargs.get("topo_with_selfloops", True) and loss_kwargs.get("topo_type", "default") not in ["forced_blank", "identity"]
        self._wer.remove_consecutive = remove_consecutive

        # collecting prior knowledge for MAPLoss
        if loss_kwargs.get("criterion_type", "ml") == "map":
            self.token_lm = None
            self.token_lm_path = self._cfg.get("token_lm", None)
            token_lm_overwrite = self._cfg.get("token_lm_overwrite", False)
            if token_lm_overwrite:
                logging.info(f"""Overwriting token_lm with {self.token_lm_path}. 
                             Previously saved token_lm, if it exists, will be ignored.""")
                self.token_lm = load_graph(self.token_lm_path)
                loss_kwargs["token_lm"] = self.token_lm

        transcribe_decode = self._cfg.get("transcribe_decode", False)
        criterion_type = self.loss.criterion_type
        if transcribe_decode and criterion_type == "ml":
            logging.warning(f"""You do not need to use transcribe_decode={transcribe_decode} 
                            with criterion_type={criterion_type}. transcribe_decode will be set to False.""")
            transcribe_decode = False
        self.transcribe_decode = transcribe_decode
        if self.transcribe_decode:
            self.transcribe_decoder = ViterbiDecoderWithGraph(num_classes=self.decoder.num_classes_with_blank - 1, dec_type='tokenlm', return_type='1best', **loss_kwargs)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """TBD"""
        state_dict = super().state_dict(destination, prefix, keep_vars)
        # fail if k2.Fsa ever supports .state_dict()
        assert "token_lm" not in state_dict
        if hasattr(self, "token_lm") and self.token_lm is not None:
            state_dict["wfst_graph.token_lm"] = self.token_lm.as_dict()
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        """TBD"""
        if "wfst_graph.token_lm" in state_dict:
            # loading only if self.token_lm is not initialized in __init__
            if self.token_lm is None:
                token_lm_dict = state_dict["wfst_graph.token_lm"]
                self.token_lm = k2.Fsa.from_dict(token_lm_dict)
                if self.loss.criterion_type == "map":
                    self.loss.update_graph(self.token_lm)
                if self.transcribe_decode:
                    self.transcribe_decoder.update_graph(self.token_lm)
            del state_dict["wfst_graph.token_lm"]
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
        raise NotImplementedError
        super().change_vocabulary(new_tokenizer_dir, new_tokenizer_type)

        del self.loss
        loss_kwargs=self._cfg.get("loss_kwargs", {})
        self.loss = LatticeLoss(
            num_classes=self.decoder.num_classes_with_blank - 1,
            reduction=self._cfg.get("ctc_reduction", "mean_batch"),
            **loss_kwargs
        )

    @typecheck()
    def forward(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None
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
        # trying to load token_lm from token_lm_path if it hasn't been loaded yet
        if self.loss.criterion_type == "map" and self.token_lm is None:
            logging.warning(f"""Loading token_lm from {self.token_lm_path} at the first .forward() call.""")
            self.token_lm = load_graph(self.token_lm_path)
            if self.token_lm is None:
                raise ValueError(f"""token_lm is empty.""")
            self.loss.update_graph(self.token_lm)
            if self.transcribe_decode:
                self.transcribe_decoder.update_graph(self.token_lm)
        log_probs, encoded_len, greedy_predictions = super().forward(input_signal=input_signal,
                                                                     input_signal_length=input_signal_length,
                                                                     processed_signal=processed_signal,
                                                                     processed_signal_length=processed_signal_length)
        if self.transcribe_decode:
            greedy_predictions, encoded_len, _ = self.transcribe_decoder.forward(log_probs=log_probs,
                                                                                 log_probs_length=encoded_len)
        return log_probs, encoded_len, greedy_predictions

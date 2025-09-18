# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.speechlm2.data.utils import get_pad_id
from omegaconf import OmegaConf

import torch
from torch import Tensor

from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTBPEDecoding
from nemo.collections.speechlm2.models.salm import SALM
from nemo.collections.speechlm2.parts.optim_setup import is_frozen
from nemo.utils import logging

from lightning import LightningModule
from omegaconf import DictConfig, open_dict
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.parts.pretrained import load_pretrained_hf, load_pretrained_nemo
from nemo.collections.speechlm2.parts.lora import maybe_install_lora

from nemo.collections.asr.models import ASRModel
from nemo.collections.speechlm2.modules.perception import AudioPerceptionModule
        

class HybridSALMTDT(SALM):    
    def __init__(self, cfg) -> None:
        # Store config first
        self.cfg = OmegaConf.create(cfg)
        
        super().__init__(cfg)
        
        # Initialize TDT head components using the same ASR model
        self._setup_tdt_head()
        
        # Task routing configuration
        self.tdt_weight = self.cfg.tdt_head.get('tdt_weight', 1.0)


    def _setup_tdt_head(self):
        # Use the already loaded ASR model
        asr_model = self.asr_model
        logging.info("Setting up TDT components from already loaded ASR model")
        
        # Extract TDT components
        self.tdt_decoder = asr_model.decoder
        self.tdt_joint = asr_model.joint
        self.tdt_tokenizer = asr_model.tokenizer
        self.tdt_tokenizer_type = asr_model.tokenizer_type
        
        # Setup TDT loss with same parameters as pretrained model
        self.tdt_loss = asr_model.loss
        
        # Setup TDT decoding
        self.tdt_decoding = RNNTBPEDecoding(
            decoding_cfg=asr_model.cfg.decoding,
            decoder=self.tdt_decoder,
            joint=self.tdt_joint,
            tokenizer=self.tdt_tokenizer,
        )
        
        logging.info("TDT head components loaded from pretrained ASR model successfully")
        
        # Verify that components are properly loaded
        self._verify_tdt_components()

    def _verify_tdt_components(self):
        # Check that encoder hidden size matches joint network input size
        encoder_hidden = self.perception.encoder.d_model
        joint_encoder_hidden = self.tdt_joint.encoder_hidden
        
        if encoder_hidden != joint_encoder_hidden:
            logging.warning(
                f"Encoder hidden size ({encoder_hidden}) doesn't match TDT joint encoder input size ({joint_encoder_hidden}). "
                f"This may cause issues during training. Consider using a TDT model trained with the same encoder."
            )
        
        # Check tokenizer compatibility
        if hasattr(self.tdt_tokenizer, 'tokenizer'):
            vocab_size = len(self.tdt_tokenizer.tokenizer.get_vocab())
            joint_vocab_size = self.tdt_joint.num_classes_with_blank - 1  # Subtract blank token
            
            if vocab_size != joint_vocab_size:
                logging.warning(
                    f"TDT tokenizer vocab size ({vocab_size}) doesn't match joint network vocab size ({joint_vocab_size}). "
                    f"This may cause issues during training."
                )
        
        logging.info("TDT component verification completed")

    def forward_tdt(self, audio_encoded: Tensor,
                    audio_encoded_len: Tensor, 
                    target_ids: Tensor,
                    target_ids_len: Tensor,
                    compute_wer: bool=False) -> dict[str, Tensor]:
        pred_out, pred_out_len, _ = self.tdt_decoder(targets=target_ids, target_length=target_ids_len)
        
        # joint_out = self.tdt_joint(encoder_outputs=audio_encoded, decoder_outputs=pred_out)
        
        # Fused joint step
        loss_value, wer, _, _ = self.tdt_joint(
            encoder_outputs=audio_encoded,
            decoder_outputs=pred_out,
            encoder_lengths=audio_encoded_len,
            transcripts=target_ids,
            transcript_lengths=target_ids_len,
            compute_wer=compute_wer,
        )
        
        print("Loss value: ", loss_value)
        print("WER: ", wer)
        
        if compute_wer:
            return {
                "loss_value": loss_value,
                "wer": wer,
            }
        else:
            return {
                "loss_value": loss_value,
            }


    def prepare_tdt_inputs(self, batch: dict):
        """Prepare inputs for TDT head (ASR tasks)."""
        
        audio_encoded, audio_encoded_len = self.perception(
            input_signal=batch["audios"], input_signal_length=batch["audio_lens"], 
            apply_modality_adapter=False
        )
        
        assert "tdt_input_ids" in batch, "tdt_input_ids must be in batch"
        assert "tdt_input_ids_len" in batch, "tdt_input_ids_len must be in batch"
        
        target_ids = batch["tdt_input_ids"]
        target_ids_len = batch["tdt_input_ids_len"]
                
        return {
            "audio_encoded": audio_encoded,
            "audio_encoded_len": audio_encoded_len,
            "target_ids": target_ids,
            "target_ids_len": target_ids_len,
        }

    def training_step(self, batch: dict, batch_idx: int):
        """
        Training step that handles separate speech and non-speech batches.
        
        Args:
            batch: Dictionary containing either:
                - speech_batch: dict with speech data (goes through both TDT and SALM)
                - non_speech_batch: dict with non-speech data (goes through SALM only)
                - batch_type: str indicating batch type
        """
        total_loss = 0.0
        loss_dict = {}
            
        # Legacy batch format - treat as speech data
        speech_loss, speech_metrics = self._process_speech_batch(batch, batch_idx)
        total_loss += speech_loss
        loss_dict.update(speech_metrics)
    
        # Final logging
        loss_dict["loss"] = total_loss
        self.log_dict(loss_dict, on_step=True)
        return loss_dict

    def _process_speech_batch(self, batch: dict, batch_idx: int) -> tuple[float, dict]:
        """Process speech batch through both TDT and SALM heads."""
        loss_dict = {}
        total_loss = 0.0
        
        # SALM training for speech data
        salm_result = super().training_step(batch, batch_idx)
        salm_loss = salm_result["loss"]
        total_loss += (1 - self.tdt_weight) * salm_loss
        loss_dict["salm_speech_loss"] = salm_loss
        
        # Add other SALM metrics with speech prefix
        for key, value in salm_result.items():
            if key != "loss":
                loss_dict[f"salm_speech_{key}"] = value
    
        # TDT training for speech data
        for m in (self.perception.preprocessor, self.perception.encoder):
            if is_frozen(m):
                m.eval()
    
        compute_wer = True if batch_idx % 100 == 0 else False
        # Prepare TDT inputs using the new structure
        tdt_loss = 0.0
        if "tdt_input_ids" in batch:
            inputs = self.prepare_tdt_inputs(batch)
            forward_outputs = self.forward_tdt(
                inputs["audio_encoded"], 
                inputs["audio_encoded_len"],
                inputs["target_ids"],
                inputs["target_ids_len"],
                compute_wer
            )
            
            tdt_loss = self.tdt_weight * forward_outputs["loss_value"]
        
        total_loss += tdt_loss
        loss_dict["tdt_loss"] = forward_outputs["loss_value"]
        if compute_wer:
            loss_dict["tdt_wer"] = forward_outputs["wer"]
        
        # Batch size info
        if "audios" in batch:
            loss_dict["speech_batch_size"] = batch["audios"].shape[0]
        
        return total_loss, loss_dict

    def _process_non_speech_batch(self, batch: dict, batch_idx: int) -> tuple[float, dict]:
        """Process non-speech batch through SALM head only."""
        loss_dict = {}
        
        # Create SALM batch (use salm_input_ids)
        salm_batch = batch.copy()
        if "salm_input_ids" in batch:
            salm_batch["input_ids"] = batch["salm_input_ids"]
        
        # SALM training for non-speech data
        salm_result = super().training_step(salm_batch, batch_idx)
        salm_loss = salm_result["loss"]
        loss_dict["salm_non_speech_loss"] = salm_loss
        
        # Add other SALM metrics with non-speech prefix
        for key, value in salm_result.items():
            if key != "loss":
                loss_dict[f"salm_non_speech_{key}"] = value
        
        # Batch size info
        if "salm_input_ids" in batch:
            loss_dict["non_speech_batch_size"] = batch["salm_input_ids"].shape[0]
        
        return salm_loss, loss_dict

    def validation_step(self, batch: dict, batch_idx: int):
        """Validation step that handles separate speech and non-speech batches."""
        # Handle different batch types
        if "batch_type" in batch:
            # New hybrid batch format
            batch_type = batch["batch_type"]
            
            # Process speech data (TDT + SALM validation)
            if "speech_batch" in batch:
                speech_batch = batch["speech_batch"]
                self._validate_speech_batch(speech_batch, batch_idx, "speech")
            
            # Process non-speech data (SALM validation only)
            if "non_speech_batch" in batch:
                non_speech_batch = batch["non_speech_batch"]
                self._validate_non_speech_batch(non_speech_batch, batch_idx, "non_speech")
                
        else:
            # Legacy batch format - treat as speech data
            self._validate_speech_batch(batch, batch_idx, "legacy")

    def _validate_speech_batch(self, batch: dict, batch_idx: int, dataset_name: str):
        """Validate speech batch through both TDT and SALM heads."""

        super().validation_step(batch, batch_idx)
        
        for dataset_name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue  # some dataset is exhausted
            # TDT validation
            inputs = self.prepare_tdt_inputs(dataset_batch)
            forward_outputs = self.forward_tdt(
                inputs["audio_encoded"], 
                inputs["audio_encoded_len"],
                inputs["target_ids"],
                inputs["target_ids_len"],
                compute_wer=True
            )
            
            self._partial_accuracies[f"{dataset_name}_tdt_val_loss"].append(forward_outputs["loss_value"])
            self._partial_accuracies[f"{dataset_name}_tdt_val_wer"].append(forward_outputs["wer"])

    def _validate_non_speech_batch(self, batch: dict, batch_idx: int, dataset_name: str):
        """Validate non-speech batch through SALM head only."""
        # Create SALM batch (use salm_input_ids)
        salm_batch = batch.copy()
        if "salm_input_ids" in batch:
            salm_batch["input_ids"] = batch["salm_input_ids"]
        
        # SALM validation - use parent class method
        super().validation_step(salm_batch, batch_idx)
        
        # No TDT validation for non-speech data


    @torch.no_grad()
    def generate_salm(self, prompts, audios=None, audio_lens=None, **generation_kwargs):
        """Generate using SALM head (delegates to parent class generate method)."""
        return self.generate(prompts, audios, audio_lens, **generation_kwargs)

    @torch.no_grad()
    def transcribe_tdt(self, audios, audio_lens):
        """Transcribe using TDT head."""
        # Audio encoding
        audio_encoded, audio_encoded_len = self.perception(audios, audio_lens, apply_modality_adapter=False)
        
        # TDT decoding
        hypotheses = self.tdt_decoding(
            encoder_output=audio_encoded,
            encoded_lengths=audio_encoded_len,
        )
        
        return hypotheses

    @property
    def oomptimizer_schema(self) -> dict:
        """Return typing schema for optimal batch size calibration."""
        # Extend parent schema with task_type
        parent_schema = super().oomptimizer_schema
        parent_schema["inputs"].append({"name": "task_type", "type": str})  # "salm", "tdt", or "both"
        return parent_schema

    # def _setup_tdt_head_from_config(self):
    #     """Fallback method to setup TDT head from config (when no pretrained model available)."""
    #     from nemo.collections.asr.modules.rnnt import RNNTJoint
    #     from nemo.collections.asr.parts.mixins.mixins import ASRBPEMixin
        
    #     # Setup TDT tokenizer using ASRBPEMixin pattern
    #     tdt_tokenizer_cfg = self.cfg.tdt_head.tokenizer
    #     if 'tokenizer' not in self.cfg.tdt_head:
    #         raise ValueError("`cfg.tdt_head` must have `tokenizer` config to create a TDT tokenizer!")
        
    #     # Create a temporary mixin instance to setup TDT tokenizer
    #     tdt_mixin = ASRBPEMixin()
    #     tdt_mixin._setup_tokenizer(tdt_tokenizer_cfg)
    #     self.tdt_tokenizer = tdt_mixin.tokenizer
    #     self.tdt_tokenizer_type = tdt_mixin.tokenizer_type
        
    #     # Get TDT vocabulary size
    #     tdt_vocabulary = self.tdt_tokenizer.tokenizer.get_vocab()
    #     tdt_vocab_size = len(tdt_vocabulary)
        
    #     # TDT predictor
    #     predictor_cfg = self.cfg.tdt_head.predictor
    #     self.tdt_predictor = self.from_config_dict(predictor_cfg)
        
    #     # TDT joint network
    #     joint_cfg = self.cfg.tdt_head.joint
    #     joint_cfg.num_classes = tdt_vocab_size
    #     joint_cfg.vocabulary = list(tdt_vocabulary.keys())
    #     joint_cfg.jointnet.encoder_hidden = self.cfg.model_defaults.enc_hidden
    #     joint_cfg.jointnet.pred_hidden = self.cfg.model_defaults.pred_hidden
        
    #     self.tdt_joint = RNNTJoint(
    #         vocab_size=tdt_vocab_size,
    #         encoder_hidden=joint_cfg.jointnet.encoder_hidden,
    #         pred_hidden=joint_cfg.jointnet.pred_hidden,
    #         joint_hidden=joint_cfg.jointnet.joint_hidden,
    #         activation=joint_cfg.jointnet.activation,
    #         dropout=joint_cfg.jointnet.dropout,
    #     )
        
    #     # TDT loss
    #     self.tdt_loss = RNNTLoss(
    #         num_classes=tdt_vocab_size,
    #         loss_name='tdt',
    #         durations=self.cfg.tdt_head.get('durations', [0, 1, 2, 3, 4]),
    #         sigma=self.cfg.tdt_head.get('sigma', 0.0),
    #     )
        
    #     # TDT decoding
    #     self.tdt_decoding = RNNTBPEDecoding(
    #         decoding_cfg=self.cfg.tdt_head.decoding,
    #         decoder=self.tdt_predictor,
    #         joint=self.tdt_joint,
    #         tokenizer=self.tdt_tokenizer,
    #     )

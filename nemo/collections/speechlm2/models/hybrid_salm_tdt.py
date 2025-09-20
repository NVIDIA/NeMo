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
        
        # Make TDT components trainable (override the .eval() from parent)
        self.tdt_decoder.train()
        self.tdt_joint.train()
        
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
        tdt_batch_audios = batch["audios"].index_select(0, batch["tdt_input_idxs"])
        tdt_batch_audio_lens = batch["audio_lens"].index_select(0, batch["tdt_input_idxs"])
        
        audio_encoded, audio_encoded_len = self.perception(
            input_signal=tdt_batch_audios, input_signal_length=tdt_batch_audio_lens, 
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
        
        # Ensure TDT components are in training mode
        if not is_frozen(self.tdt_decoder):
            self.tdt_decoder.train()
        if not is_frozen(self.tdt_joint):
            self.tdt_joint.train()
    
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
            
            tdt_loss = forward_outputs["loss_value"]
            loss_dict["tdt_loss"] = forward_outputs["loss_value"]
        
            if compute_wer:
                loss_dict["tdt_wer"] = forward_outputs["wer"]
        else:
            print("No TDT input ids in batch")
            
        total_loss += tdt_loss * self.tdt_weight
        return total_loss, loss_dict

    def validation_step(self, batch: dict, batch_idx: int):
        """Validation step that handles separate speech and non-speech batches."""
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
        # Create a comprehensive schema for the hybrid model
        return {
            "cls": dict,
            "inputs": [
                # Audio inputs (shared by both SALM and TDT)
                {"name": "audios", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                
                # SALM inputs
                {
                    "name": "input_ids",
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "vocab_size": self.text_vocab_size,
                },
                {"name": "loss_mask", "type": NeuralType(("B", "T"), MaskType()), "seq_length": "output"},
                
                # TDT inputs
                {
                    "name": "tdt_input_ids",
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "tdt_output",
                    "vocab_size": self.tdt_tokenizer.vocab_size if hasattr(self.tdt_tokenizer, 'vocab_size') else 1024,
                },
                {"name": "tdt_input_ids_len", "type": NeuralType(("B",), LengthsType()), "seq_length": "tdt_output"},
                
                # Task routing inputs
                {"name": "tdt_input_idxs", "type": NeuralType(("B",), LengthsType())},
                {"name": "task_type", "type": str},  # "salm", "tdt", or "both"
            ],
        }
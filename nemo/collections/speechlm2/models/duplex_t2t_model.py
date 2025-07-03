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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from lightning import LightningModule
from lhotse.dataset.collation import collate_vectors
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict
import os
import json
from peft import PeftModel
from torch import Tensor
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    loss_parallel,
    parallelize_module,
)
from transformers import DynamicCache

from nemo.collections.audio.parts.utils.resampling import resample
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.utils import get_pad_id
from nemo.collections.speechlm2.models.duplex_s2s_model import replace_control_speech_codes, tokens_to_str
from nemo.collections.speechlm2.modules import TransformerARSpeechDecoder
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.lora import maybe_install_lora
from nemo.collections.speechlm2.parts.metrics.asr_bleu import ASRBLEU
from nemo.collections.speechlm2.parts.metrics.bleu import BLEU
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.speechlm2.parts.pretrained import load_pretrained_hf, load_pretrained_nemo, setup_audio_codec, setup_speech_encoder
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.utils import logging
from nemo.collections.asr.modules.transformer import TransformerEncoder

class DuplexT2TModel(LightningModule, HFHubMixin):
    """
    Duplex text/token to text/speech model with frozen ASR.
    With cfg.generate_speech=True and cfg.audio_loss_weight > 0, this model can be trained to generate speech.
    
    Text to text model:
      speech → [ASR] → decoded text → [deterministic retokenization] → tokens from LLM's vocabulary → 
      [LLM's embed and combine with agent channel] → continuous representation → [LLM]

      CASE 1 input (oracle-EoU):
        <BOS><turn 1 tokens><EOS><PAD tokens to fill user turn 1 duration><PAD tokens to fill agent turn 1>
        <BOS><turn 2 tokens><EOS><PAD tokens to fill user turn 2 duration><PAD tokens to fill agent turn 2>
        ...
      CASE 2 input (oracle-aligned):
        <turn 1 tokens word-aligned><PAD tokens to fill agent turn 1>
        <turn 2 tokens word-aligned><PAD tokens to fill agent turn 2>
        ...

    Token to text model:
      speech → [ASR] → frame-level output tokens → [ASR's embed and combine with agent channel embed^] → 
      continuous representation → [shallow transformer module*] → continuous representation → [LLM]
    
      Input:
        <turn 1 tokens frame-aligned><PAD tokens to fill agent turn 1>
        <turn 2 tokens frame-aligned><PAD tokens to fill agent turn 2>
        ...

    ^Agent channel can be embedded either via LLM's tokenize+embedding or ASR's tokenization+embedding. 

    *transformer so that the self-attention can learn which tokens to combine/split etc to match LLM's vocabulary space. 
    """

    def __init__(self, cfg: dict) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to DuplexS2SModel as a Python dict to support hyperparameter serialization "
            f"in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        super().__init__()
        self.save_hyperparameters()
        self.cfg = DictConfig(cfg)

        # Add configuration for saving validation outputs
        self.save_val_outputs = cfg.get("save_val_outputs", False)
        self.val_output_dir = cfg.get("val_output_dir", "validation_outputs")
        
        self.generate_speech = cfg.get("generate_speech", True)
        self.train_retokenizer = cfg.get("train_retokenizer", False)
        if self.generate_speech:
            setup_audio_codec(self)
            self._codebook_size = self.audio_codec.vector_quantizer.codebook_size_per_group
            self._num_codebooks = self.audio_codec.vector_quantizer.num_groups

        # We load the pretrained HF LLM using "ForCausalLM" variant so that we can obtain the
        # pretrained LM head weights.
        # However, for S2S we need to access the activations before LM head directly
        # to feed them to the audio codec head.
        self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True)
        llm = load_pretrained_hf(self.cfg.pretrained_llm, pretrained_weights=self.cfg.pretrained_weights).train()
        self.llm = llm.model  # fetch PretrainedBaseModel from model "ForCausalLM"
        self.lm_head = llm.lm_head
        # Note: we have to "move out" the token embedding outside of LLM to avoid
        #       messing up FSDP/TP hooks.
        self.embed_tokens = self.llm.embed_tokens
        del self.llm.embed_tokens
        maybe_install_lora(self)

        # # Load the pretrained streaming ASR model and copy its parameters into the audio perception module.
        # setup_speech_encoder(self)
        # if self.train_retokenizer:
        #     with fp32_precision(), torch.no_grad():  # The streaming ASR model is not trained with bfloat16.
        #         self.asr_model = load_pretrained_nemo(ASRModel, self.cfg.pretrained_asr).eval()
        #         self.asr_model.eval()

        if self.generate_speech:
            self.speech_generation = TransformerARSpeechDecoder(
                speech_decoder_parms=OmegaConf.to_container(self.cfg.speech_decoder),
                lantent_dim=self.llm.config.hidden_size,
                num_audio_codebooks=self._num_codebooks,
                num_audio_tokens_per_codebook=self.speech_vocab_size,
            )
            self.embed_audio_tokens = torch.nn.ModuleList(
                [
                    torch.nn.Embedding(self.speech_vocab_size, self.embed_tokens.embedding_dim)
                    for _ in range(self._num_codebooks)
                ]
            )
            self.audio_head = torch.nn.Linear(self.llm.config.hidden_size, self.speech_vocab_size * self._num_codebooks)

            # cached for quicker audio decoding
            self.register_buffer(
                "_control_codes",
                torch.tensor([self.speech_bos_id, self.speech_eos_id, self.speech_delay_id], device=self.device),
            )

        self._use_fsdp = False
        self._use_tp = False

        # Configuration for embedding combination mode
        self.concat_embeddings = cfg.get("concat_embeddings", False)
        self.retokenize_type = cfg.get("retokenize_type", "no_llm_embed")
        self.retokenize_first = cfg.get("retokenize_first", False)
        self.num_eos_zeros_eval = cfg.get("num_eos_zeros_eval", -1)
        self.force_agent_bos = cfg.get("force_agent_bos", False)

        # Transform concatenated embeddings to LLM dimension
        if self.concat_embeddings:
            # Moved the commented out logic to the training script since
            # self.asr_embed is not set when we initialize the model.
            # if self.train_retokenizer:
            #     num_transformer_layers = self.cfg.get("train_retokenizer_transformer_layers", 0)
            #     if num_transformer_layers > 0:
            #         use_transformer = True
            #     else:
            #         use_transformer = False # Linear layer only
            #     concat_dim = self.embed_tokens.embedding_dim + self.asr_embed.embedding_dim
            #     output_dim = self.embed_tokens.embedding_dim

            #     self.map_projection = Retokenizer(
            #         input_dim=concat_dim,
            #         output_dim=output_dim,
            #         use_transformer=use_transformer,
            #         num_layers=num_transformer_layers,
            #         num_heads=4,
            #     )
            # else:
            self.map_projection = torch.nn.Linear(
                self.embed_tokens.embedding_dim * 2,  # 2x because we concatenate user and agent embeddings
                self.embed_tokens.embedding_dim
            )
            self.map_projection_retokenized = torch.nn.Linear(
                self.embed_tokens.embedding_dim * 2,  # 2x because we concatenate user and agent embeddings
                self.embed_tokens.embedding_dim
            )

    
    @property
    def speech_vocab_size(self):
        """Return the size of the audio codec codebook including extra speech BOS and EOS tokens."""
        return self._codebook_size + 3

    @property
    def speech_pad_id(self) -> int:
        """PAD ID based on streaming ASR vocabulary."""
        return 1024

    @property
    def speech_bos_id(self) -> int:
        """Indicates start of utterance generation (not start of inference!)."""
        return self._codebook_size

    @property
    def speech_eos_id(self) -> int:
        """Indicates end of utterance generation."""
        return self._codebook_size + 1

    @property
    def speech_delay_id(self) -> int:
        """Indicates start of inference (the very first frame)."""
        return self._codebook_size + 2

    @property
    def text_vocab_size(self):
        """Return the size of the text tokenizer."""
        return self.tokenizer.vocab_size

    @property
    def text_bos_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def text_eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def text_pad_id(self) -> int:
        """
        Text pad ID is used as a 'blank' for frames when the model is not speaking
        and for frames where the model is speaking but has already predicted the
        entire text channel's content.

        Example:

            flow:         |---user---||-------assistant--------||-user-|
            text channel:  0000000000  1xxxxxxx0000000000000002  000000

        Where 0 indicates PAD ID, 1 indicates BOS ID, 2 indacates EOS ID,
        and x indicates tokens corresponding to actual text

        """
        return get_pad_id(self.tokenizer)

    def _combine_embeddings(self, agent_embeds: Tensor, user_embeds: Tensor, user_tokens: Tensor) -> Tensor:
        """
        Combine agent and user embeddings based on the configuration.
        
        Args:
            agent_embeds: Agent (target) embeddings of shape (B, T, H)
            user_embeds: User (source) embeddings of shape (B, T, H)
            
        Returns:
            Combined embeddings of shape (B, T, H)
        """
        if self.train_retokenizer and self.retokenize_first:
            if self.retokenize_type == "soft_token_map":
                user_embeds = self.map_projection(user_tokens)
            else:
                user_embeds = self.map_projection(user_embeds)
        if self.concat_embeddings:
            # Concatenate embeddings and project to original dimension
            concatenated = torch.cat([agent_embeds, user_embeds], dim=-1)  # (B, T, H1+H2)
            if not self.retokenize_first:
                return self.map_projection(concatenated)  # (B, T, H)
            else:
                return self.map_projection_retokenized(concatenated)  # (B, T, H)
        else:
            # Add embeddings (original behavior)
            user_weighted = user_embeds * self.cfg.get("duplex_user_channel_weight", 1.0)
            return agent_embeds + user_weighted

    def forward(self, input_embeds: Tensor, cache=None, input_audio_tokens=None, loss_mask=None) -> dict[str, Tensor]:
        """
        Separated text and speech prediction:
            - Speech prediction is achieved by a independent AR decoder based on last_hidden_state + audio tokens
            - For KV-cache:
                (1) llm cache depends on input cache is None or Not
                (2) speech_generation cache relys on reset_input_and_kv_cache function.
        """

        out = self.llm(
            inputs_embeds=input_embeds, past_key_values=cache, use_cache=cache is not None, return_dict=True
        )
        B, T = input_embeds.shape[:2]
        text_logits = self.lm_head(out['last_hidden_state'])  # (B, T, text_vocab_size)

        if loss_mask is not None:
            # This is training Mode
            loss_mask = loss_mask[:, :, -1].reshape(loss_mask.size(0), loss_mask.size(1))
            if self.generate_speech:
                self.speech_generation.reset_input_and_kv_cache(use_cache=False)

        if self.generate_speech:
            _, audio_logits = self.speech_generation(
                out['last_hidden_state'].transpose(0, 1), loss_mask, input_audio_tokens=input_audio_tokens
            )

            audio_logits = audio_logits.view(B, T, self._num_codebooks, self.speech_vocab_size)
        else:
            audio_logits = None

        ans = {
            "text_logits": text_logits,
            "audio_logits": audio_logits,
        }
        if cache is not None:
            ans["cache"] = out["past_key_values"]

        return ans

    def get_asr_hyps(self, source_audio, source_audio_lens):
        with fp32_precision(), torch.no_grad(): # The streaming ASR model is not trained with bfloat16.
            asr_hyps = self._asr.transcribe(
                [audio[:alen] for audio, alen in zip(source_audio, source_audio_lens)],
                batch_size=source_audio.shape[0],
                return_hypotheses=True,
                verbose=False,
            )
        return asr_hyps

    def prepare_inputs(self, batch: dict):
        """
        Similar to DuplexS2SModel.prepare_inputs, with following changes:
            (1) Add 'input_audio_tokens' and 'loss_mask' in return value for TransformerARSpeechDecoder
            (2) Remove audio codec embedding from 'input_embeds'
        """

        source_encoded, source_encoded_lens = self.perception(
            input_signal=batch["source_audio"], input_signal_length=batch["source_audio_lens"]
        )

        target_tokens = batch["target_tokens"]
        if (diff := target_tokens.shape[1] - source_encoded.shape[1]) < 0:
            target_tokens = torch.cat(
                [
                    target_tokens,
                    (
                        torch.ones(source_encoded.shape[0], abs(diff), device=source_encoded.device) * self.text_pad_id
                    ).to(torch.long),
                ],
                dim=-1,
            )
        elif diff > 0:
            target_tokens = target_tokens[:, : source_encoded.shape[1]]

        with fp32_precision(), torch.no_grad():
            target_codes, target_codes_lens = self.audio_codec.encode(
                audio=batch["target_audio"], audio_len=batch["target_audio_lens"]
            )
        target_codes = target_codes.transpose(1, 2)  # (B, K, T) -> (B, T, K)

        if (tl := target_codes.shape[1]) != (sl := source_encoded.shape[1]):
            if tl < sl:
                diff = sl - tl
                source_encoded = source_encoded[:, :tl]
                target_tokens = target_tokens[:, :tl]
                torch.clamp_(source_encoded_lens, max=tl)
            else:
                diff = tl - sl
                target_codes = target_codes[:, :sl]
                torch.clamp_(target_codes_lens, max=sl)
            if diff > 2:
                logging.warning(
                    f"A mismatch between source ({sl}) and target ({tl}) sequence length greater than 2 detected. "
                    f"This may indicate significant desynchronization in longer sessions."
                )

        btt = target_tokens[..., None]
        target_codes = torch.where(btt == self.text_bos_id, self.speech_bos_id, target_codes)
        target_codes = torch.where(btt == self.text_eos_id, self.speech_eos_id, target_codes)
        target_codes = torch.cat(
            [
                torch.full(
                    [target_codes.shape[0], 1, target_codes.shape[-1]],
                    fill_value=self.speech_delay_id,
                    device=self.device,
                    dtype=torch.long,
                ),
                target_codes[:, :-1],
            ],
            dim=1,
        )

        input_ids = torch.cat([target_codes, target_tokens[..., None]], dim=-1)
        if self._use_tp:
            tp_world_size = self.device_mesh["tensor_parallel"].size()
            if (remainder := (input_ids.shape[1] - 1) % tp_world_size) != 0:
                input_ids = input_ids[:, :-remainder]
                source_encoded = source_encoded[:, :-remainder]

        text_inputs = input_ids[:, :-1, -1]  # (B, T-1)
        text_labels = input_ids[:, 1:, -1]  # (B, T-1)
        audio_inputs = input_ids[:, :-1, :-1]  # (B, T-1, K)
        audio_labels = input_ids[:, 1:, :-1]  # (B, T-1, K)

        input_embeds = self.embed_tokens(text_inputs)

        input_embeds.add_(source_encoded[:, :-1] * self.cfg.get("duplex_user_channel_weight", 1.0))

        loss_mask = torch.ones_like(
            torch.cat([text_labels.unsqueeze(-1), audio_labels], dim=-1),
            device=self.device,
            dtype=torch.bool,
        )

        return {
            "input_embeds": input_embeds,
            "input_lens": source_encoded_lens - 1,
            "output_lens": target_codes_lens - 1,
            "text_labels": text_labels,
            "input_audio_tokens": audio_inputs,
            "audio_labels": audio_labels,
            "loss_mask": loss_mask,
        }

    def prepare_t2t_inputs(self, batch: dict):
        """
        Prepare inputs for text-to-text processing where source and target sequences are of equal length.
        
        Args:
            batch: dict containing:
                - source_tokens: tensor of shape (B, T) containing source text token ids
                - target_tokens: tensor of shape (B, T) containing target text token ids
        
        Returns:
            dict containing:
                - input_embeds: tensor of shape (B, T-1, H) containing embedded input tokens
                - text_labels: tensor of shape (B, T-1) containing target text tokens for loss computation
                - input_lens: tensor of shape (B,) containing sequence lengths
        """        
        # Get source and target tokens from batch
        if self.train_retokenizer:
            asr_hyps = self.get_asr_hyps(batch["source_audio"], batch["source_audio_lens"])
            source_tokens = collate_vectors(
                [
                    hyp.y_sequence.argmax(dim=-1).to(batch["source_audio"].device) for hyp in asr_hyps
                ],
                padding_value=self.text_pad_id
                )
            source_token_lens = torch.tensor([len(hyp.y_sequence) for hyp in asr_hyps])
        else:
            source_tokens = batch["source_tokens"]  # (B, T)
        target_tokens = batch["target_tokens"]  # (B, T)
        # breakpoint()
        # Append padding to the shorter sequence
        diff = target_tokens.shape[1] - source_tokens.shape[1]
        assert diff < 2, f"Source and target sequences must have similar lengths, got {source_tokens.shape[1]} and {target_tokens.shape[1]}"
        if diff > 0:
            source_tokens = torch.cat(
                [
                    source_tokens,
                    (
                        torch.ones(source_tokens.shape[0], abs(diff), device=source_tokens.device) * self.speech_pad_id
                    ).to(torch.long),
                ],
                dim=-1,
            )
        elif diff < 0:
            target_tokens = torch.cat(
                [
                    target_tokens,
                    (
                        torch.ones(target_tokens.shape[0], abs(diff), device=target_tokens.device) * self.text_pad_id
                    ).to(torch.long),
                ],
                dim=-1,
            )

        # Verify lengths match
        assert source_tokens.shape[1] == target_tokens.shape[1], \
            f"Source and target sequences must have same length, got {source_tokens.shape[1]} and {target_tokens.shape[1]}"
        
        # Create input and label sequences
        # For autoregressive training, we use tokens up to T-1 as input and tokens from 1 to T as labels
        text_inputs = target_tokens[:, :-1]  # (B, T-1)
        text_labels = target_tokens[:, 1:]   # (B, T-1)
        
        # Get sequence lengths (all sequences are same length in this case)
        input_lens = torch.full(
            (source_tokens.shape[0],), 
            source_tokens.shape[1] - 1,  # T-1 since we use T-1 tokens as input
            device=source_tokens.device,
            dtype=torch.long
        )
        
        # Embed the input tokens
        input_embeds = self.embed_tokens(text_inputs)  # (B, T-1, H)
        if self.train_retokenizer:
            source_embeds = self.asr_embed(source_tokens[:, :-1])  # (B, T-1, H)
        else:
            source_embeds = self.embed_tokens(source_tokens[:, :-1])  # (B, T-1, H)
       
        # Add source token embeddings to input embeddings
        # This allows the model to condition on the source sequence
        input_embeds = self._combine_embeddings(input_embeds, source_embeds, source_tokens[:, :-1])
        
        return {
            "input_embeds": input_embeds,      # (B, T-1, H)
            "input_lens": input_lens,          # (B,)
            "text_labels": text_labels,        # (B, T-1)
        }

    def training_step(self, batch: dict, batch_idx: int):
        # for m in (self.perception.preprocessor, self.perception.encoder, self.llm, self.speech_generation):
        #     if is_frozen(m):
        #         m.eval()
        if is_frozen(self.llm):
            self.llm.eval()
        if self.generate_speech:
            inputs = self.prepare_inputs(batch)
            forward_outputs = self(
                inputs["input_embeds"],
                input_audio_tokens=inputs["input_audio_tokens"],
                loss_mask=inputs["loss_mask"],
            )
        else:
            inputs = self.prepare_t2t_inputs(batch)
            forward_outputs = self(inputs["input_embeds"])
        
        num_frames = inputs["input_lens"].sum()
        with loss_parallel():
            text_loss = (
                torch.nn.functional.cross_entropy(
                    forward_outputs["text_logits"].flatten(0, 1),  # (B, T, Vt) -> (*, Vt)
                    inputs["text_labels"].flatten(0, 1),
                    reduction="sum",
                )
                / num_frames
            )
            if self.cfg.audio_loss_weight > 0:
                audio_loss = torch.nn.functional.cross_entropy(
                    forward_outputs["audio_logits"].flatten(0, 2),  # (B, T, K, Vs) -> (*, Vs)
                    inputs["audio_labels"].flatten(0, 2),
                    reduction="sum",
                ) / (num_frames * self._num_codebooks)
            else:
                audio_loss = 0

        loss = self.cfg.text_loss_weight * text_loss + self.cfg.audio_loss_weight * audio_loss

        B, T = inputs["input_embeds"].shape[:2]
        ans = {
            "loss": loss,
            "learning_rate": (
                torch.as_tensor(self.trainer.optimizers[0].param_groups[0]['lr'] if self._trainer is not None else 0)
            ),
            "text_loss": text_loss,
            "audio_loss": audio_loss,
            "batch_size": B,
            "sequence_length": T,
            "num_frames": num_frames.to(torch.float32),  # avoid warning
            "padding_ratio": num_frames / (B * T),
        }
        self.log_dict(ans, on_step=True)
        return ans

    def on_train_epoch_start(self) -> None:
        if self.generate_speech:
            setup_audio_codec(self)  # potentially reloads the audio codec to make sure it's in fp32

    def on_validation_epoch_start(self) -> None:
        self.on_train_epoch_start()
        if self.generate_speech:
            self.asr_bleu = ASRBLEU(self.cfg.scoring_asr).reset()
        self.bleu = BLEU().reset()
        self.val_results_to_save = defaultdict(list)

    def on_validation_epoch_end(self, prefix="val") -> None:
        if self.generate_speech:
            asr_bleu = self.asr_bleu.compute()
            for k, m in asr_bleu.items():
                self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
        bleu = self.bleu.compute()
        for name, score in bleu.items():
            self.log(f"{prefix}_{name}", score.to(self.device), on_epoch=True, sync_dist=True)
            # Save text BLEU score for this dataset
            if self.save_val_outputs:
                with open(os.path.join(self.val_output_dir, f"{prefix}_{name}.lst"), "w") as f:
                    f.write(f"{score.item()}\n")

        # Save validation outputs
        if self.save_val_outputs:
            for name, results in self.val_results_to_save.items():
                with open(os.path.join(self.val_output_dir, f"{prefix}_{name}_outputs.jsonl"), "w") as f:
                    for result in results:
                        f.write(json.dumps(result) + "\n")

    def temp_extra_func(self, source_tokens, source_lens, num_filler_zeros, force_agent_bos):
        if num_filler_zeros != -1:
            # control number of zeros between last non-zero token and EOS token (id 2)
            for idx, token in enumerate(source_tokens[0]):
                if token != 0 and token != 2:
                    last_non_zero_token = token
                    last_non_zero_token_idx = idx
                elif token == 2:
                    eos_token_idx = idx
                    break
            # if eos_token_idx - last_non_zero_token_idx > num_filler_zeros:
            source_tokens[0, eos_token_idx] = 0
            source_tokens[0, last_non_zero_token_idx+num_filler_zeros] = 2
        if self.force_agent_bos:
            force_bos_positions = []
            for idx, cur_source_tokens in enumerate(source_tokens):
                # tmp = torch.where(cur_source_tokens == self.text_pad_id)[0]
                tmp = torch.where(cur_source_tokens == 2)[0] # end of user turns
                for user_eos_pos in tmp:
                    force_bos_positions.append(user_eos_pos.item() + self.cfg.get("force_agent_bos_offset", 5))
                    if not self.cfg.get("add_eos", False) and not self.cfg.get("add_bos_eos", False):   # if we don't add EOS or BOS, we need to remove the user EOS
                        source_tokens[idx, user_eos_pos] = self.text_pad_id
        else:
            force_bos_positions = None
        # breakpoint()
        results = self.offline_t2t_inference(
            source_tokens,
            source_lens,
            force_bos_positions = force_bos_positions,
        )
        print(source_tokens[0])
        print(results["tokens"][0])
        print(tokens_to_str(source_tokens, source_lens, tokenizer=self.tokenizer, pad_id=self.text_pad_id))
        print(results["text"][0])
        return results

    def validation_step(self, batch: dict, batch_idx: int):
        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue  # some dataset is exhausted

            if self.generate_speech:
                results = self.offline_inference(
                    dataset_batch["source_audio"],
                    dataset_batch["source_audio_lens"],
                )
            else:
                # Handle ASR retokenization for text-to-text inference
                if self.train_retokenizer and "source_audio" in dataset_batch:
                    # Use ASR to get source tokens from audio
                    asr_hyps = self.get_asr_hyps(dataset_batch["source_audio"], dataset_batch["source_audio_lens"])
                    source_tokens = collate_vectors(
                        [hyp.y_sequence.argmax(dim=-1).to(dataset_batch["source_audio"].device) for hyp in asr_hyps],
                        padding_value=self.text_pad_id
                    )
                    source_lens = torch.tensor([len(hyp.y_sequence) for hyp in asr_hyps])
                else:
                    # Use provided source tokens
                    source_tokens = dataset_batch["source_tokens"]
                    source_lens = dataset_batch["source_token_lens"]
                
                results = self.temp_extra_func(source_tokens, source_lens, self.num_eos_zeros_eval, self.force_agent_bos)
            
            if self.save_val_outputs:
                for user, agent, pred, data_id in zip(dataset_batch["source_texts"], dataset_batch["target_texts"], results["text"], dataset_batch["data_id"]):
                    self.val_results_to_save[name].append({
                        "ref_user": user,
                        "ref_agent": agent,
                        "pred_agent": pred,
                        "data_id": data_id,
                    })
            
            if self.generate_speech:
                with fp32_precision():  # resample is fragile to bfloat16 default dtype
                    self.asr_bleu.update(
                        name=name,
                        refs=dataset_batch["target_texts"],
                        pred_audio=resample(results["audio"], 22050, 16000),
                        pred_audio_lens=(results["audio_len"] / 22050 * 16000).to(torch.long),
                    )
            self.bleu.update(name=name, refs=dataset_batch["target_texts"], hyps=results["text"])

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end(prefix="test")

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def _get_bos_embedding(self) -> torch.Tensor:
        """
        Remove the audio codec embedding for the beginning of AR decoding.
        """
        text_bos = torch.full((1,), fill_value=self.text_pad_id, device=self.device)
        input_embeds = self.embed_tokens(text_bos)
        return input_embeds

    @torch.no_grad()
    def offline_inference(
        self,
        input_signal: torch.Tensor,
        input_signal_lens: torch.Tensor,
        decode_audio: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Autoregressive prediction.

        Args:
            input_signal: a batch of waveforms with shape (B, T) with source sampling rate.
            input_signal_lens: example lengths as number of samples of shape (B,).
            decode_audio: bool, whether to decode audio codes to waveform.

        Returns:
            A dict with keys:
                * "text": generated text, de-tokenized to strings, properly skipping text_pad_id; list of length B.
                * "tokens_text": generated text tokens of shape (B, T2).
                * "tokens_audio": generated audio codes of shape (B, T2, K) where `K=num_codebooks`.
                * "tokens_len" output lengths as number of tokens of shape (B,).
                * "audio": generated waveform of shape (B, T3) (`decode_audio=True`).
                * "audio_len" output lengths as number of waveform samples of shape (B,) (when `decode_audio=True`).
        """
        input_embeds, lengths = self.perception(
            input_signal=input_signal,
            input_signal_length=input_signal_lens,
        )
        B, T_local, H = input_embeds.shape

        # Determine decoding length and pad if FSDP
        if self._use_fsdp:
            T_tensor = torch.tensor([T_local], device=input_embeds.device)
            dist.all_reduce(T_tensor, op=dist.ReduceOp.MAX)
            T = int(T_tensor.item())
            if T > T_local:
                last_frame = input_embeds[:, T_local - 1 : T_local, :]  # (B,1,H)
                pad = last_frame.repeat(1, T - T_local, 1)  # (B, T-T_local, H)
                input_embeds = torch.cat([input_embeds, pad], dim=1)
        else:
            T = T_local

        # Apply channel weight
        if not self.concat_embeddings:
            input_embeds *= self.cfg.get("duplex_user_channel_weight", 1.0)

        # This cache is for self.llm
        cache = DynamicCache()
        # Call reset_input_and_kv_cache to enable cache for TransformerARSpeechDecoder
        self.speech_generation.reset_input_and_kv_cache(use_cache=True)
        gen_text = torch.empty(B, T, device=self.device, dtype=torch.long)
        gen_audio = torch.empty(B, T, self._num_codebooks, device=self.device, dtype=torch.long)

        # First step, use speech_delay token
        input_embeds[:, 0] += self._get_bos_embedding()
        first_audio = torch.full(
            [B, 1, self._num_codebooks],
            fill_value=self.speech_delay_id,
            device=self.device,
            dtype=torch.long,
        )
        ans = self(input_embeds[:, :1], cache=cache, input_audio_tokens=first_audio, loss_mask=None)
        gen_text[:, 0] = ans["text_logits"][:, -1].argmax(dim=-1)
        gen_audio[:, 0] = ans["audio_logits"][:, -1].argmax(dim=-1)

        # Autoregressive loop
        for t in range(1, T):
            last_emb = self.embed_tokens(gen_text[:, t - 1])
            input_embeds[:, t] += last_emb
            current_audio = gen_audio[:, t - 1 : t, :]
            ans = self(input_embeds[:, t : t + 1], cache=ans["cache"], input_audio_tokens=current_audio)
            gen_text[:, t] = ans["text_logits"][:, -1].argmax(dim=-1)
            gen_audio[:, t] = ans["audio_logits"][:, -1].argmax(dim=-1)

        # Trim back to local length if padded
        if self._use_fsdp and T > T_local:
            gen_text = gen_text[:, :T_local]
            gen_audio = gen_audio[:, :T_local]

        ans = {
            "text": tokens_to_str(gen_text, lengths, tokenizer=self.tokenizer, pad_id=self.text_pad_id),
            "tokens_text": gen_text,
            "tokens_audio": gen_audio,
            "tokens_len": lengths,
        }

        if decode_audio:
            gen_audio_codes = replace_control_speech_codes(gen_audio, self._control_codes)
            with fp32_precision(), torch.no_grad():
                predicted_audio, predicted_audio_lens = self.audio_codec.decode(
                    tokens=gen_audio_codes.transpose(1, 2), tokens_len=lengths
                )
            ans["audio"] = predicted_audio
            ans["audio_len"] = predicted_audio_lens

        return ans

    @torch.no_grad()
    def offline_t2t_inference(
        self,
        source_tokens: torch.Tensor,
        source_lens: torch.Tensor,
        force_bos_positions = None,
    ) -> dict[str, torch.Tensor]:
        """
        Autoregressive text-to-text prediction.

        Args:
            source_tokens: a batch of source text tokens with shape (B, T)
            source_lens: example lengths as number of tokens of shape (B,)

        Returns:
            A dict with keys:
                * "text": generated text, de-tokenized to strings, properly skipping text_pad_id; list of length B.
                * "tokens": generated text tokens of shape (B, T2).
                * "tokens_len": output lengths as number of tokens of shape (B,).
        """
        B, T = source_tokens.shape

        if force_bos_positions is not None:
            assert len(force_bos_positions) == B, "force_bos_positions must have the same length as batch size"
        
        # Embed source tokens
        if self.train_retokenizer:
            source_embeds = self.asr_embed(source_tokens)  # (B, T, H)
        else:
            source_embeds = self.embed_tokens(source_tokens)  # (B, T, H)
        
        # Apply channel weight if specified (only for addition mode)
        if not self.concat_embeddings:
            source_embeds *= self.cfg.get("duplex_user_channel_weight", 1.0)
        
        # Initialize cache for autoregressive generation
        cache = DynamicCache()
        
        # Initialize output tensor
        gen_tokens = torch.empty(B, T, device=self.device, dtype=torch.long)
        
        # First step, use BOS token
        bos_embedding = self._get_bos_embedding().expand(B, 1, -1)  # (B, 1, H)
        # breakpoint()
        first_step_embeds = self._combine_embeddings(bos_embedding, source_embeds[:, :1], source_tokens[:, :1])
        
        # Generate first token
        ans = self(first_step_embeds, cache=cache)
        gen_tokens[:, 0] = ans["text_logits"][:, -1].argmax(dim=-1)
        
        # Autoregressive loop
        for t in range(1, T):
            # Add embedding of previously generated token
            last_emb = self.embed_tokens(gen_tokens[:, t-1:t])  # (B, 1, H)
            if force_bos_positions is not None:
                for batch_idx in range(last_emb.shape[0]):
                    if force_bos_positions[batch_idx] == t:
                        last_emb[batch_idx] = self.embed_tokens(
                            torch.full((1,), fill_value=self.text_bos_id, device=self.device)
                        )
            step_embeds = self._combine_embeddings(last_emb, source_embeds[:, t:t+1], source_tokens[:, t:t+1])
            
            # Generate next token
            ans = self(step_embeds, cache=ans["cache"])
            gen_tokens[:, t] = ans["text_logits"][:, -1].argmax(dim=-1)
        pred_text = tokens_to_str(gen_tokens, source_lens, tokenizer=self.tokenizer, pad_id=self.text_pad_id)
        # if pred_text[0] == "": #empty output
        #     breakpoint()
        return {
            "text": tokens_to_str(gen_tokens, source_lens, tokenizer=self.tokenizer, pad_id=self.text_pad_id),
            "tokens": gen_tokens,
            "tokens_len": source_lens,
        }

    def backward(self, *args, **kwargs):
        with loss_parallel():
            super().backward(*args, **kwargs)

    def configure_optimizers(self):
        return configure_optimizers(self)

    @property
    def oomptimizer_schema(self) -> dict:
        if self.generate_speech:
            return {
                "cls": dict,
                "inputs": [
                    {"name": "source_audio", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                    {"name": "source_audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                    {"name": "target_audio", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                    {"name": "target_audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                    {
                        "name": "target_tokens",
                        "type": NeuralType(("B", "T"), LabelsType()),
                        "seq_length": "output",
                        "vocab_size": self.tokenizer.vocab_size,
                    },
                ],
            }
        else:
            return {
                "cls": dict,
                "inputs": [
                    {"name": "source_tokens", "type": NeuralType(("B", "T"), LabelsType()), "seq_length": "input"},
                    {"name": "source_token_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                    {"name": "target_tokens", "type": NeuralType(("B", "T"), LabelsType()), "seq_length": "output"},
                    {"name": "target_tokens_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "output"},
                ],
            }

    def configure_model(self) -> None:
        # TODO(pzelasko): refactor into separate module re-usable across models
        device_mesh = self.device_mesh
        if device_mesh is None:
            return

        llm = self.llm
        if isinstance(llm, PeftModel):
            llm = llm.base_model.model

        if (tp_mesh := device_mesh["tensor_parallel"]).size() > 1:
            self._use_tp = True

            plan = {
                "layers.0": PrepareModuleInput(
                    input_layouts=(Replicate(),),  # , None)
                    desired_input_layouts=(Shard(1),),  # , None)
                    use_local_output=True,
                ),
                "norm": SequenceParallel(),
            }
            parallelize_module(llm, tp_mesh, plan)

            for transformer_block in llm.layers:
                plan = {
                    "input_layernorm": SequenceParallel(),
                    "self_attn.q_proj": ColwiseParallel(),
                    "self_attn.k_proj": ColwiseParallel(),
                    "self_attn.v_proj": ColwiseParallel(),
                    "self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
                    "post_attention_layernorm": SequenceParallel(),
                    "mlp": PrepareModuleInput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "mlp.gate_proj": ColwiseParallel(),
                    "mlp.up_proj": ColwiseParallel(),
                    "mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
                    # "pre_feedforward_layernorm": SequenceParallel(),
                    # "post_feedforward_layernorm": SequenceParallel(),
                }

                # Adjust attention module to use the local number of heads
                attn_layer = transformer_block.self_attn
                for attr in ("num_heads", "num_key_value_heads", "hidden_size"):
                    val = getattr(attn_layer, attr)
                    if val % tp_mesh.size() != 0:
                        logging.warning(
                            f"attn_layer.{attr}={val} is not divisible by {tp_mesh.size()=}: "
                            f"set a different tensor parallelism size to avoid errors."
                        )
                    setattr(attn_layer, attr, val // tp_mesh.size())

                parallelize_module(transformer_block, tp_mesh, plan)

            for m in (self.lm_head, self.audio_head):
                parallelize_module(
                    m,
                    tp_mesh,
                    ColwiseParallel(
                        input_layouts=Shard(1),
                        output_layouts=Shard(-1),
                        use_local_output=False,
                    ),
                )

            if self.concat_embeddings:
                parallelize_module(
                    self.map_projection,
                    tp_mesh,
                    ColwiseParallel(
                        input_layouts=Shard(1),
                        output_layouts=Shard(-1),
                        use_local_output=False,
                    ),
                )

        if (dp_mesh := device_mesh["data_parallel"]).size() > 1:
            assert dp_mesh.ndim == 1
            self._use_fsdp = True

            fsdp_config = {"mesh": dp_mesh}

            for idx, layer in enumerate(llm.layers):
                llm.layers[idx] = fully_shard(layer, **fsdp_config)
            self.embed_tokens = fully_shard(self.embed_tokens, **fsdp_config)
            self.llm = fully_shard(self.llm, **fsdp_config)
            self.lm_head = fully_shard(self.lm_head, **fsdp_config)
            self.perception = fully_shard(self.perception, **fsdp_config)
            self.speech_generation = fully_shard(self.speech_generation, **fsdp_config)
            if self.concat_embeddings:
                self.map_projection = fully_shard(self.map_projection, **fsdp_config)
    
def tokens_to_turn_str(tokens: torch.Tensor, lengths: torch.Tensor, tokenizer: AutoTokenizer, pad_id: int, delimiter: str = "|") -> list[str]:
    """
    [WIP] Convert tokens to turn-delimited text
    """
    ans = []
    for hyp_ids, hyp_len in zip(tokens.cpu(), lengths.cpu()):
        # Get tokens up to length
        hyp_ids = hyp_ids[:hyp_len]
        
        # Find segments between pad tokens
        segments = []
        start_idx = 0
        for i in range(len(hyp_ids)):
            if hyp_ids[i] == pad_id:
                # Convert segment before pad token to text
                if i > start_idx:
                    segment_tokens = hyp_ids[start_idx:i]
                    segment_tokens = segment_tokens[segment_tokens != pad_id]
                    if len(segment_tokens) > 0:
                        segments.append(tokenizer.ids_to_text(segment_tokens))
                start_idx = i + 1
        
        # Add final segment if any
        if start_idx < len(hyp_ids):
            segment_tokens = hyp_ids[start_idx:]
            segment_tokens = segment_tokens[segment_tokens != pad_id]
            if len(segment_tokens) > 0:
                segments.append(tokenizer.ids_to_text(segment_tokens))
        
        # Join segments with delimiter
        ans.append(delimiter.join(segments))
    return ans

class Retokenizer(nn.Module):
    def __init__(self, input_dim, output_dim, use_transformer=False, num_layers=1, num_heads=4, dropout=0.1):
        super().__init__()
        self.use_transformer = use_transformer

        # First projection to match transformer input size
        self.linear = nn.Linear(input_dim, output_dim)

        if use_transformer:
            # Use NeMo's TransformerEncoder with causal masking
            self.transformer = TransformerEncoder(
                num_layers=num_layers,
                hidden_size=output_dim,
                inner_size=output_dim * 4,  # Standard FFN size
                mask_future=True,  # Enable causal masking
                num_attention_heads=num_heads,
                attn_score_dropout=dropout,
                attn_layer_dropout=dropout,
                ffn_dropout=dropout,
                hidden_act="gelu",  # Use GELU activation
                pre_ln=False,  # Use post-LayerNorm for consistency
            )
        else:
            self.transformer = None

    def forward(self, x):
        x = self.linear(x)  # [B, T, output_dim]
        if self.transformer:
            # Create attention mask for the transformer
            # For causal transformer, we need a mask that allows each position to attend to itself and previous positions
            B, T, _ = x.shape
            # Create a mask where each position can attend to itself and all previous positions
            # This is handled automatically by the NeMo TransformerEncoder when mask_future=True
            attention_mask = torch.ones(B, T, device=x.device, dtype=torch.bool)
            x = self.transformer(x, attention_mask)  # [B, T, output_dim]
        return x

class SoftTokenMap(nn.Module):
    def __init__(self, asr_vocab_size, llm_embed_map, temperature=1.0):
        """
        Map ASR tokens to LLM embeddings using a soft mapping.

        Lower temperature (< 1) -> more deterministic mapping
        """
        super().__init__()
        self.llm_vocab_size = llm_embed_map.weight.shape[0] # v2
        self.mapping_logits = nn.Parameter(torch.randn(asr_vocab_size, self.llm_vocab_size)) # v1 -> v2
        self.llm_embed_map = llm_embed_map # v2 x d2
        self.temperature = temperature

    def forward(self, token_ids):  # token_ids: [B, T]
        weights = F.softmax(self.mapping_logits[token_ids] / self.temperature, dim=-1)  # [B, T, v2]
        mapped_embed = torch.matmul(weights, self.llm_embed_map.weight)  # [B, T, d2]
        return mapped_embed


class SoftEmbedMap(nn.Module):
    def __init__(self, asr_embed_dim, llm_embed_map, temperature=1.0):
        """
        Map ASR embeddings to LLM embeddings using a soft mapping.

        Lower temperature (< 1) -> more deterministic mapping
        """
        super().__init__()
        self.llm_vocab_size = llm_embed_map.weight.shape[0] # v2
        self.proj = nn.Linear(asr_embed_dim, self.llm_vocab_size) # v2 x d1
        self.llm_embed_map = llm_embed_map # v2 x d2
        self.temperature = temperature

    def forward(self, asr_embed):  # asr_embed:[B, T, D]
        logits = self.proj(asr_embed) # [B, T, v2]
        probs = F.softmax(logits / self.temperature, dim=-1) # [B, T, v2]
        mapped_embed = torch.matmul(probs, self.llm_embed_map.weight)  # [B, T, d2]
        return mapped_embed
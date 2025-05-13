# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import json
import os
import random
import string
import time
from typing import List

import librosa
import numpy as np
import soundfile as sf
import torch
import wandb
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from omegaconf import DictConfig, open_dict
from torch import nn
from torch.utils.data import get_worker_info

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.tts.data.text_to_speech_dataset_lhotse import MagpieTTSLhotseDataset, setup_tokenizers
from nemo.collections.tts.losses.aligner_loss import ForwardSumLoss
from nemo.collections.tts.models import AudioCodecModel
from nemo.collections.tts.modules import transformer_2501
from nemo.collections.tts.modules.aligner import AlignmentEncoder
from nemo.collections.tts.modules.magpietts_modules import SpecialAudioToken, LocalTransformerType, cosine_schedule
from nemo.collections.tts.parts.utils.helpers import (
    binarize_attention_parallel,
    get_mask_from_lengths,
    plot_alignment_to_numpy,
)
from nemo.collections.tts.parts.utils.tts_dataset_utils import stack_tensors
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging


def worker_init_fn(worker_id):
    # For mp.set_start_method("spawn", force=True)
    # The dataset class should be picklable, so we initialize non-picklable objects here
    logging.info(f"Worker {worker_id} initializing...")
    worker_info = get_worker_info()
    dataset = worker_info.dataset  # Get the dataset instance in this worker
    tokenizer, text_conditioning_tokenizer = setup_tokenizers(
        dataset.tokenizer_config, dataset.use_text_conditioning_tokenizer, mode=dataset.dataset_type
    )
    dataset.text_tokenizer = tokenizer
    dataset.text_conditioning_tokenizer = text_conditioning_tokenizer

class MagpieTTSModel(ModelPT):
    """
    Magpie-TTS Model Base Class used for training a TTS model that can generate audio codes from transcript and a context
    audio/text

    Supports multiple model types:

    - single_encoder_sv_tts: Transcript goes into the encoder and target audio goes to the decoder. Additionally,
    speaker_embedding of target audio (or context audio if provided) from TitaNet gets added to encoder
    output(all timesteps).

    - multi_encoder_context_tts: Transcript and context audio go to different encoders. Transcript encoding feeds to
    layers given by cfg.model.transcript_decoder_layers and the context encoding feeds into the layers given by
    context_decoder_layers .Also supports text context which gets encoded by the same encoder as context audio.
    Only one of context audio or contex text is supported.

    - decoder_context_tts: Text goes into the encoder; context & target audio go to the decoder. Also supports text
    context. Supports fixed sized context so we set context_duration_min and context_duration_max to the same
    value (5 seconds). Text context, which is usually shorter than number of codec frames of 5 second of audio, is
    padded to the max context duration in this model.

    - decoder_pretrain_synthesizer: This is the model type used for pretraining the decoder only on audio data using
    next frame prediction loss.
    """

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_devices

        # load codec
        codec_model = AudioCodecModel.restore_from(cfg.get('codecmodel_path'), strict=False)
        # del codec discriminator to free memory
        del codec_model.discriminator

        # Set up codebook configuration
        self.num_audio_codebooks = codec_model.num_codebooks
        self.codec_model_samples_per_frame = codec_model.samples_per_frame
        # Our codebooks start with actual audio codec tokens, followed by special tokens.
        # The `forced_*` options are for backward compatibility for models trained with older code.
        num_audio_tokens = codec_model.codebook_size
        self.audio_bos_id = cfg.get('forced_audio_bos_id', num_audio_tokens + SpecialAudioToken.AUDIO_BOS.value)
        self.audio_eos_id = cfg.get('forced_audio_eos_id', num_audio_tokens + SpecialAudioToken.AUDIO_EOS.value)
        self.context_audio_bos_id = cfg.get('forced_context_audio_bos_id', num_audio_tokens + SpecialAudioToken.AUDIO_CONTEXT_BOS.value)
        self.context_audio_eos_id = cfg.get('forced_context_audio_eos_id', num_audio_tokens + SpecialAudioToken.AUDIO_CONTEXT_EOS.value)
        self.num_all_tokens_per_codebook = cfg.get('forced_num_all_tokens_per_codebook',num_audio_tokens + len(SpecialAudioToken))
        self.mask_token_id = cfg.get('forced_mask_token_id', num_audio_tokens + SpecialAudioToken.MASK_TOKEN.value)
    
        # Setup tokenizer
        if hasattr(cfg, 'text_tokenizer'):
            # For backward compatibility for English-only models
            with open_dict(cfg):
                cfg.text_tokenizers = {"english_phoneme": cfg.text_tokenizer}
                del cfg['text_tokenizer']

        self.use_text_conditioning_encoder = cfg.get('use_text_conditioning_encoder', False)
        # TODO @xueyang: both tokenizers are only used to get some token ids. We
        # should kill them to save a small mount of mem resources since dataloader will initialize them
        # again after the worker processes are spawned.
        self.tokenizer, self.text_conditioning_tokenizer = setup_tokenizers(
            all_tokenizers_config=cfg.text_tokenizers,
            use_text_conditioning_tokenizer=self.use_text_conditioning_encoder,
            mode='train',
        )

        num_tokens_tokenizer = len(self.tokenizer.tokens)
        num_tokens = num_tokens_tokenizer + 2  # +2 for BOS and EOS
        self.bos_id = num_tokens - 2
        self.eos_id = num_tokens - 1

        self.model_type = cfg.get('model_type', 'single_encoder_sv_tts')

        self.pad_context_text_to_max_duration = self.model_type == 'decoder_context_tts'
        self.use_kv_cache_for_inference = cfg.get('use_kv_cache_for_inference', False)

        super().__init__(cfg=cfg, trainer=trainer)

        if self.use_text_conditioning_encoder:
            self.context_text_embedding = nn.Embedding(self.text_conditioning_tokenizer.vocab_size, cfg.embedding_dim)

        # This needs to happen after super().__init__()
        self._codec_model = codec_model
        self._codec_model.freeze()  #Lightning does requires_grad = False and self.eval()

        audio_embeddings = []
        for _ in range(self.num_audio_codebooks):
            audio_embeddings.append(nn.Embedding(self.num_all_tokens_per_codebook, cfg.embedding_dim))
        self.audio_embeddings = nn.ModuleList(audio_embeddings)

        if self.model_type != 'decoder_pretrain_synthesizer':
            # Decoder pretrain synthesizer doesn't have transcript encoder/text embeddings
            self.text_embedding = nn.Embedding(num_tokens, cfg.embedding_dim)
            self.encoder = transformer_2501.Transformer(**dict(cfg.encoder))

        self.decoder = transformer_2501.Transformer(**dict(cfg.decoder))
        self.final_proj = nn.Linear(cfg.decoder.d_model, self.num_audio_codebooks * self.num_all_tokens_per_codebook)

        self.local_transformer_type = LocalTransformerType(cfg.get('local_transformer_type', 'none').lower())
        logging.info(f"Local transformer type: {self.local_transformer_type}")
        if self.local_transformer_type != LocalTransformerType.NO_LT:
            local_transformer_hidden_dim = cfg.get('local_transformer_hidden_dim', 256)
            if local_transformer_hidden_dim != cfg.decoder.d_model:
                self.local_transformer_in_projection = nn.Linear(cfg.decoder.d_model, local_transformer_hidden_dim)
            else:
                self.local_transformer_in_projection = nn.Identity()
            self.local_transformer = transformer_2501.Transformer(
                n_layers=self.cfg.get('local_transformer_n_layers', 2),
                d_model=local_transformer_hidden_dim,
                d_ffn=local_transformer_hidden_dim*4,
                sa_n_heads=self.cfg.get('local_transformer_n_heads', 1),
                kernel_size=1,
                is_causal=self.local_transformer_type == LocalTransformerType.AR,
                max_length_causal_mask=self.num_audio_codebooks+2,
                use_learnable_pos_emb=True,
            )
            local_transformer_out_projections = []
            for _ in range(self.num_audio_codebooks):
                # Have a separate projection layer for each codebook, to distinguish between them
                local_transformer_out_projections.append(nn.Linear(local_transformer_hidden_dim, self.num_all_tokens_per_codebook))
            self.local_transformer_out_projections = nn.ModuleList(local_transformer_out_projections)

        if cfg.get('use_alignment_encoder', False):
            self.alignment_encoder = AlignmentEncoder(
                n_mel_channels=cfg.embedding_dim,
                n_text_channels=cfg.embedding_dim,
                dist_type="cosine",
                temperature=15.0,
            )

        if self.model_type == 'single_encoder_sv_tts':
            self._speaker_verification_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                model_name='titanet_large'
            )
            self._speaker_verification_model.freeze()  #Lightning does requires_grad = False and self.eval()
            self.speaker_projection_layer = nn.Linear(cfg.speaker_emb_dim, cfg.embedding_dim)
            self.transcript_decoder_layers = [
                idx for idx in range(cfg.decoder.n_layers)
            ]  # All layers are used for text
        elif self.model_type == 'multi_encoder_context_tts':
            self.transcript_decoder_layers = cfg.get('transcript_decoder_layers', [3, 4, 5, 6, 7, 8])
            self.context_decoder_layers = cfg.get(
                'context_decoder_layers', [0, 1, 2, 9, 10, 11]
            )  # For backward compatibility
            multi_encoder_mapping = [None for _ in range(cfg.decoder.n_layers)]
            for layer in self.transcript_decoder_layers:
                multi_encoder_mapping[layer] = 0  # 0 means text goes to this layer, 1 means context goes to this layer
            for layer in self.context_decoder_layers:
                multi_encoder_mapping[layer] = 1
            self.multi_encoder_mapping = multi_encoder_mapping
            self.context_encoder = transformer_2501.Transformer(**dict(cfg.context_encoder))
        elif self.model_type == 'decoder_context_tts':
            self.transcript_decoder_layers = [
                idx for idx in range(cfg.decoder.n_layers)
            ]  # All layers are used for text
        elif self.model_type == 'decoder_pretrain_synthesizer':
            assert cfg.alignment_loss_scale == 0.0, "Alignment loss is not supported for decoder pretrain synthesizer"
        else:
            raise ValueError(f"Unsupported model type {self.model_type}")

        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        alignment_loss_scale = cfg.get('alignment_loss_scale', 0.0)
        alignment_encoder_loss_scale = cfg.get('alignment_encoder_loss_scale', 0.0)
        if alignment_loss_scale > 0.0:
            self.alignment_loss = ForwardSumLoss(loss_scale=alignment_loss_scale)
        if alignment_encoder_loss_scale > 0.0:
            self.alignment_encoder_loss = ForwardSumLoss(loss_scale=alignment_encoder_loss_scale)



    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Only used for saving checkpoints. On save, we remove _speaker_verification_model and _codec_model
        from the checkpoint. The codec model is saved in a separate checkpoint.
        """
        if hasattr(self, '_no_state_dict') and self._no_state_dict:
            return {}
        # Don't save the speaker verification and codec model in the state dict
        state_dict = super().state_dict(destination, prefix, keep_vars)
        keys_substrings_to_exclude = ['_speaker_verification_model', '_codec_model']
        for key in list(state_dict.keys()):
            if any([substring in key for substring in keys_substrings_to_exclude]):
                del state_dict[key]
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        """
        Modify load_state_dict so that we don't restore weights to _speaker_verification_model and _codec_model when
        strict is True.
        When strict is False, we can call pytorch's load_state_dict.
        When strict is True, we loop through all parameters and rename them to enable loading.
        """
        if strict == False:
            super().load_state_dict(state_dict, strict=False)
        for name, child in self.named_children():
            if name in ['_speaker_verification_model', '_codec_model']:
                continue
            if any(param.numel() > 0 for param in child.parameters()):
                # If the module has parameters, we want to change the default mapping so that the state_dict gets
                # loaded.
                # Ex: state_dict[encoder.position_embeddings.weight] -> new_state_dict[position_embeddings.weight]
                new_state_dict = {}
                for key in state_dict.keys():
                    if key.startswith(f"{name}."):
                        new_state_dict[key[len(name)+1:]] = state_dict[key]  # +1 for '.'
                child.load_state_dict(new_state_dict)

    def audio_to_codes(self, audio, audio_len, audio_type='target'):
        # audio: (B, T)
        # audio_len: (B,)
        if audio_type == 'target':
            audio_eos_id = self.audio_eos_id
            audio_bos_id = self.audio_bos_id
        elif audio_type == 'context':
            audio_eos_id = self.context_audio_eos_id
            audio_bos_id = self.context_audio_bos_id
        else:
            raise ValueError(f"Received audio_type of {audio_type}. Must be `target` or `context`")

        self._codec_model.eval()
        with torch.no_grad(), torch.autocast(device_type=audio.device.type, dtype=torch.float32):
            codes, codes_len = self._codec_model.encode(audio=audio, audio_len=audio_len)
            # Add a timestep to begining and end of codes tensor
            bos_tensor = torch.full(
                (codes.size(0), codes.size(1), 1), audio_bos_id, dtype=codes.dtype, device=codes.device
            )
            pad_tensor = torch.full(
                (codes.size(0), codes.size(1), 1), 0, dtype=codes.dtype, device=codes.device
            )  # 0 is the padding token in the audio codebook
            codes = torch.cat([bos_tensor, codes, pad_tensor], dim=-1)
            # codes: (B, C, T')
            # codes_len: (B,)
            for idx in range(codes.size(0)):
                codes[idx, :, codes_len[idx] + 1] = audio_eos_id
            codes_len = codes_len + 2

            return codes.long(), codes_len.long()

    def codes_to_audio(self, codes, codes_len):
        # codes: (B, C, T')
        # codes_len: (B,)
        self._codec_model.eval()
        with torch.no_grad(), torch.autocast(device_type=codes.device.type, dtype=torch.float32):
            # Make a copy to avoid modifying the original tensor if it's used elsewhere
            codes_copy = codes.clone()
            # Replace eos and bos tokens with padding in the copied tensor
            codes_copy[codes == self.audio_bos_id] = 0  # zero is the padding token
            codes_copy[codes == self.audio_eos_id] = 0
            # Pass the modified integer token IDs
            audio, audio_len = self._codec_model.decode(tokens=codes_copy, tokens_len=codes_len)
            # audio: (B, T)
            # audio_len: (B,)
            return audio, audio_len

    def embed_audio_tokens(self, audio_tokens):
        # audio_tokens: (B, C, T')
        # Add and average the embeddings of the audio tokens across the codebooks
        audio_embedding = None
        for c in range(audio_tokens.size(1)):
            embedding = self.audio_embeddings[c](audio_tokens[:, c, :])
            if audio_embedding is None:
                audio_embedding = embedding
            else:
                audio_embedding = audio_embedding + embedding
        audio_embedding = audio_embedding / audio_tokens.size(1)
        return audio_embedding

    def get_speaker_embeddings(self, audio_16khz, audio_len_16khz):
        # audio_16khz: (B, T)
        # audio_len_16khz: (B,)
        self._speaker_verification_model.eval()
        with torch.no_grad():
            _, speaker_embeddings = self._speaker_verification_model.forward(
                input_signal=audio_16khz, input_signal_length=audio_len_16khz
            )
            return speaker_embeddings

    def compute_local_transformer_logits(self, dec_out, audio_codes_target, targets_offset_by_one=False):
        """
        Predicts the logits for all codebooks using the local transformer. Used in both autoregressive (AR) and MaskGit (MG) modes.
        This function is used in training and validation, not inference/sampling.
        The sequence layout is slightly different between AR and MG modes, as shown in the diagram below,
        (using an 8-codebook setup as an example):
        +------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
        | AR target  |    0    |    1    |    2    |    3    |    4    |    5    |    6    |    7    |   none  |
        +------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
        | MG target  |  none   |    0    |    1    |    2    |    3    |    4    |    5    |    6    |    7    |
        +------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
        |   Input    | Magpie  |    0    |    1    |    2    |    3    |    4    |    5    |    6    |    7    |
        |            | Latent  | or MASK | or MASK | or MASK | or MASK | or MASK | or MASK | or MASK | or MASK |
        +------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
        | Seq. Index |    0    |    1    |    2    |    3    |    4    |    5    |    6    |    7    |    8    |
        +------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
        
        dec_out: (B, T', E)
        audio_codes_target: (B, C, T')
        targets_offset_by_one: bool, if False, the target for index 0 is codebook 0, for index 1 is codebook 1, etc. (autoregressive)
                                     if True,  the target for index 1 is codebook 0, for index 2 is codebook 1, etc. (MaskGit)
        """
        dec_out_all = dec_out.reshape(-1, dec_out.size(-1)) # (B*T', E)
        local_transformer_input = [dec_out_all]
        for codebook_num in range(audio_codes_target.size(1)):
            codes = audio_codes_target[:, codebook_num] # (B, T')
            codes = codes.reshape(-1) # (B*T',)
            codebook_embedding = self.audio_embeddings[codebook_num](codes) # (B*T', E)
            local_transformer_input.append(codebook_embedding)

        local_transformer_input = torch.stack(local_transformer_input, dim=1) # (B*T', C+1, E)
        local_transformer_input = self.local_transformer_in_projection(local_transformer_input) # (B*T', C+1, 128)
        _mask = torch.ones( local_transformer_input.size(0), local_transformer_input.size(1), device=local_transformer_input.device)
        local_transformer_output = self.local_transformer(local_transformer_input, _mask)['output'] # (B*T', C+1, E)
        if not targets_offset_by_one:
            # for autoregressive local transformer the target for index 0 is codebook 0, for index 1 is codebook 1, etc.
            local_transformer_output = local_transformer_output[:, :-1, :] # (B*T', C, E)
        else:
            # for MaskGit the target for index **1** is codebook 0, for index 2 is codebook 1, etc.
            local_transformer_output = local_transformer_output[:, 1:, :] # (B*T', C, E)
        all_code_logits = []
        for codebook_num in range(audio_codes_target.size(1)):
            # Using a separate projection layer for each codebook (to distinguish between them)
            # Checked the time - this loop is not taking much time (compared to the local transformer forward pass)
            codebook_logits = self.local_transformer_out_projections[codebook_num](local_transformer_output[:, codebook_num, :]) # (B*T', num_all_tokens_per_codebook)
            all_code_logits.append(codebook_logits)
        all_code_logits = torch.cat(all_code_logits, dim=1) # (B*T', num_codebooks * num_all_tokens_per_codebook)

        all_code_logits = all_code_logits.view(
            audio_codes_target.size(0), audio_codes_target.size(2), -1
        ) # (B, T', C * num_all_tokens_per_codebook)

        return all_code_logits

    def maskgit_create_random_mask(self, codes):
        """
        Creates a mask where True indicates the positions that should be replaced with a MASK_TOKEN.
        """
        # Codes: (B, C, T)
        B,C,T = codes.shape
        # get a uniform random vector uniformly sampled from [0,1) ## Todo does it need to be inclusive on the right?
        rand_values = torch.rand(B,T, device=codes.device)
        # apply the cosine schedule 
        frac_masked = cosine_schedule(rand_values)
        # how many positions to mask
        n_masked = torch.ceil(frac_masked * C).long() # B,T
        # start from all unmasked
        mask = torch.zeros_like(codes, dtype=torch.bool)
        # The code further below is the vectorized version of this:
        #  for b in range(B):
        #      for t in range(T):
        #          if n_masked[b,t] > 0:
        #              # get a random permutation of the codebook indices
        #              perm = torch.randperm(C)
        #              # mask the top n_masked positions
        #              mask[b, perm[:n_masked[b,t]], t] = True
        #
        # Create random permutations 
        random_permutations = torch.argsort(torch.rand(B, C, T, device=codes.device), dim=1)  # (B, C, T)        
        # Create a mask tensor where each position indicates if it should be masked        
        mask_indices = torch.arange(C, device=codes.device).view(1, C, 1)
        mask = mask_indices < n_masked.view(B, 1, T) # (B, C, T)
        # Apply the random permutations to the mask
        mask = torch.gather(mask, 1, random_permutations)
    
        return mask # (B, C, T)
    
    def maskgit_apply_random_mask(self, codes):
        # Randomly replaces some codes with the MASK_TOKEN with a proportion following the cosine schedule.
        # Codes: (B, C, T)        
        mask = self.maskgit_create_random_mask(codes)
        ## replace some tokens with MASK_TOKEN
        codes_with_mask = torch.where(mask, self.mask_token_id, codes)
        return codes_with_mask, mask

    def compute_loss(self, logits, audio_codes, audio_codes_lens, mask_tokens_mask=None):
        """
        Computes the audio codebook loss. Used by
        (1) The main Magpie-TTS transformer
        (2) The local transformer, for both autoregressive and MaskGit methods
        
        logits: (B, T', num_codebooks * num_tokens_per_codebook)
        audio_codes: (B, C, T')
        audio_codes_lens: (B,)
        mask_tokens_mask: (B, C, T') True for tokens that were replaced with the MASK_TOKEN and should
                                     therefore be the only ones included in the loss computation.
        """
        loss_mask = get_mask_from_lengths(audio_codes_lens)
        if mask_tokens_mask is not None:
            # For MaskGit we only compute loss for the masked tokens.
            # *Both* conditions must be true:
            # 1. the token is masked
            # 2. the token is not padding
            loss_mask = loss_mask.unsqueeze(1) * mask_tokens_mask
            if not loss_mask.any():
                # Without this we were very rarely getting NaNs in the loss
                logging.warning("No tokens valid were found in compute_loss()!")
                return torch.tensor(0.0, device=loss_mask.device), loss_mask 
        else:            
            # repeat loss mask for each codebook to simplify code below
            loss_mask = loss_mask.unsqueeze(1).repeat(1, audio_codes.size(1), 1)
        total_codebook_loss = None
        for codebook in range(audio_codes.size(1)):
            si = codebook * self.num_all_tokens_per_codebook
            ei = si + self.num_all_tokens_per_codebook
            codebook_logits = logits[:, :, si:ei]  # (B, T', num_tokens_per_codebook)
            codebook_targets = audio_codes[:, codebook]  # (B, T')
            codebook_loss = self.cross_entropy_loss(
                codebook_logits.permute(0, 2, 1), codebook_targets  # (B, num_tokens_per_codebook, T')
            )  # (B, T')
            codebook_loss = codebook_loss * loss_mask[:, codebook, :]
            codebook_loss = codebook_loss.sum() / loss_mask[:, codebook, :].sum()
            if total_codebook_loss is None:
                total_codebook_loss = codebook_loss
            else:
                total_codebook_loss = total_codebook_loss + codebook_loss

        total_codebook_loss = total_codebook_loss / audio_codes.size(1)
        return total_codebook_loss, loss_mask

    def forward(self, dec_input_embedded, dec_input_mask, cond, cond_mask, attn_prior, multi_encoder_mapping):
        decoder_out = self.decoder(
            dec_input_embedded,
            dec_input_mask,
            cond=cond,
            cond_mask=cond_mask,
            attn_prior=attn_prior,
            multi_encoder_mapping=multi_encoder_mapping,
        )
        attn_probabilities = decoder_out['attn_probabilities']
        all_code_logits = self.final_proj(decoder_out['output'])  # (B, T', num_codebooks * num_tokens_per_codebook)
        return all_code_logits, attn_probabilities, decoder_out['output']

    def logits_to_audio_codes(self, all_code_logits, audio_codes_lens):
        # all_code_logits: (B, T', num_codebooks * num_tokens_per_codebook)
        # audio_codes_lens: (B,)
        all_preds = []
        for idx in range(self.num_audio_codebooks):
            si = idx * self.num_all_tokens_per_codebook
            ei = si + self.num_all_tokens_per_codebook
            codebook_logits = all_code_logits[:, :, si:ei]
            codebook_probs = torch.softmax(codebook_logits, dim=-1)  # (B, T', num_tokens_per_codebook)
            # argmax to get the tokens
            codebook_preds = torch.argmax(codebook_probs, dim=-1)  # (B, T')
            all_preds.append(codebook_preds)

        all_preds = torch.stack(all_preds, dim=1)  # (B, C, T')
        audio_mask = get_mask_from_lengths(audio_codes_lens)
        all_preds = all_preds * audio_mask.unsqueeze(1)

        return all_preds

    def local_transformer_sample_maskgit(self, dec_output, temperature=0.7, topk=80, unfinished_items={}, finished_items={}, use_cfg=False, cfg_scale=1.0, n_steps=3):
        """
        Sample codes for one timestep from the local transformer using MaskGit.
        """
        # dec_output: (B, E)
        device = dec_output.device
        # disable KV cache since our transformer is not causal
        self.local_transformer.reset_cache(use_cache=False)
        dec_output = dec_output.unsqueeze(1) # (B, 1, E)
        local_transformer_input_init = self.local_transformer_in_projection(dec_output) # (B, 1, D) where D is the dimension of the local transformer
        C = self.num_audio_codebooks
        B = dec_output.size(0)

        min_confidence = float("-inf")
        max_confidence = 10000 # this needs to be large enough that unmasked items will always remain unmasked. # TODO @rfejgin: use float('inf')?
        confidences = min_confidence * torch.ones(B, C, device=device)
        # initialize to all masked
        codes = self.mask_token_id * torch.ones((B, C), device=device, dtype=torch.long)
        sampled_codes = codes.clone()
        for step in range(n_steps):
            # get mask fraction
            frac_masked = cosine_schedule(torch.tensor(step / (n_steps)))
            # how many codebooks to mask
            n_masked = torch.ceil(C * frac_masked).long() # TODO @rfejgin: should we force this to be initialized to exactly `C` (to avoid numerical issues)?
            n_unmasked = C - n_masked
            # pick top-confidence codebooks up to n_unmasked
            _, topk_indices = torch.topk(confidences, k=n_unmasked, dim=1)

            # replace masks of the top-k confident codebooks with the the codes that were sampled for them
            unmasked_codes = torch.gather(sampled_codes, dim=1, index=topk_indices)
            codes.scatter_(dim=1, index=topk_indices, src=unmasked_codes)
            
            # build transformer input 
            local_transformer_input = local_transformer_input_init
            for codebook_num in range(C):
                next_local_transformer_input = self.audio_embeddings[codebook_num](codes[:, codebook_num]).unsqueeze(1) # (B, 1, 768)
                next_local_transformer_input = self.local_transformer_in_projection(next_local_transformer_input) # (B, 1, d_local)
                local_transformer_input = torch.cat([local_transformer_input, next_local_transformer_input], dim=1) # (B, codebook_num+1, d_local)

            # run transformer
            _mask = torch.ones(B, C+1, device=device)
            local_transformer_output = self.local_transformer(local_transformer_input, _mask)['output'] # (B, C+1, d_local)
            
            # get logits
            logits = []
            for codebook_num in range(C):
                # The `codebook_num+1` is to drop first position which corresponds to the magpie latent
                codebook_logits = self.local_transformer_out_projections[codebook_num](local_transformer_output[:, codebook_num+1, :]) # (B, num_audio_tokens_per_codebook)
                logits.append(codebook_logits)
            logits = torch.stack(logits, dim=1) # (B, C, num_audio_tokens_per_codebook)

            # apply CFG
            if use_cfg:
                actual_batch_size = logits.size(0) // 2
                conditional_logits = logits[:actual_batch_size]
                unconditional_logits = logits[actual_batch_size:]
                cfg_logits = cfg_scale * conditional_logits +  (1.0 - cfg_scale) * unconditional_logits
                logits[:actual_batch_size] = cfg_logits

            # handle unfinished and finished items
            for item_idx in unfinished_items:
                logits[item_idx, self.audio_eos_id] = float('-inf')
            for item_idx in finished_items:
                logits[item_idx, :, :] = float('-inf')
                logits[item_idx, :, self.audio_eos_id] = 0.0
            
            # sample with top-k
            logits_topk = torch.topk(logits, topk, dim=-1)[0] # (B, C, topk)
            indices_to_remove = logits < logits_topk[:, :, -1].unsqueeze(-1) # (B, C, num_audio_tokens_per_codebook)
            logits_rescored = logits.clone()
            logits_rescored[indices_to_remove] = float('-inf')
            probs = torch.softmax(logits_rescored / temperature, dim=-1) # (B, C, num_audio_tokens_per_codebook)
            sampled_codes = torch.multinomial(probs.view(B*C, -1), 1).view(B, C)
            if use_cfg:
                # TODO @rfejgin: why do we need to keep second half of the batch? can probably optimize this
                sampled_codes[actual_batch_size:] = sampled_codes[:actual_batch_size]
                probs[actual_batch_size:] = probs[:actual_batch_size]
            confidences  = torch.gather(probs, dim=2, index=sampled_codes.unsqueeze(-1)).squeeze(-1)

            # set confidence to max for unmasked codebooks so that they will remain unmasked
            confidences.scatter_(index=topk_indices, dim=1, src=max_confidence*torch.ones_like(topk_indices, dtype=torch.float))

            # replace entries in sampled_codes with previously unmasked codebooks
            sampled_codes.scatter_(dim=1, index=topk_indices, src=unmasked_codes)
            # optionally: add noise to confidences here (as in token-critic paper) (not implemented)
        
        codes = sampled_codes
        assert not (codes == self.mask_token_id).any(), f"Codes contain mask tokens after completion of MaskGit sampling"
        if use_cfg:
            codes = codes[:actual_batch_size]
        return codes

    def local_transformer_sample_autoregressive(self, dec_output, temperature=0.7, topk=80, unfinished_items={}, finished_items={}, use_cfg=False, cfg_scale=1.0):
        # dec_output: (B, E)
        self.local_transformer.reset_cache(use_cache=True)
        dec_output = dec_output.unsqueeze(1) # (B, 1, E)
        local_transformer_input = self.local_transformer_in_projection(dec_output) # (B, 1, 128)
        all_preds = []
        for codebook_num in range(self.num_audio_codebooks):
            _mask = torch.ones( local_transformer_input.size(0), local_transformer_input.size(1), device=local_transformer_input.device)
            local_transformer_output = self.local_transformer(local_transformer_input, _mask)['output'] # (B, T, 128)
            codebook_logits = self.local_transformer_out_projections[codebook_num](local_transformer_output[:, -1, :]) # (B, num_all_tokens_per_codebook)
            if use_cfg:
                actual_batch_size = codebook_logits.size(0) // 2
                conditional_logits = codebook_logits[:actual_batch_size]
                unconditional_logits = codebook_logits[actual_batch_size:]
                cfg_logits = cfg_scale * conditional_logits +  (1.0 - cfg_scale) * unconditional_logits
                codebook_logits[:actual_batch_size] = cfg_logits

            for item_idx in unfinished_items:
                codebook_logits[item_idx, self.audio_eos_id] = float('-inf')
            for item_idx in finished_items:
                codebook_logits[item_idx, :] = float('-inf')
                codebook_logits[item_idx, self.audio_eos_id] = 0.0

            codebook_logits_topk = torch.topk(codebook_logits, topk, dim=-1)[0] # (B, topk)
            indices_to_remove = codebook_logits < codebook_logits_topk[:, -1].unsqueeze(-1) # (B, num_tokens_per_codebook)
            codebook_logits_rescored = codebook_logits.clone()
            codebook_logits_rescored[indices_to_remove] = float('-inf')
            codebook_probs = torch.softmax(codebook_logits_rescored / temperature, dim=-1) # (B, num_tokens_per_codebook)
            codebook_preds = torch.multinomial(codebook_probs, 1) # (B, 1)
            if use_cfg:
                codebook_preds[actual_batch_size:] = codebook_preds[:actual_batch_size]
            all_preds.append(codebook_preds)
            next_local_transformer_input = self.audio_embeddings[codebook_num](codebook_preds.squeeze(-1)).unsqueeze(1) # (B, 1, 128)
            next_local_transformer_input = self.local_transformer_in_projection(next_local_transformer_input) # (B, 1, 128)
            local_transformer_input = torch.cat([local_transformer_input, next_local_transformer_input], dim=1) # (B, T+1, 128)

        all_preds = torch.cat(all_preds, dim=1).long() # (B, num_codebooks)
        if use_cfg:
            all_preds = all_preds[:actual_batch_size]

        return all_preds


    def sample_codes_from_logits(self, all_code_logits_t, temperature=0.7, topk=80, unfinished_items={}, finished_items={}):
        # all_code_logits_t: (B, num_codebooks * num_tokens_per_codebook), logits at a given timestep
        all_preds = []
        for idx in range(self.num_audio_codebooks):
            si = idx * self.num_all_tokens_per_codebook
            ei = si + self.num_all_tokens_per_codebook
            codebook_logits = all_code_logits_t[:, si:ei]  # (B, num_tokens_per_codebook)
            for item_idx in unfinished_items:
                codebook_logits[item_idx, self.audio_eos_id] = float('-inf')
            for item_idx in finished_items:
                codebook_logits[item_idx, :] = float('-inf')
                codebook_logits[item_idx, self.audio_eos_id] = 0.0
            codebook_logits_topk = torch.topk(codebook_logits, topk, dim=-1)[0]  # (B, topk)
            indices_to_remove = codebook_logits < codebook_logits_topk[:, -1].unsqueeze(
                -1
            )  # (B, num_tokens_per_codebook)
            codebook_logits_rescored = codebook_logits.clone()
            codebook_logits_rescored[indices_to_remove] = float('-inf')

            codebook_probs = torch.softmax(codebook_logits_rescored / temperature, dim=-1)  # (B, num_tokens_per_codebook)
            codebook_preds = torch.multinomial(codebook_probs, 1)  # (B, 1)
            all_preds.append(codebook_preds)
        all_preds = torch.cat(all_preds, dim=1).long()  # (B, num_codebooks)
        return all_preds

    def log_attention_probs(self, attention_prob_matrix, audio_codes_lens, text_lens, prefix="", dec_context_size=0):
        # attention_prob_matrix List of (B, C, audio_timesteps, text_timesteps)
        wandb_images_log = {}

        with torch.no_grad():
            attention_prob_matrix = torch.cat(attention_prob_matrix, dim=1)  # (B, C, audio_timesteps, text_timesteps)
            attention_prob_matrix_mean = attention_prob_matrix.mean(dim=1)  # (B, audio_timesteps, text_timesteps)

            for logger in self.loggers:
                is_wandb = isinstance(logger, WandbLogger)
                is_tb = isinstance(logger, TensorBoardLogger)
                if not is_wandb and not is_tb:
                    raise ValueError(f"Invalid logger type for image logging: {type(logger)}. Only `WandbLogger` and `TensorBoardLogger` are supported.")

                wandb_images_log[f"Image/{prefix}/attention_matrix"] = list()
                for idx in range(min(3, attention_prob_matrix_mean.size(0))):
                    item_attn_matrix = attention_prob_matrix_mean[idx][
                        dec_context_size : dec_context_size + audio_codes_lens[idx], : text_lens[idx]
                    ]
                    item_attn_matrix = item_attn_matrix.detach().cpu().numpy()
                    img_np = plot_alignment_to_numpy(item_attn_matrix.T)

                    if is_wandb:
                        wandb_images_log[f"Image/{prefix}/attention_matrix"].append(wandb.Image(img_np, caption=f"Example_{idx}"))

                    if is_tb:
                        logger.experiment.add_image(
                            f'{prefix}/attention_matrix/Example_{idx}',
                            img_np,
                            global_step=self.global_step,
                            dataformats="HWC",
                        )

        return wandb_images_log

    def log_val_audio_example(
        self,
        logits,
        target_audio_codes,
        audio_codes_lens_target,
        context_audio_codes=None,
        context_audio_codes_lens=None,
    ):
        wandb_audio_log = {}

        pred_audio_codes = self.logits_to_audio_codes(logits, audio_codes_lens_target)
        pred_audio, pred_audio_lens = self.codes_to_audio(pred_audio_codes, audio_codes_lens_target)
        target_audio, target_audio_lens = self.codes_to_audio(target_audio_codes, audio_codes_lens_target)

        context_audio, context_audio_lens = None, None
        if context_audio_codes is not None and context_audio_codes.shape[2] > 3:
            # > 3 ensures, it is a valid context audio tensor (and not dummy tensor used in text context)
            context_audio, context_audio_lens = self.codes_to_audio(context_audio_codes, context_audio_codes_lens)

        for logger in self.loggers:
            is_wandb = isinstance(logger, WandbLogger)
            is_tb = isinstance(logger, TensorBoardLogger)
            if not is_wandb and not is_tb:
                raise ValueError(f"Invalid logger type for audio logging: {type(logger)}. Only `WandbLogger` and `TensorBoardLogger` are supported.")

            for idx in range(min(3, pred_audio.size(0))):
                pred_audio_np = pred_audio[idx].float().detach().cpu().numpy()
                target_audio_np = target_audio[idx].float().detach().cpu().numpy()
                pred_audio_np = pred_audio_np[: pred_audio_lens[idx]]
                target_audio_np = target_audio_np[: target_audio_lens[idx]]
                context_audio_np = None
                if context_audio is not None:
                    context_audio_np = context_audio[idx].float().detach().cpu().numpy()
                    context_audio_np = context_audio_np[: context_audio_lens[idx]]

                if is_wandb:
                    wandb_audio_log[f"Audio/Example_{idx}"] = list()
                    if context_audio_np is not None:
                        wandb_audio_log[f"Audio/Example_{idx}"].append(wandb.Audio(context_audio_np, sample_rate=self.cfg.sample_rate, caption="context"))
                    wandb_audio_log[f"Audio/Example_{idx}"].append(wandb.Audio(pred_audio_np, sample_rate=self.cfg.sample_rate, caption="prediction"))
                    wandb_audio_log[f"Audio/Example_{idx}"].append(wandb.Audio(target_audio_np, sample_rate=self.cfg.sample_rate, caption="target"))

                if is_tb:
                    if context_audio_np is not None:
                        logger.experiment.add_audio(
                            f'Example_{idx}/context',
                            context_audio_np,
                            global_step=self.global_step,
                            sample_rate=self.cfg.sample_rate,
                        )
                    logger.experiment.add_audio(
                        f'Example_{idx}/prediction',
                        pred_audio_np,
                        global_step=self.global_step,
                        sample_rate=self.cfg.sample_rate,
                    )
                    logger.experiment.add_audio(
                        f'Example_{idx}/target',
                        target_audio_np,
                        global_step=self.global_step,
                        sample_rate=self.cfg.sample_rate,
                    )

        return wandb_audio_log

    def scale_prior(self, prior, global_step):
        if prior is None:
            return None
        prior_end_step = self.cfg.prior_end_step
        prior_scaledown_start_step = self.cfg.prior_scaledown_start_step
        if global_step < prior_scaledown_start_step:
            return prior
        elif global_step >= prior_end_step:
            indefinite_prior_prob = self.cfg.get('indefinite_prior_prob', 0.0)
            if random.random() < indefinite_prior_prob:
                print("Using Prior")
                return prior
            else:
                print("Not using Prior")
                return None
        else:
            with torch.no_grad():
                # Interpolate between all ones and the prior
                residual = 1.0 - prior
                new_prior = prior + (
                    residual
                    * (global_step - prior_scaledown_start_step)
                    / (prior_end_step - prior_scaledown_start_step)
                )
                return new_prior

    def compute_alignment_loss(self, attention_scores, text_lens, audio_lens, dec_context_size=0):
        # attention scores: List of (B, C, audio_timesteps, text_timesteps)
        attention_scores_combined = torch.cat(attention_scores, dim=1)  # (B, C, audio_timesteps, text_timesteps)
        attention_scores_mean = attention_scores_combined.mean(
            dim=1, keepdim=True
        )  # (B, 1, audio_timesteps, text_timesteps)
        attention_scores_mean = attention_scores_mean[
            :, :, dec_context_size:, :
        ]  # Remove the context audio embeddings from the attention scores
        alignment_loss = self.alignment_loss(
            attn_logprob=attention_scores_mean, in_lens=text_lens, out_lens=audio_lens
        )
        return alignment_loss

    def prepare_context_tensors(self, batch):
        dec_context_size = 0
        additional_decoder_input = None
        addtional_decoder_mask = None
        context_audio_codes = None
        context_audio_codes_lens = None
        _attn_prior = None
        attn_prior = None
        cond = None
        cond_mask = None
        multi_encoder_mapping = None
        text = None
        text_lens = None

        # self.model_type must be one of
        # [single_encoder_sv_tts, multi_encoder_context_tts, decoder_context_tts, decoder_pretrain_synthesizer]
        if self.model_type != 'decoder_pretrain_synthesizer':
            text = batch['text']
            text_lens = batch['text_lens']
            text_embedded = self.text_embedding(text)  # (B, T, E)
            text_mask = get_mask_from_lengths(text_lens)  # (B, T)
            text_encoder_out = self.encoder(text_embedded, text_mask, cond=None, cond_mask=None)['output']  # (B, T, E)
            _attn_prior = batch.get('align_prior_matrix', None)
            _attn_prior = self.scale_prior(_attn_prior, self.global_step)

        if self.model_type == 'single_encoder_sv_tts':
            target_audio_16khz = batch['audio_16khz']
            target_audio_lens_16khz = batch['audio_lens_16khz']
            speaker_embeddings = self.get_speaker_embeddings(target_audio_16khz, target_audio_lens_16khz)
            speaker_embeddings_projected = self.speaker_projection_layer(speaker_embeddings)
            cond = text_encoder_out + speaker_embeddings_projected.unsqueeze(1)
            cond_mask = text_mask
            multi_encoder_mapping = None
            attn_prior = _attn_prior
        elif self.model_type in ['multi_encoder_context_tts', 'decoder_context_tts']:
            if 'context_audio_codes' in batch:
                context_audio_codes = batch['context_audio_codes']
                context_audio_codes_lens = batch['context_audio_codes_lens']
            else:
                context_audio_codes, context_audio_codes_lens = self.audio_to_codes(
                    batch['context_audio'], batch['context_audio_lens'], audio_type='context'
                )
            context_audio_embedded = self.embed_audio_tokens(context_audio_codes)  # (B, T', E)

            if self.use_text_conditioning_encoder:
                context_text_tokens = batch['context_text_tokens']
                context_text_lens = batch['context_text_tokens_lens']
                context_text_embedded = self.context_text_embedding(context_text_tokens)  # (B, L, E)
                # Pad context_audio_embedded or context_text_embedded so that they have same number of timesteps
                if context_audio_embedded.size(1) < context_text_embedded.size(1):
                    padding = torch.zeros(
                        context_audio_embedded.size(0),
                        context_text_embedded.size(1) - context_audio_embedded.size(1),
                        context_audio_embedded.size(2),
                        device=context_audio_embedded.device,
                    )
                    context_audio_embedded = torch.cat([context_audio_embedded, padding], dim=1)
                elif context_audio_embedded.size(1) > context_text_embedded.size(1):
                    padding = torch.zeros(
                        context_text_embedded.size(0),
                        context_audio_embedded.size(1) - context_text_embedded.size(1),
                        context_text_embedded.size(2),
                        device=context_text_embedded.device,
                    )
                    context_text_embedded = torch.cat([context_text_embedded, padding], dim=1)  # (B, T, E)
                has_text_context = batch['has_text_context'].unsqueeze(-1).unsqueeze(-1).float()  # (B, 1, 1)
                context_input_embedded = (
                    has_text_context * context_text_embedded + (1 - has_text_context) * context_audio_embedded
                )
                context_input_lens = (
                    batch['has_text_context'].float() * context_text_lens
                    + (1 - batch['has_text_context'].float()) * context_audio_codes_lens
                )  # (B,)
            else:
                context_input_embedded = context_audio_embedded
                context_input_lens = context_audio_codes_lens

            context_mask = get_mask_from_lengths(context_input_lens)

            if self.model_type == 'multi_encoder_context_tts':
                context_embeddings = self.context_encoder(
                    context_input_embedded, context_mask, cond=None, cond_mask=None
                )['output']
                cond = [text_encoder_out, context_embeddings]
                cond_mask = [text_mask, context_mask]
                multi_encoder_mapping = self.multi_encoder_mapping
                attn_prior = [_attn_prior, None]

            elif self.model_type == 'decoder_context_tts':
                dec_context_size = context_mask.size(1)
                context_embeddings = context_input_embedded
                attn_prior = _attn_prior
                if attn_prior is not None:
                    # B, audio_timesteps, text_timesteps
                    padding_zeros = torch.zeros(
                        attn_prior.size(0), dec_context_size, attn_prior.size(2), device=attn_prior.device
                    )
                    attn_prior = torch.cat([padding_zeros, attn_prior], dim=1)
                cond = text_encoder_out
                cond_mask = text_mask
                multi_encoder_mapping = None
                additional_decoder_input = context_embeddings
                addtional_decoder_mask = context_mask
        elif self.model_type == 'decoder_pretrain_synthesizer':
            pass
        else:
            raise ValueError(f"Unsupported model type {self.model_type}")

        if attn_prior is not None and self.cfg.get('ctc_prior_layer_ids', None) is not None:
            ctc_prior_layer_ids = self.cfg.ctc_prior_layer_ids
            # Convert prior to a list of tensors, one for each layer
            # Set None for layers not in ctc_prior_layer_ids
            if self.model_type == 'multi_encoder_context_tts':
                text_attn_prior = [attn_prior[0] if layer_idx in ctc_prior_layer_ids else None for layer_idx in range(self.cfg.decoder.n_layers) ]
                attn_prior = [text_attn_prior, attn_prior[1]]
            else:
                attn_prior = [attn_prior if layer_idx in ctc_prior_layer_ids else None for layer_idx in range(self.cfg.decoder.n_layers) ]

        return {
            'beta_binomial_attn_prior': batch.get('align_prior_matrix', None),
            'text_encoder_out': text_encoder_out,
            'cond': cond,
            'cond_mask': cond_mask,
            'attn_prior': attn_prior,
            'prior_used': _attn_prior is not None,
            'multi_encoder_mapping': multi_encoder_mapping,
            'additional_decoder_input': additional_decoder_input,
            'addtional_decoder_mask': addtional_decoder_mask,
            'dec_context_size': dec_context_size,
            'text': text,
            'text_embedded': text_embedded,
            'text_mask': text_mask,
            'text_lens': text_lens,
            'context_audio_codes': context_audio_codes,
            'context_audio_codes_lens': context_audio_codes_lens,
        }

    def replace_beta_binomial_prior_with_binarized(self, attn_prior, aligner_attn_hard):
        # aligner_attn_hard B, audio_timesteps, text_timesteps
        if self.model_type == 'multi_encoder_context_tts':
            text_attn_prior = attn_prior[0]
        else:
            text_attn_prior = attn_prior

        assert text_attn_prior is not None, "Prior is None"

        if isinstance(text_attn_prior, list):
            # Layer wise prior
            prior_updated = False
            for idx, prior in enumerate(text_attn_prior):
                if prior is not None:
                    text_attn_prior[idx][:,-aligner_attn_hard.size(1):,:] = aligner_attn_hard
                    prior_updated = True
            assert prior_updated, "Did not find any prior to update"
        else:
            # Same prior for all layers
            text_attn_prior[:,-aligner_attn_hard.size(1):,:] = aligner_attn_hard

        if self.model_type == 'multi_encoder_context_tts':
            attn_prior[0] = text_attn_prior
        else:
            attn_prior = text_attn_prior

        return attn_prior

    def get_binarized_prior_matrix(self, aligner_attn_soft, audio_lens, text_lens):
        # aligner_attn_soft B, 1, audio_timesteps, text_timesteps
        if self.cfg.get('binarize_attn_method', 'argmax') == 'nemo_binarize':
            binarize_repeat_audio_factor = self.cfg.get('binarize_repeat_audio_factor', 2)
            aligner_attn_soft_repeated = aligner_attn_soft.repeat_interleave(binarize_repeat_audio_factor, dim=2) # B, 1, 2*audio_timesteps, text_timesteps
            aligner_attn_hard = binarize_attention_parallel(aligner_attn_soft_repeated, text_lens, audio_lens*binarize_repeat_audio_factor).squeeze(1) # B, 2*audio_timesteps, text_timesteps
            aligner_attn_hard = aligner_attn_hard[:, ::2, :] # B, audio_timesteps, text_timesteps
        else:
            print("Binaraizing attention using argmax")
            aligner_attn_hard = torch.argmax(aligner_attn_soft.squeeze(1), dim=-1)
            aligner_attn_hard = torch.nn.functional.one_hot(aligner_attn_hard, num_classes=aligner_attn_soft.size(-1)).float()

        prior_future_decay = self.cfg.get('prior_future_decay', 1.0)
        prior_past_decay = self.cfg.get('prior_past_decay', 1.0)
        binarized_prior_epsilon = self.cfg.get('binarized_prior_epsilon', 0.0)
        aligner_attn_hard_wider = aligner_attn_hard + binarized_prior_epsilon

        for future_timestep in range(self.cfg.get('prior_future_context', 1)):
            decay_factor = prior_future_decay ** (future_timestep + 1)
            aligner_attn_hard_wider[:,:,future_timestep+1:] += decay_factor * aligner_attn_hard[:,:,:-(future_timestep+1)]

        for past_timestep in range(self.cfg.get('prior_past_context', 1)):
            decay_factor = prior_past_decay ** (past_timestep + 1)
            aligner_attn_hard_wider[:,:,:-past_timestep-1] += decay_factor * aligner_attn_hard[:,:,past_timestep+1:]

        aligner_attn_hard_wider = torch.clamp(aligner_attn_hard_wider, 0.0, 1.0)

        return aligner_attn_hard_wider

    def prepare_dummy_cond_for_cfg(self, cond, cond_mask, additional_decoder_input, additional_dec_mask):
        dummy_additional_decoder_input = None
        dummy_additional_dec_mask = None
        if additional_decoder_input is not None:
            dummy_additional_decoder_input = torch.zeros_like(additional_decoder_input)
            # all ones mask means dont ignore any timesteps (so that it is consistent with usual decoder mask)
            dummy_additional_dec_mask = torch.ones_like(additional_dec_mask)

        if isinstance(cond, list):
            # multi encoder conditioning
            dummy_cond = [torch.zeros_like(cond_item) for cond_item in cond]
            attn_prior = [None for _ in cond]
            dummy_mask = []
            for mask_item in cond_mask:
                # ignore all timesteps except the first one
                mask = torch.zeros_like(mask_item)
                mask[:, 0] = 1  # Make first timestep all zeros
                dummy_mask.append(mask)

        elif isinstance(cond, torch.Tensor):
            # single encoder conditioning
            dummy_cond = torch.zeros_like(cond)
            dummy_mask = torch.zeros_like(cond_mask)
            dummy_mask[:, 0] = 1  # ignore all timesteps except the first one
            attn_prior = None
        else:
            raise ValueError(f"Unsupported type for cond {type(cond)}")

        return dummy_cond, dummy_mask, dummy_additional_decoder_input, dummy_additional_dec_mask, attn_prior

    def process_batch(self, batch, mode="train"):
        context_tensors = self.prepare_context_tensors(batch)
        disable_alignment_loss = False
        if 'audio_codes' not in batch:
            audio_codes, audio_codes_lens = self.audio_to_codes(batch['audio'], batch['audio_lens'])
        else:
            audio_codes = batch['audio_codes']
            audio_codes_lens = batch['audio_codes_lens']

        audio_codes_input = audio_codes[:, :, :-1]  # B, C, T'
        audio_codes_target = audio_codes[:, :, 1:]
        audio_codes_lens_input = audio_codes_lens_target = audio_codes_lens - 1
        audio_codes_embedded_all = self.embed_audio_tokens(audio_codes) # (B, T, E) # Computing this to be use in the alignment encoder
        audio_codes_embedded = audio_codes_embedded_all[:, :-1, :] # (B, T', E) Input to the decoder

        audio_codes_mask = get_mask_from_lengths(audio_codes_lens_input)
        use_cfg = (
            (self.cfg.get('cfg_unconditional_prob', 0.0) > 0.0)
            and (mode == "train")
            and (context_tensors['cond'] is not None)
        )
        if use_cfg and torch.rand(1).item() < self.cfg.cfg_unconditional_prob:
            cond, cond_mask, additional_decoder_input, additional_decoder_mask, attn_prior = (
                self.prepare_dummy_cond_for_cfg(
                    context_tensors['cond'],
                    context_tensors['cond_mask'],
                    context_tensors['additional_decoder_input'],
                    context_tensors['addtional_decoder_mask'],
                )
            )
            disable_alignment_loss = True
        else:
            cond = context_tensors['cond']
            cond_mask = context_tensors['cond_mask']
            additional_decoder_input = context_tensors['additional_decoder_input']
            additional_decoder_mask = context_tensors['addtional_decoder_mask']
            attn_prior = context_tensors['attn_prior']

            if (
                mode == "train"
                and self.cfg.get('decoder_input_dropout_prob', 0.0) > 0.0
                and torch.rand(1).item() < 0.5
            ):
                # For some batches (half of them), replace decoder_input_dropout_prob of the timesteps with random tokens
                max_codebook_val = self.cfg.get('dec_random_input_max', self.num_all_tokens_per_codebook)
                # @pneekhara: Keeping dec_random_input_max configurable since num_all_tokens_per_codebook usually has padding tokens
                # which can cause errors when doing codes_to_audio for audio_codes_input. We are not currently calling codes_to_audio on
                # audio_codes_input so should not matter if we don't supply dec_random_input_max.
                random_audio_tokens = torch.randint(
                    0, max_codebook_val, audio_codes_input.size(), device=audio_codes_input.device
                )
                random_audio_tokens = random_audio_tokens * audio_codes_mask.unsqueeze(1)
                dec_dropout_mask = (
                    torch.rand((1, 1, audio_codes_input.size(2)), device=audio_codes_input.device)
                    > self.cfg.decoder_input_dropout_prob
                )
                # timestep_mask is True for timesteps to be kept
                audio_codes_input = audio_codes_input * dec_dropout_mask + random_audio_tokens * (~dec_dropout_mask)
                audio_codes_embedded = self.embed_audio_tokens(audio_codes_input) # (B, T', E)

        if context_tensors['additional_decoder_input'] is not None:
            dec_input_embedded = torch.cat([additional_decoder_input, audio_codes_embedded], dim=1)
            dec_input_mask = torch.cat([additional_decoder_mask, audio_codes_mask], dim=1)
        else:
            dec_input_embedded = audio_codes_embedded
            dec_input_mask = audio_codes_mask

        aligner_encoder_loss = None
        aligner_attn_soft = None
        aligner_attn_hard = None
        if self.cfg.get('use_alignment_encoder', False) and not disable_alignment_loss:
            aligner_prior = None
            if self.cfg.get('use_prior_for_aligner', False):
                aligner_prior = context_tensors['beta_binomial_attn_prior']
            # Passing target audio embeddings to the alignment encoder
            if self.global_step < self.cfg.get('aligner_encoder_train_steps', float('inf')):
                aligner_attn_soft, aligner_attn_logprobs = self.alignment_encoder(
                    queries=audio_codes_embedded_all[:, 1:, :].permute(0, 2, 1), # B, E, T'
                    keys=context_tensors['text_encoder_out'].permute(0, 2, 1), # B, E, T
                    mask=~context_tensors['text_mask'].unsqueeze(-1),
                    attn_prior=aligner_prior
                )

                aligner_encoder_loss = self.alignment_encoder_loss(
                    attn_logprob=aligner_attn_logprobs, in_lens=context_tensors['text_lens'], out_lens=audio_codes_lens_input
                )
            else:
                with torch.no_grad():
                    # Just get the attention matrix without computing the loss or gradients
                    aligner_attn_soft, aligner_attn_logprobs = self.alignment_encoder(
                        queries=audio_codes_embedded_all[:, 1:, :].permute(0, 2, 1), # B, E, T'
                        keys=context_tensors['text_encoder_out'].permute(0, 2, 1), # B, E, T
                        mask=~context_tensors['text_mask'].unsqueeze(-1),
                        attn_prior=aligner_prior
                    )

            with torch.no_grad():
                aligner_attn_hard = self.get_binarized_prior_matrix(
                    aligner_attn_soft, audio_codes_lens_input, context_tensors['text_lens']
                )
                if (self.global_step > self.cfg.get('binarize_prior_after_step', 0)) and context_tensors['prior_used']:
                    print("Updating Prior")
                    attn_prior = self.replace_beta_binomial_prior_with_binarized(attn_prior, aligner_attn_hard)

        logits, attn_info, dec_out = self.forward(
            dec_input_embedded=dec_input_embedded,
            dec_input_mask=dec_input_mask,
            cond=cond,
            cond_mask=cond_mask,
            attn_prior=attn_prior,
            multi_encoder_mapping=context_tensors['multi_encoder_mapping'],
        )
        # logits: (B, T', num_codebooks * num_tokens_per_codebook)
        # dec_out: (B, T', E)
        dec_context_size = context_tensors['dec_context_size']
        logits = logits[:, dec_context_size:, :]  # Remove the context audio embeddings from the logits

        codebook_loss, loss_mask = self.compute_loss(logits, audio_codes_target, audio_codes_lens_target)
        codebook_loss_scale = self.cfg.get('codebook_loss_scale', 1.0)
        alignment_loss = None
        if self.cfg.alignment_loss_scale > 0.0 and not disable_alignment_loss:
            text_lens = context_tensors['text_lens']
            ctc_prior_layer_ids = self.cfg.get('ctc_prior_layer_ids', self.transcript_decoder_layers)
            cross_attention_scores = [attn['cross_attn_probabilities'][1] for layer_idx, attn in enumerate(attn_info) if layer_idx in ctc_prior_layer_ids]
            alignment_loss = self.compute_alignment_loss(
                cross_attention_scores, text_lens, audio_codes_lens_target, dec_context_size
            )
            loss = codebook_loss_scale * codebook_loss + alignment_loss
        else:
            loss = codebook_loss_scale * codebook_loss

        local_transformer_loss = None
        local_transformer_logits = None
        if self.local_transformer_type != LocalTransformerType.NO_LT:
            if self.local_transformer_type == LocalTransformerType.MASKGIT:
                # randomly replace some positions with MASK_TOKEN
                audio_codes_masked, mask_tokens_mask = self.maskgit_apply_random_mask(audio_codes_target)
                local_transformer_logits = self.compute_local_transformer_logits(dec_out[:,dec_context_size:,:], audio_codes_masked, targets_offset_by_one=True)
                #audio_codes_masked = audio_codes_masked[:, 1:, :]
                local_transformer_loss, _ = self.compute_loss(local_transformer_logits, audio_codes_target, audio_codes_lens_target, mask_tokens_mask)
            else:
                # autoregressive
                assert self.local_transformer_type == LocalTransformerType.AR, "Unexpected local transformer type"
                local_transformer_logits = self.compute_local_transformer_logits(dec_out[:,dec_context_size:,:], audio_codes_target, targets_offset_by_one=False)
                local_transformer_loss, _ = self.compute_loss(local_transformer_logits, audio_codes_target, audio_codes_lens_target, None)
            local_transformer_loss_scale = self.cfg.get('local_transformer_loss_scale', 1.0)
            loss = loss + local_transformer_loss_scale * local_transformer_loss

        if aligner_encoder_loss is not None:
            loss = loss + aligner_encoder_loss

        return {
            'logits': logits,
            'attn_info': attn_info,
            'loss': loss,
            'codebook_loss': codebook_loss,
            'local_transformer_loss' : local_transformer_loss,
            'local_transformer_logits' : local_transformer_logits,
            'loss_mask': loss_mask,
            'alignment_loss': alignment_loss,
            'aligner_encoder_loss': aligner_encoder_loss,
            'audio_codes_target': audio_codes_target,
            'audio_codes_lens_target': audio_codes_lens_target,
            'text': context_tensors['text'],
            'text_lens': context_tensors['text_lens'],
            'context_audio_codes': context_tensors['context_audio_codes'],
            'context_audio_codes_lens': context_tensors['context_audio_codes_lens'],
            'dec_context_size': dec_context_size,
            'aligner_attn_soft': aligner_attn_soft,
            'aligner_attn_hard': aligner_attn_hard,
        }

    def training_step(self, batch, batch_idx):
        batch_output = self.process_batch(batch)
        loss = batch_output['loss']
        codebook_loss = batch_output['codebook_loss']
        self.log('train/codebook_loss', codebook_loss, prog_bar=True, sync_dist=True)
        if self.cfg.get('cfg_unconditional_prob', 0.0) == 0.0:
            # Only log alignment loss when not using cfg to avoid sync issues when
            # alignment loss is None on some ranks
            alignment_loss = batch_output['alignment_loss']
            if alignment_loss is not None:
                self.log('train/alignment_loss', alignment_loss, prog_bar=True, sync_dist=True)
        self.log('train/loss', loss, prog_bar=True, sync_dist=True)
        local_transformer_loss = batch_output['local_transformer_loss']
        if local_transformer_loss is not None:
            self.log('train/local_transformer_loss', local_transformer_loss, prog_bar=True, sync_dist=True)

        # Log batch info
        batch_size, text_token_max_len = batch["text"].shape
        text_token_total_num = batch["text_lens"].sum()
        batch_info_dict = {
            "train/batch_size": batch_size,
            "train/text_token_max_len": text_token_max_len,
            "train/text_token_total_num_in_batch": text_token_total_num,
            "train/text_token_pad_ratio_percent_in_batch": 100 * (1 - text_token_total_num / (batch_size * text_token_max_len)),
        }

        if "audio_codes" in batch:
            audio_codes_max_len = batch["audio_codes"].shape[-1]
            audio_codes_total_num = batch["audio_codes_lens"].sum()
            batch_info_dict.update({
                "train/audio_codes_max_len": audio_codes_max_len,
                "train/audio_codes_total_num_in_batch": audio_codes_total_num,
                "train/audio_codes_pad_ratio_percent_in_batch": 100 * (1 - audio_codes_total_num / (batch_size * audio_codes_max_len)),
            })
        else:
            audio_samples_max_len = batch["audio"].shape[-1]
            audio_samples_total_num = batch["audio_lens"].sum()
            batch_info_dict.update({
                "train/audio_samples_max_len": audio_samples_max_len,
                "train/audio_samples_total_num_in_batch": audio_samples_total_num,
                "train/audio_samples_pad_ratio_percent_in_batch": 100 * (1 - audio_samples_total_num / (batch_size * audio_samples_max_len)),
            })

        self.log_dict(batch_info_dict, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        batch_output = self.process_batch(batch, mode="val")
        # self.process_batch returns a dict. We currently only log "logits" which come from the parallel prediction
        # head. If we use local_transformer, then the local_transformer returns "local_transformer_logits"
        loss = batch_output['loss']
        codebook_loss = batch_output['codebook_loss']
        alignment_loss = batch_output['alignment_loss']
        aligner_encoder_loss = batch_output['aligner_encoder_loss']
        logits = batch_output['logits']
        audio_codes_target = batch_output['audio_codes_target']
        audio_codes_lens_target = batch_output['audio_codes_lens_target']
        context_audio_codes = batch_output['context_audio_codes']
        context_audio_codes_lens = batch_output['context_audio_codes_lens']
        attn_info = batch_output['attn_info']
        text_lens = batch_output['text_lens']
        dec_context_size = batch_output['dec_context_size']
        if alignment_loss is None:
            alignment_loss = torch.tensor(0.0, device=loss.device)
        if aligner_encoder_loss is None:
            aligner_encoder_loss = torch.tensor(0.0, device=loss.device)

        if batch_idx == 0 and self.global_rank == 0:
            # Prepare dictionary for aggregated wandb logging
            wandb_log_dict = {}

            # Get audio data for logging
            wandb_log_dict.update(
                self.log_val_audio_example(
                    logits, audio_codes_target, audio_codes_lens_target, context_audio_codes, context_audio_codes_lens
                )
            )

            # Get attention image data for logging
            if (
                self.model_type != 'decoder_pretrain_synthesizer'
                and len(attn_info[self.transcript_decoder_layers[0]]['cross_attn_probabilities']) > 1
            ):
                # cross_attn_probabilities only returned when not using flash attention
                ctc_prior_layer_ids = self.cfg.get('ctc_prior_layer_ids', self.transcript_decoder_layers)
                cross_attention_probs = [attn['cross_attn_probabilities'][0] for layer_idx, attn in enumerate(attn_info) if layer_idx in ctc_prior_layer_ids]
                wandb_log_dict.update(
                    self.log_attention_probs(
                        cross_attention_probs,
                        audio_codes_lens_target,
                        text_lens,
                        prefix="val",
                        dec_context_size=dec_context_size,
                    )
                )

                for layer_idx in self.transcript_decoder_layers:
                    cross_attention_probs = [ attn_info[layer_idx]['cross_attn_probabilities'][0] ]
                    wandb_log_dict.update(
                        self.log_attention_probs(
                            cross_attention_probs,
                            audio_codes_lens_target,
                            text_lens,
                            prefix=f"val/layer_{layer_idx}",
                            dec_context_size=dec_context_size
                        )
                    )

                if batch_output['aligner_attn_soft'] is not None:
                    wandb_log_dict.update(
                        self.log_attention_probs(
                            [batch_output['aligner_attn_soft']],
                            audio_codes_lens_target,
                            text_lens,
                            prefix=f"val/aligner_encoder_attn",
                        )
                    )

                if batch_output['aligner_attn_hard'] is not None:
                    wandb_log_dict.update(
                        self.log_attention_probs(
                            [batch_output['aligner_attn_hard'].unsqueeze(1)],
                            audio_codes_lens_target,
                            text_lens,
                            prefix=f"val/aligner_encoder_attn_hard",
                        )
                    )

            # Perform single wandb log call if wandb is active and there is data
            for logger in self.loggers:
                if isinstance(logger, WandbLogger) and wandb_log_dict:
                    logger.experiment.log(wandb_log_dict)

        local_transformer_loss = batch_output['local_transformer_loss']
        val_output = {
            'val_loss': loss,
            'val_codebook_loss': codebook_loss,
            'val_alignment_loss': alignment_loss,
            'val_local_transformer_loss': local_transformer_loss,
            'val_aligner_encoder_loss': aligner_encoder_loss,
        }
        self.validation_step_outputs.append(val_output)

        return val_output

    def get_cross_attention_scores(self, attn_probs, filter_layers=None):
        """
        Returns the cross attention probabilities for the last audio timestep
        """
        mean_cross_attn_scores = []
        all_heads_cross_attn_scores = []
        for lidx, layerwise_attn_prob in enumerate(attn_probs):
            if (filter_layers is not None and lidx not in filter_layers) or (lidx not in self.transcript_decoder_layers):
                continue
            cross_attn_prob = layerwise_attn_prob['cross_attn_probabilities'][0] # B, H, audio_timesteps, text_timesteps
            mean_cross_attn_scores.append(cross_attn_prob.mean(dim=1)) # B, audio_timesteps, text_timesteps
            for head_idx in range(cross_attn_prob.size(1)):
                all_heads_cross_attn_scores.append(cross_attn_prob[:, head_idx, -1, :]) # B, text_timesteps

        mean_cross_attn_scores = torch.stack(mean_cross_attn_scores, dim=1) # B, L, audio_timesteps, text_timesteps
        mean_cross_attn_scores = mean_cross_attn_scores.mean(dim=1) # B, audio_timesteps, text_timesteps
        last_audio_timestep_scores = mean_cross_attn_scores[:, -1, :] # B, text_timesteps
        return last_audio_timestep_scores, all_heads_cross_attn_scores

    def get_most_attended_text_timestep(self, alignment_attention_scores, last_attended_timesteps,
                                   text_lens, lookahead_window_size, attended_timestep_counter, batch_size):
        """
        Returns the most attended timestep for each batch item
        """
        text_time_step_attended = []
        for bidx in range(batch_size):
            last_attended_timestep = last_attended_timesteps[-1][bidx]
            if attended_timestep_counter[bidx].get(last_attended_timestep, 0) >= 8:
                # This is probably an attention sink! Move to the next timestep
                last_attended_timestep += 1
            window_size = lookahead_window_size
            window_end = min(last_attended_timestep + window_size, text_lens[bidx] - 3) # Ignore the last 3 timesteps
            item_attention_scores = alignment_attention_scores[bidx,last_attended_timestep:window_end]
            if item_attention_scores.size(0) == 0:
                # This means the sentence has ended
                attended_timestep = text_lens[bidx] - 1
            else:
                attended_timestep = item_attention_scores.argmax().item() + last_attended_timestep
            text_time_step_attended.append(attended_timestep)
            attended_timestep_counter[bidx][attended_timestep] = attended_timestep_counter[bidx].get(attended_timestep, 0) + 1
        return text_time_step_attended, attended_timestep_counter

    def construct_inference_prior(self, prior_epsilon, cross_attention_scores,
                                  text_lens, text_time_step_attended, attended_timestep_counter,
                                  unfinished_texts, finished_texts_counter, end_indices, batch_size):
        # Attn prior for the next timestep
        _attn_prior = torch.zeros(cross_attention_scores.shape[0], 1, cross_attention_scores.shape[1]) + prior_epsilon
        _attn_prior = _attn_prior.to(cross_attention_scores.device)
        for bidx in range(cross_attention_scores.shape[0]):
            if bidx < batch_size:
                _text_len = text_lens[bidx]
                if text_lens[bidx] <= 5:
                    # Very short sentences, No Prior
                    _attn_prior[bidx, 0, :] = 1.0
                else:
                    # _attn_prior[bidx, 0, max(1, text_time_step_attended[bidx]-2)] = 0.1 # Slight exposure to history for better pronounciation. Not very important.
                    _attn_prior[bidx, 0, max(1, text_time_step_attended[bidx]-1)] = 0.2 # Slight exposure to history for better pronounciation. Not very important.
                    _attn_prior[bidx, 0, text_time_step_attended[bidx]] = 0.8 # Slightly bias to continue moving forward. Not very important.
                    _attn_prior[bidx, 0, min(text_time_step_attended[bidx]+1, _text_len - 1) ] = 1.0
                    _attn_prior[bidx, 0, min(text_time_step_attended[bidx]+2, _text_len - 1) ] = 0.8

                # Penalize timesteps that have been attended to more than 10 times
                for _timestep in attended_timestep_counter[bidx]:
                    if attended_timestep_counter[bidx][_timestep] >= 10:
                        # This means the timestep has been attended to more than 10 times (To avoid getting stuck)
                        _attn_prior[bidx, 0, _timestep] = prior_epsilon

                unfinished_texts[bidx] = False
                if text_time_step_attended[bidx] < text_lens[bidx] - 3:
                    # This means the sentence has not ended
                    if bidx not in end_indices:
                        unfinished_texts[bidx] = True

                if text_time_step_attended[bidx] >= text_lens[bidx] - 5 or bidx in end_indices:
                    if bidx not in finished_texts_counter:
                        finished_texts_counter[bidx] = 0

        for bidx in finished_texts_counter:
            finished_texts_counter[bidx] += 1
            if finished_texts_counter[bidx] > 10:
                # This means we have been within the text EOS window for atleast 10 timesteps
                # We should allow EOS to be predicted now.
                unfinished_texts[bidx] = False

        return _attn_prior, unfinished_texts, finished_texts_counter

    def get_inference_attention_plots(self, cross_attention_scores_all_timesteps, all_heads_cross_attn_scores_all_timesteps, text_lens, predicted_codes_lens, batch_size, compute_all_heads_attn_maps):
        cross_attention_scores_all_timesteps = torch.stack(cross_attention_scores_all_timesteps, dim=2) # B, text_timesteps, T'
        headwise_cross_attention_scores_all_timesteps = []
        for hidx in range(len(all_heads_cross_attn_scores_all_timesteps[0])):
            head_cross_attention_all_timesteps = torch.stack([x[hidx] for x in all_heads_cross_attn_scores_all_timesteps], dim=2) # B, text_timesteps, T'
            headwise_cross_attention_scores_all_timesteps.append(head_cross_attention_all_timesteps)

        cross_attention_maps = []
        headwise_cross_attention_maps = []
        for bidx in range(batch_size):
            item_cross_attention_scores = cross_attention_scores_all_timesteps[bidx,:text_lens[bidx],:predicted_codes_lens[bidx]]
            cross_attn_np = plot_alignment_to_numpy(item_cross_attention_scores.cpu().numpy())
            cross_attention_maps.append(cross_attn_np)
            item_all_head_cross_attn_maps = []
            if compute_all_heads_attn_maps:
                for hidx in range(len(all_heads_cross_attn_scores_all_timesteps[0])):
                    item_headwise_cross_attention_scores = headwise_cross_attention_scores_all_timesteps[hidx][bidx,:text_lens[bidx],:predicted_codes_lens[bidx]]
                    headwise_cross_attn_np = plot_alignment_to_numpy(item_headwise_cross_attention_scores.cpu().numpy())
                    item_all_head_cross_attn_maps.append(headwise_cross_attn_np)
                headwise_cross_attention_maps.append(item_all_head_cross_attn_maps)

        return cross_attention_maps, headwise_cross_attention_maps

    def infer_batch(
            self,
            batch,
            max_decoder_steps=500,
            temperature=0.7,
            topk=80,
            use_cfg=False,
            cfg_scale=1.0,
            return_cross_attn_probs=False,
            apply_attention_prior=False,
            prior_epsilon=1e-5,
            lookahead_window_size=10,
            estimate_alignment_from_layers=None,
            apply_prior_to_layers=None,
            start_prior_after_n_audio_steps=10,
            compute_all_heads_attn_maps=False,
            use_local_transformer_for_inference=False,
            maskgit_n_steps=3
        ):
        with torch.no_grad():
            start_time = time.time()
            self.decoder.reset_cache(use_cache=self.use_kv_cache_for_inference)

            context_tensors = self.prepare_context_tensors(batch)
            text = context_tensors['text']
            audio_codes_bos = torch.full(
                (text.size(0), self.num_audio_codebooks, 1), self.audio_bos_id, device=text.device
            ).long()
            audio_codes_lens = torch.full((text.size(0),), 1, device=text.device).long()
            audio_codes_input = audio_codes_bos
            audio_codes_mask = get_mask_from_lengths(audio_codes_lens)

            all_predictions = []
            end_indices = {}

            if use_cfg:
                dummy_cond, dummy_cond_mask, dummy_additional_decoder_input, dummy_addition_dec_mask, _ = (
                    self.prepare_dummy_cond_for_cfg(
                        context_tensors['cond'],
                        context_tensors['cond_mask'],
                        context_tensors['additional_decoder_input'],
                        context_tensors['addtional_decoder_mask'],
                    )
                )

            cross_attention_scores_all_timesteps = []
            all_heads_cross_attn_scores_all_timesteps = []
            _attn_prior = None
            unfinished_texts = {}
            finished_texts_counter = {}
            attended_timestep_counter = [{} for _ in range(text.size(0))]
            last_attended_timesteps = [[1 for _ in range(text.size(0))]] # Maintain a list of attended timesteps as we predict audio for each batch item
            time_to_first_prediction = 0.0
            for idx in range(max_decoder_steps):
                if idx == 1:
                    time_to_first_prediction = time.time() - start_time
                if idx % 20 == 0:
                    print(f"Decoding timestep {idx}")
                audio_codes_embedded = self.embed_audio_tokens(audio_codes_input)
                if context_tensors['additional_decoder_input'] is not None:
                    _audio_codes_embedded = torch.cat(
                        [context_tensors['additional_decoder_input'], audio_codes_embedded], dim=1
                    )
                    _audio_codes_mask = torch.cat([context_tensors['addtional_decoder_mask'], audio_codes_mask], dim=1)
                else:
                    _audio_codes_embedded = audio_codes_embedded
                    _audio_codes_mask = audio_codes_mask

                if apply_prior_to_layers is not None:
                    attn_prior = [None for _ in range(self.cfg.decoder.n_layers)]
                    for layer_idx in apply_prior_to_layers:
                        attn_prior[layer_idx] = _attn_prior
                else:
                    attn_prior = _attn_prior

                if self.model_type == 'multi_encoder_context_tts':
                    attn_prior = [attn_prior, None]


                if use_cfg:
                    batch_size = audio_codes_embedded.size(0)
                    if isinstance(context_tensors['cond'], list):
                        cfg_cond = [
                            torch.cat([cond_item, dummy_cond_item], dim=0)
                            for cond_item, dummy_cond_item in zip(context_tensors['cond'], dummy_cond)
                        ]
                        cfg_cond_mask = [
                            torch.cat([cond_mask_item, dummy_cond_mask_item], dim=0)
                            for cond_mask_item, dummy_cond_mask_item in zip(
                                context_tensors['cond_mask'], dummy_cond_mask
                            )
                        ]
                    else:
                        cfg_cond = torch.cat([context_tensors['cond'], dummy_cond], dim=0)
                        cfg_cond_mask = torch.cat([context_tensors['cond_mask'], dummy_cond_mask], dim=0)
                    cfg_audio_codes_embedded = torch.cat([_audio_codes_embedded, _audio_codes_embedded], dim=0)
                    cfg_audio_codes_mask = torch.cat([_audio_codes_mask, _audio_codes_mask], dim=0)
                    if dummy_additional_decoder_input is not None:
                        cfg_audio_codes_embedded[batch_size:, : dummy_additional_decoder_input.size(1)] = (
                            dummy_additional_decoder_input
                        )
                        cfg_audio_codes_mask[batch_size:, : dummy_additional_decoder_input.size(1)] = (
                            dummy_addition_dec_mask
                        )

                    # print(f"step {idx}")
                    # print(f"use_cfg {use_cfg}")
                    # print(f"shape {cfg_audio_codes_embedded.shape}")
                    # print(f"use kv cahce? {self.use_kv_cache_for_inference}")
                    combined_logits, attn_probs, dec_out = self.forward(
                        dec_input_embedded=cfg_audio_codes_embedded,
                        dec_input_mask=cfg_audio_codes_mask,
                        cond=cfg_cond,
                        cond_mask=cfg_cond_mask,
                        attn_prior=attn_prior,
                        multi_encoder_mapping=context_tensors['multi_encoder_mapping']
                    )

                    cond_logits = combined_logits[:batch_size]
                    uncond_logits = combined_logits[batch_size:]
                    all_code_logits = (1 - cfg_scale) * uncond_logits + cfg_scale * cond_logits
                else:
                    batch_size = audio_codes_embedded.size(0)
                    all_code_logits, attn_probs, dec_out = self.forward(
                        dec_input_embedded=_audio_codes_embedded,
                        dec_input_mask=_audio_codes_mask,
                        cond=context_tensors['cond'],
                        cond_mask=context_tensors['cond_mask'],
                        attn_prior=attn_prior,
                        multi_encoder_mapping=context_tensors['multi_encoder_mapping']
                    )

                if return_cross_attn_probs or apply_attention_prior:
                    cross_attention_scores, all_heads_cross_attn_scores = self.get_cross_attention_scores(attn_probs) # B, text_timesteps
                    alignment_attention_scores = cross_attention_scores
                    if estimate_alignment_from_layers is not None:
                        alignment_attention_scores, _ = self.get_cross_attention_scores(attn_probs, filter_layers=estimate_alignment_from_layers) # B, text_timesteps

                    cross_attention_scores_all_timesteps.append(cross_attention_scores)
                    all_heads_cross_attn_scores_all_timesteps.append(all_heads_cross_attn_scores)

                if apply_attention_prior and idx >= start_prior_after_n_audio_steps:
                    text_time_step_attended, attended_timestep_counter = self.get_most_attended_text_timestep(
                        alignment_attention_scores=alignment_attention_scores,
                        last_attended_timesteps=last_attended_timesteps,
                        text_lens=context_tensors['text_lens'],
                        lookahead_window_size=lookahead_window_size,
                        attended_timestep_counter=attended_timestep_counter,
                        batch_size=batch_size
                    )
                    last_attended_timesteps.append(text_time_step_attended)
                    _attn_prior, unfinished_texts, finished_texts_counter = self.construct_inference_prior(
                        prior_epsilon=prior_epsilon,
                        cross_attention_scores=cross_attention_scores,
                        text_lens=context_tensors['text_lens'],
                        text_time_step_attended=text_time_step_attended,
                        attended_timestep_counter=attended_timestep_counter,
                        unfinished_texts=unfinished_texts,
                        finished_texts_counter=finished_texts_counter,
                        end_indices=end_indices,
                        batch_size=batch_size
                    )

                finished_items = {k: v for k, v in finished_texts_counter.items() if v >= 20} # Items that have been close to the end for atleast 20 timesteps
                unifinished_items = {k: v for k, v in unfinished_texts.items() if v}

                all_code_logits_t = all_code_logits[:, -1, :] # (B, num_codebooks * num_tokens_per_codebook)
                if use_local_transformer_for_inference:
                    if self.local_transformer_type == LocalTransformerType.AR :
                        # Autoregressive sampling with local transformer
                        audio_codes_next = self.local_transformer_sample_autoregressive(
                            dec_output=dec_out[:,-1,:],
                            temperature=temperature,
                            topk=topk,
                            unfinished_items=unifinished_items,
                            finished_items=finished_items,
                            use_cfg=use_cfg,
                            cfg_scale=cfg_scale
                        )
                    elif self.local_transformer_type == LocalTransformerType.MASKGIT:
                        audio_codes_next = self.local_transformer_sample_maskgit(
                            dec_output=dec_out[:,-1,:],
                            temperature=temperature,
                            topk=topk,
                            unfinished_items=unifinished_items,
                            finished_items=finished_items,
                            use_cfg=use_cfg,
                            cfg_scale=cfg_scale,
                            n_steps=maskgit_n_steps
                        )
                    else:
                        raise ValueError(f"Local transformer inference requested by but local transformer type is {self.local_transformer_type}")
                    # TODO @rfejgin: should we add argmax sampling for EOS here too?
                    #all_codes_next_argmax = audio_codes_next
                else:
                    # Parallel sampling from all codebooks
                    audio_codes_next = self.sample_codes_from_logits(all_code_logits_t, temperature=temperature, topk=topk, unfinished_items=unifinished_items, finished_items=finished_items) # (B, num_codebooks)
                all_codes_next_argmax = self.sample_codes_from_logits(all_code_logits_t, temperature=0.01, unfinished_items=unifinished_items, finished_items=finished_items) # (B, num_codebooks)

                for item_idx in range(all_codes_next_argmax.size(0)):
                    if item_idx not in end_indices:
                        pred_token = all_codes_next_argmax[item_idx][0].item()
                        pred_token_multinomial = audio_codes_next[item_idx][0].item()
                        if (pred_token == self.audio_eos_id) or (pred_token_multinomial == self.audio_eos_id):
                            print("End detected for item {} at timestep {}".format(item_idx, idx))
                            end_indices[item_idx] = idx

                all_predictions.append(audio_codes_next)
                audio_codes_input = torch.cat(
                    [audio_codes_input, audio_codes_next.unsqueeze(-1)], dim=-1
                )  # (B, C, T')
                audio_codes_lens = audio_codes_lens + 1
                audio_codes_mask = get_mask_from_lengths(audio_codes_lens)
                if len(end_indices) == text.size(0) and len(all_predictions) >= 4:
                    # Codec must be of atleast 4 timesteps to be decoded properly
                    print("All ends reached")
                    break
            tts_generation_time = time.time() - start_time
            tts_generation_time_per_frame = tts_generation_time / len(all_predictions)
            predicted_codes = torch.stack(all_predictions, dim=-1)  # (B, num_codebooks, T')

            predicted_lens = [end_indices.get(idx, max_decoder_steps) for idx in range(text.size(0))] #  Ensure that the codec is atleast of length 4
            predicted_codes_lens = torch.tensor(predicted_lens, device=text.device).long()

            predicted_audio, predicted_audio_lens = self.codes_to_audio(predicted_codes, predicted_codes_lens)
            end_time = time.time()
            total_audio_duration_generated = (predicted_audio_lens.max().item() * predicted_audio_lens.shape[0])/self._codec_model.sample_rate
            rtf = total_audio_duration_generated / (end_time - start_time)
            rtf_metrics = {
                'rtf': rtf,
                'time_to_first_prediction': time_to_first_prediction,
                'tts_generation_time': tts_generation_time,
                'max_frames_generated': len(all_predictions),
                'tts_generation_time_per_frame': tts_generation_time_per_frame,
                'batch_size': text.size(0),
            }
            torch.cuda.empty_cache()
            if return_cross_attn_probs:
                cross_attention_maps, headwise_cross_attention_maps = self.get_inference_attention_plots(
                    cross_attention_scores_all_timesteps, all_heads_cross_attn_scores_all_timesteps,
                    context_tensors['text_lens'], predicted_codes_lens, text.size(0), compute_all_heads_attn_maps
                )
                return predicted_audio, predicted_audio_lens, predicted_codes, predicted_codes_lens, rtf_metrics, cross_attention_maps, headwise_cross_attention_maps
            else:
                # For backward compatibility
                return predicted_audio, predicted_audio_lens, predicted_codes, predicted_codes_lens, rtf_metrics

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            test_dl_batch_size = self._test_dl.batch_size
            temperature = self.cfg.get('inference_temperature', 0.7)
            topk = self.cfg.get('inference_topk', 80)
            use_cfg = self.cfg.get('inference_use_cfg', False)
            cfg_scale = self.cfg.get('inference_cfg_scale', 1.0)
            predicted_audio, predicted_audio_lens, predicted_codes, predicted_codes_lens, _ = self.infer_batch(
                batch,
                max_decoder_steps=self.cfg.get('max_decoder_steps', 500),
                temperature=temperature,
                topk=topk,
                use_cfg=use_cfg,
                cfg_scale=cfg_scale,
            )

            for logger in self.loggers:
                is_wandb = isinstance(logger, WandbLogger)
                is_tb = isinstance(logger, TensorBoardLogger)
                if not is_wandb and not is_tb:
                    raise ValueError(f"Invalid logger type for audio logging: {type(logger)}. Only `WandbLogger` and `TensorBoardLogger` are supported.")

                for idx in range(predicted_audio.size(0)):
                    predicted_audio_np = predicted_audio[idx].float().detach().cpu().numpy()
                    predicted_audio_np = predicted_audio_np[: predicted_audio_lens[idx]]
                    item_idx = batch_idx * test_dl_batch_size + idx

                    if is_wandb:
                        log_dict = {
                            f"test/predicted_audio": wandb.Audio(
                                predicted_audio_np, sample_rate=self.cfg.sample_rate, caption=f"Predicted Audio"
                            ),
                        }
                        logger.experiment.log(log_dict, step=item_idx)

                    if is_tb:
                        logger.experiment.add_audio(
                            'test/predicted_audio',
                            predicted_audio_np,
                            global_step=item_idx,
                            sample_rate=self.cfg.sample_rate,
                        )

                    # Save the predicted audio
                    log_dir = logger.log_dir
                    audio_dir = os.path.join(log_dir, 'audios')
                    if not os.path.exists(audio_dir):
                        os.makedirs(audio_dir)
                    audio_path = os.path.join(audio_dir, f'predicted_audioRank{self.global_rank}_{item_idx}.wav')
                    sf.write(audio_path, predicted_audio_np, self.cfg.sample_rate)

    def on_validation_epoch_end(self):
        collect = lambda key: torch.stack([x[key] for x in self.validation_step_outputs]).mean()
        val_loss = collect("val_loss")
        val_codebook_loss = collect("val_codebook_loss")
        val_alignment_loss = collect("val_alignment_loss")
        val_aligner_encoder_loss = collect("val_aligner_encoder_loss")
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        self.log("val/codebook_loss", val_codebook_loss, prog_bar=True, sync_dist=True)
        self.log("val/alignment_loss", val_alignment_loss, prog_bar=True, sync_dist=True)
        self.log("val/aligner_encoder_loss", val_aligner_encoder_loss, prog_bar=True, sync_dist=True)
        if self.local_transformer_type != LocalTransformerType.NO_LT:
            val_local_transformer_loss = collect("val_local_transformer_loss")
            self.log("val/local_transformer_loss", val_local_transformer_loss, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.clear()  # free memory

    def get_dataset(self, dataset_cfg, dataset_type):
        dataset = instantiate(
            dataset_cfg.dataset,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            audio_bos_id=self.audio_bos_id,
            audio_eos_id=self.audio_eos_id,
            context_audio_bos_id=self.context_audio_bos_id,
            context_audio_eos_id=self.context_audio_eos_id,
            num_audio_codebooks=self.num_audio_codebooks,
            codec_model_samples_per_frame=self.codec_model_samples_per_frame,
            prior_scaling_factor=self.cfg.prior_scaling_factor,
            load_cached_codes_if_available=self.cfg.load_cached_codes_if_available,
            dataset_type=dataset_type,  # train or test used for setting phone prob to 1.0 in test dataset (worker_init_fn)
            use_text_conditioning_tokenizer=self.cfg.use_text_conditioning_encoder,
            pad_context_text_to_max_duration=self.pad_context_text_to_max_duration,
            context_duration_min=self.cfg.context_duration_min,
            context_duration_max=self.cfg.context_duration_max,
        )
        dataset.load_16khz_audio = self.model_type == 'single_encoder_sv_tts'
        dataset.tokenizer_config = (
            self.cfg.text_tokenizers
        )  # This will be used in worker_init_fn for instantiating tokenizer
        return dataset

    def get_lhotse_dataloader(self, dataset_cfg, mode='train') -> torch.utils.data.DataLoader:
        # TODO @xueyang: better to distinguish cfg. self.cfg is the model cfg, while cfg here is train_ds cfg. Also
        #   cfg is a classifier-free guidance.
        dataset = MagpieTTSLhotseDataset(
            sample_rate=self.cfg.sample_rate,
            volume_norm=dataset_cfg.volume_norm,
            codec_model_samples_per_frame=self.codec_model_samples_per_frame,
            codec_model_name=self.cfg.codec_model_name,
            audio_bos_id=self.audio_bos_id,
            audio_eos_id=self.audio_eos_id,
            context_audio_bos_id=self.context_audio_bos_id,
            context_audio_eos_id=self.context_audio_eos_id,
            num_audio_codebooks=self.num_audio_codebooks,
            prior_scaling_factor=self.cfg.prior_scaling_factor,
            load_cached_codes_if_available=self.cfg.load_cached_codes_if_available,
            dataset_type=mode,  # train or test used for setting phone prob to 1.0 in test dataset (worker_init_fn)
            load_16khz_audio=(self.model_type == 'single_encoder_sv_tts'),
            pad_context_text_to_max_duration=self.pad_context_text_to_max_duration,
            context_duration_min=self.cfg.context_duration_min,
            context_duration_max=self.cfg.context_duration_max,
            use_text_conditioning_tokenizer=self.cfg.use_text_conditioning_encoder,
            tokenizer_config=self.cfg.text_tokenizers,
        )
        data_loader = get_lhotse_dataloader_from_config(
            config=dataset_cfg.dataset,
            global_rank=self.global_rank,
            world_size=self.world_size,
            dataset=dataset,
        )
        return data_loader

    def setup_training_data(self, dataset_cfg):
        if dataset_cfg.get("use_lhotse", False):
            # TODO @xueyang: better to distinguish cfg. self.cfg is the model cfg, while cfg here is train_ds cfg. Also
            #   cfg is a classifier-free guidance.
            self._train_dl = self.get_lhotse_dataloader(dataset_cfg, mode='train')
        else:
            dataset = self.get_dataset(dataset_cfg, dataset_type='train')
            sampler = dataset.get_sampler(dataset_cfg.dataloader_params.batch_size, world_size=self.trainer.world_size)
            persistent_workers = True
            if dataset_cfg.dataloader_params.num_workers == 0:
                persistent_workers = False
                # For num workers > 0 tokenizer will be assigned in worker_init_fn (since it is not picklable)
                dataset.text_tokenizer, dataset.text_conditioning_tokenizer = setup_tokenizers(
                    all_tokenizers_config=self.cfg.text_tokenizers,
                    use_text_conditioning_tokenizer=self.use_text_conditioning_encoder,
                    mode='train',
                )
            self._train_dl = torch.utils.data.DataLoader(
                dataset,
                collate_fn=dataset.collate_fn,
                sampler=sampler,
                **dataset_cfg.dataloader_params,
                worker_init_fn=worker_init_fn,
                persistent_workers=persistent_workers,
            )

    def _setup_test_dataloader(self, dataset_cfg) -> torch.utils.data.DataLoader:
        if dataset_cfg.get("use_lhotse", False):
            data_loader = self.get_lhotse_dataloader(dataset_cfg, mode='test')
        else:
            dataset = self.get_dataset(dataset_cfg, dataset_type='test')
            persistent_workers = True
            if dataset_cfg.dataloader_params.num_workers == 0:
                persistent_workers = False
                # For num workers > 0 tokenizer will be assigned in worker_init_fn (since it is not picklable)
                dataset.text_tokenizer, dataset.text_conditioning_tokenizer = setup_tokenizers(
                    all_tokenizers_config=self.cfg.text_tokenizers,
                    use_text_conditioning_tokenizer=self.use_text_conditioning_encoder,
                    mode='test'
                )

            data_loader = torch.utils.data.DataLoader(
                dataset,
                collate_fn=dataset.collate_fn,
                **dataset_cfg.dataloader_params,
                worker_init_fn=worker_init_fn,
                persistent_workers=persistent_workers,
            )
        return data_loader

    def setup_validation_data(self, cfg):
        self._validation_dl = self._setup_test_dataloader(cfg)

    def setup_test_data(self, cfg):
        self._test_dl = self._setup_test_dataloader(cfg)

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        return []


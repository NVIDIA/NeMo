# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from typing import List
from math import ceil
import numpy as np
import omegaconf
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
import os

from nemo.collections.tts.parts.utils.helpers import (
    get_mask_from_lengths,
    plot_alignment_to_numpy,
)
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging, model_utils
from nemo.collections.tts.modules import t5tts_transformer, t5tts_perceiver
from nemo.collections.tts.models import AudioCodecModel
from nemo.collections.tts.losses.aligner_loss import ForwardSumLoss
import nemo.collections.asr as nemo_asr
import soundfile as sf
from torch.utils.data import get_worker_info
from transformers import DistilBertTokenizer, DistilBertModel

HAVE_WANDB = True
try:
    import wandb
except ModuleNotFoundError:
    HAVE_WANDB = False


def worker_init_fn(worker_id):
    # Access worker information
    logging.info(f"Worker {worker_id} initializing...")
    worker_info = get_worker_info()
    dataset = worker_info.dataset  # Get the dataset instance in this worker

    # Initialize a non-picklable tokenizer for this worker
    text_tokenizer_kwargs = {}
    if "g2p" in dataset.tokenizer_config:
        # for backward compatibility
        text_tokenizer_kwargs["g2p"] = instantiate(dataset.tokenizer_config.g2p)
        logging.info(f"g2p instantiated: {text_tokenizer_kwargs['g2p']}")
    tokenizer = instantiate(dataset.tokenizer_config, **text_tokenizer_kwargs)
    if dataset.dataset_type == 'test' and hasattr(tokenizer, "set_phone_prob"):
        logging.info("Setting phone prob to 1.0 for test dataset")
        tokenizer.set_phone_prob(1.0)
    logging.info(f"Tokenizer instantiated: {tokenizer}")
    dataset.text_tokenizer = tokenizer # Use for transcripts
    if dataset.use_text_conditioning_tokenizer:
        dataset.text_conditioning_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased') # Used for text conditioning

class T5TTS_Model(ModelPT):
    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_devices
        
        # Setup tokenizer
        self.tokenizer = self._setup_tokenizer(cfg)

        num_tokens_tokenizer = len(self.tokenizer.tokens)
        num_tokens = num_tokens_tokenizer + 2 # +2 for BOS and EOS
        self.bos_id = num_tokens - 2
        self.eos_id = num_tokens - 1
        self.tokenizer_pad = self.tokenizer.pad
        self.tokenizer_unk = self.tokenizer.oov
        self.audio_bos_id = cfg.num_audio_tokens_per_codebook - 2
        self.audio_eos_id = cfg.num_audio_tokens_per_codebook - 1
        self._tb_logger = None
        self.model_type = cfg.get('model_type', 'single_encoder_sv_tts')
        self.use_text_conditioning_encoder = cfg.get('use_text_conditioning_encoder', False)
        
        super().__init__(cfg=cfg, trainer=trainer)
        
        self.text_embedding = nn.Embedding(num_tokens, cfg.embedding_dim)

        audio_embeddings = []
        for _ in range(cfg.num_audio_codebooks):
            audio_embeddings.append(nn.Embedding(cfg.num_audio_tokens_per_codebook, cfg.embedding_dim))
        self.audio_embeddings = nn.ModuleList(audio_embeddings)

        self.t5_encoder = t5tts_transformer.TransformerStack(dict(cfg.t5_encoder))
        
        decoder_config = dict(cfg.t5_decoder)
        decoder_config['context_xattn'] = {'params': decoder_config['context_xattn']}
        self.t5_decoder = t5tts_transformer.TransformerStack(decoder_config)

        self.final_proj = nn.Linear(cfg.t5_decoder.d_model, cfg.num_audio_codebooks * cfg.num_audio_tokens_per_codebook)

        codec_model = AudioCodecModel.restore_from(cfg.get('codecmodel_path'), strict=False)
        codec_model.eval()
        self.freeze_model(codec_model)
        self._codec_model = codec_model

        if self.model_type == 'single_encoder_sv_tts':
            speaker_verification_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large') 
            speaker_verification_model.eval()
            self.freeze_model(speaker_verification_model)
            self._speaker_verification_model = speaker_verification_model
            self.speaker_projection_layer = nn.Linear(cfg.speaker_emb_dim, cfg.embedding_dim)
            self.transcript_decoder_layers = [idx for idx in range(cfg.t5_decoder.n_layers)] # All layers are used for text
        elif self.model_type == 'multi_encoder_context_tts':
            self.transcript_decoder_layers = cfg.get('transcript_decoder_layers', [3,4,5,6,7,8])
            self.context_decoder_layers = cfg.get('context_decoder_layers', [0,1,2,9,10,11]) # For backward compatibility
            multi_encoder_mapping = [None for _ in range(cfg.t5_decoder.n_layers)]
            for layer in self.transcript_decoder_layers:
                multi_encoder_mapping[layer] = 0 # 0 means text goes to this layer, 1 means context goes to this layer
            for layer in self.context_decoder_layers:
                multi_encoder_mapping[layer] = 1
            self.multi_encoder_mapping = multi_encoder_mapping

            self.context_encoder = t5tts_transformer.TransformerStack(dict(cfg.context_encoder))
            if cfg.use_perceiver:
                self.perceiver_resampler = t5tts_perceiver.PerceiverResampler(
                    dim=cfg.context_encoder.d_model,
                    depth=2,
                    dim_context=cfg.context_encoder.d_model,
                    num_latents=32,
                    dim_head=64,
                    heads=8,
                    ff_mult=4,
                    use_flash_attn=False,
                )
        else:
            raise ValueError(f"Unknown model type: {cfg.model_type}")
        
        if self.use_text_conditioning_encoder:
            self.text_conditioning_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.text_conditioning_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
            self.text_conditioning_encoder.train()

        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        alignment_loss_scale = cfg.get('alignment_loss_scale', 0.0)
        if alignment_loss_scale > 0.0:
            self.alignment_loss = ForwardSumLoss(loss_scale=alignment_loss_scale)

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def state_dict(self):
        # Don't save the speaker verification and codec model in the state dict
        state_dict = super().state_dict()
        keys_substrings_to_exclude = ['_speaker_verification_model', '_codec_model']
        for key in list(state_dict.keys()):
            if any([substring in key for substring in keys_substrings_to_exclude]):
                del state_dict[key]
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        # Override to load all the keys except _speaker_verification_model and _codec_model
        super().load_state_dict(state_dict, strict=False)
        
    def _setup_tokenizer(self, cfg, mode='train'):
        text_tokenizer_kwargs = {}
        if "g2p" in cfg.text_tokenizer:
            text_tokenizer_kwargs["g2p"] = instantiate(cfg.text_tokenizer.g2p)
        tokenizer = instantiate(cfg.text_tokenizer, **text_tokenizer_kwargs)
        if mode == 'test' and hasattr(tokenizer, "set_phone_prob"):
            tokenizer.set_phone_prob(1.0)
        return tokenizer

    @property
    def tb_logger(self):
        if self._tb_logger is None:
            if self.logger is None and self.logger.experiment is None:
                return None
            tb_logger = self.logger.experiment
            for logger in self.trainer.loggers:
                if isinstance(logger, TensorBoardLogger):
                    tb_logger = logger.experiment
                    break
            self._tb_logger = tb_logger
        return self._tb_logger
    
    def audio_to_codes(self, audio, audio_len):
        # audio: (B, T)
        # audio_len: (B,)
        self._codec_model.eval()
        with torch.no_grad():
            # Move codec model to same device as audio
            # self.additional_models['codec'] = self.additional_models['codec'].to(audio.device)
            codes, codes_len = self._codec_model.encode(audio=audio, audio_len=audio_len)
            # Add a timestep to begining and end of codes tensor
            bos_tensor = torch.full((codes.size(0), codes.size(1), 1), self.audio_bos_id, dtype=codes.dtype, device=codes.device)
            pad_tensor = torch.full((codes.size(0), codes.size(1), 1), 0, dtype=codes.dtype, device=codes.device) # 0 is the padding token in the audio codebook
            codes = torch.cat([bos_tensor, codes, pad_tensor], dim=-1)
            # codes: (B, C, T')
            # codes_len: (B,)
            for idx in range(codes.size(0)):
                codes[idx, :, codes_len[idx] + 1] = self.audio_eos_id
            codes_len = codes_len + 2
            
            return codes.long(), codes_len.long()
    
    def codes_to_audio(self, codes, codes_len):
        # codes: (B, C, T')
        # codes_len: (B,)
        self._codec_model.eval()
        with torch.no_grad():
            # Replace eos and bos tokens with padding in codes tensor
            codes[codes == self.audio_bos_id] = 0 # zero is the padding token in the audio codebook
            codes[codes == self.audio_eos_id] = 0
            # self.additional_models['codec'] = self.additional_models['codec'].to(codes.device)
            audio, audio_len = self._codec_model.decode(tokens=codes, tokens_len=codes_len)
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

    def compute_loss(self, logits, audio_codes, audio_codes_lens):
        # logits: (B, T', num_codebooks * num_tokens_per_codebook)
        # audio_codes: (B, C, T')
        # audio_codes_lens: (B,)
        loss_mask = get_mask_from_lengths(audio_codes_lens)
        total_codebook_loss = None
        for codebook in range(audio_codes.size(1)):
            si = codebook * self.cfg.num_audio_tokens_per_codebook
            ei = si + self.cfg.num_audio_tokens_per_codebook
            codebook_logits = logits[:, :, si:ei] # (B, T', num_tokens_per_codebook)
            codebook_targets = audio_codes[:, codebook] # (B, T')
            codebook_loss = self.cross_entropy_loss(
                codebook_logits.permute(0, 2, 1), # (B, num_tokens_per_codebook, T')
                codebook_targets
            ) # (B, T')
            codebook_loss = codebook_loss * loss_mask
            codebook_loss = codebook_loss.sum() / loss_mask.sum()
            if total_codebook_loss is None:
                total_codebook_loss = codebook_loss
            else:
                total_codebook_loss = total_codebook_loss + codebook_loss
        
        total_codebook_loss = total_codebook_loss / audio_codes.size(1)
        return total_codebook_loss

    def forward(self, text, text_lens, audio_codes=None, audio_codes_lens=None, attn_prior=None, conditioning_vector=None, context_embeddings=None, context_mask=None):
        # Either conditioning_vector or context_embeddings should be provided
        assert (conditioning_vector is not None) ^ (context_embeddings is not None)
        text_embedded = self.text_embedding(text) # (B, T, E)
        text_mask = ~get_mask_from_lengths(text_lens) # (B, T)
        encoder_out = self.t5_encoder(text_embedded, text_mask, cond=None, cond_mask=None)['output'] # (B, T, E)
        if conditioning_vector is not None:
            # conditioning_vector: (B, E) usually speaker embeddings
            encoder_out = encoder_out + conditioning_vector.unsqueeze(1)
            cond = encoder_out
            cond_mask = text_mask
            multi_encoder_mapping = None
            _attn_prior = attn_prior
        elif context_embeddings is not None:
            cond = [encoder_out, context_embeddings]
            cond_mask = [text_mask, context_mask]
            multi_encoder_mapping = self.multi_encoder_mapping
            _attn_prior = [attn_prior, None]
        
        audio_codes_mask = ~get_mask_from_lengths(audio_codes_lens)
        audio_codes_embedded = self.embed_audio_tokens(audio_codes)
        decoder_out = self.t5_decoder(
            audio_codes_embedded,
            audio_codes_mask,
            cond=cond,
            cond_mask=cond_mask,
            attn_prior=_attn_prior,
            multi_encoder_mapping=multi_encoder_mapping,
        ) # (B, T', E)
        attn_probabilities = decoder_out['attn_probabilities']
        all_code_logits = self.final_proj(decoder_out['output']) # (B, T', num_codebooks * num_tokens_per_codebook)
        return all_code_logits, attn_probabilities

    def logits_to_audio_codes(self, all_code_logits, audio_codes_lens):
        # all_code_logits: (B, T', num_codebooks * num_tokens_per_codebook)
        # audio_codes_lens: (B,)
        all_preds = []
        for idx in range(self.cfg.num_audio_codebooks):
            si = idx * self.cfg.num_audio_tokens_per_codebook
            ei = si + self.cfg.num_audio_tokens_per_codebook
            codebook_logits = all_code_logits[:, :, si:ei]
            codebook_probs = torch.softmax(codebook_logits, dim=-1) # (B, T', num_tokens_per_codebook)
            # argmax to get the tokens
            codebook_preds = torch.argmax(codebook_probs, dim=-1) # (B, T')
            all_preds.append(codebook_preds)
        
        all_preds = torch.stack(all_preds, dim=1) # (B, C, T')
        audio_mask = get_mask_from_lengths(audio_codes_lens)
        all_preds = all_preds * audio_mask.unsqueeze(1)

        return all_preds

    def sample_codes_from_logits(self, all_code_logits_t, temperature=0.7, topk=80):
        # all_code_logits_t: (B, num_codebooks * num_tokens_per_codebook), logits at a given timestep
        all_preds = []
        for idx in range(self.cfg.num_audio_codebooks):
            si = idx * self.cfg.num_audio_tokens_per_codebook
            ei = si + self.cfg.num_audio_tokens_per_codebook
            codebook_logits = all_code_logits_t[:, si:ei] # (B, num_tokens_per_codebook)
            codebook_logits_topk = torch.topk(codebook_logits, topk, dim=-1)[0] # (B, topk)
            indices_to_remove = codebook_logits < codebook_logits_topk[:, -1].unsqueeze(-1) # (B, num_tokens_per_codebook)
            codebook_logits_rescored = codebook_logits.clone()
            codebook_logits_rescored[indices_to_remove] = float('-inf')

            codebook_probs = torch.softmax(codebook_logits / temperature, dim=-1) # (B, num_tokens_per_codebook)
            codebook_preds = torch.multinomial(codebook_probs, 1) # (B, 1)
            all_preds.append(codebook_preds)
        all_preds = torch.cat(all_preds, dim=1).long() # (B, num_codebooks)
        return all_preds

    def log_attention_probs(self, attention_prob_matrix, audio_codes_lens, text_lens, prefix=""):
        # attention_prob_matrix List of (B, C, audio_timesteps, text_timesteps)
        with torch.no_grad():
            attention_prob_matrix = torch.cat(attention_prob_matrix, dim=1) # (B, C, audio_timesteps, text_timesteps)
            attention_prob_matrix_mean = attention_prob_matrix.mean(dim=1) # (B, audio_timesteps, text_timesteps)
            for idx in range(min(3, attention_prob_matrix_mean.size(0))):
                item_attn_matrix = attention_prob_matrix_mean[idx][:audio_codes_lens[idx], :text_lens[idx]]
                item_attn_matrix = item_attn_matrix.detach().cpu().numpy()
                attn_np = plot_alignment_to_numpy(item_attn_matrix.T)
                self.tb_logger.add_image(
                    f'{prefix}attention_matrix_{idx}',
                    attn_np,
                    global_step=self.global_step,
                    dataformats="HWC",
                )


    def log_train_val_example(self, logits, target_audio_codes, audio_codes_lens_target, context_audio_codes=None, context_audio_codes_lens=None):
        pred_audio_codes = self.logits_to_audio_codes(logits, audio_codes_lens_target)
        pred_audio, pred_audio_lens = self.codes_to_audio(pred_audio_codes, audio_codes_lens_target)
        target_audio, target_audio_lens = self.codes_to_audio(target_audio_codes, audio_codes_lens_target)
        context_audio, context_audio_lens = None, None
        if context_audio_codes is not None:
            context_audio, context_audio_lens = self.codes_to_audio(context_audio_codes, context_audio_codes_lens)
        for idx in range(min(3, pred_audio.size(0))):
            pred_audio_np = pred_audio[idx].float().detach().cpu().numpy()
            target_audio_np = target_audio[idx].float().detach().cpu().numpy()
            pred_audio_np = pred_audio_np[:pred_audio_lens[idx]]
            target_audio_np = target_audio_np[:target_audio_lens[idx]]
            self.tb_logger.add_audio(
                f'pred_audio_{idx}',
                pred_audio_np,
                global_step=self.global_step,
                sample_rate=self.cfg.sample_rate,
            )
            self.tb_logger.add_audio(
                f'target_audio_{idx}',
                target_audio_np,
                global_step=self.global_step,
                sample_rate=self.cfg.sample_rate,
            )
            if context_audio is not None:
                context_audio_np = context_audio[idx].float().detach().cpu().numpy()
                context_audio_np = context_audio_np[:context_audio_lens[idx]]
                self.tb_logger.add_audio(
                    f'context_audio_{idx}',
                    context_audio_np,
                    global_step=self.global_step,
                    sample_rate=self.cfg.sample_rate,
                )
    
    def scale_prior(self, prior, global_step):
        if prior is None:
            return None
        prior_end_step = self.cfg.prior_end_step
        prior_scaledown_start_step = self.cfg.prior_scaledown_start_step
        if global_step < prior_scaledown_start_step:
            return prior
        elif global_step >= prior_end_step:
            return None
        else:
            with torch.no_grad():
                # Interpolate between all ones and the prior
                residual = 1.0 - prior
                new_prior = prior + (residual * (global_step - prior_scaledown_start_step) / (prior_end_step - prior_scaledown_start_step))
                return new_prior

    def compute_alignment_loss(self, attention_scores, text_lens, audio_lens):
        # attention scores: List of (B, C, audio_timesteps, text_timesteps)
        attention_scores_combined = torch.cat(attention_scores, dim=1) # (B, C, audio_timesteps, text_timesteps)
        attention_scores_mean = attention_scores_combined.mean(dim=1, keepdim=True) # (B, 1, audio_timesteps, text_timesteps)
        alignment_loss = self.alignment_loss(
            attn_logprob=attention_scores_mean, in_lens=text_lens, out_lens=audio_lens
        )
        return alignment_loss

    def process_batch(self, batch):
        text = batch['text']
        text_lens = batch['text_lens']
        
        attn_prior = batch.get('align_prior_matrix', None)
        attn_prior = self.scale_prior(attn_prior, self.global_step)

        if 'audio_codes' not in batch:
            audio_codes, audio_codes_lens = self.audio_to_codes(batch['audio'], batch['audio_lens'])
        else:
            audio_codes = batch['audio_codes']
            audio_codes_lens = batch['audio_codes_lens']

        audio_codes_input = audio_codes[:, :, :-1]
        audio_codes_target = audio_codes[:, :, 1:]
        audio_codes_lens_input = audio_codes_lens_target = audio_codes_lens - 1
        
        context_audio_codes = None
        context_audio_codes_lens = None
        
        if self.use_text_conditioning_encoder:
            context_text_tokens = batch['context_text_tokens']
            context_text_lens = batch['context_text_tokens_lens']
            context_text_mask = get_mask_from_lengths(context_text_lens)
            context_text_encoded = self.text_conditioning_encoder(input_ids=context_text_tokens, attention_mask=context_text_mask)['last_hidden_state'] # (B, L, E)
            # Hidden dim of distilbert is same as embedding dim = 768, so no projection needed

        if self.model_type == 'single_encoder_sv_tts':
            target_audio_16khz = batch['audio_16khz']
            target_audio_lens_16khz = batch['audio_lens_16khz']
            speaker_embeddings = self.get_speaker_embeddings(target_audio_16khz, target_audio_lens_16khz)
            speaker_embeddings_projected = self.speaker_projection_layer(speaker_embeddings)
            if self.use_text_conditioning_encoder:
                # In single_encoder_sv_tts either speaker embeddings or context text embeddings are used, not both!
                has_text_context = batch['has_text_context'].unsqueeze(-1).float()
                conditioning_vector = has_text_context * context_text_encoded[:,0] + (1 - has_text_context) * speaker_embeddings_projected
            else:
                conditioning_vector = speaker_embeddings_projected
            context_embeddings = None
            context_mask = None

        elif self.model_type == 'multi_encoder_context_tts':
            if 'context_audio_codes' in batch:
                context_audio_codes = batch['context_audio_codes']
                context_audio_codes_lens = batch['context_audio_codes_lens']
            else:
                context_audio_codes, context_audio_codes_lens = self.audio_to_codes(batch['context_audio'], batch['context_audio_lens'])
            context_audio_embedded = self.embed_audio_tokens(context_audio_codes)
            context_mask = ~get_mask_from_lengths(context_audio_codes_lens)
            context_embeddings = self.context_encoder(context_audio_embedded, context_mask, cond=None, cond_mask=None)['output']
            if self.cfg.use_perceiver:
                # pneekhara: Check if we can pass mask here. Mask of size B, L doesn't work
                context_embeddings = self.perceiver_resampler(context_embeddings) # B, 32, C
                context_mask = torch.zeros(context_embeddings.size(0), context_embeddings.size(1), dtype=torch.bool, device=context_embeddings.device)
            
            if self.use_text_conditioning_encoder:
                # Concatenate the context text embeddings to the context audio embeddings
                # If context text is not provided in the manifest, context_text_encoded is just the encoding of "[NO TEXT CONTEXT]"
                context_embeddings = torch.cat([context_embeddings, context_text_encoded], dim=1) # (B, T, C)
                context_mask = torch.cat([context_mask, ~context_text_mask], dim=1) # (B, T)
                
            conditioning_vector = None

        logits, attn_info = self.forward(
            text=text,
            text_lens=text_lens,
            audio_codes=audio_codes_input,
            audio_codes_lens=audio_codes_lens_input,
            attn_prior=attn_prior,
            conditioning_vector=conditioning_vector,
            context_embeddings=context_embeddings,
            context_mask=context_mask
        )
        codebook_loss = self.compute_loss(logits, audio_codes_target, audio_codes_lens_target)
        alignment_loss = None
        if self.cfg.alignment_loss_scale > 0.0:
            cross_attention_scores = [attn['cross_attn_probabilities'][1] for layer_idx, attn in enumerate(attn_info) if layer_idx in self.transcript_decoder_layers]
            alignment_loss = self.compute_alignment_loss(cross_attention_scores, text_lens, audio_codes_lens_target)
            loss = codebook_loss + alignment_loss
        else:
            loss = codebook_loss
        
        return {
            'logits': logits,
            'attn_info' : attn_info,
            'loss': loss,
            'codebook_loss': codebook_loss,
            'alignment_loss': alignment_loss,
            'audio_codes_target': audio_codes_target,
            'audio_codes_lens_target': audio_codes_lens_target,
            'text': text,
            'text_lens': text_lens,
            'context_audio_codes': context_audio_codes,
            'context_audio_codes_lens': context_audio_codes_lens
        }
    
    def training_step(self, batch, batch_idx):
        batch_output = self.process_batch(batch)
        loss = batch_output['loss']
        codebook_loss = batch_output['codebook_loss']
        alignment_loss = batch_output['alignment_loss']
        self.log('train_codebook_loss', codebook_loss, prog_bar=True, sync_dist=True)
        if alignment_loss is not None:
            self.log('train_alignment_loss', alignment_loss, prog_bar=True, sync_dist=True)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        batch_output = self.process_batch(batch)
        loss = batch_output['loss']
        codebook_loss = batch_output['codebook_loss']
        alignment_loss = batch_output['alignment_loss']
        logits = batch_output['logits']
        audio_codes_target = batch_output['audio_codes_target']
        audio_codes_lens_target = batch_output['audio_codes_lens_target']
        context_audio_codes = batch_output['context_audio_codes']
        context_audio_codes_lens = batch_output['context_audio_codes_lens']
        attn_info = batch_output['attn_info']
        text_lens = batch_output['text_lens']
        if alignment_loss is None:
            alignment_loss = torch.tensor(0.0, device=loss.device)
        
        if batch_idx == 0 and self.global_rank == 0:
            self.log_train_val_example(logits, audio_codes_target, audio_codes_lens_target, context_audio_codes, context_audio_codes_lens)
            if len(attn_info[self.transcript_decoder_layers[0]]['cross_attn_probabilities']) > 1:
                # cross_attn_probabilities only returned when not using flash attention
                cross_attention_probs = [attn['cross_attn_probabilities'][0] for layer_idx, attn in enumerate(attn_info) if layer_idx in self.transcript_decoder_layers]
                self.log_attention_probs(cross_attention_probs, audio_codes_lens_target, text_lens, prefix="val_")

        val_output = {
            'val_loss': loss,
            'val_codebook_loss': codebook_loss,
            'val_alignment_loss': alignment_loss,
        }
        self.validation_step_outputs.append(val_output)

        return val_output
    
    def infer_batch(self, batch, max_decoder_steps=500):
        with torch.no_grad():
            text = batch['text']
            text_lens = batch['text_lens']
            audio_codes_bos = torch.full((text.size(0), self.cfg.num_audio_codebooks, 1), self.audio_bos_id, device=text.device).long()
            audio_codes_lens = torch.full((text.size(0),), 1, device=text.device).long()
            audio_codes_input = audio_codes_bos
            audio_codes_mask = ~get_mask_from_lengths(audio_codes_lens)

            text_mask = ~get_mask_from_lengths(text_lens)
            encoder_out = self.t5_encoder(self.text_embedding(text), text_mask, cond=None, cond_mask=None)['output']

            if self.use_text_conditioning_encoder:
                context_text_tokens = batch['context_text_tokens']
                context_text_lens = batch['context_text_tokens_lens']
                context_text_mask = get_mask_from_lengths(context_text_lens)
                context_text_encoded = self.text_conditioning_encoder(input_ids=context_text_tokens, attention_mask=context_text_mask)['last_hidden_state'] # (B, L, E)

            if self.model_type == 'single_encoder_sv_tts':
                target_audio_16khz = batch['audio_16khz']
                target_audio_lens_16khz = batch['audio_lens_16khz']
                
                speaker_embeddings = self.get_speaker_embeddings(target_audio_16khz, target_audio_lens_16khz)
                speaker_embeddings_projected = self.speaker_projection_layer(speaker_embeddings)
                if self.use_text_conditioning_encoder:
                    has_text_context = batch['has_text_context'].unsqueeze(-1).float()
                    conditioning_vector = has_text_context * context_text_encoded[:,0] + (1 - has_text_context) * speaker_embeddings_projected
                else:
                    conditioning_vector = speaker_embeddings_projected

                encoder_out = encoder_out + conditioning_vector.unsqueeze(1)
                cond = encoder_out
                cond_mask = text_mask
                multi_encoder_mapping = None

            elif self.model_type == 'multi_encoder_context_tts':
                if 'context_audio_codes' in batch:
                    context_audio_codes = batch['context_audio_codes']
                    context_audio_codes_lens = batch['context_audio_codes_lens']
                else:
                    context_audio_codes, context_audio_codes_lens = self.audio_to_codes(batch['context_audio'], batch['context_audio_lens'])

                context_audio_embedded = self.embed_audio_tokens(context_audio_codes)
                context_mask = ~get_mask_from_lengths(context_audio_codes_lens)
                context_embeddings = self.context_encoder(context_audio_embedded, context_mask, cond=None, cond_mask=None)['output']
                if self.cfg.use_perceiver:
                    context_embeddings = self.perceiver_resampler(context_embeddings)
                    context_mask = torch.zeros(context_embeddings.size(0), context_embeddings.size(1), dtype=torch.bool, device=context_embeddings.device)
                
                if self.use_text_conditioning_encoder:
                    context_embeddings = torch.cat([context_embeddings, context_text_encoded], dim=1)
                    context_mask = torch.cat([context_mask, ~context_text_mask], dim=1)
                
                cond = [encoder_out, context_embeddings]
                cond_mask = [text_mask, context_mask]
                multi_encoder_mapping = self.multi_encoder_mapping
            
            all_predictions = []
            end_indices = {}
            for idx in range(max_decoder_steps):
                audio_codes_embedded = self.embed_audio_tokens(audio_codes_input)
                decoder_out = self.t5_decoder(
                    audio_codes_embedded,
                    audio_codes_mask,
                    cond=cond,
                    cond_mask=cond_mask,
                    multi_encoder_mapping=multi_encoder_mapping
                )
                all_code_logits = self.final_proj(decoder_out['output']) # (B, T', num_codebooks * num_tokens_per_codebook)
                all_code_logits_t = all_code_logits[:, -1, :] # (B, num_codebooks * num_tokens_per_codebook)
                audio_codes_next = self.sample_codes_from_logits(all_code_logits_t) # (B, num_codebooks)
                all_codes_next_argmax = self.sample_codes_from_logits(all_code_logits_t, temperature=0.01) # (B, num_codebooks)
                
                for item_idx in range(all_codes_next_argmax.size(0)):
                    if item_idx not in end_indices:
                        pred_token = all_codes_next_argmax[item_idx][0].item()
                        if pred_token == self.audio_eos_id:
                            print("End detected for item {} at timestep {}".format(item_idx, idx))
                            end_indices[item_idx] = idx

                all_predictions.append(audio_codes_next)
                audio_codes_input = torch.cat([audio_codes_input, audio_codes_next.unsqueeze(-1)], dim=-1) # (B, C, T')
                audio_codes_lens = audio_codes_lens + 1
                audio_codes_mask = ~get_mask_from_lengths(audio_codes_lens)
                if len(end_indices) == text.size(0):
                    print("All ends reached")
                    break
            
            predicted_codes = torch.stack(all_predictions, dim=-1) # (B, num_codebooks, T')
            predicted_lens = [end_indices.get(idx, max_decoder_steps) for idx in range(text.size(0))]
            predicted_codes_lens = torch.tensor(predicted_lens, device=text.device).long()

            predicted_audio, predicted_audio_lens = self.codes_to_audio(predicted_codes, predicted_codes_lens)
            
            return predicted_audio, predicted_audio_lens, predicted_codes, predicted_codes_lens

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            test_dl_batch_size = self._test_dl.batch_size
            predicted_audio, predicted_audio_lens, predicted_codes, predicted_codes_lens = self.infer_batch(batch, max_decoder_steps=self.cfg.max_decoder_steps)
            for idx in range(predicted_audio.size(0)):
                predicted_audio_np = predicted_audio[idx].float().detach().cpu().numpy()
                predicted_audio_np = predicted_audio_np[:predicted_audio_lens[idx]]
                item_idx = batch_idx * test_dl_batch_size + idx
                self.tb_logger.add_audio(
                    'predicted_audio',
                    predicted_audio_np,
                    global_step=item_idx,
                    sample_rate=self.cfg.sample_rate,
                )
                # Save the predicted audio
                log_dir = self.logger.log_dir
                audio_dir = os.path.join(log_dir, 'audios')
                if not os.path.exists(audio_dir):
                    os.makedirs(audio_dir)
                audio_path = os.path.join(audio_dir, f'predicted_audio_{item_idx}.wav')
                sf.write(audio_path, predicted_audio_np, self.cfg.sample_rate)


    def on_validation_epoch_end(self):
        collect = lambda key: torch.stack([x[key] for x in self.validation_step_outputs]).mean()
        val_loss = collect("val_loss")
        val_codebook_loss = collect("val_codebook_loss")
        val_alignment_loss = collect("val_alignment_loss")
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        self.log("val_codebook_loss", val_codebook_loss, prog_bar=True, sync_dist=True)
        self.log("val_alignment_loss", val_alignment_loss, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.clear()  # free memory

    def get_dataset(self, cfg, dataset_type):
        dataset = instantiate(
            cfg.dataset,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            audio_bos_id=self.audio_bos_id,
            audio_eos_id=self.audio_eos_id,
            num_audio_codebooks=self.cfg.num_audio_codebooks,
            codec_model_downsample_factor=self.cfg.codec_model_downsample_factor,
            prior_scaling_factor=self.cfg.prior_scaling_factor,
            load_cached_codes_if_available=self.cfg.load_cached_codes_if_available,
            dataset_type=dataset_type, # train or test used for setting phone prob to 1.0 in test dataset (worker_init_fn)
            use_text_conditioning_tokenizer=self.cfg.use_text_conditioning_encoder,
        )
        dataset.load_16khz_audio = self.model_type == 'single_encoder_sv_tts'
        dataset.tokenizer_config = self.cfg.text_tokenizer # This will be used in worker_init_fn for instantiating tokenizer
        return dataset

    def _setup_train_dataloader(self, cfg):
        dataset = self.get_dataset(cfg, dataset_type='train')
        sampler = dataset.get_sampler(cfg.dataloader_params.batch_size, world_size=self.trainer.world_size)
        if cfg.dataloader_params.num_workers == 0:
            # For num workers > 0 tokenizer will be assigned in worker_init_fn (since it is not picklable)
            dataset.text_tokenizer = self._setup_tokenizer(self.cfg)
            if self.cfg.use_text_conditioning_encoder:
                dataset.text_conditioning_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        data_loader = torch.utils.data.DataLoader(
            dataset, collate_fn=dataset.collate_fn, sampler=sampler, **cfg.dataloader_params, worker_init_fn=worker_init_fn
        )
        return data_loader

    def _setup_test_dataloader(self, cfg):
        dataset = self.get_dataset(cfg, dataset_type='test')
        if cfg.dataloader_params.num_workers == 0:
            # For num workers > 0 tokenizer will be assigned in worker_init_fn (since it is not picklable)
            dataset.text_tokenizer = self._setup_tokenizer(self.cfg, mode='test')
            if self.cfg.use_text_conditioning_encoder:
                dataset.text_conditioning_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params, worker_init_fn=worker_init_fn)
        return data_loader

    def setup_training_data(self, cfg):
        self._train_dl = self._setup_train_dataloader(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self._setup_test_dataloader(cfg)

    def setup_test_data(self, cfg):
        self._test_dl = self._setup_test_dataloader(cfg)

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        return []

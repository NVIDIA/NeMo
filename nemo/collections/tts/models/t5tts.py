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
    binarize_attention,
    g2p_backward_compatible_support,
    get_mask_from_lengths,
    plot_alignment_to_numpy,
)
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging, model_utils
from nemo.core.optim.lr_scheduler import compute_max_steps, prepare_lr_scheduler
from nemo.collections.tts.parts.utils.helpers import get_batch_size, get_num_workers
from nemo.collections.tts.modules import t5_transformer
from nemo.collections.tts.models import AudioCodecModel
from nemo.collections.tts.losses.aligner_loss import ForwardSumLoss
import nemo.collections.asr as nemo_asr
import soundfile as sf

HAVE_WANDB = True
try:
    import wandb
except ModuleNotFoundError:
    HAVE_WANDB = False


class T5TTS_Model(ModelPT):
    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_devices

        # Setup tokenizer
        self.tokenizer = None
        self._setup_tokenizer(cfg)
        assert self.tokenizer is not None

        num_tokens_tokenizer = len(self.tokenizer.tokens)
        num_tokens = num_tokens_tokenizer + 2 # +2 for BOS and EOS
        self.bos_id = num_tokens - 2
        self.eos_id = num_tokens - 1
        self.tokenizer_pad = self.tokenizer.pad
        self.tokenizer_unk = self.tokenizer.oov
        self.audio_bos_id = cfg.num_audio_tokens_per_codebook - 2
        self.audio_eos_id = cfg.num_audio_tokens_per_codebook - 1
        self._tb_logger = None

        super().__init__(cfg=cfg, trainer=trainer)
        
        self.text_embedding = nn.Embedding(num_tokens, cfg.embedding_dim)

        audio_embeddings = []
        for _ in range(cfg.num_audio_codebooks):
            audio_embeddings.append(nn.Embedding(cfg.num_audio_tokens_per_codebook, cfg.embedding_dim))
        self.audio_embeddings = nn.ModuleList(audio_embeddings)

        self.speaker_projection_layer = nn.Linear(cfg.speaker_emb_dim, cfg.embedding_dim)

        self.t5_encoder = t5_transformer.TransformerStack(dict(cfg.t5_encoder))
        
        decoder_config = dict(cfg.t5_decoder)
        decoder_config['context_xattn'] = {'params': decoder_config['context_xattn']}
        self.t5_decoder = t5_transformer.TransformerStack(decoder_config)

        self.final_proj = nn.Linear(cfg.t5_decoder.d_model, cfg.num_audio_codebooks * cfg.num_audio_tokens_per_codebook)

        codec_model = AudioCodecModel.restore_from(cfg.get('codecmodel_path'), strict=False)
        codec_model.eval()
        self.freeze_model(codec_model)
        self._codec_model = codec_model

        speaker_verification_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large') 
        speaker_verification_model.eval()
        self.freeze_model(speaker_verification_model)
        self._speaker_verification_model = speaker_verification_model

        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        alignment_loss_scale = cfg.get('alignment_loss_scale', 0.0)
        if alignment_loss_scale > 0.0:
            self.alignment_loss = ForwardSumLoss(loss_scale=alignment_loss_scale)

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def on_save_checkpoint(self, checkpoint):
        keys_substrings_to_exclude = ['_speaker_verification_model', '_codec_model']
        for key in list(checkpoint['state_dict'].keys()):
            if any([substring in key for substring in keys_substrings_to_exclude]):
                del checkpoint['state_dict'][key]

    def _setup_tokenizer(self, cfg):
        text_tokenizer_kwargs = {}
        if "g2p" in cfg.text_tokenizer:
            # for backward compatibility
            if (
                self._is_model_being_restored()
                and (cfg.text_tokenizer.g2p.get('_target_', None) is not None)
                and cfg.text_tokenizer.g2p["_target_"].startswith("nemo_text_processing.g2p")
            ):
                cfg.text_tokenizer.g2p["_target_"] = g2p_backward_compatible_support(
                    cfg.text_tokenizer.g2p["_target_"]
                )

            g2p_kwargs = {}

            if "phoneme_dict" in cfg.text_tokenizer.g2p:
                g2p_kwargs["phoneme_dict"] = self.register_artifact(
                    'text_tokenizer.g2p.phoneme_dict',
                    cfg.text_tokenizer.g2p.phoneme_dict,
                )

            if "heteronyms" in cfg.text_tokenizer.g2p:
                g2p_kwargs["heteronyms"] = self.register_artifact(
                    'text_tokenizer.g2p.heteronyms',
                    cfg.text_tokenizer.g2p.heteronyms,
                )

            text_tokenizer_kwargs["g2p"] = instantiate(cfg.text_tokenizer.g2p, **g2p_kwargs)

        self.tokenizer = instantiate(cfg.text_tokenizer, **text_tokenizer_kwargs)

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
        with torch.no_grad():
            # Move codec model to same device as audio
            # self.additional_models['codec'] = self.additional_models['codec'].to(audio.device)
            codec_model = self._codec_model
            codes, codes_len = codec_model.encode(audio=audio, audio_len=audio_len)
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
        with torch.no_grad():
            # Replace eos and bos tokens with padding in codes tensor
            codes[codes == self.audio_bos_id] = 0 # zero is the padding token in the audio codebook
            codes[codes == self.audio_eos_id] = 0
            # self.additional_models['codec'] = self.additional_models['codec'].to(codes.device)
            codec_model = self._codec_model
            audio, audio_len = codec_model.decode(tokens=codes, tokens_len=codes_len)
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
        with torch.no_grad():
            speaker_verification_model = self._speaker_verification_model
            _, speaker_embeddings = speaker_verification_model.forward(
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

    def forward(self, text, text_lens, audio_codes=None, audio_codes_lens=None, attn_prior=None, conditioning_vector=None):
        # import ipdb; ipdb.set_trace()
        text_embedded = self.text_embedding(text) # (B, T, E)
        text_mask = ~get_mask_from_lengths(text_lens) # (B, T)
        encoder_out = self.t5_encoder(text_embedded, text_mask, cond=None, cond_mask=None)['output'] # (B, T, E)
        if conditioning_vector is not None:
            # conditioning_vector: (B, E)
            encoder_out = encoder_out + conditioning_vector.unsqueeze(1)

        audio_codes_mask = ~get_mask_from_lengths(audio_codes_lens)
        audio_codes_embedded = self.embed_audio_tokens(audio_codes)
        decoder_out = self.t5_decoder(
            audio_codes_embedded,
            audio_codes_mask,
            cond=encoder_out,
            cond_mask=text_mask,
            attn_prior=attn_prior
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

    def sample_codes_from_logits(self, all_code_logits_t, temperature=0.1, topk=80):
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


    def log_train_val_example(self, logits, target_audio_codes, audio_codes_lens_target):
        pred_audio_codes = self.logits_to_audio_codes(logits, audio_codes_lens_target)
        pred_audio, pred_audio_lens = self.codes_to_audio(pred_audio_codes, audio_codes_lens_target)
        target_audio, target_audio_lens = self.codes_to_audio(target_audio_codes, audio_codes_lens_target)
        for idx in range(min(3, pred_audio.size(0))):
            pred_audio_np = pred_audio[idx].detach().cpu().numpy()
            target_audio_np = target_audio[idx].detach().cpu().numpy()
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

    def training_step(self, batch, batch_idx):
        text = batch['text']
        text_lens = batch['text_lens']
        target_audio = batch['audio']
        target_audio_lens = batch['audio_lens']
        attn_prior = batch.get('align_prior_matrix', None)
        attn_prior = self.scale_prior(attn_prior, self.global_step)

        target_audio_16khz = batch['audio_16khz']
        target_audio_lens_16khz = batch['audio_lens_16khz']
        speaker_embeddings = self.get_speaker_embeddings(target_audio_16khz, target_audio_lens_16khz)
        speaker_embeddings_projected = self.speaker_projection_layer(speaker_embeddings)

        audio_codes, audio_codes_lens = self.audio_to_codes(target_audio, target_audio_lens)
        audio_codes_input = audio_codes[:, :, :-1]
        audio_codes_target = audio_codes[:, :, 1:]
        audio_codes_lens_input = audio_codes_lens_target = audio_codes_lens - 1

        logits, attn_info = self.forward(
            text=text,
            text_lens=text_lens,
            audio_codes=audio_codes_input,
            audio_codes_lens=audio_codes_lens_input,
            attn_prior=attn_prior,
            conditioning_vector=speaker_embeddings_projected
        )
        # import ipdb; ipdb.set_trace()
        codebook_loss = self.compute_loss(logits, audio_codes_target, audio_codes_lens_target)
        self.log('train_codebook_loss', codebook_loss, prog_bar=True, sync_dist=True)
        if self.cfg.alignment_loss_scale > 0.0:
            cross_attention_scores = [attn['cross_attn_probabilities'][1] for attn in attn_info]
            alignment_loss = self.compute_alignment_loss(cross_attention_scores, text_lens, audio_codes_lens_target)
            self.log('train_alignment_loss', alignment_loss, prog_bar=True, sync_dist=True)
            loss = codebook_loss + alignment_loss
        else:
            loss = codebook_loss
        
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)

        return loss
    
    
    def validation_step(self, batch, batch_idx):
        text = batch['text']
        text_lens = batch['text_lens']
        target_audio = batch['audio']
        target_audio_lens = batch['audio_lens']
        attn_prior = batch.get('align_prior_matrix', None)
        attn_prior = self.scale_prior(attn_prior, self.global_step)

        target_audio_16khz = batch['audio_16khz']
        target_audio_lens_16khz = batch['audio_lens_16khz']
        speaker_embeddings = self.get_speaker_embeddings(target_audio_16khz, target_audio_lens_16khz)
        speaker_embeddings_projected = self.speaker_projection_layer(speaker_embeddings)

        audio_codes, audio_codes_lens = self.audio_to_codes(target_audio, target_audio_lens)
        audio_codes_input = audio_codes[:, :, :-1]
        audio_codes_target = audio_codes[:, :, 1:]
        audio_codes_lens_input = audio_codes_lens_target = audio_codes_lens - 1

        logits, attn_info = self.forward(
            text=text,
            text_lens=text_lens,
            audio_codes=audio_codes_input,
            audio_codes_lens=audio_codes_lens_input,
            attn_prior=attn_prior,
            conditioning_vector=speaker_embeddings_projected
        )
        
        # import ipdb; ipdb.set_trace()
        codebook_loss = self.compute_loss(logits, audio_codes_target, audio_codes_lens_target)
        if self.cfg.alignment_loss_scale > 0.0:
            cross_attention_scores = [attn['cross_attn_probabilities'][1] for attn in attn_info]
            alignment_loss = self.compute_alignment_loss(cross_attention_scores, text_lens, audio_codes_lens_target)
            loss = codebook_loss + alignment_loss
        else:
            loss = codebook_loss
            alignment_loss = torch.tensor(0.0, device=loss.device)

        if batch_idx == 0 and self.global_rank == 0:
            self.log_train_val_example(logits, audio_codes_target, audio_codes_lens_target)
            cross_attention_probs = [attn['cross_attn_probabilities'][0] for attn in attn_info]
            self.log_attention_probs(cross_attention_probs, audio_codes_lens_target, text_lens, prefix="val_")

        val_output = {
            'val_loss': loss,
            'val_codebook_loss': codebook_loss,
            'val_alignment_loss': alignment_loss,
        }
        self.validation_step_outputs.append(val_output)

        return val_output
    
    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            test_dl_batch_size = self._test_dl.batch_size
            text = batch['text']
            text_lens = batch['text_lens']
            audio_codes_bos = torch.full((text.size(0), self.cfg.num_audio_codebooks, 1), self.audio_bos_id, device=text.device).long()
            audio_codes_lens = torch.full((text.size(0),), 1, device=text.device).long()
            audio_codes_input = audio_codes_bos
            audio_codes_mask = ~get_mask_from_lengths(audio_codes_lens)

            text_mask = ~get_mask_from_lengths(text_lens)
            encoder_out = self.t5_encoder(self.text_embedding(text), text_mask, cond=None, cond_mask=None)

            all_predictions = []
            end_indices = {}
            for idx in range(self.cfg.max_decoder_steps):
                audio_codes_embedded = self.embed_audio_tokens(audio_codes_input)
                decoder_out = self.t5_decoder(
                    audio_codes_embedded,
                    audio_codes_mask,
                    cond=encoder_out['output'],
                    cond_mask=text_mask,
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
            predicted_lens = [end_indices.get(idx, self.cfg.max_decoder_steps) for idx in range(text.size(0))]
            predicted_codes_lens = torch.tensor(predicted_lens, device=text.device).long()

            predicted_audio, predicted_audio_lens = self.codes_to_audio(predicted_codes, predicted_codes_lens)
            for idx in range(predicted_audio.size(0)):
                predicted_audio_np = predicted_audio[idx].detach().cpu().numpy()
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

    def get_dataset(self, cfg):
        dataset = instantiate(
            cfg.dataset,
            text_tokenizer=self.tokenizer,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            codec_model_downsample_factor=self.cfg.codec_model_downsample_factor,
            prior_scaling_factor=self.cfg.prior_scaling_factor,
        )
        
        return dataset

    def _setup_train_dataloader(self, cfg):
        dataset = self.get_dataset(cfg)
        sampler = dataset.get_sampler(cfg.dataloader_params.batch_size, world_size=self.trainer.world_size)
        data_loader = torch.utils.data.DataLoader(
            dataset, collate_fn=dataset.collate_fn, sampler=sampler, **cfg.dataloader_params
        )
        return data_loader

    def _setup_test_dataloader(self, cfg):
        dataset = self.get_dataset(cfg)
        data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params)
        return data_loader

    def setup_training_data(self, cfg):
        self._train_dl = self._setup_train_dataloader(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self._setup_test_dataloader(cfg)

    def setup_test_data(self, cfg):
        self._test_dl = self._setup_test_dataloader(cfg)

    @property
    def max_steps(self):
        if "max_steps" in self._cfg:
            return self._cfg.get("max_steps")

        if "max_epochs" not in self._cfg:
            raise ValueError("Must specify 'max_steps' or 'max_epochs'.")

        if "steps_per_epoch" in self._cfg:
            return self._cfg.max_epochs * self._cfg.steps_per_epoch
        return compute_max_steps(
            max_epochs=self._cfg.max_epochs,
            accumulate_grad_batches=self.trainer.accumulate_grad_batches,
            limit_train_batches=self.trainer.limit_train_batches,
            num_workers=get_num_workers(self.trainer),
            num_samples=len(self._train_dl.dataset),
            batch_size=get_batch_size(self._train_dl),
            drop_last=self._train_dl.drop_last,
        )

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        return []

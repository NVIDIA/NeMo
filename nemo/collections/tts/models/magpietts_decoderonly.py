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
from typing import List, Sequence, Tuple
import torch
import wandb
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import get_worker_info

from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.tts.data.text_to_speech_dataset_lhotse import MagpieTTSLhotseDataset, setup_tokenizers

from nemo.collections.tts.models import AudioCodecModel
from nemo.collections.tts.modules import transformer_2501

from nemo.collections.tts.modules.magpietts_modules import CharAwareSubwordEncoder, SpecialAudioToken, LocalTransformerType, cosine_schedule
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths

from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM
)
import time

def worker_init_fn(worker_id):
    # For mp.set_start_method("spawn", force=True)
    # The dataset class should be picklable, so we initialize non-picklable objects here
    logging.info(f"Worker {worker_id} initializing...")
    worker_info = get_worker_info()
    dataset = worker_info.dataset  # Get the dataset instance in this worker
    tokenizer, _ = setup_tokenizers(
        dataset.tokenizer_config, False, mode=dataset.dataset_type
    )
    dataset.text_tokenizer = tokenizer
    dataset.text_conditioning_tokenizer = tokenizer.first_tokenizer
    
class MagpieTTSDecoderModel(ModelPT):
    """
    Magpie-TTS Model Decoder Only Model
    audio/text
    """

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_devices

        # load codec
        codec_model = AudioCodecModel.restore_from(cfg.get('codecmodel_path'), strict=False)
        self.sample_rate = codec_model.sample_rate
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
        self.use_bpe_char_tokenizer = cfg.get('use_bpe_char_tokenizer', False)
        self.cfg_unconditional_prob = cfg.get('cfg_unconditional_prob', 0.0)
        
        self.tokenizer, _ = setup_tokenizers(
            all_tokenizers_config=cfg.text_tokenizers,
            use_text_conditioning_tokenizer=False,
            mode='train',
        )
        self.eos_id = self.tokenizer.first_tokenizer.eos_token_id

        self.pad_context_text_to_max_duration = False

        super().__init__(cfg=cfg, trainer=trainer)

        # This needs to happen after super().__init__()
        self._codec_model = codec_model
        self._codec_model.freeze()  #Lightning does requires_grad = False and self.eval()

        audio_embeddings = []
        for _ in range(self.num_audio_codebooks):
            audio_embeddings.append(nn.Embedding(self.num_all_tokens_per_codebook, cfg.embedding_dim))
        self.audio_embeddings = nn.ModuleList(audio_embeddings)

        self.transformer_backend_config = AutoConfig.from_pretrained(
            cfg.transformer_hf_backend,
            trust_remote_code=True,
        )

        self.decoder = AutoModelForCausalLM.from_config(self.transformer_backend_config)

        if self.use_bpe_char_tokenizer:
            # BPE char tokenizer
            assert len(self.tokenizer.tokenizers) == 1, "BPE char tokenizer should only be used with one tokenizer"
            tokenizer_name = self.tokenizer.tokenizer_names[0]
            tokenizer = self.tokenizer.tokenizers[tokenizer_name]
            subword_vocab = tokenizer.get_vocab()
            # special tokens will be stored as it is in the char_vocab
            # Each special token will only be mapped to one char id
            special_vocab = {}
            self.cas_encoder = CharAwareSubwordEncoder(
                d_embed=cfg.embedding_dim,
                llm_tokenizer_vocab=subword_vocab,
                subword_padding_idx=self.tokenizer.pad,
                special_vocab=special_vocab
            )

        self.final_proj = nn.Linear(cfg.hidden_dim, self.num_audio_codebooks * self.num_all_tokens_per_codebook)
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

        self.local_transformer_type = LocalTransformerType(cfg.get('local_transformer_type', 'none').lower())
        logging.info(f"Local transformer type: {self.local_transformer_type}")
        if self.local_transformer_type != LocalTransformerType.NO_LT:
            local_transformer_hidden_dim = cfg.get('local_transformer_hidden_dim', 256)
            if local_transformer_hidden_dim != cfg.hidden_dim:
                self.local_transformer_in_projection = nn.Linear(cfg.hidden_dim, local_transformer_hidden_dim)
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
                    name_with_dot = f"{name}."
                    if key.startswith(name_with_dot):
                        new_state_dict[key[len(name_with_dot):]] = state_dict[key]
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

    def forward(self, inputs_embeds, attention_mask, use_cache=False, past_key_values=None):
        backend_out = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )
        # hidden_states = backend_out.last_hidden_state  # (B, T_total, H)
        return backend_out
    

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
                        wandb_audio_log[f"Audio/Example_{idx}"].append(wandb.Audio(context_audio_np, sample_rate=self.sample_rate, caption="context"))
                    wandb_audio_log[f"Audio/Example_{idx}"].append(wandb.Audio(pred_audio_np, sample_rate=self.sample_rate, caption="prediction"))
                    wandb_audio_log[f"Audio/Example_{idx}"].append(wandb.Audio(target_audio_np, sample_rate=self.sample_rate, caption="target"))

                if is_tb:
                    if context_audio_np is not None:
                        logger.experiment.add_audio(
                            f'Example_{idx}/context',
                            context_audio_np,
                            global_step=self.global_step,
                            sample_rate=self.sample_rate,
                        )
                    logger.experiment.add_audio(
                        f'Example_{idx}/prediction',
                        pred_audio_np,
                        global_step=self.global_step,
                        sample_rate=self.sample_rate,
                    )
                    logger.experiment.add_audio(
                        f'Example_{idx}/target',
                        target_audio_np,
                        global_step=self.global_step,
                        sample_rate=self.sample_rate,
                    )

        return wandb_audio_log

    
    def join_embeddings_temporally(
        self,
        embeddings: Sequence[torch.Tensor],     # [ (B, Ti, E), … ]
        lengths:  Sequence[torch.Tensor],     # [ (B,), … ]  same order/size as `embeddings`
        pad_embed: torch.Tensor | None = None # (E,)  defaults to zeros
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Merges Multiple Embedding sequences into a single Embedding Sequence.

        Args:
            embeddings  : Sequence of tensors, each of shape (B, Ti, E) — batch, time, embedding
            lengths     : Sequence of tensors, each of shape (B,)
            pad_embed   : (E,)  — embedding to use for padding, defaults to zeros
        
        Returns:
            joined      : (B, max_sum_len, E)  — merged & padded
            out_lengths : (B,)  — total lengths of each batch element after merging
        """
        if len(embeddings) == 0:
            raise ValueError("contexts must be non-empty")

        B, _, E = embeddings[0].shape
        device = embeddings[0].device
        dtype = embeddings[0].dtype

        # 1. compute output sizes
        len_stack   = torch.stack(tuple(lengths), dim=0)          # (N, B)
        out_lengths = len_stack.sum(0)
        max_len     = int(out_lengths.max())

        if pad_embed is None:
            pad_embed = torch.zeros(E, dtype=dtype, device=device)

        joined = pad_embed.expand(B, max_len, E).clone()          # (B,max_len,E)

        # batch row indices
        batch_rows = torch.arange(B, device=device).unsqueeze(1)  # (B,1)

        # running offset keeps “write cursor” for each row
        offset = torch.zeros(B, dtype=torch.long, device=device)  # (B,)

        for i, (embedding_i, len_i) in enumerate(zip(embeddings, lengths)):
            Ti = embedding_i.shape[1]
            t_idx  = torch.arange(Ti, device=device) # (Ti,)
            mask   = t_idx.unsqueeze(0) < len_i.unsqueeze(1) # (B,Ti)

            # destination columns: offset + t
            dest_cols = offset.unsqueeze(1) + t_idx # (B,Ti)

            # Assign embedding_i to the correct positions in joined
            joined[batch_rows.expand_as(mask)[mask],
                dest_cols[mask]] = embedding_i[mask]

            # move cursor past this segment
            offset += len_i

        return joined, out_lengths

    def prepare_context_tensors(self, batch):
        # Transcript
        text = batch['text']
        text_lens = batch['text_lens']
        text_embedded = self.decoder.get_input_embeddings()(text)
        if self.use_bpe_char_tokenizer:
            text_mask = get_mask_from_lengths(text_lens)
            cas_embedding = self.cas_encoder(text, subword_mask=text_mask)  # (B, L, E)
            text_embedded = text_embedded + cas_embedding

        # Context Audio
        if 'context_audio_codes' in batch:
            context_audio_codes = batch['context_audio_codes']
            context_audio_codes_lens = batch['context_audio_codes_lens']
        else:
            context_audio_codes, context_audio_codes_lens = self.audio_to_codes(
                batch['context_audio'], batch['context_audio_lens'], audio_type='context'
            )
        context_audio_embedded = self.embed_audio_tokens(context_audio_codes)  # (B, T', E)

        # Context Text
        context_text_tokens = batch['context_text_tokens']
        context_text_lens = batch['context_text_tokens_lens']
        context_text_embedded = self.decoder.get_input_embeddings()(context_text_tokens)  # (B, L, E)
    
        full_context_embedding, full_context_lens = self.join_embeddings_temporally(
            embeddings=[context_audio_embedded, context_text_embedded, text_embedded],
            lengths=[context_audio_codes_lens, context_text_lens, text_lens],
        )

        return {
            'full_context_embedding': full_context_embedding,  # (B, T_total, E)
            'full_context_lens': full_context_lens,  # (B,)
            'context_audio_codes': context_audio_codes,  # (B, C, T')
            'context_audio_codes_lens': context_audio_codes_lens,  # (B,)
        }

    def slice_pred_embeddings(self, transformer_out, context_lens, target_lens):
        """
        Slices the transformer output to get the predicted embeddings for the target sequence.
        Args:
            transformer_out: (B, T, E)
            context_lens: (B,) - start index of target per batch
            target_lens: (B,) - length of target per batch
        
        Returns: (B, T_max, E) tensor where T_max = max(target_lens)
        """
        B, T, E = transformer_out.shape
        device = transformer_out.device

        # Compute max target length in batch for padding
        max_len = target_lens.max().item()

        # Build index tensor for each batch element
        # Shape: (B, max_len)
        range_indices = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)
        gather_indices = context_lens.unsqueeze(1) + range_indices  # (B, max_len)
        gather_indices = torch.clamp(gather_indices, max=transformer_out.size(1) - 1)
        
        # Expand to shape (B, max_len, E) for gather
        gather_indices_exp = gather_indices.unsqueeze(2).expand(-1, -1, E)
        sliced = torch.gather(transformer_out, dim=1, index=gather_indices_exp)
        return sliced


    def process_batch(self, batch, mode="train"):
        context_tensors = self.prepare_context_tensors(batch)
        context_embedding = context_tensors['full_context_embedding']  # (B, T_total, E)
        context_lens = context_tensors['full_context_lens']  # (B,)

        if mode == 'train' and self.cfg_unconditional_prob > 0.0:
            if torch.rand(1).item() < self.cfg_unconditional_prob:
                context_embedding = torch.zeros_like(context_embedding)  # (B, T_total, E) to be used in case of no context

        if 'audio_codes' not in batch:
            audio_codes, audio_codes_lens = self.audio_to_codes(batch['audio'], batch['audio_lens'])
        else:
            audio_codes = batch['audio_codes']
            audio_codes_lens = batch['audio_codes_lens']
        
        audio_codes_lens_input = audio_codes_lens_target = audio_codes_lens - 1
        audio_codes_target = audio_codes[:, :, 1:]  # (B, C, T') Target for the decoder
        audio_codes_input = audio_codes[:, :, :-1]  # (B, C, T') Input to the decoder
        audio_codes_input_embedded = self.embed_audio_tokens(audio_codes_input) # (B, T, E) # Computing this to be use in the alignment encoder

        context_plus_audio_embedded, context_plus_audio_lens = self.join_embeddings_temporally(
            embeddings=[context_embedding, audio_codes_input_embedded],
            lengths=[context_lens, audio_codes_lens_input],
        )

        transformer_out = self.forward(
            inputs_embeds=context_plus_audio_embedded,
            attention_mask=get_mask_from_lengths(context_plus_audio_lens),
        )
        transformer_hidden_states = transformer_out.hidden_states[-1]  # (B, T_total, E)
        
        pred_embeddings = self.slice_pred_embeddings(
            transformer_hidden_states,
            context_lens=context_lens,
            target_lens=audio_codes_lens_target,
        )
        
        logits = self.final_proj(pred_embeddings)  # (B, T', num_codebooks * num_tokens_per_codebook)
        # import ipdb; ipdb.set_trace()
        codebook_loss, loss_mask = self.compute_loss(logits, audio_codes_target, audio_codes_lens_target)
        loss = codebook_loss

        local_transformer_loss = None
        local_transformer_logits = None
        if self.local_transformer_type != LocalTransformerType.NO_LT:
            if self.local_transformer_type == LocalTransformerType.MASKGIT:
                # randomly replace some positions with MASK_TOKEN
                audio_codes_masked, mask_tokens_mask = self.maskgit_apply_random_mask(audio_codes_target)
                local_transformer_logits = self.compute_local_transformer_logits(pred_embeddings, audio_codes_masked, targets_offset_by_one=True)
                #audio_codes_masked = audio_codes_masked[:, 1:, :]
                local_transformer_loss, _ = self.compute_loss(local_transformer_logits, audio_codes_target, audio_codes_lens_target, mask_tokens_mask)
            else:
                # autoregressive
                assert self.local_transformer_type == LocalTransformerType.AR, "Unexpected local transformer type"
                local_transformer_logits = self.compute_local_transformer_logits(pred_embeddings, audio_codes_target, targets_offset_by_one=False)
                local_transformer_loss, _ = self.compute_loss(local_transformer_logits, audio_codes_target, audio_codes_lens_target, None)
            local_transformer_loss_scale = self.cfg.get('local_transformer_loss_scale', 1.0)
            loss = loss + local_transformer_loss_scale * local_transformer_loss

        return {
            'loss': loss,
            'codebook_loss': codebook_loss,
            'local_transformer_loss': local_transformer_loss,
            'local_transformer_logits': local_transformer_logits,  # (B, T', num_codebooks * num_tokens_per_codebook)
            'logits': logits,
            'audio_codes_target': audio_codes_target,  # (B, C, T')
            'audio_codes_lens_target': audio_codes_lens_target,  # (B,)
            'context_audio_codes': context_tensors['context_audio_codes'],  # (B, C, T')
            'context_audio_codes_lens': context_tensors['context_audio_codes_lens'],  # (B,)
        }

        

    def training_step(self, batch, batch_idx):
        batch_output = self.process_batch(batch)
        loss = batch_output['loss']
        codebook_loss = batch_output['codebook_loss']
        self.log('train/codebook_loss', codebook_loss, prog_bar=True, sync_dist=True)
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
        logits = batch_output['logits']
        audio_codes_target = batch_output['audio_codes_target']
        audio_codes_lens_target = batch_output['audio_codes_lens_target']
        context_audio_codes = batch_output['context_audio_codes']
        context_audio_codes_lens = batch_output['context_audio_codes_lens']
        
        if batch_idx == 0 and self.global_rank == 0:
            # Prepare dictionary for aggregated wandb logging
            wandb_log_dict = {}

            # Get audio data for logging
            wandb_log_dict.update(
                self.log_val_audio_example(
                    logits, audio_codes_target, audio_codes_lens_target, context_audio_codes, context_audio_codes_lens
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
            'val_local_transformer_loss': local_transformer_loss,
        }
        self.validation_step_outputs.append(val_output)

        return val_output

    def on_validation_epoch_end(self):
        collect = lambda key: torch.stack([x[key] for x in self.validation_step_outputs]).mean()
        val_loss = collect("val_loss")
        val_codebook_loss = collect("val_codebook_loss")
        
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        self.log("val/codebook_loss", val_codebook_loss, prog_bar=True, sync_dist=True)
        
        if self.local_transformer_type != LocalTransformerType.NO_LT:
            val_local_transformer_loss = collect("val_local_transformer_loss")
            self.log("val/local_transformer_loss", val_local_transformer_loss, prog_bar=True, sync_dist=True)
        
        self.validation_step_outputs.clear()  # free memory

    def get_dataset(self, dataset_cfg, dataset_type):
        dataset = instantiate(
            dataset_cfg.dataset,
            sample_rate=self.sample_rate,
            bos_id=None,
            eos_id=self.eos_id,
            audio_bos_id=self.audio_bos_id,
            audio_eos_id=self.audio_eos_id,
            context_audio_bos_id=self.context_audio_bos_id,
            context_audio_eos_id=self.context_audio_eos_id,
            num_audio_codebooks=self.num_audio_codebooks,
            codec_model_samples_per_frame=self.codec_model_samples_per_frame,
            prior_scaling_factor=0.0,
            load_cached_codes_if_available=self.cfg.load_cached_codes_if_available,
            dataset_type=dataset_type,  # train or test used for setting phone prob to 1.0 in test dataset (worker_init_fn)
            use_text_conditioning_tokenizer=True,
            pad_context_text_to_max_duration=self.pad_context_text_to_max_duration,
            context_duration_min=self.cfg.context_duration_min,
            context_duration_max=self.cfg.context_duration_max,
        )
        dataset.load_16khz_audio = False
        dataset.tokenizer_config = (
            self.cfg.text_tokenizers
        )  # This will be used in worker_init_fn for instantiating tokenizer
        return dataset

    def get_lhotse_dataloader(self, dataset_cfg, mode='train') -> torch.utils.data.DataLoader:
        # TODO @xueyang: better to distinguish cfg. self.cfg is the model cfg, while cfg here is train_ds cfg. Also
        #   cfg is a classifier-free guidance.
        dataset = MagpieTTSLhotseDataset(
            sample_rate=self.sample_rate,
            volume_norm=dataset_cfg.volume_norm,
            codec_model_samples_per_frame=self.codec_model_samples_per_frame,
            codec_model_name=self.cfg.codec_model_name,
            audio_bos_id=self.audio_bos_id,
            audio_eos_id=self.audio_eos_id,
            context_audio_bos_id=self.context_audio_bos_id,
            context_audio_eos_id=self.context_audio_eos_id,
            num_audio_codebooks=self.num_audio_codebooks,
            prior_scaling_factor=0.0,
            load_cached_codes_if_available=self.cfg.load_cached_codes_if_available,
            dataset_type=mode,  # train or test used for setting phone prob to 1.0 in test dataset (worker_init_fn)
            load_16khz_audio=False,
            pad_context_text_to_max_duration=self.pad_context_text_to_max_duration,
            context_duration_min=self.cfg.context_duration_min,
            context_duration_max=self.cfg.context_duration_max,
            use_text_conditioning_tokenizer=True,
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
                dataset.text_tokenizer, _ = setup_tokenizers(
                    all_tokenizers_config=self.cfg.text_tokenizers,
                    use_text_conditioning_tokenizer=False,
                    mode='train',
                )
                dataset.text_conditioning_tokenizer = dataset.text_tokenizer.first_tokenizer

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
                dataset.text_tokenizer, _ = setup_tokenizers(
                    all_tokenizers_config=self.cfg.text_tokenizers,
                    use_text_conditioning_tokenizer=False,
                    mode='test'
                )
                dataset.text_conditioning_tokenizer = dataset.text_tokenizer.first_tokenizer

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

    def infer_batch(self, batch, max_decoder_steps=500, temperature=0.7, topk=80, use_local_transformer_for_inference=False, maskgit_n_steps=3, use_cfg=False, cfg_scale=1.0):
        with torch.inference_mode():
            start_time = time.time()
            context_tensors = self.prepare_context_tensors(batch)
            context_embedding = context_tensors['full_context_embedding']  # (B, T_total, E)
            context_lens = context_tensors['full_context_lens']  # (B,)

            audio_codes_bos = torch.full(
                    (context_embedding.size(0), self.num_audio_codebooks, 1), self.audio_bos_id, device=context_embedding.device
                ).long()
            audio_codes_lens = torch.full((context_embedding.size(0),), 1, device=context_embedding.device).long()
            audio_codes_input = audio_codes_bos

            audio_codes_input_embedded = self.embed_audio_tokens(audio_codes_input)  # (B, T, E)
            
            context_plus_audio_embedded, context_plus_audio_lens = self.join_embeddings_temporally(
                embeddings=[context_embedding, audio_codes_input_embedded],
                lengths=[context_lens, audio_codes_lens],
            )
            min_context_len = context_plus_audio_lens.min().item()

            actual_batch_size = context_embedding.size(0)
            if use_cfg:
                dummy_context_embedding = torch.zeros_like(context_plus_audio_embedded)  # (B, T_total, E) to be used in case of no context
                dummy_context_plus_audio_embedded, _ = self.join_embeddings_temporally(
                    embeddings=[dummy_context_embedding, audio_codes_input_embedded],
                    lengths=[context_lens, audio_codes_lens],
                )
                first_inference_input = torch.cat(
                    [context_plus_audio_embedded, dummy_context_plus_audio_embedded],
                    dim=0
                )[:,:min_context_len, :]  # (2B, T_min, E)
            else:
                first_inference_input = context_plus_audio_embedded[:, :min_context_len, :]  # (B, T_min, E)
            # First forward pass to get the initial hidden state and past key values
            transformer_out = self.forward(
                inputs_embeds=first_inference_input,
                attention_mask=None,
                use_cache=True,
                past_key_values=None,  # No past key values for the first step
            )

            time_to_first_prediction = time.time() - start_time
            last_hidden = transformer_out.hidden_states[-1]  # (B, T_total, E)
            past_kv = transformer_out.past_key_values

            all_predictions = []
            end_indices = {}
            
            for idx in range(max_decoder_steps):
                if idx % 20 == 0:
                    print(f"Decoding timestep {idx}")

                all_code_logits_t = self.final_proj(last_hidden[:, -1, :])  # (B, num_codebooks * num_tokens_per_codebook)
                if use_cfg:
                    conditional_logits = all_code_logits_t[:actual_batch_size]
                    unconditional_logits = all_code_logits_t[actual_batch_size:]
                    all_code_logits_t = cfg_scale * conditional_logits +  (1.0 - cfg_scale) * unconditional_logits

                if use_local_transformer_for_inference:
                    if self.local_transformer_type == LocalTransformerType.AR :
                        # Autoregressive sampling with local transformer
                        audio_codes_next = self.local_transformer_sample_autoregressive(
                            dec_output=last_hidden[:, -1, :],
                            temperature=temperature,
                            topk=topk,
                            use_cfg=use_cfg,
                            cfg_scale=cfg_scale,
                        )
                    elif self.local_transformer_type == LocalTransformerType.MASKGIT:
                        audio_codes_next = self.local_transformer_sample_maskgit(
                            dec_output=last_hidden[:, -1, :],
                            temperature=temperature,
                            topk=topk,
                            n_steps=maskgit_n_steps,
                            use_cfg=use_cfg,
                            cfg_scale=cfg_scale,
                        )
                    else:
                        raise ValueError(f"Local transformer inference requested by but local transformer type is {self.local_transformer_type}")
                    # TODO @rfejgin: should we add argmax sampling for EOS here too?
                    all_codes_next_argmax = audio_codes_next
                else:
                    # Parallel sampling from logits
                    audio_codes_next = self.sample_codes_from_logits(all_code_logits_t, temperature=temperature, topk=topk) # (B, num_codebooks)
                    all_codes_next_argmax = self.sample_codes_from_logits(all_code_logits_t, temperature=0.01) # (B, num_codebooks)

                for item_idx in range(all_codes_next_argmax.size(0)):
                    if item_idx not in end_indices and idx + min_context_len > context_plus_audio_lens[item_idx]:
                        pred_token = all_codes_next_argmax[item_idx][0].item()
                        pred_token_multinomial = audio_codes_next[item_idx][0].item()
                        if (pred_token == self.audio_eos_id) or (pred_token_multinomial == self.audio_eos_id):
                            print("End detected for item {} at timestep {}".format(item_idx, idx))
                            end_indices[item_idx] = idx
                
                all_predictions.append(audio_codes_next)
                
                new_emb = self.embed_audio_tokens(audio_codes_next.unsqueeze(2))  # (B, 1, E)
                
                context_incomplete_mask = context_plus_audio_lens > idx + min_context_len # (B,)
                # True if we have not yet reached the end of the context for this item
                # import ipdb; ipdb.set_trace()
                if context_incomplete_mask.any():
                    context_incomplete_mask = context_incomplete_mask.unsqueeze(1).unsqueeze(2).float()  # (B, 1, 1)
                    context_embedding = context_plus_audio_embedded[:,min_context_len+idx:min_context_len+idx+1,:] # (B, 1, E)
                    next_input = context_incomplete_mask * context_embedding + (1 - context_incomplete_mask) * new_emb
                    if use_cfg:
                        dummy_context_embedding = torch.zeros_like(context_embedding)  # (B, 1, E) to be used in case of no context
                        next_input_unconditional = context_incomplete_mask * dummy_context_embedding + (1 - context_incomplete_mask) * new_emb
                        next_input = torch.cat([next_input, next_input_unconditional], dim=0)  # (2B, 1, E)
                else:
                    next_input = new_emb
                    if use_cfg:
                        # Duplicate the input for CFG
                        next_input = torch.cat([next_input, next_input], dim=0)  # (2B, 1, E)

                transformer_out = self.forward(
                    inputs_embeds=next_input,
                    attention_mask=None,
                    use_cache=True,
                    past_key_values=past_kv,
                )
                last_hidden = transformer_out.hidden_states[-1]
                past_kv = transformer_out.past_key_values
                if len(end_indices) == audio_codes_next.size(0):
                    print("All items finished at timestep {}".format(idx))
                    break
            
            tts_generation_time = time.time() - start_time
            tts_generation_time_per_frame = tts_generation_time / len(all_predictions)
            pred_codes_start_indices = context_plus_audio_lens - min_context_len # (B,)
            predicted_lens = [end_indices.get(idx, max_decoder_steps) for idx in range(context_embedding.size(0))] #  Ensure that the codec is atleast of length 4
            predicted_codes_lens = torch.tensor(predicted_lens, device=context_embedding.device).long()
            predicted_codes_lens = predicted_codes_lens - pred_codes_start_indices # (B,)

            predicted_codes = torch.stack(all_predictions, dim=-1)  # (B, num_codebooks, T)
            predicted_codes = self.slice_pred_embeddings(
                predicted_codes.permute(0, 2, 1),
                context_lens=pred_codes_start_indices,
                target_lens=predicted_codes_lens,
            )
            predicted_codes = predicted_codes.permute(0, 2, 1)  # (B, num_codebooks, T)
            predicted_audio, predicted_audio_lens = self.codes_to_audio(predicted_codes, predicted_codes_lens)
            
            end_time = time.time()
            total_audio_duration_generated = (predicted_audio_lens.max().item() * predicted_audio_lens.shape[0])/self.sample_rate
            rtf = total_audio_duration_generated / (end_time - start_time)

            rtf_metrics = {
                'rtf': rtf,
                'time_to_first_prediction': time_to_first_prediction,
                'tts_generation_time': tts_generation_time,
                'max_frames_generated': len(all_predictions),
                'tts_generation_time_per_frame': tts_generation_time_per_frame,
                'batch_size': context_embedding.size(0),
            }

            return predicted_audio, predicted_audio_lens, predicted_codes, predicted_codes_lens, rtf_metrics


        
    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        return []


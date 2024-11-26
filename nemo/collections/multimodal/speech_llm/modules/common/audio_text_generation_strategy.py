# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from typing import List, Optional, Tuple

import torch

import nemo.collections.nlp.modules.common.text_generation_strategy as text_generation_strategy
from nemo.collections.multimodal.speech_llm.parts.utils.data_utils import shift_tokens_by_multi_audios
from nemo.collections.nlp.modules.common.megatron.utils import build_position_ids

# the text representation of eos_id, it applies for all tokenizers
END_OF_SEQ = '<|endoftext|>'


def switch(val1, val2, boolean):
    boolean = boolean.type_as(val1)
    boolean = boolean.unsqueeze(0).unsqueeze(-1)
    return (1 - boolean) * val1 + boolean * val2


class AudioToTextGenerationStrategy(text_generation_strategy.GPTModelTextGenerationStrategy):
    def init_batch(
        self,
        context_tokens: torch.Tensor,
        context_lengths: torch.Tensor,
        audio_signal: torch.Tensor,
        audio_length: torch.Tensor,
        compute_attention_mask: bool,
        num_audios: Optional[torch.Tensor] = None,
        context_start_idx: Optional[List[List[int]]] = None,
    ):
        """initialize the batch data before the inference steps."""
        # Move to GPU.

        audio_feats, audio_feat_lens = self.model.perception(
            input_signal=audio_signal,
            input_signal_length=audio_length,
            processed_signal=None,
            processed_signal_length=None,
        )

        if num_audios is not None:
            # handle multiple audio files per sample
            audio_feats = audio_feats.split(num_audios.tolist())
            audio_feat_lens = audio_feat_lens.split(num_audios.tolist())

        encoder_input, attention_mask, _, position_ids, encoder_max_length = self.model.inject_perception_input(
            audio_feats, audio_feat_lens, context_tokens, context_lengths, context_start_idx
        )

        self.attention_mask = attention_mask
        self.position_ids = position_ids

        if num_audios is not None:
            # handle multiple audio files per sample
            new_context_tokens = shift_tokens_by_multi_audios(
                context_tokens, context_lengths, audio_feat_lens, context_start_idx, encoder_max_length
            )
            audio_feat_lens = torch.stack([torch.sum(lens) for lens in audio_feat_lens])  # [batch,]
        else:
            new_context_tokens = self.model._shift_labels_by_emb_len(
                context_tokens, context_lengths, audio_feat_lens, encoder_max_length, pad_token=0
            )

        return new_context_tokens, encoder_input, audio_feat_lens

    def clip_max_len(self, maxlen: int) -> int:
        """clip the max len based on the LM model max sequence length"""
        # for positional embedding types that allow length extrapolation, don't clip the max length
        if self.model.cfg.get("position_embedding_type", "learned_absolute") == "learned_absolute":
            if maxlen > self.model.cfg.encoder_seq_length + 1:
                maxlen = self.model.cfg.encoder_seq_length + 1
        return maxlen

    def prepare_batch_at_step(
        self,
        tokens: torch.Tensor,
        input_embeddings: torch.Tensor,
        maxlen: int,
        micro_batch_size: int,
        step: int,
        context_lengths: torch.Tensor,
        curr_context_length: int,
        compute_attention_mask: bool,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        # types2use = None
        if step == 0:
            # Allocate memory for the entire context.
            set_inference_key_value_memory = True
            tokens2use = tokens[:, :curr_context_length]
            positions2use = self.position_ids[:, :curr_context_length]
            embeddings2use = input_embeddings[:curr_context_length]
        else:
            # Set this to false so the memory is not reallocated.
            set_inference_key_value_memory = False
            tokens2use = tokens[:, curr_context_length - 1].view(micro_batch_size, -1)
            positions2use = self.position_ids[:, curr_context_length - 1].view(micro_batch_size, -1)
            embeddings2use = self.model._get_text_embeddings(tokens2use, positions2use)
            started = context_lengths <= curr_context_length
            embeddings2use = switch(input_embeddings[curr_context_length - 1].unsqueeze(0), embeddings2use, started)

        """Prepare batch for each of the inference steps"""
        setkey_value_array = torch.tensor(
            [set_inference_key_value_memory] * micro_batch_size, device=torch.cuda.current_device()
        )
        len_array = torch.tensor([maxlen] * micro_batch_size, device=torch.cuda.current_device())

        batch = [tokens2use, embeddings2use, self.attention_mask, positions2use, setkey_value_array, len_array]
        tensor_shape = [tokens2use.shape[1], micro_batch_size, self.model.cfg.hidden_size]
        return batch, tensor_shape

    def post_process(self, tokens: torch.Tensor, new_tokens: torch.Tensor, context_length: int):
        """
        At the end of the inference, post process the inference results
        """
        pass

    def end_of_generation_condition(
        self, tokens: torch.Tensor, prev: torch.Tensor, eod_id: int, end_strings: List[str]
    ) -> torch.Tensor:
        """
        return whether the generation should stop based on the previous token
        Args:
            tokens (torch.Tensor): the generated tokens so far
            prev  (torch.Tensor): the previous token
            eod_id (int): the end of document token id
            end_strings (List[str]): the list of end of generation strings
        returns:
            a boolean tensor indicating whether the generation should stop
        """
        # import pdb; pdb.set_trace()
        if len(end_strings) == 1 and end_strings[0] == END_OF_SEQ:
            return prev == eod_id
        else:
            tokenizer = self.model.tokenizer
            conditions = []
            end_tokens = set()
            end_tokens.add(eod_id)
            for end_string in end_strings:
                if len(end_string) > 1:
                    continue
                ids_1 = tokenizer.text_to_ids(f'<extra_id_1>{end_string}')
                ids_2 = tokenizer.text_to_ids('<extra_id_1>')
                if len(ids_1) <= len(ids_2):
                    continue
                token_id = ids_1[len(ids_2) :][0]

                end_tokens.add(token_id)

            for p, token_item in zip(prev, tokens):
                text = tokenizer.ids_to_text(token_item.tolist())
                conditions.append(
                    any([text.endswith(end_string) for end_string in end_strings] + [p.item() in end_tokens])
                )
            return torch.tensor(conditions, dtype=torch.bool, device=tokens.device)


class CrossAttendAudioToTextGenerationStrategy(AudioToTextGenerationStrategy):
    def init_batch(
        self,
        context_tokens: torch.Tensor,
        context_lengths: torch.Tensor,
        audio_signal: torch.Tensor,
        audio_length: torch.Tensor,
        compute_attention_mask: bool,
        num_audios: Optional[torch.Tensor] = None,
        context_start_idx: Optional[List[List[int]]] = None,
    ):
        """initialize the batch data before the inference steps."""
        # Move to GPU.
        batch = {
            'audio_signal': audio_signal,
            'audio_signal_length': audio_length,
            'tokens': context_tokens,
            'tokens_length': context_lengths,
            'labels': context_tokens,
            'loss_mask': None,
        }
        if self.model.perception.cfg.get('combine_return', True):
            (
                encoder_input,
                self.attention_mask,
                context_tokens,
                _,
                (speech_encoded, speech_encoded_len, extra_outputs),
            ) = self.model.prepare_llm_input(batch)
            self.position_ids = build_position_ids(encoder_input[:, :, 0].transpose(0, 1))
            self.extra_outputs = extra_outputs
            return (
                context_tokens,
                (encoder_input, speech_encoded, speech_encoded_len),
                torch.zeros_like(context_lengths),
            )
        else:
            (
                encoder_input,
                self.attention_mask,
                context_tokens,
                _,
                (speech_encoded, speech_encoded_len, llm_encoded_len, extra_outputs),
            ) = self.model.prepare_llm_input(batch)
            self.position_ids = build_position_ids(encoder_input[:, :, 0].transpose(0, 1))
            self.extra_outputs = extra_outputs
            return context_tokens, (encoder_input, speech_encoded, speech_encoded_len), llm_encoded_len

    def prepare_batch_at_step(
        self,
        tokens: torch.Tensor,
        input_embeddings: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        maxlen: int,
        micro_batch_size: int,
        step: int,
        context_lengths: torch.Tensor,
        curr_context_length: int,
        compute_attention_mask: bool,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        # types2use = None
        self.input_embeds_hidden = self.extra_outputs.get('input_embeds_hidden', None)
        input_embeddings, speech_encoded, speech_encoded_len = input_embeddings
        if step == 0:
            # Allocate memory for the entire context.
            set_inference_key_value_memory = True
            tokens2use = tokens[:, :curr_context_length]
            positions2use = self.position_ids[:, :curr_context_length]
            embeddings2use = input_embeddings[:curr_context_length]
        else:
            # Set this to false so the memory is not reallocated.
            set_inference_key_value_memory = False
            tokens2use = tokens[:, curr_context_length - 1].view(micro_batch_size, -1)
            positions2use = self.position_ids[:, curr_context_length - 1].view(micro_batch_size, -1)
            embeddings2use = self.model._get_text_embeddings(tokens2use, positions2use).transpose(0, 1)
            started = context_lengths <= curr_context_length
            # for seq started, first get embeddings2use, and then run cross attend, after that replace embeddings2use with the cross attended embed
            # use speech_encoded; rerun cross attend
            # [1, b, d]
            decoder_mems_list = self.extra_outputs.get('decoder_mems_list', None)
            if decoder_mems_list is not None:
                decoder_mems_list = decoder_mems_list[:, :, : curr_context_length - 1]
            # need to use audio_ratio field if to support text-only decoding
            embeddings2use, self.extra_outputs = self.model.perception_cross_attn(
                speech_encoded,
                speech_encoded_len,
                embeddings2use,
                input_lengths=tokens2use.squeeze(-1) != self.model.tokenizer.eos_id,
                decoder_mems_list=decoder_mems_list,
                return_mems=True,
            )
            self.input_embeds_hidden = self.extra_outputs.get('input_embeds_hidden', None)
            embeddings2use = switch(
                input_embeddings[curr_context_length - 1].unsqueeze(0), embeddings2use.transpose(0, 1), started
            )

        """Prepare batch for each of the inference steps"""
        setkey_value_array = torch.tensor(
            [set_inference_key_value_memory] * micro_batch_size, device=torch.cuda.current_device()
        )
        len_array = torch.tensor([maxlen] * micro_batch_size, device=torch.cuda.current_device())

        batch = [tokens2use, embeddings2use, self.attention_mask, positions2use, setkey_value_array, len_array]
        tensor_shape = [tokens2use.shape[1], micro_batch_size, self.model.cfg.hidden_size]
        return batch, tensor_shape


class AudioToAudioGenerationStrategy(AudioToTextGenerationStrategy):
    def end_of_generation_condition(
        self,
        tokens: torch.Tensor,
        prev: torch.Tensor,
        eod_id: int,
        end_strings: List[str],
        speech_eos_id: int,
    ) -> torch.Tensor:
        """
        return whether the generation should stop based on the previous token
        Args:
            tokens (torch.Tensor): the generated tokens so far
            prev  (torch.Tensor): the previous token
            eod_id (int): the end of document token id
            end_strings (List[str]): the list of end of generation strings
        returns:
            a boolean tensor indicating whether the generation should stop
        """
        from nemo.utils import logging

        text_done_token = super().end_of_generation_condition(tokens, prev, eod_id, end_strings).any(dim=1)
        speech_done_token = (prev[:, 1:] == speech_eos_id).all(dim=1)
        if speech_done_token.any():
            logging.debug(f"speech done text {text_done_token}")
        return speech_done_token

    def prepare_batch_at_step(
        self,
        tokens: torch.Tensor,
        input_embeddings: torch.Tensor,
        maxlen: int,
        micro_batch_size: int,
        step: int,
        context_lengths: torch.Tensor,
        curr_context_length: int,
        compute_attention_mask: bool,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        if step == 0:
            set_inference_key_value_memory = True
            tokens2use = tokens[:, :curr_context_length]
            positions2use = self.position_ids[:, :curr_context_length]
            embeddings2use = input_embeddings[:curr_context_length]
        else:
            set_inference_key_value_memory = False
            # handle positions2use and tokens2use differently
            tokens2use = tokens[:, curr_context_length - 1].view(micro_batch_size, 1, -1)
            positions2use = self.position_ids[:, curr_context_length - 1].view(micro_batch_size, 1, -1)
            # embedding offset and sum is handled inside
            embeddings2use = self.model._get_text_embeddings(tokens2use, positions2use)
            started = context_lengths <= curr_context_length
            embeddings2use = switch(input_embeddings[curr_context_length - 1].unsqueeze(0), embeddings2use, started)
            duplex_method = self.model.cfg.get("duplex_method", None)
            if duplex_method is None:
                pass
            elif duplex_method == "from_multiturn" or duplex_method == "from_duplex":
                encoded = self.encoded[:, curr_context_length - 1].view(micro_batch_size, 1, -1)
                embeddings2use = embeddings2use + encoded.transpose(0, 1).contiguous()
            else:
                raise ValueError(f"duplex_method {duplex_method} not supported")

        setkey_value_array = torch.tensor(
            [set_inference_key_value_memory] * micro_batch_size, device=torch.cuda.current_device()
        )
        len_array = torch.tensor([maxlen] * micro_batch_size, device=torch.cuda.current_device())
        batch = [tokens2use, embeddings2use, self.attention_mask, positions2use, setkey_value_array, len_array]
        tensor_shape = [tokens2use.shape[1], micro_batch_size, self.model.cfg.hidden_size]
        return batch, tensor_shape

    def init_batch_duplex_from_multiturn(self, context_tokens, context_lengths, audio_signal, audio_length):
        tokens_to_generate = self.model.get_inference_config()['tokens_to_generate']
        _, answer_audio_lens = self.model.get_duration_by_steps(tokens_to_generate)

        duplex_method = self.model.cfg.get("duplex_method", None)
        if duplex_method == 'from_duplex':
            answer_audio_lens = int(
                answer_audio_lens
                / self.model.cfg.data.train_ds.get("codec_sample_rate", 22050)
                * self.model.cfg.data.train_ds.get("sample_rate", 16000)
            )
            padded_audio_signal = torch.cat(
                [audio_signal, torch.zeros([audio_signal.shape[0], answer_audio_lens]).cuda()], axis=1
            )
            all_lens_answer_rate = (
                (audio_length + answer_audio_lens)
                / self.model.cfg.data.train_ds.get("sample_rate", 16000)
                * self.model.cfg.data.train_ds.get("codec_sample_rate", 22050)
            ).long()
            batch = {
                'audio_signal': padded_audio_signal,
                'audio_signal_length': audio_length + answer_audio_lens,
                'context_lengths': context_lengths,
                'target_texts_merge': torch.full(
                    [audio_signal.shape[0], self.model.get_step_from_audio_len(all_lens_answer_rate).max() + 1],
                    self.model.tokenizer.eos_id,
                ).cuda(),
                'answer_audio_lens': all_lens_answer_rate,
                'answer_audio': torch.zeros([audio_signal.shape[0], all_lens_answer_rate.max()]).cuda(),
                'loss_mask': None,
            }
        elif duplex_method == 'from_multiturn':
            batch = {
                'audio_signal': audio_signal,
                'audio_signal_length': audio_length,
                'context_lengths': context_lengths,
                'labels': context_tokens,
                'answer_audio_lens': torch.full([audio_signal.shape[0]], answer_audio_lens).cuda(),
                'answer_audio': torch.zeros([audio_signal.shape[0], answer_audio_lens]).cuda(),
                'loss_mask': None,
            }
            # pad user signal with silence of the length of answer_audio_lens and store the encoded for prepare_batch_at_step
            # in real setting, encoded has to be recomputed every time if using bidirectional encoder or incrementally computed
        else:
            raise ValueError(f"duplex_method {duplex_method} not supported")

        encoder_input, _, labels, _, (self.encoded, _) = self.model.prepare_llm_input_duplex_from_multiturn(batch)
        self.attention_mask = self.model._create_attention_mask(encoder_input.transpose(0, 1))
        self.position_ids = build_position_ids(encoder_input.transpose(0, 1)[:, :, 0])
        return labels, encoder_input, -context_lengths + 1

    def init_batch(
        self,
        context_tokens: torch.Tensor,
        context_lengths: torch.Tensor,
        audio_signal: torch.Tensor,
        audio_length: torch.Tensor,
        compute_attention_mask: bool,
        num_audios: Optional[torch.Tensor] = None,
        context_start_idx: Optional[List[List[int]]] = None,
    ):
        """initialize the batch data before the inference steps."""
        duplex_method = self.model.cfg.get("duplex_method", None)
        if duplex_method is None:
            return super().init_batch(
                context_tokens, context_lengths, audio_signal, audio_length, compute_attention_mask, num_audios
            )
        elif duplex_method == "from_multiturn" or duplex_method == "from_duplex":
            return self.init_batch_duplex_from_multiturn(context_tokens, context_lengths, audio_signal, audio_length)
        else:
            raise ValueError(f"duplex_method {duplex_method} not supported")


def model_inference_strategy_dispatcher(model, **args):
    from nemo.collections.multimodal.speech_llm.models.modular_models import (
        CrossAttendModularAudioGPTModel,
        ModularAudioGPTModel,
    )
    from nemo.collections.multimodal.speech_llm.models.modular_s2s_models import S2sModularAudioGPTModel

    if isinstance(model, CrossAttendModularAudioGPTModel):
        return CrossAttendAudioToTextGenerationStrategy(model, **args)
    elif isinstance(model, S2sModularAudioGPTModel):
        return AudioToAudioGenerationStrategy(model, **args)
    elif isinstance(model, ModularAudioGPTModel):
        return AudioToTextGenerationStrategy(model, **args)
    else:
        return text_generation_strategy.model_inference_strategy_dispatcher(model, **args)

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

from lightning import LightningModule
from torch import Tensor
from transformers import AutoModel, AutoModelForCausalLM

from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.tts.models import AudioCodecModel


class DuplexS2SModelConfig:
    pass


class DuplexS2SModel(LightningModule):
    def __init__(self, cfg: DuplexS2SModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = AutoTokenizer("TinyLlama/TinyLlama_v1.1")
        self.speech_encoder = AudioCodecModel.from_pretrained("nvidia/low-frame-rate-speech-codec-22khz")
        # self.speech_encoder = AudioCodecModel.restore_from("Low_Frame-rate_Speech_Codec++_without_speaker_encoder.nemo")
        self.llm = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama_v1.1")

    def forward(self, input_signal: Tensor, input_signal_lens: Tensor) -> tuple[Tensor, Tensor]:
        # TODO(pzelasko): implement according to
        #   https://github.com/zhehuaichen/NeMo/blob/speechllm-develop-gen_duplex2_clean/nemo/collections/multimodal/speech_llm/models/modular_s2s_models.py
        llm_input, llm_input_lens = self.speech_encoder(input_signal, input_signal_lens)
        predicted = self.llm(llm_input, llm_input_lens)
        return predicted, llm_input_lens

    def training_step(self, batch: dict, batch_idx: int):
        inputs = self.prepare_llm_input_duplex_from_multiturn(batch)
        outputs = self(*inputs)

    def configure_optimizers(self):
        raise NotImplementedError()

    def configure_model(self) -> None:
        # TODO(pzelasko): configure FSDP2
        #   Optionally the whole model can be instantiated here instead of __init__
        #   which is a start-time and peak-memory-usage optimization, helps with very large models.
        return

    def prepare_llm_input_duplex_from_multiturn(self, audio_batch):
        codec_sample_rate = self.codec_sample_rate
        decoder_reduction_factor = self.cfg.get("decoder_reduction_factor", 1)
        # make the following to be one decoding step so as to easier replace with speech bos token and eos token
        duplex_inject_silence_second = (
            self.codec_model_downsampling_factor / codec_sample_rate * decoder_reduction_factor
        )
        silence = int(codec_sample_rate * duplex_inject_silence_second)
        user_signal = audio_batch['audio_signal']
        user_signal_length = audio_batch['audio_signal_length']

        def resample(audio, audio_lens, orig_sample_rate, target_sample_rate):
            audio = torchaudio.functional.resample(audio, orig_sample_rate, target_sample_rate)
            audio_lens = (audio_lens * (target_sample_rate / orig_sample_rate)).int()
            return audio, audio_lens

        if 'target_texts_merge' not in audio_batch:  # create duplex data from single turn
            # this branch is not used anymore; duplex data should go to else:
            labels, loss_mask = (
                audio_batch['labels'],
                audio_batch['loss_mask'],
            )
            context_lengths = audio_batch['context_lengths']

            assert self.extract_codec_on_the_fly
            agent_signal = audio_batch['answer_audio']
            agent_signal_length = audio_batch['answer_audio_lens']

            if self.perception.cfg.preprocessor.sample_rate != codec_sample_rate:
                user_signal, user_signal_length = resample(
                    user_signal,
                    user_signal_length,
                    self.perception.cfg.preprocessor.sample_rate,
                    codec_sample_rate,
                )

            new_user_signal = []
            new_agent_signal = []
            new_user_signal_length = []
            new_agent_signal_length = []
            silence_value = 0
            shift_text_channel_len = []
            agent_bos_eos_step = []
            for user, agent, user_len, agent_len in zip(
                user_signal, agent_signal, user_signal_length, agent_signal_length
            ):
                user = user[:user_len]
                agent = agent[:agent_len]
                # user, silence, agent, silence -> user, bos, agent, eos
                # TODO: above design means that in real/synthetic data, we need to mark bos and eos timestamp of agent responses
                silence_piece = torch.full([silence], silence_value).cuda()
                new_user_signal.append(
                    torch.cat([user, silence_piece, torch.ones_like(agent) * silence_value, silence_piece], dim=0)
                )
                new_agent_signal.append(
                    torch.cat([torch.ones_like(user) * silence_value, silence_piece, agent, silence_piece], dim=0)
                )
                duplex_len = user_len + silence + agent_len + silence
                # make bos step -1 to be safe for silence+speech boundary
                agent_bos_eos_step.append(
                    [self.get_step_from_audio_len(user_len + silence) - 1, self.get_step_from_audio_len(duplex_len)]
                )
                new_user_signal_length.append(duplex_len)
                new_agent_signal_length.append(duplex_len)
            new_user_signal = pad_sequence(new_user_signal, batch_first=True)
            new_agent_signal = pad_sequence(new_agent_signal, batch_first=True)
            new_user_signal_length = torch.Tensor(new_user_signal_length).long().cuda()
            new_agent_signal_length = torch.Tensor(new_agent_signal_length).long().cuda()
            if self.perception.cfg.preprocessor.sample_rate != codec_sample_rate:
                new_user_signal, new_user_signal_length = resample(
                    new_user_signal,
                    new_user_signal_length,
                    codec_sample_rate,
                    self.perception.cfg.preprocessor.sample_rate,
                )
        else:  # real duplex data read from dataloader
            new_user_signal = audio_batch['audio_signal']
            new_user_signal_length = audio_batch['audio_signal_length']
            new_agent_signal = audio_batch['answer_audio']
            new_agent_signal_length = audio_batch['answer_audio_lens']
            loss_mask = None
            duplex_method = self.cfg.duplex_method
            assert duplex_method == "from_duplex"

        # [b, t, c]
        encoded, encoded_len = self.perception(
            input_signal=new_user_signal,
            input_signal_length=new_user_signal_length,
            processed_signal=None,
            processed_signal_length=None,
        )

        answer_codecs, answer_codecs_lens = self._get_codec_embeddings(
            new_agent_signal, new_agent_signal_length
        )  # list, list

        answer_codecs_lens = torch.Tensor(answer_codecs_lens).long().cuda()
        assert all(torch.isclose(answer_codecs_lens, encoded_len, atol=3))
        encoded_len = answer_codecs_lens
        if 'answer_features_lens' in audio_batch:
            assert 'target_texts_merge' not in audio_batch
            prev_answer_features_lens = (
                torch.ceil(
                    agent_signal_length / self.codec_model_downsampling_factor / decoder_reduction_factor
                ).long()
                + 1
            )  # bos
            assert all(prev_answer_features_lens == audio_batch['answer_features_lens'])
            shift_text_channel_len = answer_codecs_lens - prev_answer_features_lens - 2  # 2 is for bos and eos

        new_loss_mask = []
        all_channels = []
        for i, answer_codec in enumerate(answer_codecs):
            if 'target_texts_merge' in audio_batch:
                text_channel = audio_batch['target_texts_merge'][i]
                sliced_text_channel = text_channel[: answer_codec.shape[0]].unsqueeze(-1)
                answer_codec = torch.where(
                    sliced_text_channel == self.tokenizer.bos_id, self.cfg.data.train_ds.speech_bos_id, answer_codec
                )
                answer_codec = torch.where(
                    sliced_text_channel == self.tokenizer.eos_id, self.cfg.data.train_ds.speech_eos_id, answer_codec
                )
                if getattr(self.cfg, 'predict_source_text', False):
                    # Also use source_text
                    source_text_channel = audio_batch['source_texts_merge'][i]
                    sliced_source_text_channel = source_text_channel[: answer_codec.shape[0]].unsqueeze(-1)
            else:
                # this branch is not used anymore
                # mask bos and eos following timestamp or synthetic data mark
                answer_codec[agent_bos_eos_step[i][0]] = self.cfg.data.train_ds.speech_bos_id
                answer_codec[agent_bos_eos_step[i][1]] = self.cfg.data.train_ds.speech_eos_id
                pad_id = self.tokenizer.pad_id if self.tokenizer.pad_id > 0 else self.tokenizer.unk_id
                base_length = -1 + context_lengths[i]
                text_channel = torch.cat(
                    [
                        torch.full([shift_text_channel_len[i], 1], pad_id).cuda(),
                        torch.full([1, 1], self.tokenizer.bos_id).cuda(),
                        labels[i, base_length:, :1],
                    ],
                    dim=0,
                )
                sliced_text_channel = text_channel[: answer_codec.shape[0]]

            if getattr(self.cfg, 'predict_source_text', False):
                # TODO(kevinhu): Add delay to better predict user text.
                # Predict user text when the agent turn starts.
                all_channels.append(torch.cat([sliced_text_channel, answer_codec, sliced_source_text_channel], dim=-1))
            else:
                if getattr(self.cfg, 'speech_delay', False):
                    # TODO(kevinhu): Implement cascaded delays across all channels.
                    text_len, text_vocab = sliced_text_channel.shape
                    speech_len, speech_vocab = answer_codec.shape
                    assert text_len == speech_len
                    speech_pad_id = self.cfg.data.train_ds.speech_unk_id
                    text_pad_id = self.tokenizer.eos_id
                    answer_codec_padded = torch.full(
                        (self.cfg.speech_delay, speech_vocab), speech_pad_id, device=answer_codec.device
                    )
                    answer_codec_shifted = torch.cat([answer_codec_padded, answer_codec], dim=0)[:speech_len, :]
                    sliced_text_channel_padded = torch.full(
                        (self.cfg.speech_delay, text_vocab), text_pad_id, device=sliced_text_channel.device
                    )
                    sliced_text_channel_extended = torch.cat([sliced_text_channel, sliced_text_channel_padded], dim=0)[
                        :speech_len, :
                    ]
                    combined_channels = torch.cat([sliced_text_channel_extended, answer_codec_shifted], dim=-1)
                    all_channels.append(combined_channels)
                else:
                    # checked text_channel, loss_mask;  checked injecting bos and eos properly to control turn taking in inference
                    all_channels.append(torch.cat([sliced_text_channel, answer_codec], dim=-1))

            if 'target_texts_merge' not in audio_batch and loss_mask is not None:
                cur_loss_mask = torch.cat(
                    [torch.zeros([shift_text_channel_len[i], loss_mask.shape[-1]]).cuda(), loss_mask[i, base_length:]],
                    dim=0,
                )
                new_loss_mask.append(cur_loss_mask[: answer_codec.shape[0]])
        all_channels = pad_sequence(all_channels, batch_first=True)
        input_ids = all_channels[:, :-1]
        encoded = encoded[:, : input_ids.shape[1]]
        encoder_length = encoded_len - 1
        labels = all_channels[:, 1:]
        # assert labels.shape[1] == encoded.shape[1]
        labels = labels[:, : encoded.shape[1]]
        input_ids = input_ids[:, : encoded.shape[1]]
        if 'target_texts_merge' in audio_batch:
            loss_mask = torch.ones_like(labels)
            assert self.cfg.get(
                'duplex_loss_on_all_steps', False
            ), "only support duplex_loss_on_all_steps in real duplex data read from dataloader"
        elif loss_mask is not None:
            loss_mask = pad_sequence(new_loss_mask, batch_first=True)
            assert loss_mask.shape == labels.shape
            if self.cfg.get('duplex_loss_on_all_steps', False):
                loss_mask = torch.ones_like(labels)  # include loss on silence too
        # lookup input_ids
        if self.cfg.get('megatron_amp_O2', False):
            base_module = self.model.module
        else:
            base_module = self.model
        lm_embedding = (
            base_module.language_model.embedding if hasattr(base_module, 'language_model') else base_module.embedding
        )
        input_embeds = lm_embedding.word_embeddings(input_ids)
        # merge with encoded
        encoder_input = input_embeds + encoded * self.cfg.get("duplex_user_channel_weight", 0.3)

        scale_loss_mask_by = self.cfg.get("scale_loss_mask_by", None)
        if scale_loss_mask_by == 'bos_eos':
            for i, answer_codec in enumerate(answer_codecs):
                if 'target_texts_merge' in audio_batch:
                    text_channel = audio_batch['target_texts_merge'][i]
                    sliced_text_channel = text_channel[: loss_mask.shape[1]].unsqueeze(-1)
                    loss_mask = torch.where(sliced_text_channel == self.tokenizer.bos_id, 2.0, loss_mask)
                    loss_mask = torch.where(sliced_text_channel == self.tokenizer.eos_id, 2.0, loss_mask)
                else:
                    raise ValueError("scale_loss_mask_by=bos_eos is only supported for target_texts_merge")
        elif scale_loss_mask_by == 'non_sil':
            for i, answer_codec in enumerate(answer_codecs):
                if 'target_texts_merge' in audio_batch:
                    text_channel = audio_batch['target_texts_merge'][i]
                    sliced_text_channel = text_channel[: loss_mask.shape[1]].unsqueeze(-1)
                    loss_mask = torch.where(labels[:, :, :] != labels[:, :1, :], 2.0, loss_mask)
                else:
                    raise ValueError("scale_loss_mask_by=bos_eos is only supported for target_texts_merge")
        elif scale_loss_mask_by == None:
            pass
        else:
            raise ValueError(f"Unknown scale_loss_mask_by: {scale_loss_mask_by}")
        limit_max_seq_length = self.cfg.get("limit_max_seq_length", None)
        if limit_max_seq_length is not None and limit_max_seq_length < labels.shape[1] and self.training:
            import random

            start = random.randint(0, labels.shape[1] - limit_max_seq_length - 1)
            encoder_input = encoder_input[:, start : start + limit_max_seq_length]
            labels = labels[:, start : start + limit_max_seq_length]
            loss_mask = loss_mask[:, start : start + limit_max_seq_length]
            encoder_length = torch.minimum(encoder_length, torch.tensor(limit_max_seq_length).long().cuda())
            encoded = encoded[:, start : start + limit_max_seq_length]

        encoder_input, labels, loss_mask, encoded, encoder_length = self.inject_speaker_prompt(
            audio_batch, encoder_input, labels, loss_mask, encoded, encoder_length
        )

        attention_mask = self._create_attention_mask(encoder_input)
        if not hasattr(lm_embedding, 'transpose_batch_sequence') or lm_embedding.transpose_batch_sequence:
            encoder_input = encoder_input.transpose(0, 1).contiguous()

        return encoder_input, attention_mask, labels, loss_mask, (encoded, encoder_length)

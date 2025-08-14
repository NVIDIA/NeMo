
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

from abc import ABC
from dataclasses import dataclass
import torch

from nemo.collections.asr.models.aed_multitask_models import lens_to_mask
from nemo.collections.asr.models.aed_multitask_models import EncDecMultiTaskModel
from nemo.collections.asr.parts.submodules.multitask_decoding import AEDStreamingDecodingConfig
from nemo.utils import logging


@dataclass
class AEDStreamingState:
    decoder_input_ids: torch.Tensor = None  # tokens ids of initial canary prompt
    tgt: torch.Tensor = None  # buffer with deocoded tokens ids
    decoding_step: int = -1  # current decoding step
    decoder_mems_list: list = None  # decoder caches, helps to reduce the memory usage
    is_last_chunk_batch: torch.Tensor = False  # whether the current chunk is the last speech chunk in the audio
    max_generation_length: int = 512  # maximum number of tokens to be generated for each sample (can be bigger for long audio)
    max_tokens_per_one_second: int = 10 # maximum number of tokens to be generated per one second of audio
    max_tokens_per_alignatt_step: int = None # maximum number of tokens to be generated for each step of alignatt decoding policy
    use_avgpool_for_alignatt: bool = True # use avgpooling for alignatt decoding policy
    tokens_frame_alignment: torch.Tensor = None  # frame alignment of the predicted tokens (used for LAAL calculation in alignatt)
    prev_encoder_shift: int = 0  # previous encoder shift (used for LAAL calculation in alignatt)
    device: torch.device = None


class GreedyBatchedStreamingAEDComputer(ABC):
    """
    Batched streaming AED decoding with support for waitk and alignatt decoding policies.
    """

    def __init__(
        self,
        asr_model: EncDecMultiTaskModel,
        frame_chunk_size: int,
        decoding_cfg: AEDStreamingDecodingConfig,
        debug_mode: bool = False,
    ):
        """
        Init method.
        Args:
            asr_model: isntace of ASR model (Canary)
            frame_chunk_size: size of the frame chunk
            decoding_cfg: decoding configuration
            debug_mode: debug mode
        """
        super().__init__()

        self.asr_model = asr_model
        self.frame_chunk_size = frame_chunk_size
        self.decoding_cfg = decoding_cfg
        self.state = AEDStreamingState()
        self.debug_mode = debug_mode

    def __call__(
        self,
        encoder_output: torch.Tensor,
        encoder_output_len: torch.Tensor,
        prev_batched_state: AEDStreamingState,
    ) -> AEDStreamingState:

        self.state = prev_batched_state

        # prepare encoder embeddings for the decoding
        # enc_states = encoder_output.permute(0, 2, 1)
        encoded_speech = self.asr_model.encoder_decoder_proj(encoder_output)

        encoder_input_mask = lens_to_mask(encoder_output_len, encoded_speech.shape[1]).to(
            encoded_speech.dtype
        )

        # initiall waitk lagging. Applicable for waitk and alignatt decoding policies. Control the start of the decoding process.
        # if encoded_speech.size(-2) // self.frame_chunk_size < self.decoding_cfg.waitk_lagging and torch.any(
        #     torch.logical_not(self.state.is_last_chunk_batch)
        # ):
        if encoder_output_len.max() // self.frame_chunk_size < self.decoding_cfg.waitk_lagging and torch.any(
            torch.logical_not(self.state.is_last_chunk_batch)
        ):
            # need to wait for more speech
            if self.debug_mode:
                logging.info(f"!!! need to accumulate more speech to start the decoding process !!!")
                logging.info(f"[encoder_output_len]: {encoder_output_len}")

        # wait-k streaming decoding policy
        elif self.decoding_cfg.streaming_policy == "waitk":
            if self.state.decoding_step < 0:
                # first decoding step
                tgt, batch_size, _ = self.asr_model.decoding.decoding.greedy_search._prepare_for_search(
                    self.state.decoder_input_ids,
                    encoded_speech,
                )
                input_ids = tgt
            else:
                input_ids = self.state.tgt[
                    self.state.batch_idxs, self.state.current_context_lengths - 1
                ].unsqueeze(-1)

            active_samples_inner_loop = (
                torch.ones(self.state.batch_size, dtype=torch.bool, device=self.state.device) * self.state.active_samples
            )
            decoder_mems_list = self.state.decoder_mems_list

            # define start and max generation lengths
            start_from = self.state.decoding_step + 1
            if torch.any(torch.logical_not(self.state.is_last_chunk_batch)):
                # predict only one token per speech chunk if not the last one
                max_generation_length = start_from + 1
            else:
                max_generation_length = self.decoding_cfg.max_generation_length

            # inner deocding loop (with same speech chunk)
            for i in range(start_from, max_generation_length):

                if not decoder_mems_list:
                    positional_indexes = torch.zeros_like(self.state.current_context_lengths)
                else:
                    positional_indexes = self.state.current_context_lengths - 1

                logits, decoder_mems_list, xatt_scores_list = (
                    self.asr_model.decoding.decoding.greedy_search._one_step_forward(
                        input_ids,
                        encoded_speech,
                        encoder_input_mask,
                        decoder_mems_list,
                        positional_indexes,
                        return_scores=False,
                        return_xatt_scores=True,
                    )
                )
                next_tokens = torch.argmax(logits[:, -1], dim=-1)
                text_tokens = self.asr_model.tokenizer.ids_to_tokens(next_tokens.tolist())

                # compute eos tokens mask
                is_eos_tokens = next_tokens == self.asr_model.tokenizer.eos
                # rearange active samples (inner loop) depends on eos prediction
                active_samples_inner_loop *= torch.logical_not(is_eos_tokens)
                # disable samples (upper loop) with eos and end of speech
                eos_and_end_speech_mask = is_eos_tokens * self.state.is_last_chunk_batch
                self.state.active_samples = self.state.active_samples * torch.logical_not(
                    eos_and_end_speech_mask
                )

                if self.debug_mode:
                    logging.info(f"-------------" * 5)
                    logging.info(f"decoding step (i)        : {i}")
                    logging.info(f"start_from               : {start_from}")
                    logging.info(f"max_generation_length    : {max_generation_length}")
                    logging.info(f"[encoder_output_len]     : {encoder_output_len}")
                    logging.info(f"[is_last_chunk_batch]    : {self.state.is_last_chunk_batch}")
                    logging.info(f"[active_samples]         : {self.state.active_samples}")
                    logging.info(f"[current_context_lengths]: {self.state.current_context_lengths}")
                    logging.info(f"[predicted token]        : {text_tokens}")
                    logging.info(f"[predicted token id]     : {next_tokens}")

                if not torch.any(active_samples_inner_loop):
                    if self.debug_mode:
                        logging.info(f"!#! no active samples in inner loop, do next upper step !#!")
                    break

                # write predicted tokens to the tgt tensor
                torch.where(
                    active_samples_inner_loop, next_tokens, self.state.eos_tokens, out=next_tokens
                )
                self.state.tgt[self.state.batch_idxs, self.state.current_context_lengths] = next_tokens

                # canary_data.decoding_step = i
                self.state.decoding_step += input_ids.size(-1)
                # input_ids = next_tokens.unsqueeze(-1)
                # input_ids = canary_data.tgt[canary_data.batch_idxs, canary_data.current_context_lengths].unsqueeze(-1)

                # check for hallucinations
                # TODO add more consequtive tokens? Now we are checking only 3 same tokens
                hallucination_mask = torch.logical_and(
                    self.state.tgt[self.state.batch_idxs, self.state.current_context_lengths]
                    == self.state.tgt[self.state.batch_idxs, self.state.current_context_lengths - 1],
                    self.state.tgt[self.state.batch_idxs, self.state.current_context_lengths - 1]
                    == self.state.tgt[self.state.batch_idxs, self.state.current_context_lengths - 2],
                )
                if torch.any(hallucination_mask):
                    logging.info(f"!!! hallucination detected !!!")
                    self.state.active_samples *= torch.logical_not(hallucination_mask)
                    active_samples_inner_loop *= torch.logical_not(hallucination_mask)

                self.state.current_context_lengths += active_samples_inner_loop
                input_ids = self.state.tgt[
                    self.state.batch_idxs, self.state.current_context_lengths - 1
                ].unsqueeze(-1)

                # disable samples with maximum context length
                samples_with_max_context_length = (
                    self.state.current_context_lengths == self.decoding_cfg.max_generation_length - 1
                )
                if torch.any(samples_with_max_context_length * self.state.active_samples):
                    logging.info(f"!!! maximum context length reached !!!")
                    self.state.active_samples *= torch.logical_not(samples_with_max_context_length)
                    active_samples_inner_loop *= torch.logical_not(samples_with_max_context_length)

                # zero out decoder_mems_list for non active samples
                if torch.any(torch.logical_not(active_samples_inner_loop)):
                    for j in range(len(decoder_mems_list)):
                        decoder_mems_list[j][:, -1] *= active_samples_inner_loop.unsqueeze(-1)
                self.state.decoder_mems_list = decoder_mems_list

                if self.debug_mode:
                    # import ipdb; ipdb.set_trace()
                    pass

        # alignatt streaming decoding policy
        elif self.decoding_cfg.streaming_policy == "alignatt":
            if self.state.decoding_step < 0:
                # first decoding step
                tgt, batch_size, _ = self.asr_model.decoding.decoding.greedy_search._prepare_for_search(
                    self.state.decoder_input_ids,
                    encoded_speech,
                )
                input_ids = tgt
                start_from = 0
            else:
                input_ids = self.state.tgt[
                    self.state.batch_idxs, self.state.current_context_lengths - 1
                ].unsqueeze(-1)
                start_from = torch.min(self.state.current_context_lengths).item() - 1

            decoder_mems_list = self.state.decoder_mems_list
            self.state.steps_per_inner_loop = torch.zeros(self.state.batch_size, dtype=torch.long, device=self.state.device)
            self.state.active_samples_inner_loop = (
                torch.ones(self.state.batch_size, dtype=torch.bool, device=self.state.device) * self.state.active_samples
            )

            for i in range(start_from, self.state.max_generation_length):
                # prepare positional indexes offset for attention decoder
                if not decoder_mems_list:
                    positional_indexes = torch.zeros_like(self.state.current_context_lengths)
                else:
                    positional_indexes = self.state.current_context_lengths - 1

                logits, decoder_mems_list, xatt_scores_list = (
                    self.asr_model.decoding.decoding.greedy_search._one_step_forward(
                        input_ids,
                        encoded_speech,
                        encoder_input_mask,
                        decoder_mems_list,
                        positional_indexes,
                        return_scores=False,
                        return_xatt_scores=True,
                    )
                )

                next_tokens = torch.argmax(logits[:, -1], dim=-1)
                text_token = self.asr_model.tokenizer.ids_to_tokens(next_tokens.tolist())

                # compute the most attended encoder token
                xatt_scores = xatt_scores_list[self.decoding_cfg.xatt_scores_layer]
                xatt_scores = torch.mean(xatt_scores, 1)
                if i == 0 and xatt_scores.shape[-1] <= self.decoding_cfg.exclude_sink_frames:
                    exclude_sink_frames = xatt_scores.shape[-1] // 2
                else:
                    exclude_sink_frames = self.decoding_cfg.exclude_sink_frames if self.state.prev_encoder_shift == 0 else 0
                most_attended_idxs = (
                    torch.argmax(xatt_scores[:, :, exclude_sink_frames:], dim=-1) + exclude_sink_frames
                )

                if self.decoding_cfg.use_avgpool_for_alignatt:
                    average_pooling_xatt_scores = self.state.avgpool2d(xatt_scores[:, :, exclude_sink_frames:])
                    most_attended_idxs_avgpool = (
                        torch.argmax(average_pooling_xatt_scores, dim=-1) + exclude_sink_frames
                    )
                    most_attended_idxs = most_attended_idxs_avgpool

                # select the last attended token for each sample
                if most_attended_idxs.size(-1) > 1:
                    most_attended_idxs = most_attended_idxs[:, -1]
                else:
                    most_attended_idxs = most_attended_idxs.squeeze(-1)

                # aligatt condition (True -- continue decoding, False -- wait for more speech)
                # TODO: consider only active samples
                alignatt_condition = (
                    encoder_output_len - (most_attended_idxs + 1) >= self.decoding_cfg.alignatt_thr
                )

                # alignatt condition is always True for the last speech chunk
                alignatt_condition += self.state.is_last_chunk_batch

                # applay alignatt condition for inner loop
                self.state.active_samples_inner_loop *= alignatt_condition

                if self.debug_mode:
                    logging.info(f"========================" * 5)
                    logging.info(f"self.state.decoding_step   : {self.state.decoding_step}")
                    logging.info(f"decoding step i            : {i}")
                    logging.info(f"[encoded_speech.shape]     : {encoded_speech.shape}")
                    logging.info(f"[encoder_output_len]       : {encoder_output_len}")
                    logging.info(f"[positional_indexes]       : {positional_indexes}")
                    logging.info(f"[most_attended_idxs]       : {most_attended_idxs}")
                    logging.info(f"[is_last_chunk_batch]      : {self.state.is_last_chunk_batch}")
                    logging.info(f"[active_samples]           : {self.state.active_samples}")
                    logging.info(f"[active_samples_inner_loop]: {self.state.active_samples_inner_loop}")
                    logging.info(f"[current_context_lengths]  : {self.state.current_context_lengths}")
                    logging.info(f"[predicted tokens]         : {text_token}")
                    logging.info(f"[predicted tokens id]      : {next_tokens}")
                    # import ipdb; ipdb.set_trace()

                # increase speech chunk if no active samples in the inner loop
                if not torch.any(self.state.active_samples_inner_loop):
                    if self.debug_mode:
                        logging.info(f"!#! no active samples in inner loop, do next upper step !#!")
                    break

                # compute eos tokens mask
                # TODO add a case of "." + EOS prediction. It is the important case for AST tasl with PC support
                is_eos_tokens = next_tokens == self.asr_model.tokenizer.eos
                # rearange active samples (inner loop) depends on eos prediction
                self.state.active_samples_inner_loop *= torch.logical_not(is_eos_tokens)
                # disable samples (upper loop) with eos and end of speech
                eos_and_end_speech_mask = is_eos_tokens * self.state.is_last_chunk_batch
                self.state.active_samples *= torch.logical_not(eos_and_end_speech_mask)

                if not torch.any(self.state.active_samples_inner_loop):
                    if self.debug_mode:
                        logging.info(f"!#! no active samples in inner loop, do next upper step !#!")
                        logging.info(f"[active_samples]           : {self.state.active_samples}")
                        logging.info(f"[active_samples_inner_loop]: {self.state.active_samples_inner_loop}")
                        logging.info(f"________________________")
                    break

                # write predicted tokens to the tgt tensor
                torch.where(
                    self.state.active_samples_inner_loop, next_tokens, self.state.eos_tokens, out=next_tokens
                )
                self.state.tgt[self.state.batch_idxs, self.state.current_context_lengths] = next_tokens

                # update tokens frame alignment based on current encoder step (this alignment is used for LAAL calculation)
                # self.state.tokens_frame_alignment[self.state.batch_idxs, self.state.current_context_lengths] = (
                #     encoded_speech.size(-2) + self.state.prev_encoder_shift # we need to add the real frame position in the audio signal
                # )
                self.state.tokens_frame_alignment[self.state.batch_idxs, self.state.current_context_lengths] = (
                    encoder_output_len + self.state.prev_encoder_shift # we need to add the real frame position in the audio signal
                )

                self.state.decoding_step += input_ids.size(-1)

                # # check for hallucinations
                # # TODO add more consequtive tokens? Now we are checking only 3 same tokens
                if self.decoding_cfg.hallucinations_detector:
                    hallucination_mask = self.detect_hallucinations(self.state.tgt, self.state.batch_idxs, self.state.current_context_lengths)
                    if torch.any(hallucination_mask):
                        logging.info(f"!!! hallucination detected !!!")
                        self.state.active_samples *= torch.logical_not(hallucination_mask)
                        self.state.active_samples_inner_loop *= torch.logical_not(hallucination_mask)

                # disable samples with maximum context length
                samples_with_max_context_length = (
                    self.state.current_context_lengths == self.state.max_generation_length - 1
                )
                if torch.any(samples_with_max_context_length * self.state.active_samples):
                    logging.info(f"!!! maximum context length reached !!!")
                    self.state.active_samples *= torch.logical_not(samples_with_max_context_length)
                    self.state.active_samples_inner_loop *= torch.logical_not(samples_with_max_context_length)

                # zero out decoder_mems_list for non active samples
                # TODO batched decoding works wrong if first token was EOS for one of the samples
                if torch.any(torch.logical_not(self.state.active_samples_inner_loop)):
                    for j in range(len(decoder_mems_list)):
                        decoder_mems_list[j][:, -1] *= self.state.active_samples_inner_loop.unsqueeze(-1)

                self.state.decoder_mems_list = decoder_mems_list
                self.state.current_context_lengths += self.state.active_samples_inner_loop
                # TODO model does not predicts any real tokens in the case of first EOS prediction
                input_ids = self.state.tgt[
                    self.state.batch_idxs, self.state.current_context_lengths - 1
                ].unsqueeze(-1)

                # # limit number of steps per inner loop if not end of speech
                # if self.state.max_tokens_per_alignatt_step is not None:
                #     self.state.steps_per_inner_loop += self.state.active_samples_inner_loop
                #     disable_samples_mask = self.state.steps_per_inner_loop >= self.state.max_tokens_per_alignatt_step
                #     disable_samples_mask *= torch.logical_not(self.state.is_last_chunk_batch)
                #     self.state.active_samples_inner_loop *= torch.logical_not(disable_samples_mask)

                if self.debug_mode:
                    logging.info(f"-------------" * 5)
                    logging.info(f"self.state.decoding_step   : {self.state.decoding_step}")
                    logging.info(f"decoding step i            : {i}")
                    logging.info(f"[encoded_speech.shape]     : {encoded_speech.shape}")
                    logging.info(f"[encoder_output_len]       : {encoder_output_len}")
                    logging.info(f"[positional_indexes]       : {positional_indexes}")
                    logging.info(f"[most_attended_idxs]       : {most_attended_idxs}")
                    logging.info(f"[is_last_chunk_batch]      : {self.state.is_last_chunk_batch}")
                    logging.info(f"[active_samples]           : {self.state.active_samples}")
                    logging.info(f"[active_samples_inner_loop]: {self.state.active_samples_inner_loop}")
                    logging.info(f"[current_context_lengths]  : {self.state.current_context_lengths}")
                    logging.info(f"[predicted tokens]         : {text_token}")
                    logging.info(f"[predicted tokens id]: {next_tokens}")

                
                if self.debug_mode:
                    pass
                    # import ipdb; ipdb.set_trace()

                if not torch.any(self.state.active_samples_inner_loop):
                    if self.debug_mode:
                        import ipdb; ipdb.set_trace()
                        logging.info(f"!#! no active samples in inner loop, do next upper step !#!")
                    break

        else:
            raise ValueError("Canary streaming decoding supports only alignatt or waitk decodong policy")
        
        return self.state

    def detect_hallucinations(self, tgt, batch_idxs, current_context_lengths):

        ccl = current_context_lengths
        # pattern 1: four consequtive tokens are the same: "a a a a"
        hallucination_mask_1 = (tgt[batch_idxs, ccl] == tgt[batch_idxs, ccl-1]) * \
            (tgt[batch_idxs, ccl] == tgt[batch_idxs, ccl-2]) * \
            (tgt[batch_idxs, ccl] == tgt[batch_idxs, ccl-3]) * \
            (tgt[batch_idxs, ccl] == tgt[batch_idxs, ccl-4])
        # pattern 2: "a b a b a b"
        hallucination_mask_2 = (tgt[batch_idxs, ccl] == tgt[batch_idxs, ccl-2]) * \
            (tgt[batch_idxs, ccl-1] == tgt[batch_idxs, ccl-3]) * \
            (tgt[batch_idxs, ccl] == tgt[batch_idxs, ccl-4]) * \
            (tgt[batch_idxs, ccl-1] == tgt[batch_idxs, ccl-5])
        # pattern 3: "a b c a b c"
        hallucination_mask_3 = (tgt[batch_idxs, ccl] == tgt[batch_idxs, ccl-3]) * \
            (tgt[batch_idxs, ccl-1] == tgt[batch_idxs, ccl-4]) * \
            (tgt[batch_idxs, ccl-2] == tgt[batch_idxs, ccl-5])

        hallucination_mask = hallucination_mask_1 + hallucination_mask_2 + hallucination_mask_3
        return hallucination_mask


    def compute_laal(
            self,
            delays,
            source_length,
            target_length
        ):
        if delays[0] > source_length:
            return delays[0]
        LAAL = 0
        gamma = max(len(delays), target_length) / source_length
        tau = 0
        for t_minus_1, d in enumerate(delays):
            LAAL += d - t_minus_1 / gamma
            tau = t_minus_1 + 1
            if d >= source_length:
                break
        LAAL /= tau
        return LAAL


    def compute_alignatt_lagging(
            self,
            records,
            predicted_token_ids,
            tokens_frame_alignment,
            context_encoder_frames,
            audio_encoder_fs,
            BOW_PREFIX="\u2581"
        ):
        # import ipdb; ipdb.set_trace()
        tokens_idx_shift = self.state.decoder_input_ids.size(-1)
        target_length_word = [len(item['text'].split()) for item in records]
        audio_signal_lengths = [float(item['duration'])*1000 for item in records]
        # import ipdb; ipdb.set_trace()
        tokenizer_vocab = self.asr_model.tokenizer.vocab
        eos_token = tokenizer_vocab[self.asr_model.tokenizer.eos_id]
        laal_list = []
        for i, tokens in enumerate(predicted_token_ids):
            if len(tokens) == 0:
                laal_list.append(5000)
                continue
            audio_signal_length = audio_signal_lengths[i]
            # obtain lagging for alignatt
            lagging = []
            for j, cur_t in enumerate(tokens):
                pred_idx = tokens_frame_alignment[i][tokens_idx_shift + j] + context_encoder_frames.right # TODO: check right_context
                cur_t = tokenizer_vocab[cur_t.item()]
                if (cur_t.startswith(BOW_PREFIX) and cur_t != BOW_PREFIX) or cur_t == eos_token:  # word boundary
                    lagging.append(pred_idx * audio_encoder_fs)
                if cur_t == eos_token:
                    break
            if len(lagging) == 0:
                lagging.append(0)
            laal = self.compute_laal(lagging, audio_signal_length, target_length_word[i])
            if torch.is_tensor(laal):
                laal_list.append(laal.item())
            else:
                laal_list.append(laal)
        return laal_list


    def compute_waitk_lagging(
            self,
            records,
            predicted_token_ids,
            context_encoder_frames,
            audio_encoder_fs,
            BOW_PREFIX="\u2581"
        ):
        waitk_lagging = self.decoding_cfg.waitk_lagging
        pre_decision_ratio = context_encoder_frames.chunk
        target_length_word = [len(item['text'].split()) for item in records]
        audio_signal_lengths = [float(item['duration'])*1000 for item in records]
        tokenizer_vocab = self.asr_model.tokenizer.vocab
        laal_list = []
        for i, tokens in enumerate(predicted_token_ids):
            lagging = []
            audio_signal_length = audio_signal_lengths[i]
            for j, cur_t in enumerate(tokens):
                cur_src_len = (j + waitk_lagging) * pre_decision_ratio + context_encoder_frames.right
                cur_src_len *= audio_encoder_fs # to ms
                cur_src_len = min(audio_signal_length, cur_src_len)
                spm = tokenizer_vocab[cur_t.item()]
                # reach word boundary
                if (
                    spm.startswith(BOW_PREFIX) and spm != BOW_PREFIX
                ) or cur_t == self.asr_model.tokenizer.eos_id:  # word boundary
                    lagging.append(cur_src_len)
                if cur_t == self.asr_model.tokenizer.eos_id:
                    break
            if len(lagging) == 0:
                lagging.append(0)
            laal = self.compute_laal(lagging, audio_signal_length, target_length_word[i])
            laal_list.append(laal)
        return laal_list

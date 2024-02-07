# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import itertools
import json
import os
from typing import Any, List
import time

import editdistance
import numpy as np
import soundfile as sf
import torch
from encodec import EncodecModel
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import open_dict
from pytorch_lightning.trainer.trainer import Trainer

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.nlp.data.language_modeling.megatron.t5_speechlm_dataset import (
    T5SpeechLMDataset,
    Lang,
)
from nemo.collections.nlp.data.language_modeling.megatron.t5_speechlm_tarred_dataset import T5SpeechLMTarredDataset
from nemo.collections.nlp.models.language_modeling.megatron_base_prompt_learning_model import (
    MegatronBasePromptLearningModel,
)
from nemo.collections.nlp.models.language_modeling.megatron_base_speechlm_prompt_model import MegatronBaseSpeechLM
from nemo.collections.nlp.models.language_modeling.megatron_t5_sft_model import MegatronT5SFTModel
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model
from nemo.collections.nlp.modules.common.megatron.token_level_encoder_decoder import MegatronTokenLevelHead
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_iterator_k_split,
    init_method_normal,
)
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.collections.tts.parts.utils.helpers import plot_alignment_to_numpy, plot_encodec_to_numpy
from nemo.utils import AppState, logging
from nemo.collections.tts.losses.aligner_loss import ForwardSumLoss
from nemo.collections.tts.models import AudioCodecModel

try:
    from apex.transformer.pipeline_parallel.utils import (
        _reconfigure_microbatch_calculator,
        get_micro_batch_size,
        get_num_microbatches,
    )

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

try:
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.enums import ModelType
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

try:
    import dac
except:
    logging.warning("DAC not found, only use Encodec")

__all__ = ['MegatronT5SpeechLMModel']


# MegatronBasePromptLearningModel):
class MegatronT5SpeechLMModel(MegatronBaseSpeechLM):
    """
    Model class for prompt-tuning or p-tuning a pretrained Megatron T5 model.

    Prompt Tuning initalizes virtual prompt embeddings directly from a copy of
    certain token embeddings from the the pretrained T5 model's vocabulary
    and directly tunes these embedding weights. The token embeddings used in
    initalization are specified by the user in the config file. The model can
    be prompt-tuned for multiple tasks at once. Virtual prompts are stored in a
    prompt table and can be added or deleted without disrupting virtual prompts
    for other tasks.

    P-tuning initializes an LSTM encoder model that generates virtual prompt
    embeddings for every task. Each task shares the same encoder. After p-tuning
    is compelete, the learned virtual prompts can be saved to the prompt table
    using add_ptuned_prompts_to_prompt_table(). Thus, if a user wants to add a
    new virtual prompt via p-tuning, they do not need to retrain on all previous
    tasks. This gives p-tuning the same task flexiblity as prompt-tuning.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        # torch.autograd.set_detect_anomaly(True)
        super().__init__(cfg, trainer)
        self.model_type = ModelType.encoder_and_decoder
        speech_codebook_size = cfg.data.get('speech_codebook_size', 1024)
        num_speech_codebooks = cfg.data.get('num_speech_codebooks', 8)
        speech_offset = cfg.data.get('speech_offset', 30000)
        speech_head_type = cfg.get('speech_head_type', 'token_level')  # token_level, linear
        codecmodel_type = cfg.get('codecmodel_type', 'encodec')  # encodec, dac

        attn_prior_scaledown_start_step = cfg.get('attn_prior_scaledown_start_step', 10000)
        attn_prior_end_step = cfg.get('attn_prior_end_step', 11000)
        return_all_crossattention_probs = cfg.get('return_all_crossattention_probs', False)
        num_cross_attention_heads = cfg.get('num_cross_attention_heads', 12)
        self.lm_vocab_size = cfg.get('lm_vocab_size', 30000)
        self.context_pattern = cfg.data.get('context_pattern', 'parallel')

        self.speech_offset = speech_offset
        self.speech_codebook_size = speech_codebook_size
        self.num_speech_codebooks = num_speech_codebooks
        self.codecmodel_type = codecmodel_type

        self.frozen_model.enc_dec_model.speech_offset = speech_offset
        self.frozen_model.enc_dec_model.speech_codebook_size = speech_codebook_size
        self.frozen_model.enc_dec_model.num_speech_codebooks = num_speech_codebooks
        self.frozen_model.enc_dec_model.cross_entropy_type = cfg.get('cross_entropy_type', 'regular')
        self.frozen_model.enc_dec_model.seq_pattern = cfg.get('seq_pattern', 'parallel')
        self.frozen_model.enc_dec_model.speech_head_type = speech_head_type

        self.frozen_model.enc_dec_model.attn_prior_scaledown_start_step = attn_prior_scaledown_start_step
        self.frozen_model.enc_dec_model.attn_prior_end_step = attn_prior_end_step
        self.frozen_model.enc_dec_model.return_all_crossattention_probs = return_all_crossattention_probs
        self.frozen_model.enc_dec_model.num_cross_attention_heads = num_cross_attention_heads

        self.alignment_loss_start_step = 0
        self.alignment_loss_end_step = float('inf')
        if cfg.get('use_alignment_loss', False):
            alignment_loss_scale = cfg.get('alignment_loss_scale', 1.0)
            self.frozen_model.enc_dec_model.forward_sum_loss = ForwardSumLoss(loss_scale=alignment_loss_scale)
            self.frozen_model.enc_dec_model.alignment_text_end_offset = cfg.get('alignment_text_end_offset', 0)
            self.frozen_model.enc_dec_model.align_every_n_head = cfg.get('align_every_n_head', 1)
            self.frozen_model.enc_dec_model.alignment_decoder_layerids = cfg.get('alignment_decoder_layerids', list(range(0,12)))
            self.alignment_loss_start_step = cfg.get('alignment_loss_start_step', 0)
            self.alignment_loss_end_step = cfg.get('alignment_loss_end_step', float('inf'))


        # Parallel output is used only for vocab parallel cross entropy.
        self.frozen_model.enc_dec_model.parallel_output = (
            self.frozen_model.enc_dec_model.cross_entropy_type == 'vocab_parallel'
        )
        # Need to explicitly set this since it is already initialiazed
        self.frozen_model.enc_dec_model.tokens_head.parallel_output = self.frozen_model.enc_dec_model.parallel_output

        list_of_speech_heads = []
        list_of_speech_tokens_embeddings = []
        for _ in range(self.num_speech_codebooks-1):
            # init is NOT used since we overwrite the weight below anywas
            _speech_head_embedding = tensor_parallel.VocabParallelEmbedding(
                speech_codebook_size,
                embedding_dim=self.word_embeddings.embedding_dim,
                init_method=lambda x: x.data.fill_(0),
                config=self.model_parallel_config,
            )
            _speech_head_embedding.weight.data.fill_(0)
            _speech_head_embedding.shared = True
            list_of_speech_tokens_embeddings.append(_speech_head_embedding)
            if speech_head_type == 'token_level':
                list_of_speech_heads.append(MegatronTokenLevelHead(_speech_head_embedding.weight.size(0), False))
            elif speech_head_type == 'linear':
                # Linear layer that maps from hidden size to speech codebook size
                hidden_size = self.frozen_model.enc_dec_model.decoder_cfg.hidden_size
                init_method_std = self.frozen_model.enc_dec_model.decoder_cfg.init_method_std
                # Changing to ColumnParallelLinear instead of Linear to support 3b Tensor Parallelism
                _speech_head = tensor_parallel.ColumnParallelLinear(
                    input_size=hidden_size,
                    output_size=speech_codebook_size,
                    bias=True,
                    gather_output=not self.frozen_model.enc_dec_model.parallel_output,
                    init_method=init_method_normal(init_method_std),
                    config=self.model_parallel_config,
                    # use_cpu_initialization=False,
                    # params_dtype=self.frozen_model.enc_dec_model.dtype,
                )
                list_of_speech_heads.append(_speech_head)

        self.frozen_model.enc_dec_model.speech_tokens_heads = torch.nn.ModuleList(list_of_speech_heads)
        self.frozen_model.enc_dec_model.speech_tokens_embeddings = torch.nn.ModuleList(
            list_of_speech_tokens_embeddings
        )

        if speech_head_type == 'token_level':
            self.frozen_model.enc_dec_model.speech_residual_model_1 = SimplestModule(
                self.frozen_model.enc_dec_model.decoder_cfg.hidden_size, speech_offset + speech_codebook_size
            )
            self.frozen_model.enc_dec_model.speech_residual_model_2 = SimplestModule(
                self.frozen_model.enc_dec_model.decoder_cfg.hidden_size, speech_codebook_size
            )

        self.sample_rate = 24000
        if codecmodel_type == 'dac':
            codec_model = dac.DAC.load(cfg.get('codecmodel_path'))
            codec_model.to('cuda')
            self.sample_rate = 44100
        elif codecmodel_type == 'encodec':
            codec_model = EncodecModel.encodec_model_24khz()
            codec_model.set_target_bandwidth(6.0)
            codec_model.cuda()
            codec_model.eval()
            self.sample_rate = 24000
        elif codecmodel_type == 'nemo_codec':
            codec_model = AudioCodecModel.restore_from(cfg.get('codecmodel_path'))
            codec_model.to('cuda')
            codec_model.eval()
            self.sample_rate = 22050
        else:
            raise NotImplementedError()

        self.additional_models = {'codec': codec_model}
        self.train_check_interval  = self.cfg.get('train_check_interval', 500)
        self.plot_alignments_sliced  = self.cfg.get('plot_alignments_sliced', True)
        app_state = AppState()
        self.is_rank_zero = app_state.global_rank == 0
        self.predict_step_outputs = []
        self.phoneme_tokenizer = None

    def decode_wav_from_codec_model(self, codes):
        codec_model = self.additional_models['codec']
        if self.codecmodel_type == 'dac':
            _z = codec_model.quantizer.from_codes(codes.unsqueeze(0))[0]
            wav = codec_model.decoder(_z)[0][0]
        elif self.codecmodel_type == 'encodec':
            wav = codec_model.decode([[codes.unsqueeze(0), None]])[0, 0]
        elif self.codecmodel_type == 'nemo_codec':
            codec_len = torch.Tensor([codes.shape[1]]).long().cuda()
            wav, _ = codec_model.decode(tokens=codes.unsqueeze(0), tokens_len=codec_len)
            wav = wav[0]
        else:
            raise NotImplementedError()
        return wav

    def first_stage_of_pipeline(self):
        if self.frozen_model.enc_dec_model.pre_process and parallel_state.get_pipeline_model_parallel_rank() == 0:
            return True
        return False

    def forward(
        self,
        virtual_tokens,
        context_and_question_tokens,
        enc_mask,
        dec_input,
        dec_mask,
        position_ids,
        taskname_ids,
        labels=None,
        speech_mask=None,
        inference=False,
        inference_step=0,
        cross_attention_prior=None,
        text_limits=None,
        decoder_max_sequence_len=None,
        encoder_max_sequence_len=None,
    ):
        """
        Special forward method for p-tuning/prompt-tuning pretrained
        T5 style models.
        """
        enc_output = None
        if self.first_stage_of_pipeline() and inference_step==0:
            # Get embeddings for text tokens and insert virtual token embeddings
            # import ipdb; ipdb.set_trace()
            input_embeds = self.get_embeddings_and_combine(
                [virtual_tokens, context_and_question_tokens], taskname_ids, inference
            )
            # TODO: This check needs to be revisited with PP support.
            if hasattr(self.frozen_model.enc_dec_model.encoder_embedding, 'position_embeddings'):
                position_embeddings = self.frozen_model.enc_dec_model.encoder_embedding.position_embeddings(
                    position_ids
                )
                encoder_input = input_embeds + position_embeddings
            else:
                encoder_input = input_embeds
        else:
            encoder_input = None
            if inference_step != 0:
                enc_output = context_and_question_tokens

        # If the decoder input starts with <pad> instead of <bos>, which is the case for huggingface T5 models, we don't want to mask the first token.
        # For NeMo-Megatron, the sequence starts with <bos>, which is never masked so we can always set index 0 to be unmasked.
        dec_mask[:, 0] = 1

        if not self.cfg.data.get('use_attention_prior', False):
            cross_attention_prior = None

        # Call forward on T5 model with preprocessed embeddings
        if self.autocast_dtype == torch.float32:
            output, out_logits = self.frozen_model.enc_dec_model(
                enc_input_ids=None,
                enc_attn_mask=enc_mask,
                dec_input_ids=dec_input,
                dec_attn_mask=dec_mask,
                token_type_ids=None,
                labels=labels,
                output_enc_hidden_only=False,
                enc_input=encoder_input,
                speech_mask=speech_mask,
                cross_attention_prior=cross_attention_prior,
                text_limits=text_limits,
                global_step=self.global_step,
                set_inference_key_value_memory=True if inference and inference_step == 0 else False,
                decoder_max_sequence_len=decoder_max_sequence_len,
                encoder_max_sequence_len=encoder_max_sequence_len,
            )
        else:
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                output, out_logits = self.frozen_model.enc_dec_model(
                    enc_input_ids=None,
                    enc_attn_mask=enc_mask,
                    dec_input_ids=dec_input,
                    dec_attn_mask=dec_mask,
                    token_type_ids=None,
                    labels=labels,
                    output_enc_hidden_only=False,
                    enc_input=encoder_input,
                    enc_output=enc_output,
                    speech_mask=speech_mask,
                    cross_attention_prior=cross_attention_prior,
                    text_limits=text_limits,
                    global_step=self.global_step,
                    set_inference_key_value_memory=True if inference and inference_step == 0 else False,
                    decoder_max_sequence_len=decoder_max_sequence_len,
                    encoder_max_sequence_len=encoder_max_sequence_len,
                )

        return output, encoder_input, out_logits

    def load_frozen_model(self, cfg, trainer):
        self.megatron_amp_O2 = cfg.get('megatron_amp_o2', None)
        if self.megatron_amp_O2 == None:
            self.megatron_amp_O2 = cfg.get('megatron_amp_O2', None)
        if self.megatron_amp_O2 == None:
            self.megatron_amp_O2 = False

        # TODO: Fix this once apex patches FusedScaledMaskedSoftmax.
        # This is a workaround for the fact that `masked_softmax_fusion` has issues with certain input sizes that may be present while finetuning.
        t5_cfg = MegatronT5Model.restore_from(cfg.get('language_model_path'), trainer=trainer, return_config=True)
        OmegaConf.set_struct(t5_cfg, True)
        with open_dict(t5_cfg):
            if hasattr(t5_cfg, 'encoder') and hasattr(t5_cfg, 'decoder'):
                t5_cfg.encoder.masked_softmax_fusion = False
                t5_cfg.decoder.masked_softmax_fusion = False
            else:
                t5_cfg.masked_softmax_fusion = False
            t5_cfg.megatron_amp_O2 = self.megatron_amp_O2
            # hack to make the _GLOBAL_NUM_MICROBATCHES_CALCULATOR initialize
            t5_cfg.micro_batch_size = cfg.get('micro_batch_size', 4)
            t5_cfg.global_batch_size = cfg.get('global_batch_size', 4)
            t5_cfg.precision = trainer.precision
            t5_cfg.tokenizer.num_sentinel_tokens = cfg.get('num_sentinel_tokens', 39184 - 29056)
            t5_cfg.seq_length = cfg.data.max_seq_length
            t5_cfg.max_position_embeddings = cfg.data.max_seq_length
            t5_cfg.use_flash_attention = cfg.get('use_flash_attention', False)
            if cfg.get('override_token_model', None):
                t5_cfg.tokenizer.model = cfg['override_token_model']
            if cfg.get('override_tokenizer_vocab_file', None):
                t5_cfg.tokenizer.vocab_file = cfg['override_tokenizer_vocab_file']

        if cfg.get('train_from_scratch', False):
            print("Training from scratch!")
            # Defaults for 220m model
            # To override any of these, add +model.override_<key>=<value> to the config file.
            # Eg. +model.override_hidden_size=1024
            overide_keys = [
                'hidden_size', # 768
                'num_layers', # 12
                'num_attention_heads', # 12
                'hidden_dropout', # 0.1
                'attention_dropout', # 0.1
                'kv_channels' # 64
                'ffn_hidden_size', # 2048
            ]
            # Defaults for 220m model
            for k in overide_keys:
                if cfg.get(f'override_{k}') is not None:
                    t5_cfg[k] = cfg.get(f'override_{k}')

            self.frozen_model = MegatronT5Model(t5_cfg, trainer=trainer)
            num_params = sum(p.numel() for p in self.frozen_model.parameters() if p.requires_grad)
            print(f"Number of parameters: {num_params}")
        else:
            print("Loading from pretrained checkpoint!")
            self.frozen_model = MegatronT5Model.restore_from(
                cfg.get('language_model_path'),
                trainer=trainer,
                override_config_path=t5_cfg,
                save_restore_connector=NLPSaveRestoreConnector(),
            )

        if not cfg.get('english_only_model', False):
            self.frozen_model.tokenizer.update_phone_tokens()

        logging.info(f"self.frozen_model {self.frozen_model}")

    def fwd_bwd_step(self, dataloader_iter, batch_idx, forward_only):
        """
            Dataloader produces a global batch which is turned into a list of microbatches.
            The list of microbatches is then piped through the pipeline using megatron-core fwd/bwd functions.
        """
        # Get seq length of batch
        batch = next(dataloader_iter)
        _, seq_length = batch[0].shape
        if batch[4].dim() > 2:
            _, _, dec_seq_length = batch[4].shape
        else:
            _, dec_seq_length = batch[4].shape
        data_iter = get_iterator_k_split(batch, get_num_microbatches())

        fwd_bwd_function = get_forward_backward_func()

        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(forward_only),
            data_iterator=data_iter,
            model=[self],
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            seq_length=seq_length,
            micro_batch_size=get_micro_batch_size(),
            decoder_seq_length=dec_seq_length,
        )

        # only the last stages of the pipeline return losses
        if losses_reduced_per_micro_batch:
            # average loss across micro batches
            loss_tensors_list = [loss_reduced['avg'] for loss_reduced in losses_reduced_per_micro_batch]
            loss_tensor = torch.concat(loss_tensors_list)
            loss_mean = loss_tensor.mean()
        else:
            # we're not on the last pipeline stage so no losses
            loss_mean = torch.tensor(0.0).cuda()

        return loss_mean

    def convert_tokens_to_range(self, tokens, apply_offset_correction=True, pattern=None):
        # convert tokens to range [0, 1024]
        output_tokens = tokens.clone()
        if apply_offset_correction:
            output_tokens[0] = output_tokens[0] - self.speech_offset
        output_tokens = torch.clamp(output_tokens, min=0, max=1023)
        if pattern is None:
            pattern = self.cfg.get('seq_pattern', 'delay_parallel')
        if pattern == "delay_parallel":
            output_tokens_new = []
            for _c in range(output_tokens.shape[0]):
                si = _c
                ei = _c + output_tokens.shape[1] - self.num_speech_codebooks
                output_tokens_new.append(output_tokens[_c, si:ei])
            output_tokens_new = torch.stack(output_tokens_new)
            output_tokens = output_tokens_new

        return output_tokens

    def get_forward_output_and_loss_func(self, validation_step=False):
        def fwd_output_and_loss_func(dataloader_iter, model):
            batch = next(dataloader_iter)
            batch = [x.cuda(non_blocking=True) for x in batch]
            (
                virtual_tokens,
                context_and_question_tokens,
                enc_mask,
                dec_input,
                dec_input_mask,
                labels,
                loss_mask,
                position_ids,
                taskname_ids,
                speech_mask,
                _,
                cross_attention_prior,
                text_limits,
                _,  # TODO: text limit and lang not in tarred dataset
            ) = batch

            if self.trainer.global_step % self.train_check_interval == 0 and not validation_step and self.is_rank_zero:
                self.frozen_model.enc_dec_model.logging_step = True
            output_tensor, encoder_input, out_logits = model(
                virtual_tokens,
                context_and_question_tokens,
                enc_mask,
                dec_input,
                dec_input_mask,
                position_ids,
                taskname_ids,
                labels=labels,
                speech_mask=speech_mask,
                cross_attention_prior=cross_attention_prior,
                text_limits=text_limits,
                inference=False,
            )
            output_tensor = output_tensor.contiguous()

            alignment_loss = out_logits[3]
            if alignment_loss is not None:
                self.logger.experiment.add_scalar('train_alignment_loss', alignment_loss, self.global_step)

            if self.trainer.global_step % self.train_check_interval == 0 and not validation_step and self.is_rank_zero:
                self.frozen_model.enc_dec_model.logging_step = False
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=False):
                        # Encodec does not work with fp16, so we disable autocast for logging audio
                        if torch.count_nonzero(speech_mask) == 0:
                            text_labels = labels[:, 0, :]  # [B, 8, T] -> [B, T]
                            token_logits = out_logits[0] * 1  # [T, B, V]
                            if self.frozen_model.enc_dec_model.parallel_output:
                                # Gather from tensor parallel region
                                token_logits = tensor_parallel.gather_from_tensor_model_parallel_region(token_logits)
                            token_logits = token_logits.argmax(dim=2)  # [T, B]
                            token_logits = token_logits.t()  # [B, T]
                            score = 0
                            for i in range(text_labels.size()[0]):
                                r = text_labels[i].long()
                                nzm = r != 0
                                r = r.tolist()
                                h = token_logits[i].long() * nzm
                                h = h.tolist()
                                score += editdistance.eval(r, h)
                            score /= text_labels.size()[0]
                            logging.info(f"wer score : {score}")
                            self.logger.experiment.add_scalar('WER', score, self.global_step)
                        else:
                            audio_len = (labels[0][0] != 0).sum().item()
                            labels_to_1024 = self.convert_tokens_to_range(labels[0, :, 0:audio_len])
                            label_wav = self.decode_wav_from_codec_model(labels_to_1024)
                            dec_input_to_1024 = self.convert_tokens_to_range(dec_input[0, :, 0:audio_len])
                            dec_input_wav = self.decode_wav_from_codec_model(dec_input_to_1024)
                            self.logger.experiment.add_audio("train_label_wav", label_wav, self.global_step, self.sample_rate)
                            self.logger.experiment.add_audio("train_dec_input_wav", dec_input_wav, self.global_step, self.sample_rate)

                            input_token_list_all = [
                                context_and_question_tokens[0, 0, i].item()
                                for i in range(context_and_question_tokens.shape[2])
                            ]
                            input_token_list = [
                                (ti, t) for ti, t in enumerate(input_token_list_all) if t != 0 and t < self.speech_offset
                            ]
                            context_end_step = input_token_list[0][0]
                            if context_end_step > self.num_speech_codebooks:
                                _context_tokens = context_and_question_tokens[0][:, :context_end_step]
                                _context_tokens = self.convert_tokens_to_range(_context_tokens, pattern=self.context_pattern)
                                _context_wav = self.decode_wav_from_codec_model(_context_tokens)
                                self.logger.experiment.add_audio("train_context_wav", _context_wav, self.global_step, self.sample_rate)

                            # question_si = (
                            #     input_token_list[0][0] + virtual_tokens.shape[1]
                            # )
                            # question_ei = input_token_list[-1][0] + virtual_tokens.shape[1]
                            question_si = text_limits[0, 0].item() - virtual_tokens.shape[1]
                            question_ei = text_limits[0, 1].item() - virtual_tokens.shape[1]
                            text_si = text_limits[0, 0].item()
                            text_ei = text_limits[0, 1].item()
                            input_text = self.frozen_model.tokenizer.ids_to_text(
                                [v for v in input_token_list_all[question_si:question_ei] if v < self.lm_vocab_size]
                            )
                            self.logger.experiment.add_text("Train Input Text", input_text, self.global_step)

                            input_phoneme_tokens = [
                                v - self.lm_vocab_size for v in input_token_list_all[question_si:question_ei] if v >= self.lm_vocab_size
                            ]

                            if len(input_phoneme_tokens) > 0:
                                phoneme_text = self.phoneme_tokenizer.decode(input_phoneme_tokens)
                                self.logger.experiment.add_text("Train Input Phoneme Text", phoneme_text, self.global_step)

                            token_logits = out_logits[0]
                            speech_logits_list = out_logits[1]

                            # if self.trainer.global_step % 500 == 0:
                            attention_probs_list = out_logits[2]  # list of (BS, 12, out_length, in_length)
                            if attention_probs_list is not None:
                                attention_sliced_list = []
                                for lidx in range(len(attention_probs_list)):
                                    attention_probs = attention_probs_list[lidx]
                                    for _i in range(attention_probs.shape[1]):
                                        # alignment_image = plot_alignment_to_numpy(attention_probs[0, _i, :, :].cpu().float().numpy().T)
                                        # self.logger.experiment.add_image(
                                        #     f"Attention Probs Layer {lidx} Head {_i}", alignment_image, self.global_step, dataformats="HWC",
                                        # )
                                        name = f"Attention Probs Layer {lidx} Head {_i}"
                                        attention_to_plot = attention_probs[0, _i, :audio_len, :text_ei]
                                        if self.plot_alignments_sliced:
                                            attention_to_plot = attention_probs[
                                                0, _i, 0 : audio_len, text_si: text_ei
                                            ]
                                            # 4 to offset "Text to Speech this"
                                            name += " Sliced"
                                        alignment_image = plot_alignment_to_numpy(
                                            attention_to_plot.cpu().float().numpy().T,
                                            phoneme_ver=0 if self.plot_alignments_sliced else 1,
                                            phoneme_seq=None if self.plot_alignments_sliced else [text_si]
                                        )
                                        self.logger.experiment.add_image(
                                            name,
                                            alignment_image,
                                            self.global_step,
                                            dataformats="HWC",
                                        )
                                        attention_sliced_list.append(attention_probs[
                                            0, _i, 0 : audio_len, text_si: text_ei
                                        ])
                                attention_sliced = torch.stack(attention_sliced_list)
                                attention_sliced = torch.mean(attention_sliced, 0)
                                text = None
                                if len(input_text) > 0:
                                    text = self.frozen_model.tokenizer.ids_to_tokens([v for v in input_token_list_all[question_si:question_ei] if v < self.lm_vocab_size])
                                if len(input_phoneme_tokens) > 0:
                                    text = phoneme_text.split("|")
                                alignment_image_sliced = plot_alignment_to_numpy(
                                    attention_sliced.cpu().float().numpy().T, phoneme_seq=text, phoneme_ver=2, vmin=0., phone_offset=0, h_offset=False
                                )
                                self.logger.experiment.add_image(
                                    f"Attention Probs Average Sliced",
                                    alignment_image_sliced,
                                    self.global_step,
                                    dataformats="HWC",
                                )
                            if self.frozen_model.enc_dec_model.parallel_output:
                                # Gather from tensor parallel region
                                token_logits = tensor_parallel.gather_from_tensor_model_parallel_region(token_logits)
                                for _i in range(len(speech_logits_list)):
                                    speech_logits_list[_i] = tensor_parallel.gather_from_tensor_model_parallel_region(
                                        speech_logits_list[_i]
                                    )
                            speech_logits = torch.stack(speech_logits_list, dim=-1)  # (t, b, 1024, 7)
                            token_logits_example = token_logits[:, 0, :] * 1
                            speech_logits_example = speech_logits[:, 0, :, :] * 1
                            first_layer_tokens = token_logits_example.argmax(dim=1) - self.speech_offset
                            other_layer_tokens = []
                            for _i in range(speech_logits_example.shape[2]):
                                other_layer_tokens.append(speech_logits_example[:, :, _i].argmax(dim=1))

                            all_layer_tokens = torch.stack([first_layer_tokens] + other_layer_tokens)  # (8, t)
                            all_layer_tokens = self.convert_tokens_to_range(
                                all_layer_tokens, apply_offset_correction=False
                            )
                            # all_layer_tokens = torch.clip(all_layer_tokens, 0, 1023)
                            predicted_wav = self.decode_wav_from_codec_model(all_layer_tokens)
                            self.logger.experiment.add_audio("train_tf_pred_wav", predicted_wav, self.global_step, self.sample_rate)

            def loss_func(loss_args):
                output_tensor, out_logits, curr_step = loss_args
                alignment_loss = out_logits[3]
                loss = self.frozen_model.loss_func(loss_mask, output_tensor)
                if (alignment_loss is not None) and (curr_step > self.alignment_loss_start_step) and (curr_step < self.alignment_loss_end_step):
                    logging.debug(f"Adding alignment loss. cur:{curr_step} start:{self.alignment_loss_start_step}")
                    loss = loss + alignment_loss
                reduced_loss = average_losses_across_data_parallel_group([loss])
                return loss, {'avg': reduced_loss}

            return [output_tensor, out_logits, self.global_step], loss_func

        return fwd_output_and_loss_func

    def get_forward_output_only_func(self):
        """ Used in inference / predict """
        def fwd_output_only_func(dataloader_iter, model):
            batch = next(dataloader_iter)
            batch = [x.cuda(non_blocking=True) if isinstance(x, torch.Tensor) else x for x in batch]
            (
                decoder_max_sequence_len,
                encoder_max_sequence_len,
                context_and_question_tokens,
                enc_mask,
                dec_input,
                dec_input_mask,
                position_ids,
                taskname_ids,
                speech_mask,
            ) = batch

            output_logits, _, token_and_speech_logits = model(
                context_and_question_tokens,
                context_and_question_tokens,
                enc_mask,
                dec_input,
                dec_input_mask,
                position_ids,
                taskname_ids,
                labels=None,
                speech_mask=speech_mask,
                inference=True,
                inference_step=1,
                decoder_max_sequence_len=decoder_max_sequence_len,
                encoder_max_sequence_len=encoder_max_sequence_len,
            )
            output_tensor = [output_logits, token_and_speech_logits]

            def id_func(output_tensor):
                return 0, {'output_logits': output_tensor[0], 'token_and_speech_logits': output_tensor[1]}

            return output_tensor, id_func

        return fwd_output_only_func

    def backward(self, *args, **kwargs):
        """ LightningModule hook to do backward.
            We want this to do nothing since we run backward in the fwd/bwd functions from megatron-core.
            No need to call it here.
        """
        return

    def optimizer_zero_grad(self, *args, **kwargs):
        """ LightningModule hook to zero grad.
            We want this to do nothing as we are zeroing grads during the training_step.
        """
        return

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.
        When using pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.frozen_model.enc_dec_model.set_input_tensor(input_tensor)

    def on_train_epoch_start(self) -> None:
        gbs = self.cfg.global_batch_size
        mbs = self.cfg.micro_batch_size
        self._reconfigure_batch_sizes(gbs, mbs)
        return super().on_train_epoch_start()

    def on_validation_epoch_start(self) -> None:
        gbs = self.cfg.get('validation_global_batch_size', self.cfg.global_batch_size)
        mbs = self.cfg.get('validation_micro_batch_size', self.cfg.micro_batch_size)
        self._reconfigure_batch_sizes(gbs, mbs)
        return super().on_validation_epoch_start()

    def training_step(self, dataloader_iter, batch_idx):
        self._optimizer.zero_grad()
        batch = next(dataloader_iter)
        loss_mean = self.fwd_bwd_step(itertools.chain([batch]), batch_idx, forward_only=False)
        self.allreduce_gradients()

        ## logging
        # we can only log on one rank if it is rank zero so we broadcast from last rank
        # we can avoid this broadcast by updating the PTL log function to accept specific ranks
        torch.distributed.broadcast(loss_mean, get_last_rank())

        if self.cfg.precision == 16 and hasattr(self.trainer.precision_plugin.scaler, "_scale"):
            loss_scale = self.trainer.precision_plugin.scaler._scale
            if loss_scale is not None:
                self.log('loss_scale', loss_scale, batch_size=1)

        self.log('reduced_train_loss', loss_mean, prog_bar=True, rank_zero_only=True, batch_size=1)
        lr = self._optimizer.param_groups[0]['lr']
        self.log('lr', lr, rank_zero_only=True, batch_size=1)
        # logging.info(f'global_step {self.trainer.global_step}')
        self.log('global_step', self.trainer.global_step, prog_bar=True, rank_zero_only=True, batch_size=1)
        return loss_mean

    def get_predictions(self, input_ids, enc_mask, encoder_input, labels):
        predicted_token_ids, log_probs = self.frozen_model.decode(
            tokens_enc=input_ids,
            enc_mask=enc_mask,
            num_tokens_to_generate=self.decoder_seq_length,
            encoder_input=encoder_input,
            bos_id=self.tokenizer.pad_id
            if self.cfg.data.get('decoder_starts_with_pad', False)
            else self.tokenizer.bos_id,
        )
        # Special ids to text function to handle stripping <eos> and special tokens with sentencepiece tokenizers.
        preds_text = MegatronT5SFTModel.ids_to_text(predicted_token_ids, self.tokenizer)
        labels_text = MegatronT5SFTModel.ids_to_text(labels, self.tokenizer)
        input_text = MegatronT5SFTModel.ids_to_text(input_ids, self.tokenizer)
        return {
            'predicted_token_ids': preds_text,
            'labels': labels_text,
            'enc_inputs': input_text,
        }

    def get_embeddings(self, tokens, taskname_ids, inference=False):
        out = None
        if tokens.dim() > 2:
            for i in range(tokens.size()[1]):  # for 8 channels
                if i == 0:
                    # Embed first layer using word embeddings
                    out = self.embed_input(tokens[:, i, :], taskname_ids, inference)  # (B, T, D)
                else:
                    # Embed other layers using speech embeddings
                    cur = self.frozen_model.enc_dec_model.speech_tokens_embeddings[i - 1](tokens[:, i, :])
                    # do not add embeddings of zero tokens of other channels (except the first channel)
                    non_zero_flag = tokens[:, i, :] != 0  # (B, T)
                    cur = cur * non_zero_flag.unsqueeze(2)
                    out = out + cur
        else:
            out = self.embed_input(tokens, taskname_ids, inference)
        return out

    def get_embeddings_and_combine(self, token_list, taskname_ids, inference):
        embedding_list = []
        for tokens in token_list:
            embedding_list.append(self.get_embeddings(tokens, taskname_ids, inference))
        return torch.cat(embedding_list, dim=1)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # batch = [x.cuda(non_blocking=True) for x in batch]
        (
            virtual_tokens,
            context_and_question_tokens,
            enc_mask,
            dec_input,
            dec_input_mask,
            labels,
            loss_mask,
            position_ids,
            taskname_ids,
            speech_mask,
            _,
            cross_attention_prior,
            text_limits,
            _,
        ) = batch
        # loss_mask (b, t)
        # does not use dataloader_iter due to device placement issues arising from PTL
        mode = self.training
        self.eval()
        gbs = self.cfg.get('validation_global_batch_size', self.cfg.global_batch_size)
        self._reconfigure_and_process_inference_batch(virtual_tokens.size(0), gbs)
        # loss_mean = self.fwd_bwd_step(
        #     itertools.chain([batch]), batch_idx, forward_only=True
        # )  # comment this out and add custom forward function to calculate WER
        # # logging.info (f'loss_mean {loss_mean}')

        if batch_idx == 0 and self.is_rank_zero:
            self.frozen_model.enc_dec_model.logging_step = True

        labels_original = labels.clone()  # (b, 8, t)
        output_loss, _, output_logits = self.forward(
            virtual_tokens,
            context_and_question_tokens,
            enc_mask,
            dec_input,
            dec_input_mask,
            position_ids,
            taskname_ids,
            labels=labels,
            speech_mask=speech_mask,
            cross_attention_prior=cross_attention_prior,
            text_limits=text_limits,
            inference=False,
        )

        if batch_idx == 0 and self.is_rank_zero:
            self.frozen_model.enc_dec_model.logging_step = False
            with torch.cuda.amp.autocast(enabled=False):
                # Encodec does not work with fp16, so we disable autocast for logging audio
                if torch.count_nonzero(speech_mask) == 0:
                    text_labels = labels[:, 0, :]  # [B, 8, T] -> [B, T]
                    token_logits = output_logits[0] * 1  # [T, B, V]
                    if self.frozen_model.enc_dec_model.parallel_output:
                        # Gather from tensor parallel region
                        token_logits = tensor_parallel.gather_from_tensor_model_parallel_region(token_logits)
                    token_logits = token_logits.argmax(dim=2)  # [T, B]
                    token_logits = token_logits.t()  # [B, T]
                    score = 0
                    for i in range(text_labels.size()[0]):
                        r = text_labels[i].long()
                        nzm = r != 0
                        r = r.tolist()
                        h = token_logits[i].long() * nzm
                        h = h.tolist()
                        score += editdistance.eval(r, h)
                    score /= text_labels.size()[0]
                    logging.info(f"wer score : {score}")
                    self.logger.experiment.add_scalar('WER', score, self.global_step)
                else:
                    audio_len = (labels[0][0] != 0).sum().item()
                    labels_to_1024 = self.convert_tokens_to_range(labels[0, :, 0:audio_len])
                    label_wav = self.decode_wav_from_codec_model(labels_to_1024)
                    dec_input_to_1024 = self.convert_tokens_to_range(dec_input[0, :, 0:audio_len])
                    dec_input_wav = self.decode_wav_from_codec_model(dec_input_to_1024)
                    self.logger.experiment.add_audio("val_label_wav", label_wav, self.global_step, self.sample_rate)
                    self.logger.experiment.add_audio("val_dec_input_wav", dec_input_wav, self.global_step, self.sample_rate)

                    input_token_list_all = [
                        context_and_question_tokens[0, 0, i].item()
                        for i in range(context_and_question_tokens.shape[2])
                    ]
                    input_token_list = [
                        (ti, t) for ti, t in enumerate(input_token_list_all) if t != 0 and t < self.speech_offset
                    ]
                    context_end_step = input_token_list[0][0]
                    if context_end_step > self.num_speech_codebooks:
                        _context_tokens = context_and_question_tokens[0][:, :context_end_step]
                        _context_tokens = self.convert_tokens_to_range(_context_tokens, pattern=self.context_pattern)
                        _context_wav = self.decode_wav_from_codec_model(_context_tokens)
                        self.logger.experiment.add_audio("val_context_wav", _context_wav, self.global_step, self.sample_rate)

                    # question_si = (
                    #     input_token_list[0][0] + virtual_tokens.shape[1]
                    # )
                    # question_ei = input_token_list[-1][0] + virtual_tokens.shape[1]
                    question_si = text_limits[0, 0].item() - virtual_tokens.shape[1]
                    question_ei = text_limits[0, 1].item() - virtual_tokens.shape[1]

                    text_si = text_limits[0, 0].item()
                    text_ei = text_limits[0, 1].item()

                    input_text = self.frozen_model.tokenizer.ids_to_text(
                        [v for v in input_token_list_all[question_si:question_ei] if v < self.lm_vocab_size]
                    )
                    self.logger.experiment.add_text("Val Input Text", input_text, self.global_step)

                    input_phoneme_tokens = [
                        v - self.lm_vocab_size for v in input_token_list_all[question_si:question_ei] if v >= self.lm_vocab_size
                    ]
                    if len(input_phoneme_tokens) > 0:
                        phoneme_text = self.phoneme_tokenizer.decode(input_phoneme_tokens)
                        self.logger.experiment.add_text("Val Input Phoneme Text", phoneme_text, self.global_step)

                    token_logits = output_logits[0]
                    speech_logits_list = output_logits[1]

                    # if self.trainer.global_step % 500 == 0:
                    attention_probs_list = output_logits[2]  # list of (BS, 12, out_length, in_length)
                    if attention_probs_list is not None:
                        attention_sliced_list = []
                        for lidx in range(len(attention_probs_list)):
                            attention_probs = attention_probs_list[lidx]
                            for _i in range(attention_probs.shape[1]):
                                # alignment_image = plot_alignment_to_numpy(attention_probs[0, _i, :, :].cpu().float().numpy().T)
                                # self.logger.experiment.add_image(
                                #     f"Attention Probs Layer {lidx} Head {_i}", alignment_image, self.global_step, dataformats="HWC",
                                # )
                                # name = f"Attention Probs Layer {lidx} Head {_i}"
                                # attention_to_plot = attention_probs[0, _i, :audio_len, :question_ei]
                                # if self.plot_alignments_sliced:
                                #     attention_to_plot = attention_probs[
                                #         0, _i, 0 : audio_len, question_si + 4: question_ei
                                #     ]
                                #     # 4 to offset "Text to Speech this"
                                #     name += " Sliced"
                                # alignment_image = plot_alignment_to_numpy(
                                #     attention_to_plot.cpu().float().numpy().T
                                # )
                                # self.logger.experiment.add_image(
                                #     name,
                                #     alignment_image,
                                #     self.global_step,
                                #     dataformats="HWC",
                                #     phoneme_ver=0 if self.plot_alignments_sliced else 1,
                                #     phoneme_seq=None if self.plot_alignments_sliced else [question_si]
                                # )
                                attention_sliced_list.append(attention_probs[
                                    0, _i, 0 : audio_len, text_si : text_ei
                                ])
                        attention_sliced = torch.stack(attention_sliced_list)
                        attention_sliced = torch.mean(attention_sliced, 0)
                        text = None
                        if len(input_text) > 0:
                            text = self.frozen_model.tokenizer.ids_to_tokens([v for v in input_token_list_all[question_si:question_ei] if v < self.lm_vocab_size])
                        if len(input_phoneme_tokens) > 0:
                            text = phoneme_text.split("|")
                        alignment_image_sliced = plot_alignment_to_numpy(
                            attention_sliced.cpu().float().numpy().T, phoneme_seq=text, phoneme_ver=2, vmin=0., phone_offset=0, h_offset=False
                        )
                        self.logger.experiment.add_image(
                            f"Val Attention Probs Average Sliced",
                            alignment_image_sliced,
                            self.global_step,
                            dataformats="HWC",
                        )
                    if self.frozen_model.enc_dec_model.parallel_output:
                        # Gather from tensor parallel region
                        token_logits = tensor_parallel.gather_from_tensor_model_parallel_region(token_logits)
                        for _i in range(len(speech_logits_list)):
                            speech_logits_list[_i] = tensor_parallel.gather_from_tensor_model_parallel_region(
                                speech_logits_list[_i]
                            )
                    speech_logits = torch.stack(speech_logits_list, dim=-1)  # (t, b, 1024, 7)
                    token_logits_example = token_logits[:, 0, :] * 1
                    speech_logits_example = speech_logits[:, 0, :, :] * 1
                    first_layer_tokens = token_logits_example.argmax(dim=1) - self.speech_offset
                    other_layer_tokens = []
                    for _i in range(speech_logits_example.shape[2]):
                        other_layer_tokens.append(speech_logits_example[:, :, _i].argmax(dim=1))

                    all_layer_tokens = torch.stack([first_layer_tokens] + other_layer_tokens)  # (8, t)
                    all_layer_tokens = self.convert_tokens_to_range(
                        all_layer_tokens, apply_offset_correction=False
                    )
                    all_layer_tokens = torch.clip(all_layer_tokens, 0, 1023)
                    predicted_wav = self.decode_wav_from_codec_model(all_layer_tokens)
                    self.logger.experiment.add_audio("val_tf_pred_wav", predicted_wav, self.global_step, self.sample_rate)

        first_layer_logits = output_logits[0]
        speech_logits_list = output_logits[1]

        if self.frozen_model.enc_dec_model.parallel_output:
            # Gather from tensor parallel region
            first_layer_logits = tensor_parallel.gather_from_tensor_model_parallel_region(first_layer_logits)
            if torch.count_nonzero(speech_mask) > 0:
                for _i in range(len(speech_logits_list)):
                    speech_logits_list[_i] = tensor_parallel.gather_from_tensor_model_parallel_region(
                        speech_logits_list[_i]
                    )
                speech_logits = torch.stack(speech_logits_list, dim=-1)  # (t, b, 1024, 7)
        first_layer_preds = first_layer_logits.argmax(dim=2)  # (t,bs)
        first_layer_preds = first_layer_preds.transpose(0, 1)  # (bs,t)
        labels_first_layer = labels_original[:, 0, :]  # (bs,t)
        correct_predictions = first_layer_preds == labels_first_layer  # (bs,t)
        correct_predictions = correct_predictions * loss_mask  # (bs,t)
        total_correct_predictions = torch.sum(correct_predictions)
        total_predictions = torch.sum(loss_mask)
        first_layer_accuracy = total_correct_predictions / total_predictions
        first_layer_loss = torch.nn.functional.cross_entropy(
            first_layer_logits.permute(1, 2, 0), labels_first_layer, reduction='none'
        )  # (bs,t)
        first_layer_loss = torch.sum(first_layer_loss * loss_mask) / total_predictions

        metrics = {
            # 'loss': loss_mean * 0.14 if torch.count_nonzero(speech_mask) > 0 else loss_mean,
            # 'loss': loss_fnc(output_loss),
            'first_layer_accuracy': first_layer_accuracy,
            'first_layer_loss': first_layer_loss,
        }
        loss_total = first_layer_loss
        for i in range(self.num_speech_codebooks-1):
            if torch.count_nonzero(speech_mask) > 0:
                speech_logits_i = speech_logits[:, :, :, i]
                speech_preds_i = speech_logits_i.argmax(dim=2)  # (t,bs)
                speech_preds_i = speech_preds_i.transpose(0, 1)  # (bs,t)
                labels_i = labels_original[:, i + 1, :]  # (bs,t)
                correct_predictions_i = speech_preds_i == labels_i  # (bs,t)
                correct_predictions_i = correct_predictions_i * loss_mask * speech_mask  # (bs,t)
                total_correct_predictions_i = torch.sum(correct_predictions_i)
                total_predictions_i = torch.sum(loss_mask * speech_mask)
                speech_accuracy_i = total_correct_predictions_i / total_predictions_i
                loss_i = torch.nn.functional.cross_entropy(
                    speech_logits_i.permute(1, 2, 0), labels_i, reduction='none'
                )  # (bs,t)
                loss_i = torch.sum(loss_i * loss_mask * speech_mask) / total_predictions_i
            else:
                speech_accuracy_i = torch.tensor(0.0)
                loss_i = torch.tensor(0.0)
            metrics[f'speech_accuracy_{i+1}'] = speech_accuracy_i
            metrics[f'speech_loss_{i+1}'] = loss_i
            loss_total += loss_i

        metrics['loss'] = loss_total
        self.validation_step_outputs.append(metrics)
        self.train(mode=mode)
        self.frozen_model.train()
        return metrics['loss']

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        if self.cfg.get('pipeline_model_parallel_size', 1) > 1:
            if parallel_state.is_pipeline_last_stage():
                # only the last pipeline parallel stages return loss
                averaged_loss = torch.stack([item['loss'] for item in outputs]).mean()
                # averaged_loss_total_check = torch.stack([item['loss_total_check'] for item in outputs]).mean()
                averaged_first_layer_accuracy = torch.stack([item['first_layer_accuracy'] for item in outputs]).mean()

                self.log(
                    'val_first_layer_accuracy',
                    averaged_first_layer_accuracy,
                    prog_bar=True,
                    rank_zero_only=True,
                    batch_size=1,
                )
                # self.log(
                #     'val_loss_total_check', averaged_loss_total_check, prog_bar=True, rank_zero_only=True, batch_size=1
                # )
                logging.info(f'Validation first_layer_accuracy: {averaged_first_layer_accuracy}')
                logging.info(f'Validation loss_total_check: {averaged_loss_total_check}')

                for i in range(1, self.num_speech_codebooks):
                    averaged_speech_accuracy = torch.stack([item[f'speech_accuracy_{i}'] for item in outputs]).mean()
                    averaged_speech_loss = torch.stack([item[f'speech_loss_{i}'] for item in outputs]).mean()
                    self.log(
                        f'val_speech_accuracy_{i}',
                        averaged_speech_accuracy,
                        prog_bar=True,
                        rank_zero_only=True,
                        batch_size=1,
                    )
                    self.log(
                        f'val_speech_loss_{i}', averaged_speech_loss, prog_bar=True, rank_zero_only=True, batch_size=1
                    )
                    logging.info(f'Validation speech_accuracy_{i}: {averaged_speech_accuracy}')
                    logging.info(f'Validation speech_loss_{i}: {averaged_speech_loss}')
            else:
                averaged_loss = torch.tensor(0.0).cuda()

            # we can only log on one rank if it is rank zero so we broadcast from last rank
            torch.distributed.broadcast(averaged_loss, get_last_rank())

            self.log('val_loss', averaged_loss, prog_bar=True, rank_zero_only=True, batch_size=1)
            logging.info(f'Validation loss: {averaged_loss}')

        else:
            if len(outputs) > 0:
                averaged_loss = torch.stack([item['loss'] for item in outputs]).mean()
                # averaged_loss_total_check = torch.stack([item['loss_total_check'] for item in outputs]).mean()
                logging.info(f'Validation loss: {averaged_loss}')
                self.log('val_loss', averaged_loss, prog_bar=True, rank_zero_only=True, batch_size=1)
                # self.log(
                #     'val_loss_total_check', averaged_loss_total_check, prog_bar=True, rank_zero_only=True, batch_size=1
                # )

                averaged_first_layer_accuracy = torch.stack([item['first_layer_accuracy'] for item in outputs]).mean()
                logging.info(f'Validation first_layer_accuracy: {averaged_first_layer_accuracy}')
                self.log(
                    'val_first_layer_accuracy',
                    averaged_first_layer_accuracy,
                    prog_bar=True,
                    rank_zero_only=True,
                    batch_size=1,
                )

                for i in range(1, self.num_speech_codebooks):
                    averaged_speech_accuracy = torch.stack([item[f'speech_accuracy_{i}'] for item in outputs]).mean()
                    averaged_speech_loss = torch.stack([item[f'speech_loss_{i}'] for item in outputs]).mean()
                    logging.info(f'Validation speech_accuracy_{i}: {averaged_speech_accuracy}')
                    logging.info(f'Validation speech_loss_{i}: {averaged_speech_loss}')
                    self.log(
                        f'val_speech_accuracy_{i}',
                        averaged_speech_accuracy,
                        prog_bar=True,
                        rank_zero_only=True,
                        batch_size=1,
                    )
                    self.log(
                        f'val_speech_loss_{i}', averaged_speech_loss, prog_bar=True, rank_zero_only=True, batch_size=1
                    )

        if self.cfg.get("report_validation_metric", False):
            gather_results = [None for _ in range(parallel_state.get_data_parallel_world_size())]

            all_preds = list(itertools.chain(*[item['predicted_token_ids'] for item in outputs]))
            all_labels = list(itertools.chain(*[item['labels'] for item in outputs]))
            all_inputs = list(itertools.chain(*[item['enc_inputs'] for item in outputs]))

            assert len(all_preds) == len(all_labels)
            assert len(all_preds) == len(all_inputs)

            # Gather inputs, preds, labels from all workers
            torch.distributed.all_gather_object(
                gather_results,
                [(input, pred, label) for (input, pred, label) in zip(all_inputs, all_preds, all_labels)],
                group=parallel_state.get_data_parallel_group(),
            )

            # Deduplicate sentences that may have been distributed across multiple data parallel ranks.
            if parallel_state.get_data_parallel_rank() == 0:

                gather_results_dedup = list(set(itertools.chain(*gather_results)))

                val_metric_dict = self.validation_metric.get_score(
                    [i[2] for i in gather_results_dedup], [i[1] for i in gather_results_dedup],
                )

                for metric, val in val_metric_dict.items():
                    logging.info(f'Validation {metric}: {val}')
                val_metric = list(val_metric_dict.items())[0][1]
                metric_name = list(val_metric_dict.items())[0][0]
            else:
                val_metric = torch.tensor(0.0).cuda()
                metric_name = ''

            self.log(f'val_{metric_name}', val_metric, prog_bar=True, rank_zero_only=True, batch_size=1)

        gbs = self.cfg.global_batch_size
        mbs = self.cfg.micro_batch_size
        self._reconfigure_batch_sizes(gbs, mbs)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        # torch.cuda.memory._record_memory_history(
        #     max_entries=100000
        # )
        result = self.predict_step(batch, batch_idx)
        # torch.cuda.memory._dump_snapshot(f"memory2")
        # torch.cuda.memory._record_memory_history(enabled=None)
        # exit()
        return result

    def on_test_epoch_end(self):
        """
        This might still be broken for lightning 2.0. to fix: see
        https://github.com/NVIDIA/NeMo/blob/9bdf4d12276ee8f95a340cf2f7f340e9b5b74a7e/docs/source/starthere/migration-guide.rst
        """
        outputs = self.predict_step_outputs
        average_metrics = {}
        for output in outputs:
            for key in output:
                if key not in average_metrics:
                    average_metrics[key] = []
                if isinstance(output[key], torch.Tensor):
                    average_metrics[key].append(output[key].item())
                elif output[key] is None:
                    continue
                else:
                    average_metrics[key].append(output[key])

        for key in average_metrics:
            average_metrics[key] = np.mean(average_metrics[key]).item()
            logging.info(f'Test {key}: {average_metrics[key]}')
            self.log(f'test_{key}', average_metrics[key], prog_bar=True, rank_zero_only=True, batch_size=1)
            self.logger.experiment.add_scalar(f'Inf Cumulative {key}', average_metrics[key], 0)

        # save average metrics into json file
        with open(os.path.join(self.logger.log_dir, 'output_metrics.json'), 'w') as f:
            json.dump(average_metrics, f)

    def build_virtual_prompt_dataset(
        self, dataset_paths, batch_size, for_train, drop_last, shuffle, num_workers, pin_memory
    ):
        dataset = T5SpeechLMDataset(
            datasets=dataset_paths,
            tokenizer=self.tokenizer,
            sample_rate=self.cfg.data.get('sample_rate', 24000),
            virtual_prompt_source=self.virtual_prompt_source,
            task_templates=self.task_templates,
            pseudo_tokens=self.pseudo_tokens,
            pad_token_id=self.pad_token_id,
            max_seq_length=self.cfg.data.get('max_seq_length', self.frozen_model.cfg.max_position_embeddings),
            min_seq_length=self.cfg.data.get('min_seq_length', 1),
            add_bos=self.cfg.data.get('add_bos', False),
            add_eos=self.cfg.data.get('add_eos', True),
            decoder_starts_with_pad=self.cfg.data.get('decoder_starts_with_pad', False),
            add_eos_to_decoder_output=self.cfg.data.get('add_eos_to_decoder_output', True),
            add_sentinel_to_input=self.cfg.data.get('add_sentinel_to_input', True),
            ul2_prompt_token=self.cfg.data.get('ul2_prompt_token', None),
            for_train=for_train,
            segment_max_duration=self.cfg.data.get('segment_max_duration', None),
            trim=self.cfg.data.get('trim', None),
            trim_ref=self.cfg.data.get('trim_ref', None),
            trim_top_db=self.cfg.data.get('trim_top_db', None),
            trim_frame_length=self.cfg.data.get('trim_frame_length', None),
            trim_hop_length=self.cfg.data.get('trim_hop_length', None),
            pad_multiple=self.cfg.data.get('pad_multiple', 1),
            pitch_augment=self.cfg.data.get('pitch_augment', None),
            sup_data_path=self.cfg.data.get('sup_data_path', None),
            codec_folder=self.cfg.data.get('codec_folder', None),
            speech_offset=self.cfg.data.get('speech_offset', None),
            train_task=self.cfg.data.get('train_task', "tts"),
            seq_pattern=self.cfg.get('seq_pattern', 'delay_parallel'),
            use_attention_prior=self.cfg.data.get('use_attention_prior', False),
            attention_prior_scaling_factor=self.cfg.data.get('attention_prior_scaling_factor', 1.0),
            cross_attention_epsilon=self.cfg.data.get('cross_attention_epsilon', 0.0),
            lm_vocab_size=self.lm_vocab_size,
            num_speech_codebooks=self.num_speech_codebooks,
            codebook_fps=self.cfg.data.get('codebook_fps', 75),
            add_special_tokens_to_only_first_codebook=self.cfg.data.get('add_special_tokens_to_only_first_codebook', False),
            context_pattern=self.cfg.data.get('context_pattern', 'parallel'),
            context_duration_min=self.cfg.data.get('context_duration_min', 3.0),
            context_duration_max=self.cfg.data.get('context_duration_max', 5.0),
            g2p=self.cfg.data.get('g2p', None),
            skip_datasets=self.cfg.data.get('skip_datasets', []),
            english_only_model=self.cfg.get('english_only_model', False),
        )

        rank = parallel_state.get_data_parallel_rank()
        world_size = parallel_state.get_data_parallel_world_size()
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=self.cfg.seed
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            sampler=sampler,
            batch_size=batch_size // world_size,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True
            if num_workers > 0
            else False,  # (@adithyare and @eharper) We need to set this to True to get around issues with spawn=True
        )
        logging.info(f'build success: {len(dataloader)} {dataset_paths}')
        if self.phoneme_tokenizer is None:
            self.phoneme_tokenizer = dataset.phoneme_tokenizer
        return dataset, dataloader

    def build_virtual_prompt_tarred_dataset(
        self, dataset_paths, audio_path, batch_size, for_train, drop_last, shuffle, num_workers, pin_memory
    ):
        dataset = T5SpeechLMTarredDataset(
            audio_tar_filepaths=audio_path,
            manifest_filepath=dataset_paths,
            tokenizer=self.tokenizer,
            sample_rate=self.cfg.data.get('sample_rate', 24000),
            virtual_prompt_source=self.virtual_prompt_source,
            task_templates=self.task_templates,
            pseudo_tokens=self.pseudo_tokens,
            pad_token_id=self.pad_token_id,
            max_seq_length=self.cfg.data.get('max_seq_length', self.frozen_model.cfg.max_position_embeddings),
            min_seq_length=self.cfg.data.get('min_seq_length', 1),
            shuffle_n=shuffle,
            add_bos=self.cfg.data.get('add_bos', False),
            add_eos=self.cfg.data.get('add_eos', True),
            decoder_starts_with_pad=self.cfg.data.get('decoder_starts_with_pad', False),
            add_eos_to_decoder_output=self.cfg.data.get('add_eos_to_decoder_output', True),
            add_sentinel_to_input=self.cfg.data.get('add_sentinel_to_input', True),
            ul2_prompt_token=self.cfg.data.get('ul2_prompt_token', None),
            for_train=for_train,
            segment_max_duration=self.cfg.data.get('segment_max_duration', None),
            trim=self.cfg.data.get('trim', None),
            trim_ref=self.cfg.data.get('trim_ref', None),
            trim_top_db=self.cfg.data.get('trim_top_db', None),
            trim_frame_length=self.cfg.data.get('trim_frame_length', None),
            trim_hop_length=self.cfg.data.get('trim_hop_length', None),
            pad_multiple=self.cfg.data.get('pad_multiple', 1),
            pitch_augment=self.cfg.data.get('pitch_augment', None),
            speech_offset=self.cfg.data.get('speech_offset', None),
            train_task=self.cfg.data.get('train_task', "tts"),
            seq_pattern=self.cfg.get('seq_pattern', 'delay_parallel'),
            use_attention_prior=self.cfg.data.get('use_attention_prior', False),
            attention_prior_scaling_factor=self.cfg.data.get('attention_prior_scaling_factor', 1.0),
            cross_attention_epsilon=self.cfg.data.get('cross_attention_epsilon', 0.0),
            lm_vocab_size=self.lm_vocab_size,
            num_speech_codebooks=self.num_speech_codebooks,
        )
        rank = parallel_state.get_data_parallel_rank()
        world_size = parallel_state.get_data_parallel_world_size()
        # sampler = torch.utils.data.distributed.DistributedSampler(
        #     dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=self.cfg.seed
        # )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            batch_size=batch_size // world_size,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True
            if num_workers > 0
            else False,  # (@adithyare and @eharper) We need to set this to True to get around issues with spawn=True
        )
        logging.info(f'build success: {len(dataloader)} {dataset_paths}')

        return dataset, dataloader

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        with torch.no_grad():
            (
                virtual_tokens,
                context_and_question_tokens,
                enc_mask,
                dec_input_raw,
                dec_input_mask_raw,
                labels,
                loss_mask,
                position_ids,
                taskname_ids,
                speech_mask,
                _,
                cross_attention_prior,
                text_limits,
                lang,
            ) = batch
            dec_input = dec_input_raw * 1  # (B, 8, T)
            dec_input_mask = dec_input_mask_raw * 1  # (B, T)
            dec_input_mask[:, :] = 1  # Does not really matter
            output_token_list = []

            end_indices = {}
            # pad dec_input (B, 8, T) to 1000 timesteps
            max_inference_timesteps = self.cfg.get('max_inference_timesteps', 1200)
            dec_input = torch.nn.functional.pad(dec_input, (0, max_inference_timesteps - dec_input.shape[2]), value=0)
            dec_input[:,:,1:].zero_()
            dec_input_mask = torch.nn.functional.pad(
                dec_input_mask, (0, max_inference_timesteps - dec_input_mask.shape[1]), value=1
            )

            end_inference_loop_at = None
            fwd_bwd_function = get_forward_backward_func()
            encoder_output = None
            for t in range(dec_input.shape[2] - 1):
                if t % 100 == 0:
                    logging.info("Timestep {}".format(t))
                if t == end_inference_loop_at:
                    print("All ends detected")
                    break
                # print(t)
                # print(dec_input[:, :, : t + 1])
                # import ipdb; ipdb.set_trace()
                if t == 0:
                    # Run first step manually
                    output_logits, _, token_and_speech_logits = self.forward(
                        virtual_tokens,
                        context_and_question_tokens,
                        enc_mask,
                        dec_input[:, :, :t+1],
                        dec_input_mask[:, : t + 1],
                        position_ids,
                        taskname_ids,
                        labels=None,
                        speech_mask=speech_mask,
                        inference=True,
                        decoder_max_sequence_len=max_inference_timesteps,
                        encoder_max_sequence_len=enc_mask.size(1),
                    )
                    encoder_output = token_and_speech_logits[-1].transpose(0,1)
                else:
                    # Prepare batch
                    batch = [
                        max_inference_timesteps,
                        enc_mask.size(1),
                        encoder_output,
                        enc_mask,
                        dec_input[:, :, : t + 1],
                        dec_input_mask[:, : t + 1],
                        position_ids,
                        taskname_ids,
                        speech_mask,
                    ]
                    output_tensor = fwd_bwd_function(
                        forward_step_func=self.get_forward_output_only_func(),
                        data_iterator=iter([batch,]),
                        model=[self],
                        num_microbatches=get_num_microbatches(),
                        forward_only=True,
                        seq_length=t,
                        micro_batch_size=dec_input.shape[0],
                    )
                    output_logits = output_tensor[0]['output_logits']
                    token_and_speech_logits = output_tensor[0]['token_and_speech_logits']
                # output_logits (B, T, V, 8)
                token_logits = token_and_speech_logits[0]  # (B, T, V)
                token_logits_currtimestep = token_logits[:, -1, :]  # (B, V)
                token_preds = token_logits_currtimestep.argmax(dim=1)  # (B,)
                # logging.info(f"Token preds {token_preds}")

                if torch.count_nonzero(speech_mask) > 0:
                    # output_logits (B, T, V, 8)
                    output_logits_currtimestep = (
                        output_logits[:, -1, :, :].permute(0, 2, 1).contiguous().view(-1, self.speech_codebook_size)
                    )  # (B*8, V)
                else:
                    output_logits_currtimestep = token_logits_currtimestep  # (B, V)

                top_k = self.cfg.get('top_k', 10)

                output_logits_currtimestep_topk = torch.topk(output_logits_currtimestep, top_k, dim=1)[0]
                # (B*8, 10) or (B, 10)

                # find indices which are not top k
                indices_to_remove = output_logits_currtimestep < output_logits_currtimestep_topk[:, -1].unsqueeze(1)
                # (B*8, 1024) or (B, 1024)

                output_logits_currtimestep_rescored = output_logits_currtimestep.clone()
                output_logits_currtimestep_rescored[indices_to_remove] = -float('Inf')

                temperature = self.cfg.get('temperature', 0.7)  # Set temp 0.01 for greedy decoding
                output_logits_currtimestep_rescored = output_logits_currtimestep_rescored / temperature
                output_logits_currtimestep_rescored = torch.nn.functional.softmax(
                    output_logits_currtimestep_rescored, dim=1
                )
                output_tokens_curr_timestep = torch.multinomial(
                    output_logits_currtimestep_rescored, num_samples=1
                )  # (B*8, 1)
                # import ipdb; ipdb.set_trace()

                if torch.count_nonzero(speech_mask) > 0:
                    # Convert back to (B, 8)
                    output_tokens_curr_timestep = output_tokens_curr_timestep.view(output_logits.shape[0], self.num_speech_codebooks)

                for _b in range(token_preds.shape[0]):
                    if t > 10 and token_preds[_b] == self.tokenizer.eos_id:
                        if _b not in end_indices:
                            logging.info("End detected for item {}".format(_b) + " at timestep {}".format(t))
                            end_indices[_b] = t
                            if len(end_indices) == token_preds.shape[0]:
                                end_inference_loop_at = t + self.num_speech_codebooks

                output_token_list.append(output_tokens_curr_timestep)

                if torch.count_nonzero(speech_mask) > 0:
                    dec_input_next_timestep = output_tokens_curr_timestep * 1  # (B,8)
                    dec_input_next_timestep[:, 0] = (
                        dec_input_next_timestep[:, 0] + self.speech_offset
                    )  # add offset to first codebook
                    dec_input[:, :, t + 1] = dec_input_next_timestep * 1
                else:
                    dec_input[:, 0, t + 1] = output_tokens_curr_timestep.squeeze(1)
                # # TF
                # if t+1 < 10:
                #     dec_input[:, :, t + 1] = dec_input_raw[:, :, t+1]
                #     import ipdb; ipdb.set_trace()

            output_tokens_combined = torch.stack(output_token_list)  # (T, B, 8) if speech else (T, B)
            if torch.count_nonzero(speech_mask) > 0:
                output_tokens_combined = output_tokens_combined.permute(1, 2, 0)  # (B, 8, T)
            else:
                output_tokens_combined = output_tokens_combined.squeeze(2)
                output_tokens_combined = output_tokens_combined.permute(1, 0)  # (B, T)

            # Layerwise token error rate
            ter_dict = {}
            for i in range(self.num_speech_codebooks):
                ter_dict[i] = {'hypothesis': [], 'gt': []}

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if 'nemo_sv_model' not in self.additional_models:
                nemo_sv_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large')
                nemo_sv_model = nemo_sv_model.to(device)
                nemo_sv_model.eval()
                self.additional_models['nemo_sv_model'] = nemo_sv_model
            else:
                nemo_sv_model = self.additional_models['nemo_sv_model']

            if 'asr_model' not in self.additional_models:
                asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name="stt_multilingual_fastconformer_hybrid_large_pc_blend_eu")
                asr_model = asr_model.to(device)
                asr_model.eval()
                self.additional_models['asr_model'] = asr_model
            else:
                asr_model = self.additional_models['asr_model']

            asr_model_zh = None
            if Lang.zh.value in lang:
                if 'asr_model_zh' not in self.additional_models:
                    asr_model_zh = nemo_asr.models.EncDecRNNTModel.from_pretrained(model_name="stt_zh_conformer_transducer_large")
                    asr_model_zh = asr_model_zh.to(device)
                    asr_model_zh.eval()
                    self.additional_models['asr_model_zh'] = asr_model_zh
                else:
                    asr_model_zh = self.additional_models['asr_model_zh']
            _exp_dir_path = self.logger.log_dir
            _exp_dir_path = _exp_dir_path + '/Sample_Audios'
            if not os.path.exists(_exp_dir_path):
                os.mkdir(_exp_dir_path)
            # hyp_pred_transcript_list = []
            # gt_transcript_list = []
            similarity_list = []
            question_type = []
            # dataset_names = []

            # predicting audio
            batch_size = output_tokens_combined.shape[0]
            wer_score = 0
            audio_to_pred = []
            audio_to_pred_zh = []
            for i in range(batch_size):
                audio_len = (labels[i][0] != 0).sum().item()
                step = batch_idx * self.test_dataloader().batch_size + i
                if torch.count_nonzero(speech_mask) > 0:
                    dec_input_to_1024 = self.convert_tokens_to_range(dec_input_raw[i, :, 0:audio_len])
                    dec_input_wav = self.decode_wav_from_codec_model(dec_input_to_1024)
                    self.logger.experiment.add_audio("Inf Dec Input Wav", dec_input_wav, step, self.sample_rate)

                    predicted_tokens = output_tokens_combined[i]
                    if i in end_indices:
                        logging.info(f"Clipping until end index for audio {i}")
                        predicted_tokens = predicted_tokens[:, 0 : end_indices[i] + 1]  # trim to audio length

                    pred_img = predicted_tokens.data.cpu().float().numpy()
                    dec_inp_img = dec_input_to_1024.data.cpu().float().numpy()

                    predicted_tokens = self.convert_tokens_to_range(predicted_tokens, apply_offset_correction=False)
                    predicted_wav = self.decode_wav_from_codec_model(predicted_tokens)
                    self.logger.experiment.add_audio("Inf Pred Wav", predicted_wav, step, self.sample_rate)
                    self.logger.experiment.add_image(
                        "Inf Pred Tokens", plot_encodec_to_numpy(pred_img), step, dataformats="HWC",
                    )
                    self.logger.experiment.add_image(
                        "Inf Dec Input Tokens", plot_encodec_to_numpy(dec_inp_img), step, dataformats="HWC",
                    )

                    # save predicted_wav and gt_wav to a wav files in dir_path
                    # asr_model_i = asr_model_zh if lang[i] == Lang.zh.value else asr_model
                    audio_fp_pred = os.path.join(_exp_dir_path, f'predicted_wav_{step}.wav')
                    sf.write(audio_fp_pred, predicted_wav.cpu().numpy(), self.sample_rate)
                    audio_fp_gt = os.path.join(_exp_dir_path, f'dec_input_wav_{step}.wav')
                    sf.write(audio_fp_gt, dec_input_wav.cpu().numpy(), self.sample_rate)

                    # speaker verification evaluation
                    spk_embedding_pred = nemo_sv_model.get_embedding(audio_fp_pred)
                    spk_embedding_pred = spk_embedding_pred.cpu().detach().numpy().flatten()
                    spk_embedding_gt = nemo_sv_model.get_embedding(audio_fp_gt)
                    spk_embedding_gt = spk_embedding_gt.cpu().detach().numpy().flatten()
                    similarity = np.dot(spk_embedding_pred, spk_embedding_gt) / (
                        np.linalg.norm(spk_embedding_pred) * np.linalg.norm(spk_embedding_gt)
                    )
                    self.logger.experiment.add_scalar(f'Inf SV Cossim Individual Sample', similarity, step)
                    similarity_list.append(similarity)

                    if lang[i] == Lang.zh.value:
                        audio_to_pred_zh.append({"step":i, "audio":audio_fp_pred})
                        audio_to_pred_zh.append({"step":i, "audio":audio_fp_gt})
                    else:
                        audio_to_pred.append({"step":i, "audio":audio_fp_pred})
                        audio_to_pred.append({"step":i, "audio":audio_fp_gt})

                    # transcribe predicted_wav and gt_wav using asr_model
                    # pred_transcript = asr_model_i.transcribe([audio_fp_pred])[0][0]
                    # gt_transcript = asr_model_i.transcribe([audio_fp_gt])[0][0]
                    # self.logger.experiment.add_text("Inf Predicted Text", pred_transcript, step)
                    # self.logger.experiment.add_text("Inf GT Text", gt_transcript, step)
                    # hyp_pred_transcript_list.append(pred_transcript)
                    # gt_transcript_list.append(gt_transcript)

                    input_token_list = [
                        context_and_question_tokens[i, 0, j].item()
                        for j in range(context_and_question_tokens.shape[2])
                    ]
                    input_token_list = [
                        (ti, t) for ti, t in enumerate(input_token_list) if t != 0 and t < self.speech_offset
                    ]
                    context_end_step = input_token_list[0][0]
                    if context_end_step > self.num_speech_codebooks:
                        _context_tokens = context_and_question_tokens[i][:, :context_end_step]
                        _context_tokens = self.convert_tokens_to_range(_context_tokens, pattern=self.context_pattern)
                        _context_wav = self.decode_wav_from_codec_model(_context_tokens)
                        self.logger.experiment.add_audio("Context Wav", _context_wav, step, self.sample_rate)

                    task_question = self.frozen_model.tokenizer.ids_to_text(
                        [v[1] for v in input_token_list if v[1] < self.lm_vocab_size]
                    )
                    self.logger.experiment.add_text("Task Question", task_question, step)
                    if "Phoneme TTS" in task_question:
                        question_type.append("Phoneme TTS")
                    elif "Text to speech this" in task_question:
                        question_type.append("Text to speech this")
                    else:
                        question_type.append("Other")

                    task_question_phoneme_tokens = [
                        v[1] - self.lm_vocab_size for v in input_token_list if v[1] >= self.lm_vocab_size
                    ]
                    if len(task_question_phoneme_tokens) > 0:
                        phoneme_text = self.phoneme_tokenizer.decode(task_question_phoneme_tokens)
                        self.logger.experiment.add_text("Task Question Phoneme Text", phoneme_text, step)

                    # store predicted_tokens for each layer to compute token error rate
                    for layer_idx in range(self.num_speech_codebooks):
                        ter_dict[layer_idx]['hypothesis'].append(predicted_tokens[layer_idx].cpu().numpy().tolist())
                        ter_dict[layer_idx]['gt'].append(dec_input_to_1024[layer_idx].cpu().numpy().tolist())

                else:
                    r = labels[i, 0].long()
                    nzm = r != 0
                    r = r.tolist()[:-1]
                    nzm = nzm[:-1]
                    h = output_tokens_combined[i].long() * nzm
                    h = h.tolist()
                    cur_wer_score = editdistance.eval(r, h)
                    self.logger.experiment.add_scalar('WER', cur_wer_score, step)
                    logging.info(f"current wer score : {cur_wer_score}")
                    wer_score += cur_wer_score
            if wer_score > 0:
                wer_score /= batch_size
                self.logger.experiment.add_scalar('AVG WER', wer_score, step)
                logging.info(f"average wer score : {wer_score}")

            # compute token error rate for each layer
            for layer_idx in range(self.num_speech_codebooks):
                wer = word_error_rate(ter_dict[layer_idx]['hypothesis'], ter_dict[layer_idx]['gt'], use_cer=True)
                self.logger.experiment.add_scalar(f'Inf TER Layer {layer_idx}', wer, 0)

            greedy_transcripts = []
            if len(audio_to_pred) > 0:
                greedy_transcripts.extend(asr_model.transcribe([i["audio"] for i in audio_to_pred])[0])
            if len(audio_to_pred_zh) > 0:
                greedy_transcripts.extend(asr_model_zh.transcribe([i["audio"] for i in audio_to_pred_zh])[0])

            all_audio_to_pred = audio_to_pred + audio_to_pred_zh
            # Note WER over the batch is not equal to WER(sample) / batch_size, but approx. here
            wer_batch = []
            cer_batch = []
            cer_phoneme = []
            wer_phoneme = []
            cer_tts = []
            wer_tts = []
            # import ipdb; ipdb.set_trace()
            for i in range(0, len(greedy_transcripts)-1, 2):
                assert all_audio_to_pred[i]["step"] == all_audio_to_pred[i+1]["step"]
                step = batch_idx * self.test_dataloader().batch_size + all_audio_to_pred[i]["step"]
                cer_sample = word_error_rate([greedy_transcripts[i]], [greedy_transcripts[i+1]], use_cer=True)
                wer_sample = word_error_rate([greedy_transcripts[i]], [greedy_transcripts[i+1]], use_cer=False)
                self.logger.experiment.add_text("Inf Predicted Text", greedy_transcripts[i], step)
                self.logger.experiment.add_text("Inf GT Text", greedy_transcripts[i+1], step)
                self.logger.experiment.add_scalar(f'Inf CER Transcript', cer_sample, step)
                self.logger.experiment.add_scalar(f'Inf WER Transcript', wer_sample, step)
                cer_batch.append(cer_sample)
                wer_batch.append(wer_sample)
                if question_type[all_audio_to_pred[i]["step"]] == "Phoneme TTS":
                    self.logger.experiment.add_scalar(f'Inf CER Phoneme Task', cer_sample, step)
                    self.logger.experiment.add_scalar(f'Inf WER Phoneme Task', wer_sample, step)
                    cer_phoneme.append(cer_sample)
                    wer_phoneme.append(wer_sample)
                elif question_type[all_audio_to_pred[i]["step"]] == "Text to speech this":
                    self.logger.experiment.add_scalar(f'Inf CER TTS Task', cer_sample, step)
                    self.logger.experiment.add_scalar(f'Inf WER TTS Task', wer_sample, step)
                    cer_tts.append(cer_sample)
                    wer_tts.append(wer_sample)
            # all_audio_to_pred = sorted(all_audio_to_pred, key=lambda x: x["step"])

            # # compute character/word error rate for predicted transcript and gt transcript
            # cer_glob = word_error_rate(hyp_pred_transcript_list, gt_transcript_list, use_cer=True)
            # self.logger.experiment.add_scalar(f'Inf CER Transcript', cer_glob, batch_idx)
            # wer_glob = word_error_rate(hyp_pred_transcript_list, gt_transcript_list, use_cer=False)
            # self.logger.experiment.add_scalar(f'Inf WER Transcript', wer_glob, batch_idx)

            # phoneme_task_pred_transcript_list = [ t for t, q in zip(hyp_pred_transcript_list, question_type) if q == "Phoneme TTS"]
            # phoneme_task_gt_transcript_list = [ t for t, q in zip(gt_transcript_list, question_type) if q == "Phoneme TTS"]

            # tts_task_pred_transcript_list = [ t for t, q in zip(hyp_pred_transcript_list, question_type) if q == "Text to speech this"]
            # tts_task_gt_transcript_list = [ t for t, q in zip(gt_transcript_list, question_type) if q == "Text to speech this"]

            # cer_phoneme = None
            # wer_phoneme = None
            # cer_tts = None
            # wer_tts = None
            # if len(phoneme_task_pred_transcript_list) > 0:
            #     cer_phoneme = word_error_rate(phoneme_task_pred_transcript_list, phoneme_task_gt_transcript_list, use_cer=True)
            #     wer_phoneme = word_error_rate(phoneme_task_pred_transcript_list, phoneme_task_gt_transcript_list, use_cer=False)
            #     self.logger.experiment.add_scalar(f'Inf CER Phoneme Task', cer_phoneme, batch_idx)
            #     self.logger.experiment.add_scalar(f'Inf WER Phoneme Task', wer_phoneme, batch_idx)

            # if len(tts_task_pred_transcript_list) > 0:
            #     cer_tts = word_error_rate(tts_task_pred_transcript_list, tts_task_gt_transcript_list, use_cer=True)
            #     wer_tts = word_error_rate(tts_task_pred_transcript_list, tts_task_gt_transcript_list, use_cer=False)
            #     self.logger.experiment.add_scalar(f'Inf CER TTS Task', cer_tts, batch_idx)
            #     self.logger.experiment.add_scalar(f'Inf WER TTS Task', wer_tts, batch_idx)


            # compute average similarity
            similarity_avg = np.mean(similarity_list)
            self.logger.experiment.add_scalar(f'Inf SV Avg Cossim', similarity_avg, batch_idx)
            self.predict_step_outputs.append({
                'sv_avg_cossim': similarity_avg,
                'cer_transcript': np.mean(cer_batch),
                'wer_transcript': np.mean(wer_batch),
                'cer_phoneme': np.mean(cer_phoneme) if len(cer_phoneme) > 0 else None,
                'wer_phoneme': np.mean(wer_phoneme) if len(wer_phoneme) > 0 else None,
                'cer_tts': np.mean(cer_tts) if len(cer_tts) > 0 else None,
                'wer_tts': np.mean(wer_tts) if len(wer_tts) > 0 else None,
            })

    def predict_step_old(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:

        input_ids, dec_input, labels, loss_mask, enc_mask, dec_mask, position_ids, taskname_ids = batch

        batch_size, seq_length = input_ids.shape
        if self.first_stage_of_pipeline():
            input_embeds = self.embed_input(input_ids, taskname_ids, use_cached_reps=True)

            # TODO: This check needs to be revisited with PP support.
            if hasattr(self.frozen_model.enc_dec_model.encoder_embedding, 'position_embeddings'):
                position_embeddings = self.frozen_model.enc_dec_model.encoder_embedding.position_embeddings(
                    position_ids
                )
                encoder_input = input_embeds + position_embeddings
            else:
                encoder_input = input_embeds

        else:
            encoder_input = torch.zeros((batch_size, seq_length, self.hidden_size), dtype=self.autocast_dtype).cuda()

        predicted_token_ids, log_probs = self.frozen_model.decode(
            tokens_enc=input_ids,
            enc_mask=enc_mask,
            num_tokens_to_generate=self.decoder_seq_length,
            encoder_input=encoder_input,
            bos_id=self.tokenizer.pad_id
            if self.cfg.data.get('decoder_starts_with_pad', False)
            else self.tokenizer.bos_id,
        )
        # Special ids to text function to handle stripping <eos> and special tokens with sentencepiece tokenizers.
        preds_text = MegatronT5SFTModel.ids_to_text(predicted_token_ids, self.tokenizer)
        input_text = MegatronT5SFTModel.ids_to_text(input_ids, self.tokenizer)

        if labels is not None:
            labels_text = MegatronT5SFTModel.ids_to_text(labels, self.tokenizer)
        else:
            labels_text = [None] * len(preds_text)

        return {
            'input_text': input_text,
            'preds_text': preds_text,
            'labels_text': labels_text,
        }

    def on_predict_epoch_end(self, outputs: List[Any]) -> None:

        gather_results = [None for _ in range(parallel_state.get_data_parallel_world_size())]
        all_preds = list(itertools.chain(*[item['preds_text'] for item in outputs[0]]))
        all_labels = list(itertools.chain(*[item['labels_text'] for item in outputs[0]]))
        all_inputs = list(itertools.chain(*[item['input_text'] for item in outputs[0]]))

        assert len(all_preds) == len(all_labels)
        assert len(all_preds) == len(all_inputs)

        # Gather inputs, predictions, and ground truths from all workers
        torch.distributed.all_gather_object(
            gather_results,
            [(input, pred, label) for (input, pred, label) in zip(all_inputs, all_preds, all_labels)],
            group=parallel_state.get_data_parallel_group(),
        )

        # Deduplicate sentences that may have been distributed across multiple data parallel ranks.
        if parallel_state.get_data_parallel_rank() == 0:
            gather_results_dedup = list(set(itertools.chain(*gather_results)))

            input_prediction_pair = []
            correct = 0
            for (input, pred, label) in gather_results_dedup:
                input_prediction_pair.append((input, pred))
                if label:
                    if pred == label:
                        correct += 1

            acc = correct / len(gather_results_dedup) if all_labels[0] else None
            logging.info(f'Prediction results: {acc}')
            logging.info(f'Test finish')


class SimplestModule(torch.nn.Module):
    def __init__(self, dec_hid_size, token_input_size=1, kernel_size=15, dropout=0.5):
        super().__init__()
        self.conv = torch.nn.Conv1d(
            dec_hid_size + token_input_size + 1, dec_hid_size, kernel_size=kernel_size, padding=(kernel_size // 2)
        )
        self.norm = torch.nn.LayerNorm(dec_hid_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, dec_hidden, dec_logits, layer_i, mask):
        layer_index_tensor = torch.tile(
            torch.tensor([layer_i], requires_grad=True, dtype=dec_hidden.dtype, device=dec_hidden.device),
            [*dec_hidden.shape[:-1], 1],
        )
        # dec_prediction = torch.argmax(dec_logits, dim=-1, keepdim=True)
        out = torch.cat([dec_hidden, dec_logits, layer_index_tensor], dim=-1) * mask.T.unsqueeze(-1)
        # import ipdb; ipdb.set_trace()
        out = torch.nn.functional.relu(self.conv(out.transpose(1, 2)))
        out = self.norm(out.transpose(1, 2))
        out = self.dropout(out) * mask.T.unsqueeze(-1)

        return out

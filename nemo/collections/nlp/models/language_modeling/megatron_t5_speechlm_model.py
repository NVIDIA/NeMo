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
import os
from typing import Any, List

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
from nemo.collections.nlp.data.language_modeling.megatron.t5_speechlm_dataset import T5SpeechLMDataset
from nemo.collections.nlp.models.language_modeling.megatron_base_prompt_learning_model import (
    MegatronBasePromptLearningModel,
)
from nemo.collections.nlp.models.language_modeling.megatron_base_speechlm_prompt_model import MegatronBaseSpeechLM
from nemo.collections.nlp.models.language_modeling.megatron_finetune_model import MegatronT5FinetuneModel
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model
from nemo.collections.nlp.modules.common.megatron.token_level_encoder_decoder import MegatronTokenLevelHead
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_iterator_k_split,
    init_method_normal,
)
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.collections.tts.parts.utils.helpers import plot_encodec_to_numpy
from nemo.utils import AppState, logging

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
        speech_offset = cfg.data.get('speech_offset', 30000)
        speech_head_type = cfg.get('speech_head_type', 'token_level')  # token_level, linear

        self.speech_offset = speech_offset
        self.speech_codebook_size = speech_codebook_size
        self.frozen_model.enc_dec_model.speech_offset = speech_offset
        self.frozen_model.enc_dec_model.speech_codebook_size = speech_codebook_size
        self.frozen_model.enc_dec_model.cross_entropy_type = cfg.get('cross_entropy_type', 'regular')
        self.frozen_model.enc_dec_model.seq_pattern = cfg.get('seq_pattern', 'parallel')
        self.frozen_model.enc_dec_model.speech_head_type = speech_head_type

        # Parallel output is used only for vocab parallel cross entropy.
        self.frozen_model.enc_dec_model.parallel_output = (
            self.frozen_model.enc_dec_model.cross_entropy_type == 'vocab_parallel'
        )
        # Need to explicitly set this since it is already initialiazed
        self.frozen_model.enc_dec_model.tokens_head.parallel_output = self.frozen_model.enc_dec_model.parallel_output

        list_of_speech_heads = []
        list_of_speech_tokens_embeddings = []
        for _ in range(7):
            _speech_head_embedding = tensor_parallel.VocabParallelEmbedding(
                speech_codebook_size, embedding_dim=self.word_embeddings.embedding_dim
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
                    use_cpu_initialization=False,
                    params_dtype=self.frozen_model.enc_dec_model.dtype,
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

        encodec_model = EncodecModel.encodec_model_24khz()
        encodec_model.set_target_bandwidth(6.0)
        encodec_model.cuda()
        encodec_model.eval()

        self.additional_models = {'encodec': encodec_model}

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
    ):
        """
        Special forward method for p-tuning/prompt-tuning pretrained
        T5 style models.
        """

        if self.first_stage_of_pipeline():
            # Get embeddings for text tokens and insert virtual token embeddings
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

        # If the decoder input starts with <pad> instead of <bos>, which is the case for huggingface T5 models, we don't want to mask the first token.
        # For NeMo-Megatron, the sequence starts with <bos>, which is never masked so we can always set index 0 to be unmasked.
        dec_mask[:, 0] = 1

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
                    speech_mask=speech_mask,
                )

        return output, encoder_input, out_logits

    def load_frozen_model(self, cfg, trainer):
        self.megatron_amp_o2 = cfg.get('megatron_amp_O2', False)

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
            t5_cfg.megatron_amp_O2 = self.megatron_amp_o2
            # hack to make the _GLOBAL_NUM_MICROBATCHES_CALCULATOR initialize
            t5_cfg.micro_batch_size = cfg.get('micro_batch_size', 4)
            t5_cfg.global_batch_size = cfg.get('global_batch_size', 4)
            t5_cfg.precision = trainer.precision
            t5_cfg.tokenizer.num_sentinel_tokens = 39184 - 29056  # cfg.num_speech_tokens 39168
            t5_cfg.seq_length = 1536
            t5_cfg.max_position_embeddings = 1536

        self.frozen_model = MegatronT5Model.restore_from(
            cfg.get('language_model_path'),
            trainer=trainer,
            override_config_path=t5_cfg,
            save_restore_connector=NLPSaveRestoreConnector(),
        )
        print(f"self.frozen_model {self.frozen_model}")

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
        tensor_shape = [seq_length, get_micro_batch_size(), self.hidden_size]
        data_iter = get_iterator_k_split(batch, get_num_microbatches())

        fwd_bwd_function = get_forward_backward_func()

        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(),
            data_iterator=data_iter,
            model=[self],
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            tensor_shape=tensor_shape,
            decoder_seq_length=dec_seq_length,
            dtype=self.autocast_dtype,
            grad_scaler=self.trainer.precision_plugin.scaler.scale if self.cfg.precision == 16 else None,
            sequence_parallel=self.cfg.get('sequence_parallel', False),
            enable_autocast=self.enable_autocast,
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

    def convert_tokens_to_range(self, tokens, apply_offset_correction=True):
        # convert tokens to range [0, 1024]
        output_tokens = tokens.clone()
        if apply_offset_correction:
            output_tokens[0] = output_tokens[0] - self.speech_offset
        output_tokens = torch.clamp(output_tokens, min=0, max=1023)
        if self.cfg.seq_pattern == "delay_parallel":
            output_tokens_new = []
            for _c in range(output_tokens.shape[0]):
                si = _c
                ei = _c + output_tokens.shape[1] - 8
                output_tokens_new.append(output_tokens[_c, si:ei])
            output_tokens_new = torch.stack(output_tokens_new)
            output_tokens = output_tokens_new

        return output_tokens

    def get_forward_output_and_loss_func(self):
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
            ) = batch

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
                inference=False,
            )
            output_tensor = output_tensor.contiguous()

            if self.trainer.global_step % 100 == 0:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=False):
                        # Encodec does not work with fp16, so we disable autocast for logging audio
                        audio_len = (labels[0][0] != 0).sum().item()
                        labels_to_1024 = self.convert_tokens_to_range(labels[0, :, 0:audio_len])
                        label_wav = self.additional_models['encodec'].decode([[labels_to_1024[None], None]])[0, 0]
                        dec_input_to_1024 = self.convert_tokens_to_range(dec_input[0, :, 0:audio_len])
                        dec_input_wav = self.additional_models['encodec'].decode([[dec_input_to_1024[None], None]])[
                            0, 0
                        ]
                        self.logger.experiment.add_audio("Target Wav", label_wav, self.global_step, 24000)
                        self.logger.experiment.add_audio("Dec Input Wav", dec_input_wav, self.global_step, 24000)

                        input_token_list = [
                            context_and_question_tokens[0, 0, i].item()
                            for i in range(context_and_question_tokens.shape[2])
                        ]
                        input_token_list = [t for t in input_token_list if t != 0 and t < 30000]
                        input_text = self.frozen_model.tokenizer.ids_to_text(input_token_list)
                        self.logger.experiment.add_text("Input Text", input_text, self.global_step)

                        token_logits = out_logits[0]
                        speech_logits_list = out_logits[1]
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
                        first_layer_tokens = token_logits_example.argmax(dim=1) - 30000
                        other_layer_tokens = []
                        for _i in range(speech_logits_example.shape[2]):
                            other_layer_tokens.append(speech_logits_example[:, :, _i].argmax(dim=1))

                        all_layer_tokens = torch.stack([first_layer_tokens] + other_layer_tokens)  # (8, t)
                        all_layer_tokens = self.convert_tokens_to_range(
                            all_layer_tokens, apply_offset_correction=False
                        )
                        all_layer_tokens = torch.clip(all_layer_tokens, 0, 1023)
                        predicted_wav = self.additional_models['encodec'].decode([[all_layer_tokens[None], None]])[
                            0, 0
                        ]
                        self.logger.experiment.add_audio("Pred Wav", predicted_wav, self.global_step, 24000)

            def loss_func(output_tensor):
                loss = self.frozen_model.loss_func(loss_mask, output_tensor)
                reduced_loss = average_losses_across_data_parallel_group([loss])
                return loss, {'avg': reduced_loss}

            return output_tensor, loss_func

        return fwd_output_and_loss_func

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
        print(f'global_step {self.trainer.global_step}')
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
        preds_text = MegatronT5FinetuneModel.ids_to_text(predicted_token_ids, self.tokenizer)
        labels_text = MegatronT5FinetuneModel.ids_to_text(labels, self.tokenizer)
        input_text = MegatronT5FinetuneModel.ids_to_text(input_ids, self.tokenizer)
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

    def validation_step(self, batch, batch_idx, inference=False):
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
        ) = batch

        # loss_mask (b, t)
        # does not use dataloader_iter due to device placement issues arising from PTL
        mode = self.training
        self.eval()
        gbs = self.cfg.get('validation_global_batch_size', self.cfg.global_batch_size)
        self._reconfigure_and_process_inference_batch(virtual_tokens.size(0), gbs)
        loss_mean = self.fwd_bwd_step(
            itertools.chain([batch]), batch_idx, forward_only=True
        )  # comment this out and add custom forward function to calculate WER
        # print (f'loss_mean {loss_mean}')

        labels_original = labels.clone()  # (b, 8, t)
        output_loss, encoder_input, output_logits = self.forward(
            virtual_tokens,
            context_and_question_tokens,
            enc_mask,
            dec_input,
            dec_input_mask,
            position_ids,
            taskname_ids,
            labels=labels,
            speech_mask=speech_mask,
            inference=True,
        )
        first_layer_logits, speech_logits_list = output_logits  # first_layer_logits: (t,bs,vocab_size)
        if self.frozen_model.enc_dec_model.parallel_output:
            # Gather from tensor parallel region
            first_layer_logits = tensor_parallel.gather_from_tensor_model_parallel_region(first_layer_logits)
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
            'loss': loss_mean,
            'first_layer_accuracy': first_layer_accuracy,
            'first_layer_loss': first_layer_loss,
        }
        loss_total = first_layer_loss
        for i in range(7):
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
            metrics[f'speech_accuracy_{i+1}'] = speech_accuracy_i
            metrics[f'speech_loss_{i+1}'] = loss_i
            loss_total += loss_i

        metrics['loss_total_check'] = loss_total
        self.train(mode=mode)
        self.frozen_model.train()
        return metrics

    def validation_epoch_end(self, outputs):
        if self.cfg.get('pipeline_model_parallel_size', 1) > 1:
            if parallel_state.is_pipeline_last_stage():
                # only the last pipeline parallel stages return loss
                averaged_loss = torch.stack([item['loss'] for item in outputs]).mean()
                averaged_loss_total_check = torch.stack([item['loss_total_check'] for item in outputs]).mean()
                averaged_first_layer_accuracy = torch.stack([item['first_layer_accuracy'] for item in outputs]).mean()

                self.log(
                    'val_first_layer_accuracy',
                    averaged_first_layer_accuracy,
                    prog_bar=True,
                    rank_zero_only=True,
                    batch_size=1,
                )
                self.log(
                    'val_loss_total_check', averaged_loss_total_check, prog_bar=True, rank_zero_only=True, batch_size=1
                )
                logging.info(f'Validation first_layer_accuracy: {averaged_first_layer_accuracy}')
                logging.info(f'Validation loss_total_check: {averaged_loss_total_check}')

                for i in range(1, 8):
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
                averaged_loss_total_check = torch.stack([item['loss_total_check'] for item in outputs]).mean()
                logging.info(f'Validation loss: {averaged_loss}')
                self.log('val_loss', averaged_loss, prog_bar=True, rank_zero_only=True, batch_size=1)
                self.log(
                    'val_loss_total_check', averaged_loss_total_check, prog_bar=True, rank_zero_only=True, batch_size=1
                )

                averaged_first_layer_accuracy = torch.stack([item['first_layer_accuracy'] for item in outputs]).mean()
                logging.info(f'Validation first_layer_accuracy: {averaged_first_layer_accuracy}')
                self.log(
                    'val_first_layer_accuracy',
                    averaged_first_layer_accuracy,
                    prog_bar=True,
                    rank_zero_only=True,
                    batch_size=1,
                )

                for i in range(1, 8):
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

    def test_step(self, batch, batch_idx):
        return self.predict_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        average_metrics = {}
        for output in outputs:
            for key in output:
                if key not in average_metrics:
                    average_metrics[key] = []
                if isinstance(output[key], torch.Tensor):
                    average_metrics[key].append(output[key].item())
                else:
                    average_metrics[key].append(output[key])

        for key in average_metrics:
            average_metrics[key] = np.mean(average_metrics[key])
            logging.info(f'Test {key}: {average_metrics[key]}')

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
            sup_data_path=self.cfg.data.get('sup_data_path', '/sup_data_path'),
            speech_offset=self.cfg.data.get('speech_offset', None),
            train_task=self.cfg.data.get('train_task', "tts"),
            seq_pattern=self.cfg.seq_pattern,
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
        print('build success', len(dataloader), dataset_paths)
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
            ) = batch
            dec_input = dec_input_raw * 1 # (B, 8, T)
            dec_input_mask = dec_input_mask_raw * 1 # (B, T)
            dec_input_mask[:, :] = 1  # Does not really matter
            output_token_list = []
            
            end_indices = {}
            # pad dec_input (B, 8, T) to 1000 timesteps
            max_inference_timesteps = self.cfg.get('max_inference_timesteps', 1000)
            dec_input = torch.nn.functional.pad(dec_input, (0, max_inference_timesteps - dec_input.shape[2]), value=0)
            dec_input_mask = torch.nn.functional.pad(dec_input_mask, (0, max_inference_timesteps - dec_input_mask.shape[1]), value=1)
            
            for t in range(dec_input.shape[2]-1):
                output_logits, _, token_and_speech_logits = self.forward(
                    virtual_tokens,
                    context_and_question_tokens,
                    enc_mask,
                    dec_input[:, :, : t + 1], # Slice until the current timestep
                    dec_input_mask[:, : t + 1],
                    position_ids,
                    taskname_ids,
                    labels=None,
                    speech_mask=speech_mask,
                    inference=True,
                )
                # output_logits (B, T, V, 8)
                token_logits = token_and_speech_logits[0]  # (B, T, V)
                token_logits_currtimestep = token_logits[:, t, :]  # (B, V)
                token_preds = token_logits_currtimestep.argmax(dim=1)  # (B,)
                # print("Token preds", token_preds)

                output_logits_currtimestep = (
                    output_logits[:, t, :, :].permute(0, 2, 1).contiguous().view(-1, self.speech_codebook_size)
                )  # (B*8, V)
                temperature = self.cfg.get('temperature', 0.7)  # Set temp 0.01 for greedy decoding
                output_logits_currtimestep = output_logits_currtimestep / temperature
                output_logits_currtimestep = torch.nn.functional.softmax(output_logits_currtimestep, dim=1)
                output_tokens_curr_timestep = torch.multinomial(output_logits_currtimestep, num_samples=1)  # (B*8, 1)
                # Convert back to (B, 8)
                output_tokens_curr_timestep = output_tokens_curr_timestep.view(output_logits.shape[0], 8)

                for _b in range(token_preds.shape[0]):
                    if t > 10 and token_preds[_b] == self.tokenizer.eos_id:
                        if _b not in end_indices:
                            print("End detected for item {}".format(_b) + " at timestep {}".format(t))
                            end_indices[_b] = t

                # output_tokens = output_logits.argmax(dim=2)  # (B,T,8)
                # output_tokens_curr_timestep = output_tokens[:, t]
                output_token_list.append(output_tokens_curr_timestep)  # for later predicting audio

                dec_input_next_timestep = output_tokens_curr_timestep * 1  # (B,8)
                dec_input_next_timestep[:, 0] = (
                    dec_input_next_timestep[:, 0] + self.speech_offset
                )  # add offset to first codebook
                dec_input[:, :, t + 1] = dec_input_next_timestep * 1

            output_tokens_combined = torch.stack(output_token_list)  # (T, B, 8)
            output_tokens_combined = output_tokens_combined.permute(1, 2, 0)  # (B, 8, T)

            # Layerwise token error rate
            ter_dict = {}
            for i in range(8):
                ter_dict[i] = {'hypothesis': [], 'gt': []}

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            nemo_sv_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large')
            nemo_sv_model = nemo_sv_model.to(device)
            nemo_sv_model.eval()

            asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
                model_name="stt_en_conformer_transducer_large"
            )
            asr_model = asr_model.to(device)
            asr_model.eval()
            _exp_dir_path = self.logger.save_dir
            _exp_dir_path = _exp_dir_path + '/Sample_Audios'
            if not os.path.exists(_exp_dir_path):
                os.mkdir(_exp_dir_path)
            hyp_pred_transcript_list = []
            gt_transcript_list = []
            similarity_list = []

            # predicting audio
            batch_size = output_tokens_combined.shape[0]
            for i in range(batch_size):
                audio_len = (labels[i][0] != 0).sum().item()
                step = dataloader_idx + i
                dec_input_to_1024 = self.convert_tokens_to_range(dec_input_raw[i, :, 0:audio_len])
                dec_input_wav = self.additional_models['encodec'].decode([[dec_input_to_1024[None], None]])[0, 0]
                self.logger.experiment.add_audio("Inf Dec Input Wav", dec_input_wav, step, 24000)

                predicted_tokens = output_tokens_combined[i]
                if i in end_indices:
                    print("Clipping until end index for audio", i)
                    predicted_tokens = predicted_tokens[:, 0:end_indices[i]+1] # trim to audio length

                pred_img = predicted_tokens.data.cpu().float().numpy()
                dec_inp_img = dec_input_to_1024.data.cpu().float().numpy()

                predicted_tokens = self.convert_tokens_to_range(predicted_tokens, apply_offset_correction=False)
                predicted_wav = self.additional_models['encodec'].decode([[predicted_tokens[None], None]])[0, 0]
                self.logger.experiment.add_audio("Inf Pred Wav", predicted_wav, step, 24000)
                self.logger.experiment.add_image(
                    "Inf Pred Tokens", plot_encodec_to_numpy(pred_img), step, dataformats="HWC",
                )
                self.logger.experiment.add_image(
                    "Inf Dec Input Tokens", plot_encodec_to_numpy(dec_inp_img), step, dataformats="HWC",
                )

                # save predicted_wav and gt_wav to a wav files in dir_path
                audio_fp_pred = os.path.join(_exp_dir_path, f'predicted_wav_{step}.wav')
                sf.write(audio_fp_pred, predicted_wav.cpu().numpy(), 24000)
                audio_fp_gt = os.path.join(_exp_dir_path, f'dec_input_wav_{step}.wav')
                sf.write(audio_fp_gt, dec_input_wav.cpu().numpy(), 24000)

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

                # transcribe predicted_wav and gt_wav using asr_model
                pred_transcript = asr_model.transcribe([audio_fp_pred])[0][0]
                gt_transcript = asr_model.transcribe([audio_fp_gt])[0][0]
                self.logger.experiment.add_text("Inf Predicted Text", pred_transcript, step)
                self.logger.experiment.add_text("Inf GT Text", gt_transcript, step)
                hyp_pred_transcript_list.append(pred_transcript)
                gt_transcript_list.append(gt_transcript)

                # store predicted_tokens for each layer to compute token error rate
                for layer_idx in range(8):
                    ter_dict[layer_idx]['hypothesis'].append(predicted_tokens[layer_idx].cpu().numpy().tolist())
                    ter_dict[layer_idx]['gt'].append(dec_input_to_1024[layer_idx].cpu().numpy().tolist())

            # compute token error rate for each layer
            for layer_idx in range(8):
                wer = word_error_rate(ter_dict[layer_idx]['hypothesis'], ter_dict[layer_idx]['gt'], use_cer=True)
                self.logger.experiment.add_scalar(f'Inf TER Layer {layer_idx}', wer, 0)

            # compute character/word error rate for predicted transcript and gt transcript
            cer_glob = word_error_rate(hyp_pred_transcript_list, gt_transcript_list, use_cer=True)
            self.logger.experiment.add_scalar(f'Inf CER Transcript', cer_glob, batch_idx)
            wer_glob = word_error_rate(hyp_pred_transcript_list, gt_transcript_list, use_cer=False)
            self.logger.experiment.add_scalar(f'Inf WER Transcript', wer_glob, batch_idx)

            # compute average similarity
            similarity_avg = np.mean(similarity_list)
            self.logger.experiment.add_scalar(f'Inf SV Avg Cossim', similarity_avg, batch_idx)

            return {
                'sv_avg_cossim': similarity_avg,
                'cer_transcript': cer_glob,
                'wer_transcript': wer_glob,
            }

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
        preds_text = MegatronT5FinetuneModel.ids_to_text(predicted_token_ids, self.tokenizer)
        input_text = MegatronT5FinetuneModel.ids_to_text(input_ids, self.tokenizer)

        if labels is not None:
            labels_text = MegatronT5FinetuneModel.ids_to_text(labels, self.tokenizer)
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

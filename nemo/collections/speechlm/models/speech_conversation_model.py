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


import json
import os
import time
from dataclasses import dataclass
from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.utilities import rank_zero_only
from megatron.core import parallel_state
from megatron.core.inference_params import InferenceParams
from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel
from megatron.core.num_microbatches_calculator import (
    get_current_global_batch_size,
    get_num_microbatches,
    reconfigure_num_microbatches_calculator,
)
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from omegaconf import DictConfig

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.utils.eval_utils import remove_punctuations
from nemo.collections.common.data.utils import move_data_to_device
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.llm.gpt.model.base import GPTConfig, get_packed_seq_params
from nemo.collections.speechlm.data.dataset.data_utils import pad_or_trim_to_max_length
from nemo.collections.speechlm.models.speech_to_text_llm_model import (
    MCoreSpeechToTextLLM,
    SpeechToTextLLM,
    SpeechToTextLLMConfig,
)
from nemo.collections.speechlm.modules.asr_module import ASRModuleConfig, HFWrappedEncoder
from nemo.collections.speechlm.modules.modality_adapter import ModalityAdapterConfig
from nemo.collections.speechlm.utils.io import get_nested_attr, load_distributed_ckpt
from nemo.collections.speechlm.utils.text_generation.audio_text_generation_strategy import (
    SpeechToTextGenerationStrategy,
)
from nemo.collections.speechlm.utils.text_generation.audio_text_generation_utils import (
    clean_end_string,
    default_inference_config,
    generate,
    get_computeprob_response,
)
from nemo.lightning.megatron_parallel import (
    MaskedTokenLossReduction,
    masked_token_loss,
    masked_token_loss_context_parallel,
)
from nemo.lightning.pytorch.optim import MegatronOptimizerModule, OptimizerModule
from nemo.utils import AppState, logging
from nemo.utils.get_rank import get_last_rank


def set_input_tensor(self, tensor: torch.Tensor):
    """
    Placeholder function for pipeline parallel, not implemented yet.
    """
    pass


def speech_conversation_data_step(dataloader_iter) -> Dict[str, Any]:
    # Based on: https://github.com/NVIDIA/Megatron-LM/blob/main/pretrain_gpt.py#L87
    # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L828-L842
    # used in SpeechToTextLLMConfig
    batch = next(dataloader_iter)
    _batch: dict
    batch_idx, dataloader_idx = None, None

    if isinstance(batch, tuple) and len(batch) == 3:
        _batch, batch_idx, dataloader_idx = batch
    else:
        _batch = batch

    required_keys = set(
        [
            "sample_ids",
            "attention_mask",
            "position_ids",
            "metadata",
            "inference_params",
            "max_length",
        ]
    )
    # "context", "context_length", "answers", "max_length",
    if parallel_state.is_pipeline_first_stage():
        required_keys.update(
            (
                "audio_signal",
                "audio_signal_length",
                "processed_signal",
                "processed_signal_length",
                "tokens",
                "tokens_length",
                "context_start_idx",
                "num_audios",
                "answers",
                "contexts",
                "context_lengths",
            )
        )
    if parallel_state.is_pipeline_last_stage():
        required_keys.update(("labels", "loss_mask", "dm_labels", "dm_loss_mask"))

    _batch = {
        key: move_data_to_device(val, "cuda", non_blocking=True) if key in required_keys and val is not None else None
        for key, val in _batch.items()
    }

    # inject num_valid_tokens_in_ub for context parallelism,
    # which refers to the total number of valid tokens in the current batch
    if parallel_state.get_context_parallel_world_size() > 1:
        num_valid_tokens_in_ub = None
        if "loss_mask" in _batch and _batch["loss_mask"] is not None:
            num_valid_tokens_in_ub = _batch["loss_mask"].sum()
        _batch["num_valid_tokens_in_ub"] = num_valid_tokens_in_ub

    _batch["dataloader_idx"] = dataloader_idx
    _batch["batch_idx"] = batch_idx
    return _batch


def speech_conversation_forward_step(model: pl.LightningModule, batch: Dict[str, Any]) -> torch.Tensor:
    forward_args = {
        "input_ids": batch.get("tokens", None),
        "input_length": batch.get("tokens_length", None),
        "loss_mask": batch.get("loss_mask", None),
        "attention_mask": batch.get("attention_mask", None),
        "audio_signal": batch.get("audio_signal", None),
        "audio_signal_length": batch.get("audio_signal_length", None),
        "processed_signal": batch.get("processed_signal", None),
        "processed_signal_length": batch.get("processed_signal_length", None),
        "labels": batch.get("labels", None),
        "num_audios": batch.get("num_audios", None),
        "context_start_idx": batch.get("context_start_idx", None),
        "inference_params": batch.get("inference_params", None),
        "dm_labels": batch.get("dm_labels", None),
        "dm_loss_mask": batch.get("dm_loss_mask", None),
    }

    if 'cu_seqlens' in batch:
        forward_args['packed_seq_params'] = get_packed_seq_params(batch)

    return model(**forward_args)


@dataclass
class SpeechConversationLLMConfig(SpeechToTextLLMConfig):

    language_model_config: Optional[GPTConfig] = None
    language_model_class: Optional[str] = None
    speech_model_config: Optional[ASRModuleConfig] = None
    modality_adapter_config: Optional[ModalityAdapterConfig] = None

    dialogue_manager_config: Optional[GPTConfig] = None
    dialogue_manager_audio_only_loss: bool = False
    dialogue_manager_loss_scale: float = 1.0
    language_model_loss_scale: float = 1.0

    language_model_from_pretrained: Optional[str] = None
    language_model_hub: Optional[str] = 'hf://'

    freeze_language_model: bool = True
    freeze_speech_model: bool = True
    freeze_modality_adapter: bool = False
    freeze_dialogue_manager: bool = False

    forward_step_fn: Callable = speech_conversation_forward_step
    data_step_fn: Callable = speech_conversation_data_step

    text_generation_strategy: SpeechToTextGenerationStrategy = SpeechToTextGenerationStrategy

    inference_config: Optional[Dict[str, Any]] = None

    data_config: Optional[DictConfig] = None

    resume_speech_model_from_path: Optional[str] = None
    resume_modality_adapter_from_path: Optional[str] = None
    resume_dialogue_manager_from_path: Optional[str] = None

    def _maybe_load_dialogue_manager(self, dialogue_manager: MegatronModule) -> MegatronModule:
        if self.resume_dialogue_manager_from_path:
            logging.info(f"Loading dialogue manager weights from {self.resume_dialogue_manager_from_path}")
            state_dict, _ = load_distributed_ckpt(self.resume_dialogue_manager_from_path)
            prefix = 'module.dialogue_manager.'
            dialogue_manager_state_dict = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
            dialogue_manager.load_state_dict(dialogue_manager_state_dict, strict=True)
            logging.info(f"Restored dialogue manager weights from {self.resume_dialogue_manager_from_path}")

        return dialogue_manager

    def _propagate_model_configs(self) -> TransformerConfig:
        """
        propagate key attributes to the language/speech model config
        """
        # LLM
        self.language_model_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.language_model_config.sequence_parallel = self.sequence_parallel
        self.language_model_config.pipeline_model_parallel_size = self.pipeline_model_parallel_size
        self.language_model_config.context_parallel_size = self.context_parallel_size

        # ASR
        self.speech_model_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.speech_model_config.sequence_parallel = self.sequence_parallel
        self.speech_model_config.pipeline_model_parallel_size = self.pipeline_model_parallel_size
        self.speech_model_config.context_parallel_size = self.context_parallel_size

        # modality adapter
        self.modality_adapter_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.modality_adapter_config.sequence_parallel = self.sequence_parallel
        self.modality_adapter_config.pipeline_model_parallel_size = self.pipeline_model_parallel_size
        self.modality_adapter_config.context_parallel_size = self.context_parallel_size

        # dialogue manager
        if self.dialogue_manager_config:
            self.dialogue_manager_config.tensor_model_parallel_size = self.tensor_model_parallel_size
            self.dialogue_manager_config.sequence_parallel = self.sequence_parallel
            self.dialogue_manager_config.pipeline_model_parallel_size = self.pipeline_model_parallel_size
            self.dialogue_manager_config.context_parallel_size = self.context_parallel_size

    def configure_model(
        self, tokenizer: TokenizerSpec, speech_model: Optional[ASRModel] = None
    ) -> "MCoreSpeechConversationLLM":
        self._propagate_model_configs()
        language_model = self.language_model_config.configure_model(tokenizer=tokenizer)  # type: "MCoreGPTModel"
        language_model = self._maybe_load_pretrained_llm(language_model)

        if speech_model is None:
            # propagate key attributes to the speech model config
            speech_model = self.speech_model_config.configure_model()  # type: MCoreASRModel
        speech_model.set_input_tensor = MethodType(set_input_tensor, speech_model)

        self.modality_adapter_config.output_dim = self.language_model_config.hidden_size
        input_key = self.modality_adapter_config.input_key_from
        if input_key:
            if hasattr(self.speech_model_config.config, input_key):
                self.modality_adapter_config.input_dim = getattr(self.speech_model_config.config, input_key)
            else:
                if isinstance(speech_model.encoder, HFWrappedEncoder):
                    self.modality_adapter_config.input_dim = get_nested_attr(speech_model.encoder.encoder, input_key)
                else:
                    self.modality_adapter_config.input_dim = get_nested_attr(speech_model.encoder, input_key)

        modality_adapter = self.modality_adapter_config.configure_model()
        modality_adapter.set_input_tensor = MethodType(set_input_tensor, modality_adapter)

        speech_model, modality_adapter = self._maybe_load_asr_and_modality_adapter(speech_model, modality_adapter)

        if self.dialogue_manager_config:
            dialogue_manager = self.dialogue_manager_config.configure_model(
                tokenizer=tokenizer
            )  # type: "MCoreGPTModel"
            dialogue_manager = self._maybe_load_dialogue_manager(dialogue_manager)
        else:
            dialogue_manager = None

        model = MCoreSpeechConversationLLM(
            config=self,
            language_model=language_model,
            speech_model=speech_model,
            modality_adapter=modality_adapter,
            dialogue_manager=dialogue_manager,
            tokenizer=tokenizer,
        )

        if self.freeze_language_model:
            self._freeze_module(model.language_model)
        if self.freeze_speech_model:
            self._freeze_module(model.speech_model)
        if self.freeze_modality_adapter:
            self._freeze_module(model.modality_adapter)
        if self.freeze_dialogue_manager:
            self._freeze_module(model.dialogue_manager)

        return model


class MCoreSpeechConversationLLM(MCoreSpeechToTextLLM):
    def __init__(
        self,
        config: SpeechConversationLLMConfig,
        language_model: MCoreGPTModel,
        speech_model: MegatronModule,
        modality_adapter: MegatronModule,
        tokenizer: TokenizerSpec,
        dialogue_manager: Optional[MCoreGPTModel] = None,
    ):
        super().__init__(
            config=config,
            language_model=language_model,
            speech_model=speech_model,
            modality_adapter=modality_adapter,
            tokenizer=tokenizer,
        )
        self.config = config
        self.dialogue_manager = dialogue_manager

    def forward(
        self,
        input_ids: torch.Tensor,
        input_length: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        audio_signal: Optional[torch.Tensor] = None,
        audio_signal_length: Optional[torch.Tensor] = None,
        processed_signal: Optional[torch.Tensor] = None,
        processed_signal_length: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        num_audios: Optional[torch.Tensor] = None,
        context_start_idx: Optional[List[List[int]]] = None,
        inference_params: Optional[InferenceParams] = None,
        packed_seq_params: Optional[Dict[str, Any]] = None,
        dm_labels: Optional[torch.Tensor] = None,
        dm_loss_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            input_ids: input text tokens for LLM, shape [b, t] or [n_audio, t]
            input_length: input text lengths, shape [b] or [n_audio]
            loss_mask: mask for output answer loss computation, shape [b, t]
            attention_mask: attention mask for LLM, shape [b, t]
            audio_signal: raw audio signal, shape [b, t]
            audio_signal_length: audio signal lengths, shape [b]
            processed_signal: processed audio features, shape [b, d, t]
            processed_signal_length: processed audio feature lengths, shape [b]
            labels: output answer tokens, shape [b, t]
            num_audios: number of audio files per sample, shape [b]
            context_start_idx: start index of each context in the input_ids, shape [b, num_text_segments]
            inference_params: inference parameters
            packed_seq_params: packed sequence parameters
            dm_labels: dialogue manager output tokens, shape [b, t] or [n_audio, t]
            dm_loss_mask: mask for dialogue manager output loss computation, shape [b, t] or [n_audio, t],
                usually is None since it'll be calculated in forward()
        """
        encoded, encoded_len = self.perception(
            input_signal=audio_signal,
            input_signal_length=audio_signal_length,
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
        )

        if num_audios is not None:
            # split the encoded and encoded_len by num_audios, used when there're multiple audio files per sample
            encoded = encoded.split(num_audios.tolist())
            encoded_len = encoded_len.split(num_audios.tolist())

        combined_embeddings, attention_mask, _, _, max_length = self.inject_perception_input(
            encoded, encoded_len, input_ids, input_length, context_start_idx
        )

        if num_audios is not None:
            # sum up the audio_feat_lens for each sample in the batch
            total_encoded_len = torch.stack([torch.sum(lens) for lens in encoded_len])

        if labels is not None:
            # Shift labels to the right
            final_labels = self._shift_labels_by_emb_len(
                labels, input_length, total_encoded_len, max_length, pad_token=0
            )
        else:
            final_labels = None

        if loss_mask is not None:
            # Loss mask where answer tokens are 1.0 and all other tokens are 0.0
            final_loss_mask = self._shift_labels_by_emb_len(
                loss_mask, input_length, total_encoded_len, max_length, pad_token=0
            )
        else:
            final_loss_mask = None

        final_dm_labels, final_dm_loss_mask = self._get_dm_label_mask(
            dm_labels,
            dm_loss_mask,
            num_audios,
            encoded,
            encoded_len,
            input_ids,
            input_length,
            context_start_idx,
            combined_embeddings,
            final_labels,
            final_loss_mask,
        )

        attention_mask, combined_embeddings, final_labels, final_loss_mask = self._get_llm_input_for_context_parallel(
            attention_mask, combined_embeddings, final_labels, final_loss_mask, max_length
        )

        _, _, final_dm_labels, final_dm_loss_mask = self._get_llm_input_for_context_parallel(
            attention_mask, combined_embeddings, final_dm_labels, final_dm_loss_mask, max_length
        )

        self.language_model.post_process = False  # disable post_process for LLM to output last hidden states
        output = self.language_model(
            input_ids=None,
            position_ids=None,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=final_labels,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
        )

        if self.dialogue_manager is not None:
            dm_output = self.dialogue_manager(
                input_ids=None,
                position_ids=None,
                attention_mask=attention_mask,
                decoder_input=output,
                labels=final_dm_labels,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
            )
        else:
            dm_output = None

        if labels is None or loss_mask is None:
            # final_output is the logits, [s b h] => [b s h]
            final_output = output.transpose(0, 1).contiguous()
        else:
            # final_output is the loss for LLM
            final_output = self._get_llm_final_output(self.language_model, output, final_labels)

        results = {
            "llm_output": final_output if labels is None else None,
            "dm_output": dm_output if dm_labels is None else None,
            "llm_loss": final_output if labels is not None else None,
            "dm_loss": dm_output if dm_labels is not None else None,
            "llm_loss_mask": final_loss_mask,
            "dm_loss_mask": final_dm_loss_mask,
        }

        return results

    def _get_llm_final_output(self, llm: MCoreGPTModel, hidden_states, labels):
        """
        Compute loss for the LLM, given output hidden_stats and labels.
        Refer to `forward` method in MCoreGPTModel.
        """
        # logits and loss
        output_weight = None
        if llm.share_embeddings_and_output_weights:
            output_weight = llm.shared_embedding_or_output_weight()
        logits, _ = llm.output_layer(hidden_states, weight=output_weight)

        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        loss = llm.compute_language_model_loss(labels, logits)
        return loss

    def _get_dm_label_mask(
        self,
        dm_labels: torch.Tensor,
        dm_loss_mask: torch.Tensor,
        num_audios: torch.Tensor,
        encoded: Union[torch.Tensor, List[torch.Tensor]],
        encoded_len: Union[torch.Tensor, List[torch.Tensor]],
        input_ids: torch.Tensor,
        input_length: torch.Tensor,
        context_start_idx: List[List[int]],
        combined_embeddings: torch.Tensor,
        final_labels: torch.Tensor,
        final_loss_mask: torch.Tensor,
    ):
        """
        Get dialogue manager labels and loss mask. The output final_dm_labels and final_dm_loss_mask should have
        the same length as the combined_embeddings. The values of final_dm_labels for text tokens are set to 0, whereas
        the values for audio tokens are set to the original dm_labels. The values of final_dm_loss_mask are 1 for non-answer
        tokens and 0 for answer tokens. There's also a `dialogue_manager_audio_only_loss` flag, which is used to calculate
        the loss only for audio tokens.

        Args:
            dm_labels: dialogue manager output tokens, shape [b, t] or [n_audio, t]
            dm_loss_mask: mask for dialogue manager output loss computation, shape [b, t] or [n_audio, t],
                usually is None since it'll be calculated in forward()
            num_audios: number of audio files per sample, shape [b] or None
            encoded: encoded audio features, tensor of [b, t, d], or list of tensors of [n_audio, t, d]
            encoded_len: encoded audio feature lengths, tensor of [b], or list tensors of [n_audio]
            input_ids: input text tokens, shape [b, t]
            input_length: input text lengths, shape [b]
            context_start_idx: start index of each context in the input_ids, shape [b, num_text_segments]
            combined_embeddings: combined embeddings of text and audio, shape [s, b, d]
            final_loss_mask: mask for output answer loss computation in LLM, shape [b, t]
        Returns:
            final_dm_labels: dialogue manager output tokens, shape [b, s]
            final_dm_loss_mask: mask for dialogue manager output loss computation, shape [b, s]
        """
        if dm_labels is None:
            return None, None

        if dm_loss_mask is None:
            dm_loss_mask = torch.ones_like(dm_labels, dtype=torch.float32, device=input_length.device)

        if num_audios is None:
            # signle audio per sample, and audio is always before text
            dm_labels = pad_or_trim_to_max_length(dm_labels, combined_embeddings.size(0), 0)
            dm_loss_mask = pad_or_trim_to_max_length(dm_loss_mask, combined_embeddings.size(0), 0)
            return dm_labels, dm_loss_mask

        if isinstance(encoded, torch.Tensor):
            encoded = encoded.split(num_audios.tolist())
        if isinstance(encoded_len, torch.Tensor):
            encoded_len = encoded_len.split(num_audios.tolist())

        dm_labels = dm_labels.split(num_audios.tolist())
        dm_loss_mask = dm_loss_mask.split(num_audios.tolist())

        # total length of sequence
        max_seq_len = combined_embeddings.size(0)

        final_dm_labels, final_dm_mask = [], []
        audio_only_mask = []
        batch_size = input_length.size(0)
        for i in range(batch_size):
            start_idx_list_i = context_start_idx[i] + [input_ids.size(1)]
            input_len_list = [start_idx_list_i[j + 1] - start_idx_list_i[j] for j in range(len(start_idx_list_i) - 1)]
            dm_labels_i = [torch.zeros(input_len_list[0], device=combined_embeddings.device, dtype=torch.int64)]
            dm_mask_i = [torch.zeros(input_len_list[0], device=combined_embeddings.device, dtype=torch.float32)]
            audio_mask_i = [torch.zeros(input_len_list[0], device=combined_embeddings.device, dtype=torch.float32)]
            for j in range(1, len(input_len_list)):
                dm_labels_i.append(dm_labels[i][j - 1][: encoded_len[i][j - 1]])
                dm_labels_i.append(
                    torch.zeros(input_len_list[0], device=combined_embeddings.device, dtype=torch.int64)
                )
                dm_mask_i.append(dm_loss_mask[i][j - 1][: encoded_len[i][j - 1]])
                audio_mask_i.append(torch.ones_like(dm_mask_i[-1]))
                dm_mask_i.append(torch.ones(input_len_list[0], device=combined_embeddings.device, dtype=torch.float32))
                audio_mask_i.append(torch.zeros_like(dm_mask_i[-1]))
            dm_labels_i = torch.cat(dm_labels_i)  # T
            dm_mask_i = torch.cat(dm_mask_i)  # T
            audio_mask_i = torch.cat(audio_mask_i)  # T
            dm_labels_i = pad_or_trim_to_max_length(dm_labels_i, max_seq_len, 0)
            dm_mask_i = pad_or_trim_to_max_length(dm_mask_i, max_seq_len, 0)
            audio_mask_i = pad_or_trim_to_max_length(audio_mask_i, max_seq_len, 0)
            final_dm_labels.append(dm_labels_i)
            final_dm_mask.append(dm_mask_i)
            audio_only_mask.append(audio_mask_i)

        final_dm_labels = torch.stack(final_dm_labels)
        final_dm_mask = torch.stack(final_dm_mask)
        audio_only_mask = torch.stack(audio_only_mask)

        if self.config.dialogue_manager_audio_only_loss:
            final_dm_mask = audio_only_mask

        if final_dm_mask.shape != final_loss_mask.shape:
            raise RuntimeError(
                f"final_loss_mask shape {final_loss_mask.shape} does not match final_dm_loss_mask shape {final_dm_mask.shape}"
            )
        if final_dm_labels.shape != final_labels.shape:
            raise RuntimeError(
                f"final_labels shape {final_labels.shape} does not match final_dm_labels shape {final_dm_labels.shape}"
            )

        final_dm_mask = final_dm_mask * (1 - final_loss_mask)  # only calculate loss for non-answer tokens

        return final_dm_labels, final_dm_mask

    def _align_dm_labels(self, dm_label, encoded, encoded_len):
        """
        pad or trim dm_labels to the same length as encoded
        Args:
            dm_label: shape [b, t]
            encoded: shape [b, t, d]
            encoded_len: shape [b]
        Returns:
            dm_label: shape [b, t]
            dm_loss_mask: shape [b, t]
        """
        max_seq_len = encoded.size(1)
        dm_loss_mask = torch.arange(max_seq_len, device=encoded.device).unsqueeze(0) < encoded_len.unsqueeze(1)
        dm_loss_mask = dm_loss_mask.float()
        if dm_label.size(1) > max_seq_len:
            dm_label = dm_label[:, :max_seq_len]
        elif dm_label.size(1) < max_seq_len:
            # repeat the last label in dm_label to match the length of encoded
            dm_label = torch.cat(
                [dm_label, dm_label[:, -1].unsqueeze(1).repeat(1, max_seq_len - dm_label.size(1))], dim=1
            )
        return dm_label, dm_loss_mask


class SpeechConversationLossReduction(MaskedTokenLossReduction):
    def __init__(
        self,
        validation_step: bool = False,
        val_drop_last: bool = True,
        llm_loss_scale: float = 1.0,
        dm_loss_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.validation_step = validation_step
        self.val_drop_last = val_drop_last
        self.llm_loss_scale = llm_loss_scale
        self.dm_loss_scale = dm_loss_scale

    def get_loss_value(self, loss, mask, num_valid_tokens_in_ub: Optional[int] = None):
        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size == 1:
            loss_for_ub = masked_token_loss(loss, mask)
        else:
            loss_for_ub = masked_token_loss_context_parallel(loss, mask, num_valid_tokens_in_ub)
        return loss_for_ub

    def forward(
        self, batch: Dict[str, torch.Tensor], forward_out: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Taken from: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L951-L976 ."""  # pylint: disable=line-too-long
        from megatron.core import parallel_state

        from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group

        # neva returns (logits, loss_mask)
        if isinstance(forward_out, tuple):
            forward_out, loss_mask = forward_out
            batch["loss_mask"] = loss_mask

        llm_loss = forward_out["llm_loss"]
        llm_loss_mask = forward_out["llm_loss_mask"]
        dm_loss = forward_out["dm_loss"]
        dm_loss_mask = forward_out["dm_loss_mask"]
        num_valid_tokens_in_ub = batch.get('num_valid_tokens_in_ub', None)

        llm_loss_for_ub = self.get_loss_value(llm_loss, llm_loss_mask, num_valid_tokens_in_ub)
        dm_loss_for_ub = self.get_loss_value(dm_loss, dm_loss_mask, num_valid_tokens_in_ub)

        loss_for_ub = self.llm_loss_scale * llm_loss_for_ub + self.dm_loss_scale * dm_loss_for_ub

        cp_size = parallel_state.get_context_parallel_world_size()
        if self.validation_step and not self.val_drop_last:
            num_valid_tokens_in_ub = batch["loss_mask"].sum()
            if loss_for_ub.isnan():
                assert batch["loss_mask"].count_nonzero() == 0, "Got NaN loss with non-empty input"
                loss_sum_for_ub = torch.zeros_like(num_valid_tokens_in_ub)
            else:
                loss_sum_for_ub = num_valid_tokens_in_ub * loss_for_ub

            loss_sum_and_ub_size_all_gpu = torch.cat(
                [
                    loss_sum_for_ub.clone().detach().view(1),
                    torch.tensor([num_valid_tokens_in_ub], device=torch.cuda.current_device()).clone().detach(),
                ]
            )
            torch.distributed.all_reduce(loss_sum_and_ub_size_all_gpu, group=parallel_state.get_data_parallel_group())
            return loss_for_ub * cp_size, {"loss_sum_and_ub_size": loss_sum_and_ub_size_all_gpu}

        reduced_loss = average_losses_across_data_parallel_group([loss_for_ub])
        return loss_for_ub * cp_size, {"avg": reduced_loss}


class SpeechConversationLLM(SpeechToTextLLM):
    def __init__(
        self,
        config: SpeechConversationLLMConfig,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.optim = optim or MegatronOptimizerModule(config=OptimizerConfig(lr=1e-4, use_distributed_optimizer=True))
        self.optim.connect(self)  # This will bind the `configure_optimizers` method
        self.model_transform = model_transform
        self._training_loss_reduction = None
        self._validation_loss_reduction = None
        self._inference_config = None
        self._speech_model = self.config.speech_model_config.configure_model()

    def configure_model(self) -> None:
        if not hasattr(self, "module"):
            self.module = self.config.configure_model(
                self.tokenizer, self._speech_model
            )  # type: MCoreSpeechConversationLLM
            self.module.language_model = self.module.language_model.to(self.device)
            self.module.speech_model = self.module.speech_model.to(self.device)
            self.module.modality_adapter = self.module.modality_adapter.to(self.device)
            if self.config.dialogue_manager_config:
                self.module.dialogue_manager = self.module.dialogue_manager.to(self.device)
            del self._speech_model

    def forward(
        self,
        input_ids: torch.Tensor,
        input_length: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        audio_signal: Optional[torch.Tensor] = None,
        audio_signal_length: Optional[torch.Tensor] = None,
        processed_signal: Optional[torch.Tensor] = None,
        processed_signal_length: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        num_audios: Optional[torch.Tensor] = None,
        context_start_idx: Optional[List[List[int]]] = None,
        inference_params: Optional[InferenceParams] = None,
        dm_labels: Optional[torch.Tensor] = None,
        dm_loss_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        output = self.module(
            input_ids=input_ids,
            input_length=input_length,
            loss_mask=loss_mask,
            attention_mask=attention_mask,
            audio_signal=audio_signal,
            audio_signal_length=audio_signal_length,
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            labels=labels,
            num_audios=num_audios,
            context_start_idx=context_start_idx,
            inference_params=inference_params,
            dm_labels=dm_labels,
            dm_loss_mask=dm_loss_mask,
        )
        return output

    @property
    def training_loss_reduction(self) -> SpeechConversationLossReduction:
        if not self._training_loss_reduction:
            self._training_loss_reduction = SpeechConversationLossReduction(
                llm_loss_scale=self.config.language_model_loss_scale,
                dm_loss_scale=self.config.dialogue_manager_loss_scale,
            )

        return self._training_loss_reduction

    @property
    def validation_loss_reduction(self) -> SpeechConversationLossReduction:
        if not self._validation_loss_reduction:
            self._validation_loss_reduction = SpeechConversationLossReduction(
                llm_loss_scale=self.config.language_model_loss_scale,
                dm_loss_scale=self.config.dialogue_manager_loss_scale,
                validation_step=True,
            )

        return self._validation_loss_reduction

    def freeze_dialogue_manager(self):
        module = self.module
        while not hasattr(module, "dialogue_manager"):
            module = module.module
        self.freeze_module(module.dialogue_manager)

    def unfreeze_dialogue_manager(self):
        module = self.module
        while not hasattr(module, "dialogue_manager"):
            module = module.module
        self.unfreeze_module(module.dialogue_manager)

    def trainable_parameters(self) -> List[Tuple[str, torch.Tensor]]:
        """
        This function returns all trainable parameters of the model,
        including some params that don't require gradients (e.g., batchnorm).
        This function is used for PEFT to determine what parameters to load/save.
        See `nemo/collections/speechlm/utils/model_transform.py` for more details.
        The name of this function is set to align with the PEFT API.
        """
        trainable_params = []
        # must use state_dict() to include params like batchnorm running mean/var
        for name, param in self.state_dict().items():
            if name.startswith("module.speech_model.") and not self.config.freeze_speech_model:
                trainable_params.append((name, param))
            elif name.startswith("module.modality_adapter.") and not self.config.freeze_modality_adapter:
                trainable_params.append((name, param))
            elif name.startswith("module.language_model.") and not self.config.freeze_language_model:
                trainable_params.append((name, param))
            elif (
                name.startswith("module.language_model.")
                and self.config.freeze_language_model
                and (".adapter." in name or name.endswith(".adapters"))
            ):
                trainable_params.append((name, param))
            elif name.startswith("module.dialogue_manager.") and not self.config.freeze_dialogue_manager:
                trainable_params.append((name, param))

        return trainable_params

    def inference_step(self, batch, mode):
        """
        Used for validation and test steps, added postprocessing after calling self.predict_step().
        """
        batch_idx = batch.pop("batch_idx", None)
        dataloader_idx = batch.pop("dataloader_idx", None)

        data_cfg = self.cfg.data.validation_ds if mode == 'validation' else self.cfg.data.test_ds

        self._reconfigure_and_process_inference_batch(batch, data_cfg)
        # Meta data from dataset
        metadata = batch.get('metadata', [{}] * len(batch['tokens']))
        forward_output = self.forward_step(batch)

        if isinstance(forward_output, tuple):
            # reduce validation loss
            loss = self.validation_loss_reduction.forward(batch=batch, forward_out=forward_output)[1]['avg']
        else:
            # no labels provided, use a dummy loss value
            loss = 0.0

        metric_name = self.val_metric_name if mode == 'validation' else self.test_metric_name
        preds_text = []
        labels_text = []
        inputs_text = []
        if metric_name != "loss":
            # We need _inference_config to get generation params, tokens_to_generate are set in dataset
            if self.get_inference_config() is None:
                logging.warning(f'inference_config is not set. Use default: {default_inference_config}')
                self.set_inference_config(inference_config=default_inference_config)
            self._inference_config['tokens_to_generate'] = data_cfg.get('tokens_to_generate')

            output = self.predict_step(batch, batch_idx, dataloader_idx)

            inputs_text = [self.tokenizer.ids_to_text(c.tolist()) for c in batch['contexts']]
            labels_text = [self.tokenizer.ids_to_text(a.tolist()) for a in batch['answers']]
            preds_text = [
                self.tokenizer.ids_to_text(t[l.item() :][: data_cfg.get('tokens_to_generate')])
                for t, l in zip(output['token_ids'], batch['context_lengths'])
            ]

            if data_cfg.get("end_string", None):
                # sometimes data_cfg.end_string != self.tokenizer.ids_to_text(self.tokenizer.text_to_ids(data_cfg.end_string))
                # for example when data_cfg.end_string = "<end>", the end_string_re will start with " ?? "
                preds_text = clean_end_string(preds_text, self.tokenizer, data_cfg.end_string)
                labels_text = clean_end_string(labels_text, self.tokenizer, data_cfg.end_string)

            if data_cfg.get("remove_text_pc", False):
                preds_text = [remove_punctuations(p.lower(), data_cfg.get("punctuations", None)) for p in preds_text]
                labels_text = [remove_punctuations(l.lower(), data_cfg.get("punctuations", None)) for l in labels_text]

            if data_cfg.get("log_every_n_steps", None) is not None:
                if batch_idx % data_cfg.log_every_n_steps == 0:
                    logging.info(f"Input: `{inputs_text[0]}`")
                    logging.info(f"Label: `{labels_text[0]}`")
                    logging.info(f"Pred: `{preds_text[0]}`")

        outputs = {
            'loss': loss,
            'preds': preds_text,  # [str]
            'labels': labels_text,  # [str]
            'inputs': inputs_text,  # [str]
            'metadata': metadata,  # [dict]
        }

        if mode == 'validation':
            if self._num_validation_dl > 1:
                self.validation_step_outputs[dataloader_idx].append(outputs)
            else:
                self.validation_step_outputs.append(outputs)
        else:
            if self._num_test_dl > 1:
                self.test_step_outputs[dataloader_idx].append(outputs)
            else:
                self.test_step_outputs.append(outputs)
        return forward_output

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: Optional[int] = None):
        """
        Used to get LLM predictions for validation and test steps based on the given inference config.
        """
        inference_config = self.get_inference_config()
        if inference_config is not None:
            # need to overwrite some configuration, make it immutable
            inference_config = inference_config.copy()
        else:
            self.set_inference_config(inference_config=default_inference_config)
            logging.warning(f'inference_config is not set. Use default: {default_inference_config}')
            inference_config = self.get_inference_config()

        if self.cfg.data.get('end_string', None):
            inference_config['end_strings'] = [self.cfg.data.end_string]

        inference_config['strategy'] = self.get_inference_strategy()

        global_batch_size_per_gpu = batch['tokens'].size(0)
        num_micro_batches_before_decode = get_num_microbatches()

        compute_logprob = inference_config.get('compute_logprob', False)

        if compute_logprob:
            inference_config['inputs'] = batch
            inference_config['tokens_to_generate'] = 1
            inference_config['all_probs'] = True
            inference_config['greedy'] = True
            response = generate(self, **inference_config)
            response = get_computeprob_response(self.tokenizer, response, batch)
        else:
            if isinstance(batch, list):
                inference_config['inputs'] = batch
            elif 'num_audios' in batch:
                inference_config['inputs'] = (
                    batch['contexts'].cuda(),
                    batch['context_lengths'].cuda(),
                    batch['audio_signal'].cuda(),
                    batch['audio_signal_length'].cuda(),
                    batch['num_audios'].cuda(),
                    batch['context_start_idx'],
                )
            else:
                inference_config['inputs'] = (
                    batch['contexts'].cuda(),
                    batch['context_lengths'].cuda(),
                    batch['audio_signal'].cuda(),
                    batch['audio_signal_length'].cuda(),
                )
            response = generate(self, **inference_config)

        app_state = AppState()
        reconfigure_num_microbatches_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=global_batch_size_per_gpu * parallel_state.get_data_parallel_world_size(),
            micro_batch_size=global_batch_size_per_gpu // num_micro_batches_before_decode,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )

        # add audio offsets to context lengths for properly decoding only the response
        batch['context_lengths'] = batch['context_lengths'].cuda() + response['audio_feat_lens']

        return response

    def inference_epoch_end(self, outputs, mode, data_cfg):
        # Parent class will handle logging of the loss.
        if not outputs or (all([not x for x in outputs])):
            return None

        if isinstance(outputs[0], dict):
            outputs = [outputs]

        averaged_loss = []
        averaged_metric = []
        # Log metrics for each provided validation/test dataset.
        for dataloader_idx, output in enumerate(outputs):
            if len(output) == 0:
                logging.warning(f"Empty output for dataloader_idx: {dataloader_idx}")
                continue
            # Expand on_validation_epoch_end from parent class MegatronGPTModel as on_validation_epoch_end doesnt take outputs arg
            loss_vals = [x['loss'].view(-1, 1) for x in output]  # each loss is [1, B]
            if parallel_state.is_pipeline_last_stage():
                # only the last pipeline parallel stages return loss with their batch size
                loss = torch.vstack(loss_vals).mean().type(torch.float32).cuda()
            else:
                loss = torch.tensor(0.0, dtype=torch.float32).cuda()

            # we can only log on one rank if it is rank zero so we broadcast from last rank
            torch.distributed.broadcast(loss, get_last_rank())

            # Determine the key used to log the loss based on the user provided name of the dataset or the dataloader index.
            loss_log_key = self._determine_log_key(dataloader_idx, "loss", mode)
            self.log(loss_log_key, loss, batch_size=1)
            averaged_loss.append(loss)

            metric_name = self.val_metric_name if mode == 'validation' else self.test_metric_name
            if metric_name != 'loss':
                self.gather_and_maybe_write_predictions(data_cfg, output, averaged_metric, mode, dataloader_idx)

            torch.distributed.barrier(group=parallel_state.get_data_parallel_group())
            outputs[dataloader_idx].clear()  # free memory

        # Logging of the averaged metrics:
        averaged_loss = sum(averaged_loss) / len(averaged_loss)
        averaged_metric = sum(averaged_metric) / len(averaged_metric) if len(averaged_metric) > 0 else None
        averaged_loss = averaged_loss.to(self.device)
        if averaged_metric is not None:
            averaged_metric = averaged_metric.to(self.device)

        # Handle case where metrics can be nan or inf. This can break checkpoint save/load.
        if averaged_metric is not None and (torch.isinf(averaged_metric) or torch.isnan(averaged_metric)):
            app_state = AppState()
            monitor_mode = app_state.checkpoint_callback_params.mode
            assert monitor_mode in ['min', 'max']
            averaged_metric = 0.0 if monitor_mode == 'max' else 1e5

        if mode == 'validation':
            self.log("val_loss", averaged_loss, batch_size=1, sync_dist=True)
            if averaged_metric is not None:
                self.log(f"val_{self.val_metric_name}", averaged_metric, sync_dist=True, batch_size=1)
        elif mode == 'test':
            self.log("test_loss", averaged_loss, batch_size=1, sync_dist=True)
            if averaged_metric is not None:
                self.log(f"test_{self.test_metric_name}", averaged_metric, sync_dist=True, batch_size=1)

        # Merge the functionality of previous on_inference_epoch_end() within inference_epoch_end() func here
        app_state = AppState()
        # self._restore_activation_checkpointing_args()
        if hasattr(self.cfg.data, "train_ds"):
            reconfigure_num_microbatches_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=self.cfg.data.train_ds.global_batch_size,
                micro_batch_size=self.cfg.data.train_ds.micro_batch_size,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )
        # When running `trainer.validate()`, the training dataset is not available.
        else:
            logging.warning('No training data found, reconfiguring microbatches based on validation batch sizes.')
            reconfigure_num_microbatches_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=data_cfg.global_batch_size,
                micro_batch_size=data_cfg.micro_batch_size,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )
        return averaged_loss, averaged_metric

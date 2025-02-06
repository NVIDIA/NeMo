import copy
import itertools
import json
import math
import os
import re
import tempfile
from collections import OrderedDict
from typing import List, Optional, Union

import hydra
import numpy as np
import sacrebleu
import soundfile as sf
import torch
import torchaudio
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torchaudio.pipelines import SQUIM_SUBJECTIVE

from nemo.collections.asr.parts.utils.eval_utils import remove_punctuations
from nemo.collections.common.metrics import MetricStringToTorchMetric, TextMetricsSet
from nemo.collections.common.parts.utils import apply_rope_scaling, extend_instance
from nemo.collections.multimodal.speech_llm.models.modular_models import ModularAudioGPTModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import EmbeddingScalingMixin, get_specs
from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
from nemo.collections.nlp.modules.common.transformer import transformer_modules
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.utils import AppState, logging, model_utils

try:
    from megatron.core import InferenceParams, parallel_state, tensor_parallel
    from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
    from megatron.core.models.gpt import GPTModel as MCoreGPTModel
    from megatron.core.transformer.transformer_config import TransformerConfig

    try:
        from megatron.core.num_microbatches_calculator import (
            get_num_microbatches,
            reconfigure_num_microbatches_calculator,
        )

    except (ImportError, ModuleNotFoundError):
        logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
        from apex.transformer.pipeline_parallel.utils import (
            _reconfigure_microbatch_calculator as reconfigure_num_microbatches_calculator,
        )
        from apex.transformer.pipeline_parallel.utils import get_num_microbatches
    from megatron.core.packed_seq_params import PackedSeqParams

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False

default_inference_config = {'tokens_to_generate': 30}


class SumVocabParallelEmbedding(tensor_parallel.VocabParallelEmbedding):

    def __init__(
        self,
        proj_head_dims,
        include_proj=False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.proj_head_dims = proj_head_dims
        self.include_proj = include_proj
        if include_proj:
            self.output_proj = tensor_parallel.ColumnParallelLinear(
                kwargs['embedding_dim'] * len(proj_head_dims),
                output_size=kwargs['embedding_dim'],
                config=kwargs['config'],
                init_method=kwargs['init_method'],
            )

    def forward(self, input_):

        if input_.ndim == 3:
            assert input_.shape[2] == len(self.proj_head_dims)
            input_ = input_.clone()
            for i in range(len(self.proj_head_dims)):
                # shuold consider the offset of previous projection heads
                input_[:, :, i] += sum(self.proj_head_dims[:i])
            assert input_.max() < sum(self.proj_head_dims)
        embeddings = super().forward(input_)
        if input_.ndim == 3:
            if self.include_proj:
                new_embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[1], -1)
                new_embeddings, _ = self.output_proj(new_embeddings)
                embeddings = embeddings[:, :, 0] + new_embeddings
            else:
                # sum the multi proj embeddings as the final embeddings
                embeddings = torch.sum(embeddings, axis=2)
        return embeddings


class SumMultiEmbedding(LanguageModelEmbedding):
    """Language model embeddings with multiple tokens at each time step. The embeddings of the tokens of the same time step will be computed separately and then be summed together."""

    def __init__(
        self,
        proj_head_dims,
        include_proj=False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        del self.word_embeddings
        self.word_embeddings = SumVocabParallelEmbedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.config.hidden_size,
            init_method=self.config.init_method,
            reduce_scatter_embeddings=self.reduce_scatter_embeddings,
            config=self.config,
            proj_head_dims=proj_head_dims,
            include_proj=include_proj,
        )


class S2sMCoreGPTModel(MCoreGPTModel):

    def __init__(
        self,
        config: TransformerConfig,
        proj_head_dims: List[int],
        proj_head_loss_weights: List[float],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(config=config, *args, **kwargs)
        self.n_proj_heads = len(proj_head_dims)
        self.proj_head_dims = proj_head_dims
        self.proj_head_loss_weights = proj_head_loss_weights
        self.output_layers = torch.nn.ModuleList(
            [
                tensor_parallel.ColumnParallelLinear(
                    config.hidden_size,
                    output_size=self.proj_head_dims[i],
                    config=config,
                    init_method=config.init_method,
                    bias=False,
                    skip_bias_add=False,
                    gather_output=not self.parallel_output,
                    skip_weight_param_allocation=self.pre_process
                    and self.share_embeddings_and_output_weights,  # if skip_weight_param_allocation=True, weights are initialized from setup_embeddings_and_output_layer
                    embedding_activation_buffer=self.embedding_activation_buffer,
                    grad_output_buffer=self.grad_output_buffer,
                )
                for i in range(self.n_proj_heads)
            ]
        )

    # TODO rewrite setup_embeddings_and_output_layer to include self.output_layers

    def extend_embedding(self, vocab_size: int, include_proj=False):
        """Extend the embedding layer with new vocab size."""

        # Extend word embedding table if self.padded_vocab_size is larger than the size of the pre-trained word embedding
        pretrained_emb = self.embedding

        self.embedding = SumMultiEmbedding(
            config=self.config,
            vocab_size=vocab_size,
            max_sequence_length=self.max_sequence_length,
            position_embedding_type=self.position_embedding_type,
            proj_head_dims=self.proj_head_dims,
            include_proj=include_proj,
        )
        self.embedding.word_embeddings.weight.data[: pretrained_emb.word_embeddings.weight.shape[0]] = (
            pretrained_emb.word_embeddings.weight.data
        )
        # Zero out the new embeddings to make the model behave the same as it was pre-trained
        self.embedding.word_embeddings.weight.data[pretrained_emb.word_embeddings.weight.shape[0] :].zero_()
        del pretrained_emb
        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
    ) -> Tensor:
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoeder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units
        """
        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.

        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        if self.position_embedding_type == 'rope':
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params, self.decoder, decoder_input, self.config
            )
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )

        if not self.post_process:
            return hidden_states

        # logits and loss
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        else:
            output_weight = None
        if input_ids is not None and input_ids.dim() == 2:  # pure text example
            logits, _ = self.output_layer(
                hidden_states, weight=output_weight[: self.vocab_size] if output_weight is not None else None
            )

            if labels is None:
                # [s b h] => [b s h]
                return logits.transpose(0, 1).contiguous()

            loss = self.compute_language_model_loss(labels, logits)

            return loss
        else:
            all_logits = []
            cur_dims = 0
            for i in range(self.n_proj_heads):
                cur_output_weight = (
                    output_weight[cur_dims : cur_dims + self.proj_head_dims[i]] if output_weight is not None else None
                )
                all_logits.append(self.output_layers[i](hidden_states, weight=cur_output_weight)[0])
                cur_dims += self.proj_head_dims[i]
            assert self.vocab_size == self.proj_head_dims[0]
            all_logits[0], _ = self.output_layer(
                hidden_states, weight=output_weight[: self.vocab_size] if output_weight is not None else None
            )

            if labels is None:
                # [s b h] => [b s h]
                return_logits = [logits.transpose(0, 1).contiguous() for logits in all_logits]
                return torch.cat(return_logits, dim=-1)  # cat the last dim together to make other mcore code happy

            tokens_loss = torch.stack(
                [self.compute_language_model_loss(labels[:, :, i], all_logits[i]) for i in range(self.n_proj_heads)],
                axis=2,
            )
            tokens_loss = (
                tokens_loss
                * torch.FloatTensor(self.proj_head_loss_weights).to(tokens_loss.device)
                / sum(self.proj_head_loss_weights)
            )
            return tokens_loss


class S2sMCoreGPTModelDepth(S2sMCoreGPTModel):

    def __init__(
        self,
        config: TransformerConfig,
        proj_head_dims: List[int],
        proj_head_loss_weights: List[float],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(config, proj_head_dims, proj_head_loss_weights, *args, **kwargs)
        from nemo.collections.nlp.modules.common.transformer.transformer_encoders import TransformerEncoder

        self.proj_head_dims = proj_head_dims
        self.depth = TransformerEncoder(
            hidden_size=config.hidden_size,
            num_layers=1,
            inner_size=1 * config.hidden_size,
            num_attention_heads=8,
            mask_future=True,
        )
        self.position_embedding = transformer_modules.FixedPositionalEncoding(config.hidden_size, 128)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
    ) -> Tensor:
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoeder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units
        """
        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.

        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        if self.position_embedding_type == 'rope':
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params, self.decoder, decoder_input, self.config
            )
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )

        if not self.post_process:
            return hidden_states

        depth_hidden_states = (
            hidden_states.squeeze(2)
            .tile([1, 1, len(self.proj_head_dims), 1])
            .reshape(-1, len(self.proj_head_dims), hidden_states.shape[-1])
        )
        position_ids = torch.arange(
            start=0, end=len(self.proj_head_dims), dtype=torch.long, device=hidden_states.device
        )
        position_ids = position_ids.unsqueeze(0).repeat(depth_hidden_states.size(0), 1)

        position_embeddings = self.position_embedding(position_ids)
        depth_hidden_states = depth_hidden_states + position_embeddings

        y = self.depth(
            depth_hidden_states,
            torch.ones_like(depth_hidden_states[:, :, 0]),
        )
        depth_hidden_states = y.reshape(hidden_states.shape[0], hidden_states.shape[1], len(self.proj_head_dims), -1)
        # logits and loss
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        else:
            output_weight = None
        all_logits = []
        cur_dims = 0
        for i in range(self.n_proj_heads):
            cur_output_weight = (
                output_weight[cur_dims : cur_dims + self.proj_head_dims[i]] if output_weight is not None else None
            )
            all_logits.append(self.output_layers[i](depth_hidden_states[:, :, i], weight=cur_output_weight)[0])
            cur_dims += self.proj_head_dims[i]
        assert self.vocab_size == self.proj_head_dims[0]
        all_logits[0], _ = self.output_layer(
            depth_hidden_states[:, :, 0],
            weight=output_weight[: self.vocab_size] if output_weight is not None else None,
        )

        if labels is None:
            # [s b h] => [b s h]
            return_logits = [logits.transpose(0, 1).contiguous() for logits in all_logits]
            return torch.cat(return_logits, dim=-1)  # cat the last dim together to make other mcore code happy

        tokens_loss = torch.stack(
            [self.compute_language_model_loss(labels[:, :, i], all_logits[i]) for i in range(self.n_proj_heads)],
            axis=2,
        )
        tokens_loss = (
            tokens_loss
            * torch.FloatTensor(self.proj_head_loss_weights).to(tokens_loss.device)
            / sum(self.proj_head_loss_weights)
        )
        return tokens_loss


class S2sModularAudioGPTModel(ModularAudioGPTModel):
    """S2S version of Modularized speech GPT model."""

    gpt_model_cls = S2sMCoreGPTModel

    def model_provider_func(self, pre_process, post_process):
        """Model depends on pipeline paralellism."""
        if self.mcore_gpt:
            if not hasattr(self.cfg, 'decoder_reduction_factor'):
                self.decoder_reduction_factor = 1
            else:
                self.decoder_reduction_factor = self.cfg.decoder_reduction_factor
            self.proj_head_dims = self.cfg.proj_head_dims
            self.proj_head_loss_weights = self.cfg.get('proj_head_loss_weights', [1.0])
            if self.decoder_reduction_factor != 1:
                if getattr(self.cfg, 'predict_source_text', False):
                    self.proj_head_dims = (
                        [self.proj_head_dims[0]]
                        + self.proj_head_dims[1:-1] * self.decoder_reduction_factor
                        + [self.proj_head_dims[-1]]
                    )
                    self.proj_head_loss_weights = (
                        [self.cfg.proj_head_loss_weights[0]]
                        + self.cfg.proj_head_loss_weights[1:-1] * self.decoder_reduction_factor
                        + [self.cfg.proj_head_loss_weights[-1]]
                    )
                else:
                    self.proj_head_dims = [self.proj_head_dims[0]] + self.proj_head_dims[
                        1:
                    ] * self.decoder_reduction_factor
                    self.proj_head_loss_weights = [
                        self.cfg.proj_head_loss_weights[0]
                    ] + self.cfg.proj_head_loss_weights[1:] * self.decoder_reduction_factor

            model = self.gpt_model_cls(
                config=self.transformer_config,
                transformer_layer_spec=get_specs(
                    self.spec_name,
                    self.transformer_config,
                    self.transformer_engine,
                    self.cfg.get('hyena', None),
                ),
                vocab_size=self.padded_vocab_size,  # later can be updated to s2s_vocab_size
                max_sequence_length=self.cfg.get('encoder_seq_length', 512),
                pre_process=pre_process,
                post_process=post_process,
                parallel_output=True,
                share_embeddings_and_output_weights=self.cfg.get('share_embeddings_and_output_weights', True),
                position_embedding_type=self.cfg.get('position_embedding_type', 'learned_absolute'),
                rotary_percent=self.cfg.get('rotary_percentage', 1.0),
                seq_len_interpolation_factor=self.cfg.get('seq_len_interpolation_factor', None),
                rotary_base=self.cfg.get('rotary_base', 10000),
                proj_head_dims=self.proj_head_dims,
                proj_head_loss_weights=self.proj_head_loss_weights,
            )

            if self.cfg.get('scale_positional_embedding', False):
                model.rotary_pos_emb.inv_freq = apply_rope_scaling(model.rotary_pos_emb.inv_freq)

            if self.cfg.get("apply_embedding_scaling", False) and parallel_state.is_pipeline_first_stage():
                extend_instance(model.embedding, EmbeddingScalingMixin)
        else:
            raise ValueError("S2S ModularAudioGPTModel requires Megatron-core GPT model.")
        return model

    def post_restore_from_pretrained_models(cls, model, cfg):

        codec_model, codec_model_cfg = cls.get_codec_models_and_configs(cfg)
        logging.info(f"Loaded Codec Model: {codec_model}")

        asr_model, asr_model_cfg = cls.get_asr_models_and_configs(cfg)
        logging.info(f"Loaded ASR Model: {asr_model}")

        mos_model = cls.get_mos_models_and_configs(cfg)
        logging.info(f"Loaded MOS Model: {mos_model}")

        if cfg.model.get('salm_model_path') is not None:
            # this may only work for tp=1
            # check scripts/nlp_language_modeling/merge_lora_weights/merge_salm.py on tp>1
            salm_model_path = cfg.model.get('salm_model_path')
            if '.nemo' in salm_model_path:
                with tempfile.TemporaryDirectory() as tmpdir:
                    NLPSaveRestoreConnector._unpack_nemo_file(salm_model_path, tmpdir)
                    salm_model_path = f"{tmpdir}/model_weights.ckpt"
                    torch_state_dict = torch.load(salm_model_path)
            else:
                torch_state_dict = torch.load(salm_model_path)['state_dict']
            model.setup_complete = False
            model.load_state_dict(torch_state_dict, strict=False)
            logging.info(f"loading from {cfg.model.get('salm_model_path')}: {torch_state_dict.keys()}")

        model.padded_vocab_size = cfg.model.s2s_vocab_size

        if cfg.model.get('megatron_amp_O2', False):
            base_model = model.model.module
        else:
            base_model = model.model

        base_model.extend_embedding(model.padded_vocab_size, include_proj=cfg.model.get('combine_emb_by_proj', False))
        # print out params in more details
        model.summarize(max_depth=2)

        cls.codec_model = codec_model.cuda()
        cls.asr_model = asr_model.cuda()
        cls.mos_model = mos_model.cuda()

    @classmethod
    def restore_from_pretrained_models(
        cls,
        cfg: Optional[Union[OmegaConf, str]] = None,
        trainer: Optional[Trainer] = None,
    ):
        trainer.time_event_callback.logtimeevent.on_model_init_start()
        model = super().restore_from_pretrained_models(cfg, trainer)
        trainer.time_event_callback.logtimeevent.on_model_init_end()
        trainer.time_event_callback.logtimeevent.on_load_checkpoint_start()
        cls.post_restore_from_pretrained_models(cls, model, cfg)
        trainer.time_event_callback.logtimeevent.on_load_checkpoint_end()
        return model

    def load_state_dict(self, state_dict, strict: bool = True):
        try:
            super().load_state_dict(state_dict, strict=strict)
        except RuntimeError as e:
            logging.info(f"Error loading model: {e} retrying with extend_embedding")
            with open_dict(self.cfg):
                self.cfg.model = self.cfg
            self.post_restore_from_pretrained_models(self, self.cfg)
            super().load_state_dict(state_dict, strict=strict)

    # change to add one more dimension
    def _shift_labels_by_emb_len(self, labels, label_lens, emb_lens, max_len, pad_token=0):
        """Shift labels to the right by the length of the audio embeddings."""
        shifted_labels = []
        for label, label_len, emb_len in zip(labels, label_lens, emb_lens):
            shifted_label = torch.full([max_len, label[0].shape[0]], pad_token, device=label.device)
            shifted_label[emb_len : emb_len + label_len] = label[:label_len]
            shifted_labels.append(shifted_label)
        shifted_labels = torch.stack(shifted_labels, dim=0)
        return shifted_labels

    def inference_step(self, dataloader_iter, mode):
        """
        Used for validation and test steps, added postprocessing after calling self.predict_step().
        """

        # Evaluation of multimodal data follows the same pattern as training except predict_step
        batch, batch_idx, dataloader_idx = next(dataloader_iter)
        data_cfg = self.cfg.data.validation_ds if mode == 'validation' else self.cfg.data.test_ds
        self._reconfigure_and_process_inference_batch(batch, data_cfg)
        # Meta data from dataset
        metadata = batch.get('metadata', [{}] * len(batch['tokens']))
        loss = super(MegatronGPTSFTModel, self).validation_step(itertools.chain([batch]), dataloader_idx)

        # We need _inference_config to get generation params
        # add_BOS and tokens_to_generate are set in dataset
        if self.get_inference_config() is None:
            logging.warning(f'inference_config is not set. Use default: {default_inference_config}')
            self.set_inference_config(inference_config=default_inference_config)
        self._inference_config['add_BOS'] = data_cfg.add_bos
        self._inference_config['tokens_to_generate'] = data_cfg.get('tokens_to_generate')

        output = self.predict_step(batch, batch_idx, dataloader_idx)

        inputs_text = (
            [self.tokenizer.ids_to_text(c.tolist()) for c in batch['instructions']]
            if batch['instructions'] is not None
            else [""] * len(batch['target_texts'])
        )
        labels_text = [self.tokenizer.ids_to_text(a.tolist()) for a in batch['target_texts']]
        # only do ids_to_text on the first channel which is text
        output['token_ids_text'] = (np.array(output['token_ids'])[:, :, 0]).tolist()
        preds_text = [
            self.tokenizer.ids_to_text(t[l.item() :][: data_cfg.get('tokens_to_generate')])
            for t, l in zip(output['token_ids_text'], batch['context_lengths'])
        ]

        if data_cfg.get("end_string", None):
            # sometimes data_cfg.end_string != self.tokenizer.ids_to_text(self.tokenizer.text_to_ids(data_cfg.end_string))
            # for example when data_cfg.end_string = "<end>", the end_string_re will start with " ?? "
            end_string_re = self.tokenizer.ids_to_text(self.tokenizer.text_to_ids(data_cfg.end_string))
            preds_text_cleaned = []
            labels_text_cleaned = []
            for p, l in zip(preds_text, labels_text):
                # remove end_string from the end of the string
                for es in [end_string_re, data_cfg.end_string]:
                    if p.endswith(es):
                        p = p[: -len(es)].strip()
                    if l.endswith(es):
                        l = l[: -len(es)].strip()
                preds_text_cleaned.append(p)
                labels_text_cleaned.append(l)
            # TODO: remove preds_text here since it is not used. the real preds_text is obtained by parse_decoder_outputs()
            preds_text = preds_text_cleaned
            labels_text = labels_text_cleaned

        if data_cfg.get("remove_text_pc", False):
            preds_text = [remove_punctuations(p.lower(), data_cfg.get("punctuations", None)) for p in preds_text]
            labels_text = [remove_punctuations(l.lower(), data_cfg.get("punctuations", None)) for l in labels_text]

        # if loss is nan, print the input, label and pred
        if loss.isnan():
            logging.info("++++++++++++++ NaN loss detected ++++++++++++++")
            for i in range(len(inputs_text)):
                logging.info(f"Input: `{inputs_text[i]}`")
                logging.info(f"Label: `{labels_text[i]}`")
                logging.info(f"Pred: `{preds_text[i]}`")
            logging.info("++++++++++++++++++++++++++++++++++++++++++++++++")

        outputs = {
            'loss': loss,
            'preds': output['token_ids'],
            'context_lengths': batch['context_lengths'],
            'labels': batch['answers'],  # [str]
            'labels_text': labels_text,  # [str]
            'inputs': inputs_text,  # [str]
            'metadata': metadata,  # [dict]
            'batch_idx': batch_idx,
        }

        if mode == 'validation':
            if len(self._validation_dl) > 1:
                # super().validation_step appends just loss to self.validation_step_outputs, replace the last appended loss with the outputs dict
                self.validation_step_outputs[dataloader_idx][-1] = outputs
            else:
                # super().validation_step appends just loss to self.validation_step_outputs, replace the last appended loss with the outputs dict
                self.validation_step_outputs[-1] = outputs
        else:
            if len(self._test_dl) > 1:
                self.test_step_outputs[dataloader_idx][-1] = outputs
            else:
                self.test_step_outputs[-1] = outputs
        return outputs

    def post_inference_step(self, list_outputs, mode, data_cfg):
        deduplicated_outputs = {
            'preds': [],
            'labels': [],
            'inputs': [],
            'metadata': [],
            'speech_preds': [],
            'speech_answers': [],
            'text_answers': [],
            'batch_idx': [],
        }
        for outputs in list_outputs:
            for answer, pred, input, metadata, labels_text, pred_context_length in zip(
                outputs['labels'],
                outputs['preds'],
                outputs['inputs'],
                outputs['metadata'],
                outputs['labels_text'],
                outputs['context_lengths'],
            ):
                context_length = 0
                batch_idx = outputs['batch_idx']
                text_answer, speech_answer = self.parse_decoder_outputs(
                    answer,
                    self.tokenizer.eos_id,
                    context_length,
                    self.cfg.data.train_ds.speech_pad_id,
                    self.cfg.data.train_ds.speech_eos_id,
                )
                key = input + self.tokenizer.ids_to_text(text_answer) + str(metadata)

                text_pred, speech_pred = self.parse_decoder_outputs(
                    torch.Tensor(pred),
                    self.tokenizer.eos_id,
                    pred_context_length,
                    self.cfg.data.train_ds.speech_pad_id,
                    self.cfg.data.train_ds.speech_eos_id,
                )

                def normalize_text(text):
                    return text.strip().replace('⁇', '')

                # TODO
                if speech_answer == None:
                    speech_answer = torch.zeros_like(speech_pred)
                text_pred_text = self.tokenizer.ids_to_text(text_pred)
                deduplicated_outputs['preds'].append(normalize_text(text_pred_text))
                deduplicated_outputs['labels'].append(normalize_text(labels_text))
                text_answer_text = self.tokenizer.ids_to_text(text_answer)
                deduplicated_outputs['text_answers'].append(normalize_text(text_answer_text))
                deduplicated_outputs['speech_preds'].append(speech_pred.cpu().numpy())
                deduplicated_outputs['speech_answers'].append(speech_answer.cpu().numpy())

                deduplicated_outputs['inputs'].append(input)
                deduplicated_outputs['metadata'].append(metadata)
                deduplicated_outputs['batch_idx'].append(batch_idx)

        # Compute metric score
        metric_name = self.val_metric_name if mode == 'validation' else self.test_metric_name
        metric = self.val_metric if mode == 'validation' else self.test_metric
        averaged_metric = [[] for _ in range(len(metric_name))]
        output_dir = data_cfg.get("output_dir", "./")
        run_codec = any(("asr" in metric_name or "mos" in metric_name) for metric_name in metric_name)
        run_asr = any("asr" in metric_name for metric_name in metric_name)
        run_mos = any("mos" in metric_name for metric_name in metric_name)

        # TODO: move the following model init code to init() function
        if run_codec:
            self.additional_models['codec_model'] = self.codec_model
            assert 'codec_model' in self.additional_models
            codec_model = self.additional_models['codec_model']
            codec_model.to(self.device)
            codec_model.eval()

            with torch.no_grad():
                logging.info(f"Decoding and saving audio")
                pred_wavs = self.decode_and_save_wavs(
                    codec_model,
                    deduplicated_outputs['speech_preds'],
                    os.path.join(output_dir, "wav", "pred"),
                    deduplicated_outputs['metadata'],
                )
                answer_wavs = self.decode_and_save_wavs(
                    codec_model,
                    deduplicated_outputs['speech_answers'],
                    os.path.join(output_dir, "wav", "answer"),
                    deduplicated_outputs['metadata'],
                )

        if run_asr:
            self.additional_models['asr_model'] = self.asr_model
            assert 'asr_model' in self.additional_models
            asr_model = self.additional_models['asr_model']

            with torch.no_grad():
                logging.info(f"Running ASR on speech preds")
                asr_batch_size = min(64, len(pred_wavs))
                speech_preds_transcribed = asr_model.transcribe(pred_wavs, batch_size=asr_batch_size)
                speech_answers_transcribed = asr_model.transcribe(answer_wavs, batch_size=asr_batch_size)
                deduplicated_outputs['speech_preds_transcribed'] = speech_preds_transcribed
                deduplicated_outputs['speech_answers_transcribed'] = speech_answers_transcribed

        if run_mos:
            self.additional_models['squim_mos_model'] = self.mos_model
            assert 'squim_mos_model' in self.additional_models
            squim_mos_model = self.additional_models['squim_mos_model']
            codec_sample_rate = self.codec_sample_rate

            with torch.no_grad():
                logging.info(f"Running MOS prediction")

                pred_wavs_resampled = [
                    torchaudio.functional.resample(wav.cuda(), codec_sample_rate, 16000).unsqueeze(0)
                    for wav in pred_wavs
                ]
                answer_wavs_resampled = [
                    torchaudio.functional.resample(wav.cuda(), codec_sample_rate, 16000).unsqueeze(0)
                    for wav in answer_wavs
                ]
                squim_mos_scores = [
                    squim_mos_model(pred_wav, answer_wav).cpu()
                    for pred_wav, answer_wav in zip(pred_wavs_resampled, answer_wavs_resampled)
                ]
                deduplicated_outputs['mos_scores'] = squim_mos_scores
        return deduplicated_outputs

    def parse_decoder_outputs(
        self, input_decoder_output, text_separator, context_length, speech_pad_id=1001, speech_eos_id=1004
    ):
        # remove text context
        max_len = input_decoder_output.shape[0]
        if len(input_decoder_output.shape) == 1:
            return input_decoder_output, None
        decoder_output = input_decoder_output[-1:].tile([max_len, 1])
        decoder_output[: max_len - context_length] = input_decoder_output[context_length:]

        # Do not split because text and speech are now aligned
        # Split text and speech part based on the position of the first separator token
        # sep_pos = (decoder_output[:, 0] == text_separator).long()
        # if torch.any(sep_pos):
        #     first_sep_pos = torch.argmax(sep_pos)
        #     text_tokens = decoder_output[:first_sep_pos, 0]
        #     speech_tokens = decoder_output[first_sep_pos + 1 :, 1:]
        # else:
        #     text_tokens = decoder_output[:, 0]
        #     speech_tokens = decoder_output[:, 1:]
        text_tokens = decoder_output[:, 0]
        if self.cfg.get('predict_source_text', False):
            speech_tokens = decoder_output[:, 1:-1]
        else:
            speech_tokens = decoder_output[:, 1:]
        # Get speech token ids
        if self.cfg.get('megatron_amp_O2', False):
            n_speech_codebooks = self.model.module.n_proj_heads - 1
        else:
            n_speech_codebooks = self.model.n_proj_heads - 1
        duplex_method = self.cfg.duplex_method
        if duplex_method != 'from_duplex':
            # Remove padded parts of speech tokens
            speech_eos_pos = torch.sum(speech_tokens == speech_eos_id, axis=1) == n_speech_codebooks
            speech_mask = torch.cumsum(speech_eos_pos, 0) == 0
            speech_tokens = speech_tokens[speech_mask]
        # Revert decoder output reduction
        new_shape = (
            speech_tokens.shape[0] * self.cfg.decoder_reduction_factor,
            speech_tokens.shape[1] // self.cfg.decoder_reduction_factor,
        )
        speech_tokens = speech_tokens.reshape(new_shape)
        if speech_tokens.shape[0] == 0:
            speech_tokens = torch.zeros([1, new_shape[1]]).long().cuda()
        return text_tokens.long(), speech_tokens.long()

    def decode_and_save_wavs(self, codec_model, codes_list, wav_dir, metadata_list):
        sample_rate = self.codec_sample_rate
        os.makedirs(wav_dir, exist_ok=True)
        wavs = []
        for codes, metadata in zip(codes_list, metadata_list):
            codes = torch.tensor(codes).to(codec_model.device).T
            codec_len = torch.Tensor([codes.shape[1]]).long().to(codec_model.device)

            # get rid of bos and eos ids in the codec decoding
            def replace_speech_code(codes, id):
                return torch.where(codes == id, codes[:, :1], codes)

            codes = replace_speech_code(codes, self.cfg.data.train_ds.speech_bos_id)
            codes = replace_speech_code(codes, self.cfg.data.train_ds.speech_eos_id)
            codes = replace_speech_code(codes, self.cfg.data.train_ds.speech_unk_id)
            codes = replace_speech_code(codes, self.cfg.data.train_ds.speech_pad_id)
            wav, _ = codec_model.decode(tokens=codes.unsqueeze(0), tokens_len=codec_len)
            wav = wav[0]
            wavs.append(wav)
            sf.write(
                os.path.join(
                    wav_dir, re.sub("_repeat\d*", "", metadata['audio_filepath'].split('.wav')[0]) + ".gen.wav"
                ),
                wav.detach().cpu().numpy(),
                sample_rate,
            )

        return wavs

    def inference_epoch_end(self, outputs, mode, data_cfg):
        # Parent class will handle logging of the loss.
        if not outputs or (all([not x for x in outputs])):
            return None

        if isinstance(outputs[0], dict):
            outputs = [outputs]

        averaged_loss = []
        # Log metrics for each provided validation/test dataset.
        for dataloader_idx, output in enumerate(outputs):
            if len(output) == 0:
                logging.warning(f"Empty output for dataloader_idx: {dataloader_idx}")
                continue
            # Expand on_validation_epoch_end from parent class MegatronGPTModel as on_validation_epoch_end doesnt take outputs arg
            loss_vals = [x['loss'] for x in output]
            if parallel_state.is_pipeline_last_stage():
                # only the last pipeline parallel stages return loss with their batch size
                if self.cfg.data.get('validation_drop_last', True):
                    loss = torch.stack(loss_vals).mean()
                else:
                    # Compute the avg loss by total_loss across all samples / total number of samples
                    total_loss_and_total_samples = torch.vstack(loss_vals).sum(axis=0)
                    avg_loss = total_loss_and_total_samples[0] / total_loss_and_total_samples[1]
                    loss = avg_loss.type(torch.float32).cuda()
            else:
                loss = torch.tensor(0.0, dtype=torch.float32).cuda()

            # we can only log on one rank if it is rank zero so we broadcast from last rank
            torch.distributed.broadcast(loss, get_last_rank())

            self.log('val_loss', loss, prog_bar=True, rank_zero_only=True, batch_size=1, sync_dist=True)

            # Determine the key used to log the loss based on the user provided name of the dataset or the dataloader index.
            loss_log_key = self._determine_log_key(data_cfg, dataloader_idx, "loss", mode)
            self.log(loss_log_key, loss, batch_size=1)
            averaged_loss.append(loss)

            output = self.post_inference_step(output, mode, data_cfg)

            # Gather the outputs object from all data parallel ranks since we are using the DistributedSampler which splits data across DDP ranks.
            gathered_outputs = [None for _ in range(parallel_state.get_data_parallel_world_size())]
            torch.distributed.all_gather_object(
                gathered_outputs,
                output,
                group=parallel_state.get_data_parallel_group(),
            )

            # Remove duplicate examples due to distributed sampler.
            inp_label_set = set()

            deduplicated_outputs = {}
            total_size = 0
            for rank in range(0, parallel_state.get_data_parallel_world_size()):
                for k, v in gathered_outputs[rank].items():
                    # TODO: add deduplication
                    if k not in deduplicated_outputs:
                        deduplicated_outputs[k] = []
                    deduplicated_outputs[k].extend(v)  # use extend for the b dim

            # Compute metric score
            metric_name = self.val_metric_name if mode == 'validation' else self.test_metric_name
            metric = self.val_metric if mode == 'validation' else self.test_metric
            averaged_metric = [[] for _ in range(len(metric_name))]

            if self.global_rank == 0:
                for (
                    labels,
                    text_answer_text,
                    preds,
                    speech_preds_transcribed,
                    speech_answer,
                    speech_pred,
                    inputs,
                    batch_idx,
                    speech_answers_transcribed,
                ) in zip(
                    deduplicated_outputs['labels'],
                    deduplicated_outputs['text_answers'],
                    deduplicated_outputs['preds'],
                    deduplicated_outputs['speech_preds_transcribed'],
                    deduplicated_outputs['speech_answers'],
                    deduplicated_outputs['speech_preds'],
                    deduplicated_outputs['inputs'],
                    deduplicated_outputs['batch_idx'],
                    deduplicated_outputs['speech_answers_transcribed'],
                ):
                    if (
                        data_cfg.get("log_every_n_steps", None) is not None
                        and batch_idx % data_cfg.log_every_n_steps == 0
                    ):
                        logging.info(f"Input: `{inputs}`")
                        logging.info(f"Label: `{labels}` text_answer_text: `{text_answer_text}`")
                        logging.info(f"Pred: `{preds}`")
                        logging.info(f"speech_preds_transcribed: `{speech_preds_transcribed}`")
                        logging.info(f"speech_answers_transcribed: `{speech_answers_transcribed}`")
                        logging.info(f"Speech out len: pred {speech_pred.shape} label {speech_answer.shape}")

            # Compute metric score
            for metric_name, metric_fn, averaged_metric in zip(metric_name, metric, averaged_metric):
                if metric_name != 'loss':
                    metric_log_key = self._determine_log_key(data_cfg, dataloader_idx, metric_name, mode)
                    labels = deduplicated_outputs['labels']
                    # sacrebleu.corpus_bleu is commonly used which does not share
                    # the same interface as other metrics. We handle it separately.
                    text_preds = deduplicated_outputs['preds']
                    if "asr-" in metric_name:
                        text_preds = deduplicated_outputs['speech_preds_transcribed']

                    text_metric_name = metric_name.replace("asr-", "")

                    def get_turn_split(input_preds, num_turn):
                        if all([t > num_turn for t in get_num_turn(input_preds)]):
                            return [re.split('   *', pred)[num_turn] for pred in input_preds]
                        else:
                            return input_preds

                    def get_num_turn(input_preds):
                        return [len(re.split('   *', pred)) for pred in input_preds]

                    if text_metric_name == 'bleu':  # asr-bleu, bleu
                        metric_result = torch.Tensor([sacrebleu.corpus_bleu(text_preds, [labels]).score]).to(
                            self.device
                        )
                    elif text_metric_name == 'wer':  # asr-wer, wer
                        for pred, label in zip(text_preds, labels):
                            _ = metric_fn(pred, label)

                        metric_result = metric_fn.compute()
                        metric_fn.reset()
                    elif metric_name == 'mos':
                        metric_result = sum(deduplicated_outputs['mos_scores']) / len(
                            deduplicated_outputs['mos_scores']
                        )
                    elif metric_name == 'bleu2':
                        metric_result = torch.Tensor(
                            [sacrebleu.corpus_bleu(get_turn_split(text_preds, 2), [get_turn_split(labels, 2)]).score]
                        ).to(self.device)
                    elif metric_name == 'turndiff':
                        metric_result = torch.Tensor(
                            [np.abs(np.mean(np.subtract(get_num_turn(text_preds), get_num_turn(labels))))]
                        )
                    else:
                        for pred, label in zip(deduplicated_outputs['preds'], labels):
                            _ = metric_fn(pred, label)

                        metric_result = metric_fn.compute()
                        metric_fn.reset()

                    self.log(metric_log_key, metric_result.item(), sync_dist=True)
                    logging.info(f"{mode} {metric_name}: {metric_result.item()}")

                    averaged_metric.append(metric_result)

            # Write predictions to file
            if self.global_rank == 0 and data_cfg.get("write_predictions_to_file", False):
                logging.info(
                    f"Total deduplicated inference data size: {total_size} to {len(deduplicated_outputs['inputs'])}"
                )

                # Check if the user provided a prefix path to the file(s) they want to write.
                if not hasattr(data_cfg, "output_file_path_prefix") or data_cfg.output_file_path_prefix is None:
                    raise ValueError(
                        f"Cannot write predictions to file when output_file_path_prefix is not set or present in the yaml config file."
                    )
                filename_log_key = self._determine_log_key(data_cfg, dataloader_idx, None, mode)
                output_dir = data_cfg.get("output_dir", "./")
                self.write_predictions_to_file(
                    deduplicated_outputs, f"{data_cfg.output_file_path_prefix}_{filename_log_key}", output_dir
                )

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
            self.log("validation_loss", averaged_loss, batch_size=1, sync_dist=True)
            if averaged_metric is not None:
                self.log(f"validation_{self.val_metric_name}", averaged_metric, sync_dist=True, batch_size=1)
        elif mode == 'test':
            self.log("test_loss", averaged_loss, batch_size=1, sync_dist=True)
            if averaged_metric is not None:
                self.log(f"test_{self.test_metric_name}", averaged_metric, sync_dist=True, batch_size=1)

        # Merge the functionality of previous on_inference_epoch_end() within inference_epoch_end() func here
        app_state = AppState()
        self._restore_activation_checkpointing_args()
        if hasattr(self, "_train_ds"):
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

    # consistent with speech models
    @rank_zero_only
    def write_predictions_to_file(self, outputs, output_file_path_prefix, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for folder_name in ['speech_pred', 'speech_answer', 'speaker_contexts']:
            os.makedirs(os.path.join(output_dir, 'npy', folder_name), exist_ok=True)
        output_file_path = output_file_path_prefix + "_inputs_preds_labels.jsonl"
        output_file_path = os.path.join(output_dir, output_file_path)
        with open(output_file_path, "w") as f_json:
            assert (
                len(outputs['inputs']) == len(outputs['preds']) == len(outputs['labels']) == len(outputs['metadata'])
            )
            for i, p, l, m, speech_preds_transcribed, speech_answers_transcribed in zip(
                outputs['inputs'],
                outputs['preds'],
                outputs['labels'],
                outputs['metadata'],
                outputs['speech_preds_transcribed'],
                outputs['speech_answers_transcribed'],
            ):
                json_string = {
                    'input': i,
                    'pred_text': p,
                    'text': l,
                    'speech_preds_transcribed': speech_preds_transcribed,
                    'speech_answers_transcribed': speech_answers_transcribed,
                }
                for k, v in m.items():
                    if k not in json_string:
                        json_string[k] = v
                f_json.write(json.dumps(json_string) + '\n')

        logging.info(f'Predictions saved to {output_file_path}')

    def de_concat_multiproj_logits(self, logits):
        logits_list = []
        prev = 0

        if self.cfg.get('megatron_amp_O2', False):
            base_model = self.model.module
        else:
            base_model = self.model

        for i in base_model.proj_head_dims:
            logits_list.append(logits[:, prev : prev + i])
            prev += i
        return logits_list

    def setup_metric(self, data_cfg):
        metric_name = "exact_string_match"
        if not hasattr(data_cfg, "metrics"):
            metrics = [(MetricStringToTorchMetric["exact_string_match"], "exact_string_match")]
        else:
            metrics = []
            for metric in data_cfg.metrics:
                if not hasattr(metric, "name"):
                    raise ValueError("Metric name is not provided in the metric config.")
                base_metric_name = metric.name.replace("asr-", "")
                if metric.name == "loss" or metric.name == "mos":
                    metrics.append((None, metric.name))
                    continue
                if base_metric_name not in MetricStringToTorchMetric:
                    raise KeyError(
                        f"{metric.name} is not supported. List of supported metrics: {MetricStringToTorchMetric.keys()}"
                    )
                if base_metric_name in self._metrics_require_string2category_map:
                    if metric.average is None:
                        raise ValueError(
                            f"{metric.name} requires specifying whether you want to compute a micro or macro average. Found None."
                        )
                if (
                    metric.get('labels_are_strings', False)
                    and base_metric_name in self._metrics_require_string2category_map
                ):
                    if metric.num_classes is None:
                        raise ValueError(
                            "Number of classes is not provided in the metric section within the data config. "
                            f"Please provide the number of classes in the data config to use the {metric.name} metric."
                        )
                    if metric.get('class_labels', None) is None or not isinstance(
                        metric.get('class_labels', None), ListConfig
                    ):
                        raise ValueError(
                            "Class labels are not provided properly in the metric section witnin the data config. "
                            f"Please provide the class labels as a list of strings in the data config to use the {metric.name} metric."
                        )
                    if len(metric.get('class_labels', None)) != metric.num_classes:
                        raise ValueError(
                            f"Number of class labels {len(metric.get('class_labels', None))} does not match `num_classes` : {metric.num_classes}"
                        )

                metric_cls = MetricStringToTorchMetric[base_metric_name]
                if base_metric_name not in TextMetricsSet:
                    metric_fn = metric_cls(**data_cfg.metric)
                else:
                    metric_fn = metric_cls()
                metrics.append((metric_fn, metric.name))
        return zip(*metrics)

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        self.cfg = cfg
        self.additional_models = {}
        self.extract_codec_on_the_fly = cfg.get('extract_codec_on_the_fly', False)
        self.codec_model_downsampling_factor = cfg.get('codec_model_downsampling_factor', 1023.5)
        self.codec_sample_rate = cfg.data.train_ds.get("codec_sample_rate", 22050)
        super().__init__(cfg, trainer)
        if cfg.get('fixed_speaker_prompt', False):
            self.speaker_embeddings = nn.Embedding(16, cfg.hidden_size)

    def _get_codec_embeddings(self, audio_signal, audio_signal_length):
        """Get codec embeddings for the input audio signal."""
        if 'codec_model' not in self.additional_models:
            self.additional_models['codec_model'] = self.codec_model
            self.additional_models['codec_model'].to(self.device)
            self.additional_models['codec_model'].eval()
        codec_model = self.additional_models['codec_model']
        codec_model.eval()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                original_codec_codes, _ = codec_model.encode(audio=audio_signal, audio_len=audio_signal_length)
                original_codec_codes = original_codec_codes.transpose(1, 2)
        out_codec_codes = []
        out_codec_lens = []
        n_speech_codebooks = original_codec_codes.shape[-1]
        decoder_reduction_factor = self.cfg.get("decoder_reduction_factor", 1)
        speech_pad_id = self.cfg.data.train_ds.speech_pad_id
        padded_original_codec_codes = torch.cat(
            [
                original_codec_codes,
                torch.ones([original_codec_codes.shape[0], decoder_reduction_factor, n_speech_codebooks]).long().cuda()
                * speech_pad_id,
            ],
            axis=1,
        )
        for sidx in range(audio_signal.shape[0]):
            codec_len = min(
                torch.ceil(audio_signal_length[sidx] / self.codec_model_downsampling_factor / decoder_reduction_factor)
                .int()
                .to(self.device),
                math.ceil(original_codec_codes[sidx].shape[0] / decoder_reduction_factor),
            )
            out_codec_codes.append(
                padded_original_codec_codes[sidx, : codec_len * decoder_reduction_factor]
                .reshape((-1, n_speech_codebooks * decoder_reduction_factor))
                .to(self.device)
            )
            out_codec_lens.append(codec_len)

        return out_codec_codes, out_codec_lens

    def get_duration_by_steps(self, steps):
        codec_model_downsampling_factor = self.codec_model_downsampling_factor
        decoder_reduction_factor = self.cfg.get("decoder_reduction_factor", 1)
        codec_sample_rate = self.codec_sample_rate
        seconds = steps * codec_model_downsampling_factor / codec_sample_rate * decoder_reduction_factor
        return seconds, int(seconds * codec_sample_rate)

    def get_step_from_audio_len(self, audio_len):
        decoder_reduction_factor = self.cfg.get("decoder_reduction_factor", 1)
        return torch.ceil(audio_len / self.codec_model_downsampling_factor / decoder_reduction_factor).int() - 1

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

    def inject_speaker_prompt(self, audio_batch, encoder_input, labels, loss_mask, encoded, encoder_length):
        if self.cfg.get('fixed_speaker_prompt', False):
            speaker_ids = audio_batch['speaker_ids']
            speaker_embeds = self.speaker_embeddings(speaker_ids).unsqueeze(1)
            encoder_input = torch.cat([speaker_embeds, encoder_input], dim=1)
            labels = torch.cat([labels[:, :1], labels], dim=1)
            loss_mask = torch.cat([loss_mask[:, :1], loss_mask], dim=1)
            encoder_length += 1
            encoded = torch.cat([speaker_embeds, encoded], dim=1)
        return encoder_input, labels, loss_mask, encoded, encoder_length

    def prepare_llm_input(self, audio_batch):
        # handle duplex and singleturn s2s
        assert self.perception.cfg.preprocessor.sample_rate == self.cfg.data.train_ds.sample_rate
        duplex_method = self.cfg.duplex_method

        if duplex_method == 'from_duplex':
            # duplex data should go here
            assert 'target_texts_merge' in audio_batch
            return self.prepare_llm_input_duplex_from_multiturn(audio_batch)
        # the following branches are not used anymore
        elif duplex_method == 'from_multiturn':
            return self.prepare_llm_input_duplex_from_multiturn(audio_batch)
        elif duplex_method == None:
            pass
        else:
            raise ValueError(f"Unknown duplex method: {duplex_method}")

        # the following branch is used in single turn and multiturn but not duplex
        input_signal = audio_batch['audio_signal']
        logging.debug(f'input_signal.shape: {input_signal.shape}')
        input_signal_length = audio_batch['audio_signal_length']

        input_ids, input_length, labels, loss_mask = (
            audio_batch['tokens'],
            audio_batch['tokens_length'],
            audio_batch['labels'],
            audio_batch['loss_mask'],
        )
        context_lengths = audio_batch['context_lengths']

        if self.extract_codec_on_the_fly:
            answer_signal = audio_batch['answer_audio']
            answer_signal_length = audio_batch['answer_audio_lens']
            target_text_lengths = audio_batch['target_text_lengths']

            answer_codecs, answer_codecs_lens = self._get_codec_embeddings(
                answer_signal, answer_signal_length
            )  # list, list
            for i, answer_codec in enumerate(answer_codecs):
                base_length = target_text_lengths[i] + context_lengths[i]
                input_ids[i, base_length + 1 : base_length + 1 + answer_codecs_lens[i], 1:] = answer_codec
                labels[i, base_length : base_length + answer_codecs_lens[i], 1:] = answer_codec

        num_audios = audio_batch.get("num_audios", None)
        context_start_idx = audio_batch.get("context_start_idx", None)

        # [b, t, c]
        encoded, encoded_len = self.perception(
            input_signal=input_signal,
            input_signal_length=input_signal_length,
            processed_signal=None,
            processed_signal_length=None,
        )

        logging.debug(f'encoded.shape: {encoded.shape}')
        logging.debug(f'encoded_len.shape: {encoded_len.shape}')
        logging.debug(f'num_audios: {num_audios}')
        if num_audios is not None:
            # split the encoded and encoded_len by num_audios, used when there're multiple audio files per sample
            encoded = encoded.split(num_audios.tolist())
            encoded_len = encoded_len.split(num_audios.tolist())

        encoder_input, attention_mask, encoder_length, _, encoder_max_length = self.inject_perception_input(
            encoded, encoded_len, input_ids, input_length, context_start_idx
        )
        if num_audios is not None:
            # sum up the audio_feat_lens for each sample in the batch
            encoded_len = torch.stack([torch.sum(lens) for lens in encoded_len])

        # Shift labels to the right
        labels = self._shift_labels_by_emb_len(labels, input_length, encoded_len, encoder_max_length, pad_token=0)
        # Loss mask where answer tokens are 1.0 and all other tokens are 0.0
        loss_mask = self._shift_labels_by_emb_len(
            loss_mask, input_length, encoded_len, encoder_max_length, pad_token=0
        )
        # return if it is single turn or duplex
        if (
            all(audio_batch['num_turns'] == 2) or 'target_texts_merge' in audio_batch
        ):  # real duplex data read from dataloader
            return encoder_input, attention_mask, labels, loss_mask, encoder_length
        # special logic to handle multiturn half-duplex s2s
        # use num_turns to recover multiturn format and then merge them back to one sequence as LLM input/output
        new_encoder_input = []
        new_labels = []
        new_loss_mask = []
        new_encoder_length = []
        cnt = 0
        for num_turns in audio_batch['num_turns']:
            tmp_encoder_input = []
            tmp_labels = []
            tmp_loss_mask = []
            tmp_encoder_length = []
            for i in range(0, num_turns, 2):
                input_len = encoder_length[cnt]
                if i != num_turns - 2:  # last turn
                    input_len -= 1  # remove the last token as it is eos between the turns
                tmp_encoder_input.append(encoder_input.transpose(0, 1)[cnt][:input_len])
                tmp_labels.append(labels[cnt][:input_len])
                tmp_loss_mask.append(loss_mask[cnt][:input_len])
                tmp_encoder_length.append(input_len)
                cnt += 1
            new_encoder_input.append(torch.cat(tmp_encoder_input, dim=0))
            new_encoder_length.append(sum(tmp_encoder_length))
            new_labels.append(torch.cat(tmp_labels, dim=0))
            new_loss_mask.append(torch.cat(tmp_loss_mask, dim=0))
        new_encoder_input = pad_sequence(new_encoder_input, batch_first=True)
        new_encoder_length = torch.Tensor(new_encoder_length).long()
        new_labels = pad_sequence(new_labels, batch_first=True)
        new_loss_mask = pad_sequence(new_loss_mask, batch_first=True)
        assert cnt == encoder_length.shape[0]
        new_attention_mask = self._create_attention_mask(new_encoder_input)
        return (
            new_encoder_input.transpose(0, 1).contiguous(),
            new_attention_mask,
            new_labels,
            new_loss_mask,
            new_encoder_length,
        )

    @classmethod
    def get_codec_models_and_configs(cls, cfg):
        pretrained_codec_model = cfg.model.get("codec_model_path", None)
        pretrained_codec_model_class = cfg.model.get(
            "pretrained_codec_model_target", "nemo.collections.tts.models.audio_codec.AudioCodecModel"
        )

        model_class = hydra.utils.get_class(pretrained_codec_model_class)
        if pretrained_codec_model.endswith('.nemo'):
            logging.info(f'Loading pretrained codec model from local file: {pretrained_codec_model}')
            codec_model = model_class.restore_from(pretrained_codec_model, map_location='cpu')
        else:
            logging.info(f'Loading pretrained codec model from NGC: {pretrained_codec_model}')
            codec_model = model_class.from_pretrained(pretrained_codec_model, map_location='cpu')
        return codec_model, codec_model.cfg

    @classmethod
    def get_asr_models_and_configs(cls, cfg):

        pretrained_asr_model = cfg.model.get("asr_model_path", None)
        pretrained_asr_model_class = cfg.model.get(
            "pretrained_asr_model_target", "nemo.collections.asr.models.ASRModel"
        )

        model_class = hydra.utils.get_class(pretrained_asr_model_class)
        if pretrained_asr_model.endswith('.nemo'):
            logging.info(f'Loading pretrained codec model from local file: {pretrained_asr_model}')
            asr_model = model_class.restore_from(pretrained_asr_model, map_location='cpu')
        else:
            logging.info(f'Loading pretrained asr model from NGC: {pretrained_asr_model}')
            asr_model = model_class.from_pretrained(pretrained_asr_model, map_location='cpu')
        return asr_model, asr_model.cfg

    @classmethod
    def get_mos_models_and_configs(cls, cfg):

        squim_mos_model = SQUIM_SUBJECTIVE.get_model()
        return squim_mos_model

    def setup_optimizer_param_groups(self):
        super().setup_optimizer_param_groups()
        freeze_llm = self.cfg.get('freeze_llm', True)
        if freeze_llm:
            # needs to be updated since vocab is changed
            for param in self.model.embedding.parameters():
                param.requires_grad = True
            for param in self.model.output_layers.parameters():
                param.requires_grad = True
            if hasattr(self.model, "output_proj"):
                for param in self.model.output_proj.parameters():
                    param.requires_grad = True


class S2sModularAudioGPTModelDepth(S2sModularAudioGPTModel):
    """S2S version of Modularized speech GPT model."""

    gpt_model_cls = S2sMCoreGPTModelDepth

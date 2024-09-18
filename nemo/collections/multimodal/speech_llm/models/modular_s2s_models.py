import itertools
import json
import os
from collections import OrderedDict
from typing import List, Optional, Union

import numpy as np
import sacrebleu
import torch
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor

from nemo.collections.asr.parts.utils.eval_utils import remove_punctuations
from nemo.collections.common.metrics import MetricStringToTorchMetric, TextMetricsSet
from nemo.collections.common.parts.utils import apply_rope_scaling, extend_instance
from nemo.collections.multimodal.speech_llm.models.modular_models import ModularAudioGPTModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import EmbeddingScalingMixin, get_specs
from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
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
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.proj_head_dims = proj_head_dims

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
            # sum the multi proj embeddings as the final embeddings
            embeddings = torch.sum(embeddings, axis=2)
        return embeddings


class SumMultiEmbedding(LanguageModelEmbedding):
    """Language model embeddings with multiple tokens at each time step. The embeddings of the tokens of the same time step will be computed separately and then be summed together."""

    def __init__(
        self,
        proj_head_dims,
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
                    skip_weight_param_allocation=self.pre_process and self.share_embeddings_and_output_weights,
                    embedding_activation_buffer=self.embedding_activation_buffer,
                    grad_output_buffer=self.grad_output_buffer,
                )
                for i in range(self.n_proj_heads)
            ]
        )

    # TODO rewrite setup_embeddings_and_output_layer to include self.output_layers

    def extend_embedding(self, vocab_size: int):
        """Extend the embedding layer with new vocab size."""

        # Extend word embedding table if self.padded_vocab_size is larger than the size of the pre-trained word embedding
        pretrained_emb = self.embedding

        self.embedding = SumMultiEmbedding(
            config=self.config,
            vocab_size=vocab_size,
            max_sequence_length=self.max_sequence_length,
            position_embedding_type=self.position_embedding_type,
            proj_head_dims=self.proj_head_dims,
        )
        self.embedding.word_embeddings.weight.data[: pretrained_emb.word_embeddings.weight.shape[0]] = (
            pretrained_emb.word_embeddings.weight.data
        )
        # Zero out the new embeddings to make the model behave the same as it was pre-trained
        self.embedding.word_embeddings.weight.data[pretrained_emb.word_embeddings.weight.shape[0] :].zero_()
        del pretrained_emb

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
        all_logits = [self.output_layers[i](hidden_states)[0] for i in range(self.n_proj_heads)]
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        all_logits[0], _ = self.output_layer(hidden_states, weight=output_weight)

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
                self.proj_head_dims = [self.proj_head_dims[0]] + self.proj_head_dims[
                    1:
                ] * self.decoder_reduction_factor
                self.proj_head_loss_weights = [self.cfg.proj_head_loss_weights[0]] + self.cfg.proj_head_loss_weights[
                    1:
                ] * self.decoder_reduction_factor

            model = S2sMCoreGPTModel(
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

    @classmethod
    def restore_from_pretrained_models(
        cls,
        cfg: Optional[Union[OmegaConf, str]] = None,
        trainer: Optional[Trainer] = None,
    ):
        model = super().restore_from_pretrained_models(cfg, trainer)
        if cfg.model.get('salm_model_path') is not None:
            torch_state_dict = torch.load(cfg.model.get('salm_model_path'))['state_dict']
            model.setup_complete = False
            model.load_state_dict(torch_state_dict, strict=False)
            logging.info(f"loading from {cfg.model.get('salm_model_path')}: {torch_state_dict.keys()}")

        model.padded_vocab_size = cfg.model.s2s_vocab_size
        model.model.extend_embedding(model.padded_vocab_size)
        # print out params in more details
        model.summarize(max_depth=2)
        return model

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

        inputs_text = [self.tokenizer.ids_to_text(c.tolist()) for c in batch['instructions']]
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

    def parse_decoder_outputs(
        self, input_decoder_output, text_separator, context_length, speech_pad_id=1001, speech_eos_id=1004
    ):
        # remove text context
        max_len = input_decoder_output.shape[0]
        decoder_output = input_decoder_output[-1:].tile([max_len, 1])
        decoder_output[: max_len - context_length] = input_decoder_output[context_length:]

        # Split text and speech part based on the position of the first separator token
        sep_pos = (decoder_output[:, 0] == text_separator).long()
        if torch.any(sep_pos):
            first_sep_pos = torch.argmax(sep_pos)
            text_tokens = decoder_output[:first_sep_pos, 0]
            speech_tokens = decoder_output[first_sep_pos + 1 :, 1:]
        else:
            text_tokens = decoder_output[:, 0]
            speech_tokens = decoder_output[:, 1:]

        # Get speech token ids
        n_speech_codebooks = self.model.n_proj_heads - 1

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
        return text_tokens.long(), speech_tokens.long()

    def decode_and_save_wavs(self, codec_model, codes_list, wav_dir, metadata_list):
        import soundfile as sf

        sample_rate = 22050
        os.makedirs(wav_dir, exist_ok=True)
        wavs = []
        for codes, metadata in zip(codes_list, metadata_list):
            codes = torch.tensor(codes).to(codec_model.device).T
            codec_len = torch.Tensor([codes.shape[1]]).long().to(codec_model.device)
            wav, _ = codec_model.decode(tokens=codes.unsqueeze(0), tokens_len=codec_len)
            wav = wav[0]
            wavs.append(wav)
            sf.write(
                os.path.join(wav_dir, metadata['audio_filepath'].split('.wav')[0] + ".gen.wav"),
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

            # Gather the outputs object from all data parallel ranks since we are using the DistributedSampler which splits data across DDP ranks.
            gathered_outputs = [None for _ in range(parallel_state.get_data_parallel_world_size())]
            torch.distributed.all_gather_object(
                gathered_outputs,
                [
                    {
                        'preds': x['preds'],
                        'labels': x['labels'],
                        'inputs': x['inputs'],
                        'metadata': x['metadata'],
                        'context_lengths': x['context_lengths'],
                        'batch_idx': x['batch_idx'],
                        'labels_text': x['labels_text'],
                    }
                    for x in output
                ],
                group=parallel_state.get_data_parallel_group(),
            )

            # Remove duplicate examples due to distributed sampler.
            inp_label_set = set()
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
            total_size = 0
            for rank in range(0, parallel_state.get_data_parallel_world_size()):
                for batch in gathered_outputs[rank]:
                    for pred, answer, input, metadata, pred_context_length, labels_text in zip(
                        batch['preds'],
                        batch['labels'],
                        batch['inputs'],
                        batch['metadata'],
                        batch['context_lengths'],
                        batch['labels_text'],
                    ):
                        context_length = len(self.tokenizer.text_to_ids(input))
                        text_answer, speech_answer = self.parse_decoder_outputs(
                            answer,
                            self.tokenizer.eos_id,
                            context_length,
                            self.cfg.data.train_ds.speech_pad_id,
                            self.cfg.data.train_ds.speech_eos_id,
                        )
                        key = input + self.tokenizer.ids_to_text(text_answer) + str(metadata)
                        total_size += 1
                        if key not in inp_label_set:
                            inp_label_set.add(key)

                            text_pred, speech_pred = self.parse_decoder_outputs(
                                torch.Tensor(pred),
                                self.tokenizer.eos_id,
                                pred_context_length,
                                self.cfg.data.train_ds.speech_pad_id,
                                self.cfg.data.train_ds.speech_eos_id,
                            )
                            text_pred_text = self.tokenizer.ids_to_text(text_pred)
                            deduplicated_outputs['preds'].append(text_pred_text.strip())
                            deduplicated_outputs['labels'].append(labels_text.strip())
                            text_answer_text = self.tokenizer.ids_to_text(text_answer)
                            deduplicated_outputs['text_answers'].append(text_answer_text.strip())
                            deduplicated_outputs['speech_preds'].append(speech_pred.cpu().numpy())
                            deduplicated_outputs['speech_answers'].append(speech_answer.cpu().numpy())

                            deduplicated_outputs['inputs'].append(input)
                            deduplicated_outputs['metadata'].append(metadata)
                            deduplicated_outputs['batch_idx'].append(batch['batch_idx'])

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
                if 'codec_model' not in self.additional_models:
                    from nemo.collections.tts.models import AudioCodecModel

                    codec_model = AudioCodecModel.restore_from(self.cfg.codec_model_path)
                    codec_model.to(self.device)
                    codec_model.eval()
                    self.additional_models['codec_model'] = codec_model
                    logging.info(f"Loaded Codec Model: {codec_model}")
                else:
                    codec_model = self.additional_models['codec_model']

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
                if 'asr_model' not in self.additional_models:
                    import nemo.collections.asr as nemo_asr

                    asr_model = nemo_asr.models.ASRModel.from_pretrained(self.cfg.asr_model_path).to(self.device)
                    asr_model.encoder.disable_torch_distributed = True  # For multi-gpu training validation
                    asr_model.eval()
                    self.additional_models['asr_model'] = asr_model
                    logging.info(f"Loaded ASR Model: {asr_model}")
                else:
                    asr_model = self.additional_models['asr_model']

                with torch.no_grad():
                    logging.info(f"Running ASR on speech preds")
                    speech_preds_transcribed = asr_model.transcribe(pred_wavs)[0]
                    speech_answers_transcribed = asr_model.transcribe(answer_wavs)[0]
                    deduplicated_outputs['speech_preds_transcribed'] = speech_preds_transcribed
                    deduplicated_outputs['speech_answers_transcribed'] = speech_answers_transcribed

            if run_mos:
                if 'squim_mos_model' not in self.additional_models:
                    from torchaudio.pipelines import SQUIM_SUBJECTIVE

                    squim_mos_model = SQUIM_SUBJECTIVE.get_model().to(self.device)
                    self.additional_models['squim_mos_model'] = squim_mos_model
                else:
                    squim_mos_model = self.additional_models['squim_mos_model']

                import torchaudio

                with torch.no_grad():
                    logging.info(f"Running MOS prediction")
                    pred_wavs_resampled = [
                        torchaudio.functional.resample(wav, 22050, 16000).unsqueeze(0) for wav in pred_wavs
                    ]
                    answer_wavs_resampled = [
                        torchaudio.functional.resample(wav, 22050, 16000).unsqueeze(0) for wav in answer_wavs
                    ]
                    squim_mos_scores = [
                        squim_mos_model(pred_wav, answer_wav)
                        for pred_wav, answer_wav in zip(pred_wavs_resampled, answer_wavs_resampled)
                    ]
                    deduplicated_outputs['mos_scores'] = squim_mos_scores

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
        for i in self.model.proj_head_dims:
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
        super().__init__(cfg, trainer)

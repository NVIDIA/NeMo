import itertools
import json
import os
from collections import OrderedDict

import numpy as np
import sacrebleu
import torch
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor

from nemo.collections.asr.parts.utils.eval_utils import remove_punctuations
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


class SumMultiEmbedding(LanguageModelEmbedding):
    """Language model embeddings with multiple tokens at each time step. The embeddings of the tokens of the same time step will be computed separately and then be summed together."""

    def forward(self, input_ids: Tensor, position_ids: Tensor, tokentype_ids: int = None) -> Tensor:
        embeddings = super().forward(input_ids, position_ids, tokentype_ids)
        return torch.sum(embeddings, axis=2)


class S2sMCoreGPTModel(MCoreGPTModel):

    def __init__(self, config: TransformerConfig, n_proj_heads: int, *args, **kwargs) -> None:
        super().__init__(config=config, *args, **kwargs)
        # TODO: confirm the state dict is loaded and stored
        self.output_layers = torch.nn.ModuleList(
            [
                tensor_parallel.ColumnParallelLinear(
                    config.hidden_size,
                    self.vocab_size,
                    config=config,
                    init_method=config.init_method,
                    bias=False,
                    skip_bias_add=False,
                    gather_output=not self.parallel_output,
                    skip_weight_param_allocation=self.pre_process and self.share_embeddings_and_output_weights,
                    embedding_activation_buffer=self.embedding_activation_buffer,
                    grad_output_buffer=self.grad_output_buffer,
                )
                for i in range(n_proj_heads)
            ]
        )
        self.n_proj_heads = n_proj_heads

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
        all_logits = [self.output_layer[i](hidden_states) for i in range(len(self.output_layer))]
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        all_logits[0], _ = self.output_layer(hidden_states, weight=output_weight)

        if labels is None:
            # [s b h] => [b s h]
            return [logits.transpose(0, 1).contiguous() for logits in all_logits]

        # labels[:, :, i]-sum(self.proj_head_dims[:i]) is the label for the i-th projection head
        # which shuold consider the offset of previous projection heads
        tokens_loss = torch.stack(
            [
                self.compute_language_model_loss(labels[:, :, i] - sum(self.proj_head_dims[:i]), all_logits[i])
                for i in range(self.n_proj_heads)
            ],
            axis=2,
        )
        tokens_loss = (
            tokens_loss
            * torch.FloatTensor(self.proj_head_loss_weights).to(tokens_loss.device)
            / sum(self.proj_head_loss_weights)
        )
        breakpoint()
        return tokens_loss


class S2sModularAudioGPTModel(ModularAudioGPTModel):
    """S2S version of Modularized speech GPT model."""

    def model_provider_func(self, pre_process, post_process):
        """Model depends on pipeline paralellism."""
        if self.mcore_gpt:
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
                n_proj_heads=self.cfg.get('n_proj_heads', 1),
            )

            if self.cfg.get('scale_positional_embedding', False):
                model.rotary_pos_emb.inv_freq = apply_rope_scaling(model.rotary_pos_emb.inv_freq)

            if self.cfg.get("apply_embedding_scaling", False) and parallel_state.is_pipeline_first_stage():
                extend_instance(model.embedding, EmbeddingScalingMixin)
        else:
            raise ValueError("S2S ModularAudioGPTModel requires Megatron-core GPT model.")
        return model

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        self.cfg = cfg
        super().__init__(cfg, trainer)
        if cfg.get('salm_model_path') is not None:
            torch_state_dict = torch.load(cfg.get('salm_model_path'))['state_dict']
            self.setup_complete = False
            # breakpoint()
            self.load_state_dict(torch_state_dict, strict=False)
            logging.info(f"loading from {cfg.get('salm_model_path')}: {torch_state_dict.keys()}")

        self.padded_vocab_size = cfg.s2s_vocab_size
        self.model.extend_embedding(self.padded_vocab_size)
        # print out params in more details
        self.summarize(max_depth=2)

    # change to add one more dimension
    def _shift_labels_by_emb_len(self, labels, label_lens, emb_lens, max_len, pad_token=0):
        """Shift labels to the right by the length of the audio embeddings."""
        shifted_labels = []
        for label, label_len, emb_len in zip(labels, label_lens, emb_lens):
            shifted_label = torch.full([max_len, self.cfg.proj_head_dims], pad_token, device=label.device)
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

        inputs_text = [self.tokenizer.ids_to_text(c.tolist()) for c in batch['source_texts']]
        labels_text = [self.tokenizer.ids_to_text(a.tolist()) for a in batch['target_texts']]
        preds_text = [
            self.tokenizer.ids_to_text(t[l.item() :][: data_cfg.get('tokens_to_generate')])
            for t, l in zip(output['token_ids'], batch['context_lengths'])
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
            preds_text = preds_text_cleaned
            labels_text = labels_text_cleaned

        if data_cfg.get("remove_text_pc", False):
            preds_text = [remove_punctuations(p.lower(), data_cfg.get("punctuations", None)) for p in preds_text]
            labels_text = [remove_punctuations(l.lower(), data_cfg.get("punctuations", None)) for l in labels_text]

        if data_cfg.get("log_every_n_steps", None) is not None:
            if batch_idx % data_cfg.log_every_n_steps == 0:
                logging.info(f"Input: `{inputs_text[0]}`")
                logging.info(f"Label: `{labels_text[0]}`")
                logging.info(f"Pred: `{preds_text[0]}`")

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
            'preds': preds_text,  # [str]
            'labels': batch['answers'],  # [str]
            'inputs': inputs_text,  # [str]
            'metadata': metadata,  # [dict]
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

    def parse_decoder_outputs(self, decoder_output, text_separator, speech_pad_id=1001, speech_eos_id=1004):
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
            speech_tokens.shape[0] * self.model.decoder_reduction_factor,
            speech_tokens.shape[1] // self.model.decoder_reduction_factor,
        )
        speech_tokens = speech_tokens.reshape(new_shape)
        return text_tokens, speech_tokens

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
                    {'preds': x['preds'], 'labels': x['labels'], 'inputs': x['inputs'], 'metadata': x['metadata']}
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
            }
            total_size = 0
            for rank in range(0, parallel_state.get_data_parallel_world_size()):
                for batch in gathered_outputs[rank]:
                    for pred, answer, input, metadata in zip(
                        batch['preds'], batch['labels'], batch['inputs'], batch['metadata']
                    ):
                        key = input + label + str(metadata)
                        total_size += 1
                        if key not in inp_label_set:
                            inp_label_set.add(key)
                            #  Remove leading BOS
                            pred = pred[1:]

                            text_pred, speech_pred = self.parse_decoder_outputs(
                                pred,
                                self.tokenizer.eos_id,
                                self.cfg.data.train_ds.speech_pad_id,
                                self.cfg.data.train_ds.speech_eos_id,
                            )
                            text_answer, speech_answer = self.parse_decoder_outputs(
                                answer,
                                self.tokenizer.eos_id,
                                self.cfg.data.train_ds.speech_pad_id,
                                self.cfg.data.train_ds.speech_eos_id,
                            )
                            deduplicated_outputs['preds'].append(
                                self.tokenizer.ids_to_text(text_pred.unsqueeze(0), self.tokenizer)
                            )
                            deduplicated_outputs['labels'].append(
                                self.tokenizer.ids_to_text(text_answer.unsqueeze(0), self.tokenizer)
                            )
                            deduplicated_outputs['speech_preds'].append(speech_pred.cpu().numpy())
                            deduplicated_outputs['speech_answers'].append(speech_answer.cpu().numpy())

                            deduplicated_outputs['inputs'].append(input)
                            deduplicated_outputs['metadata'].append(metadata)

            # Compute metric score
            metric_name = self.val_metric_name if mode == 'validation' else self.test_metric_name
            metric_label_key = self.val_metric_label_key if mode == 'validation' else self.test_metric_label_key
            if metric_name != 'loss':
                metric_log_key = self._determine_log_key(data_cfg, dataloader_idx, metric_name, mode)
                metric_fn = self.val_metric[0] if mode == 'validation' else self.test_metric[0]
                if metric_label_key in deduplicated_outputs['metadata'][0]:
                    labels = [m[metric_label_key] for m in deduplicated_outputs['metadata']]
                else:
                    labels = deduplicated_outputs['labels']

                # sacrebleu.corpus_bleu is commonly used which does not share
                # the same interface as other metrics. We handle it separately.
                if metric_name == 'bleu':
                    metric_result = torch.Tensor(
                        [sacrebleu.corpus_bleu(deduplicated_outputs['preds'], [labels]).score]
                    ).to(self.device)
                else:
                    for pred, label in zip(deduplicated_outputs['preds'], labels):
                        _ = metric_fn(pred, label)

                    metric_result = metric_fn.compute()

                if metric_name == 'rouge':
                    for k, v in metric_result.items():
                        if 'fmeasure' in k:
                            self.log(metric_log_key + f'_{k}', v.item(), sync_dist=True, batch_size=1)
                            logging.info(f"{mode} {metric_name} {k}: {v.item()}")
                    metric_result = metric_result['rouge1_fmeasure']
                else:
                    self.log(metric_log_key, metric_result.item(), sync_dist=True, batch_size=1)
                    logging.info(f"{mode} {metric_name}: {metric_result.item()}")

                metric_fn.reset()
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
        # speaker_contexts_path = os.path.join(output_dir, 'npy', 'speaker_contexts', output_file_path_prefix + "_speaker_context_{}.npy")
        speech_pred_path = os.path.join(
            output_dir, 'npy', 'speech_pred', output_file_path_prefix + "_speech_pred_{}.npy"
        )
        speech_answer_path = os.path.join(
            output_dir, 'npy', 'speech_answer', output_file_path_prefix + "_speech_answer_{}.npy"
        )

        output_file_path = output_file_path_prefix + "_inputs_preds_labels.jsonl"
        output_file_path = os.path.join(output_dir, output_file_path)
        with open(output_file_path, "w") as f_json:
            assert (
                len(outputs['inputs']) == len(outputs['preds']) == len(outputs['labels']) == len(outputs['metadata'])
            )
            for i, p, l, m, speech_pred, speech_answer in zip(
                outputs['inputs'],
                outputs['preds'],
                outputs['labels'],
                outputs['metadata'],
                outputs['speech_preds'],
                outputs['speech_answers'],
            ):
                json_string = {'input': i, 'pred_text': p, 'text': l}
                for k, v in m.items():
                    if k not in json_string:
                        json_string[k] = v
                f_json.write(json.dumps(json_string) + '\n')

                # np.save(speaker_contexts_path.format(i), speaker_context.cpu().numpy())
                np.save(speech_pred_path.format(i), speech_pred)
                np.save(speech_answer_path.format(i), speech_answer)

        logging.info(f'Predictions saved to {output_file_path}')

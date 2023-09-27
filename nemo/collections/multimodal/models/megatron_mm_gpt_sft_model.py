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

import itertools
from typing import Any, List, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.common.metrics import MetricStringToTorchMetric
from nemo.collections.nlp.data.language_modeling.megatron.base_dataset_utils import (
    get_datasets_weights_and_num_samples,
)
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    average_losses_across_data_parallel_group,
    get_all_params_for_weight_decay_optimization,
    get_ltor_masks_and_position_ids,
    get_params_for_weight_decay_optimization,
)

#from nemo.collections.nlp.modules.common.text_generation_utils import (
#    generate,
#    get_computeprob_response,
#    get_default_length_params,
#    get_default_sampling_params,
#    megatron_gpt_generate,
#)

from nemo.collections.multimodal.modules.common.mm_generation_utils import (
    generate,
    get_computeprob_response,
    get_default_length_params,
    get_default_sampling_params,
    megatron_gpt_generate,
)


from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
from nemo.collections.nlp.modules.common.megatron.language_model import Embedding
from nemo.collections.multimodal.data.audio_language_modeling.gpt_mm_sft_dataset import MMGPTSFTDataset
from nemo.utils import AppState, logging


try:
    from apex.transformer.pipeline_parallel.utils import (
        _reconfigure_microbatch_calculator,
        get_current_global_batch_size,
        get_micro_batch_size,
        get_num_microbatches,
    )

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


try:
    from megatron.core import InferenceParams, parallel_state
    from megatron.core.models.gpt import GPTModel as MCoreGPTModel
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.transformer.module import Float16Module as MCoreFloat16Module
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.utils import init_method_normal, scaled_init_method_normal

    # TODO @tmoon: Use once available in Megatron-LM
    # from megatron.core.pipeline_parallel.schedules import DataIteratorList

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    TransformerConfig = ApexGuardDefaults

    HAVE_MEGATRON_CORE = False

__all__ = ['MegatronMMGPTSFTModel']


class MegatronMMGPTSFTModel(MegatronGPTSFTModel):
    """
    Megatron MultiModal GPT Supervised Fine-Tuning
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)
        
        # audio related cfg
        self.num_audio_codebooks = cfg.get('num_audio_codebooks', 1)
        self.audio_codebook_size = cfg.get('audio_codebook_size', 1024)     
        self.audio_token_offset = cfg.get('audio_token_offset', 0)     
        
        audio_embedding_layers = []
        for _ in range(self.num_audio_codebooks-1):
            audio_embedding_layers.append(
                Embedding(
                    config=self.model_parallel_config,
                    hidden_size=self.cfg.hidden_size,
                    vocab_size=self.audio_codebook_size,
                    max_sequence_length=self.cfg.max_position_embeddings,
                    init_method=init_method_normal(self.cfg.get('init_method_std', 0.02)),
                    num_tokentypes=0,
                    embedding_dropout_prob=self.cfg.get('hidden_dropout', 0.1),
                    position_embedding_type=None,
                    fp32_residual_connection=self.cfg.get('fp32_residual_connection', False),
                    dtype=self.model.module.language_model.dtype,
                )
            )
        
        # not sure about the following placement should it be 
        # self.audio_embedding_layers or self.model.module.audio_embedding_layers
        # are there any places where it is self.model.eval..? 
        self.model.module.audio_embedding_layers = torch.nn.ModuleList(audio_embedding_layers)

    def _build_dataset(self, data_cfg, is_train=True):
        datasets = []
        # Determine if we are using a single dataset or a list of datasets.
        is_list_config = isinstance(data_cfg.file_names, ListConfig)
        if not is_list_config:
            raise ValueError(f"SFT train/validation datasets must be provided as a list of individual JSONL files.")

        if is_train:
            # Construct the data prefix list for `get_datasets_weights_and_num_samples()`
            # that is of the format [weight1,file_name1,weight2,file_name2,...]
            if data_cfg.concat_sampling_probabilities is None or not isinstance(
                data_cfg.concat_sampling_probabilities, ListConfig
            ):
                raise ValueError(
                    (
                        f"concat_sampling_probabilities must be a ListConfig with the same number of files in file_names."
                        f"Found: {data_cfg.concat_sampling_probabilities}"
                    )
                )

            if len(data_cfg.get('concat_sampling_probabilities', None)) != len(data_cfg.file_names):
                raise ValueError(
                    (
                        f"concat_sampling_probabilities must be of the same size as file_names.",
                        f"Provided size {len(data_cfg.concat_sampling_probabilities)}, number of datasets {len(data_cfg.file_names)}",
                    )
                )

            data_prefix = []
            for weight, prefix in zip(data_cfg.concat_sampling_probabilities, data_cfg.file_names):
                data_prefix.append(weight)
                data_prefix.append(prefix)

            if self.trainer.max_steps is None or self.trainer.max_steps <= 0:
                raise ValueError(
                    f'Trainer max_steps must be set to a positive integer. Found {self.trainer.max_steps}'
                )
            num_train_samples = [self.trainer.max_steps * data_cfg.global_batch_size]
            _, _, num_train_samples_per_dataset = get_datasets_weights_and_num_samples(data_prefix, num_train_samples)
            num_train_samples_after_blend = sum([x[0] for x in num_train_samples_per_dataset])
        else:
            num_train_samples_per_dataset = [[None]] * len(data_cfg.file_names)

        for file_path, num_samples in zip(data_cfg.file_names, num_train_samples_per_dataset):
            dataset = MMGPTSFTDataset(
                file_path=file_path,
                tokenizer=self.tokenizer,
                max_seq_length=data_cfg.max_seq_length,
                min_seq_length=data_cfg.min_seq_length,
                add_bos=data_cfg.get('add_bos', False),
                add_eos=data_cfg.get('add_eos', True),
                add_sep=data_cfg.get('add_sep', False),
                sep_id=self.sep_id,
                max_num_samples=num_samples[0],
                seed=data_cfg.get('seed', 1234),
                label_key=data_cfg.get('label_key', 'answer'),
                answer_only_loss=self.cfg.get('answer_only_loss', True),
                truncation_field=data_cfg.get('truncation_field', 'context'),
                pad_to_max_length=data_cfg.get('pad_to_max_length', False),
                index_mapping_dir=data_cfg.get('index_mapping_dir', None),
                prompt_template=data_cfg.get('prompt_template', None),
                virtual_tokens=self.virtual_tokens,
                tokens_to_generate=data_cfg.get(
                    'tokens_to_generate', 0
                ),  # used at inference time to allocate tensor positions for tokens that will be generated by inf procedure.
                memmap_workers=data_cfg.get(
                    'memmap_workers', None
                ),  # used to set num. of workers to create the memmap index files
                hf_dataset=data_cfg.get(
                    'hf_dataset', False
                ),  # Whether to load the json file with the HuggingFace dataset. otherwise, will load the jsonl file with the JSONLMemMapDataset.
                truncation_method=data_cfg.get(
                    'truncation_method', 'right'
                ),  # used to choose truncation method. Options: ['random', 'left', 'right']
                num_audio_codebooks=data_cfg.get('num_audio_codebooks', 1),
                audio_codebook_size=data_cfg.get('audio_codebook_size', 1024),
                audio_token_offset=data_cfg.get('audio_token_offset', 256003),
            )
            datasets.append(dataset)

        if is_train:
            dataset = BlendableDataset(
                datasets=datasets, weights=data_cfg.concat_sampling_probabilities, size=num_train_samples_after_blend
            )
            return dataset
        else:
            return datasets

    
    def _get_audio_embedding_per_codebook(self, embedding_module, token_ids):
        # dumping the hacky logic of adjusting token_ids here
        token_ids = token_ids - self.audio_token_offset
        # now some of token_ids are negative, so we need to mask them
        # and set them to 0
        mask = token_ids < 0
        token_ids = token_ids * (~mask).to(token_ids.dtype)
        # now we can pass token_ids to embedding module
        # however, this 0 conflicts with real 0 audio token id
        # handling it with 'additinal_tokens_mask' in _get_encoder_input

        return embedding_module(token_ids, position_ids=None, token_type_ids=None)


    def _get_encoder_input(self, batch):
        # first codebook (text + first audio codebook)
        input_ids = batch['tokens']
        position_ids = batch['position_ids']
        enc_input = [self.model.module.language_model.embedding(input_ids, position_ids=None, token_type_ids=None)]        # dont pass position_ids here
        # the Embedding module from language_model.py transposes batch and sequence and gives [s b h]

        # remaining audio codebooks
        additional_tokens = batch['additional_tokens']
        num_additional_codebooks = additional_tokens.shape[-1]
        for i in range(num_additional_codebooks):    # additional tokens are [B, T, num_codebooks]
            additional_tokens_mask = batch['additional_tokens_mask'][:, :, i].transpose(0, 1).unsqueeze(-1)  # [s b 1]
            additional_tokens_embedding = self._get_audio_embedding_per_codebook(self.model.module.audio_embedding_layers[i], additional_tokens[:, :, i])
            # mask with additional_tokens_mask
            additional_tokens_embedding = additional_tokens_embedding * additional_tokens_mask.to(additional_tokens_embedding.dtype) 
            enc_input.append(additional_tokens_embedding)

        # stack and sum along hidden dimension
        enc_input = torch.stack(enc_input, dim=-1).sum(dim=-1)

        # only 'learned_absolute' pos_emb are inside Embedding module,
        # other are in TransformerLanguageModel module  after embedding layer
        # and don't need to be added here
        if self.cfg.position_embedding_type == 'learned_absolue':
            raise NotImplementedError
        
        return enc_input
        
        

    def get_forward_output_and_loss_func(self, validation_step=False):
        def fwd_output_and_loss_func(dataloader_iter, model, checkpoint_activations_all_layers=None):

            # Get data batch
            batch = next(dataloader_iter)

            # Transfer needed data to GPU
            required_keys = set()
            if parallel_state.get_pipeline_model_parallel_world_size() == 1:
                required_keys.update(batch.keys())
            else:
                required_keys.add('attention_mask')
                if parallel_state.is_pipeline_first_stage():
                    required_keys.update(('tokens', 'position_ids'))
                if parallel_state.is_pipeline_last_stage():
                    required_keys.update(('labels', 'loss_mask'))
            if self.get_attention_mask_from_fusion:
                required_keys.remove('attention_mask')
            batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in batch.items()}

            # Model forward pass

            # If only one codebook pass through GPT pipeline in the form of audio tokens
            # @kpuvvada: combine the below if-else by using _get_encoder_inputs which will
            # return input_ids, position_ids, encoder_input for both cases
            if self.num_audio_codebooks == 1:
                forward_args = {
                    'input_ids': batch['tokens'],
                    'position_ids': batch['position_ids'],
                    'attention_mask': batch['attention_mask'],
                    'labels': batch['labels'],
                    'loss_mask': batch['loss_mask'],
                }
            else:
                # compute and pass encoder_input (don't need input_ids and position_ids)
                forward_args = {
                    'input_ids': None,
                    'position_ids': None,
                    'encoder_input': self._get_encoder_input(batch),
                    'attention_mask': batch['attention_mask'],
                    'labels': batch['labels'],
                    'loss_mask': batch['loss_mask'],
                }

            if not self.mcore_gpt:
                forward_args['checkpoint_activations_all_layers'] = checkpoint_activations_all_layers
                if not self.use_loss_mask:
                    forward_args.pop('loss_mask')
            else:
                # TODO: @eharper can we add this to mcore?
                forward_args.pop('loss_mask')
            output_tensor = model(**forward_args)
            if isinstance(output_tensor, tuple):
                output_tensor = output_tensor[0]

            def loss_func(output_tensor):
                # Loss for a micro-batch (ub)
                loss_for_ub = self.loss_func(batch['loss_mask'], output_tensor)
                if validation_step and not self.cfg.data.get('validation_drop_last', True):
                    num_valid_tokens_in_ub = batch['loss_mask'].sum()
                    if loss_for_ub.isnan():
                        assert batch['loss_mask'].count_nonzero() == 0, 'Got NaN loss with non-empty input'
                        loss_sum_for_ub = torch.zeros_like(num_valid_tokens_in_ub)
                    else:
                        loss_sum_for_ub = num_valid_tokens_in_ub * loss_for_ub

                    loss_sum_and_ub_size_all_gpu = torch.cat(
                        [
                            loss_sum_for_ub.clone().detach().view(1),
                            torch.tensor([num_valid_tokens_in_ub]).cuda().clone().detach(),
                        ]
                    )
                    # Could potentially reduce num_valid_samples_in_microbatch and use that to aggregate instead of len(self._validation_ds)
                    torch.distributed.all_reduce(
                        loss_sum_and_ub_size_all_gpu, group=parallel_state.get_data_parallel_group()
                    )
                    return loss_for_ub, {'loss_sum_and_ub_size': loss_sum_and_ub_size_all_gpu}
                else:
                    reduced_loss = average_losses_across_data_parallel_group([loss_for_ub])
                    return loss_for_ub, {'avg': reduced_loss}

            return output_tensor, loss_func

        return fwd_output_and_loss_func


    def get_forward_output_only_func(self):
        def fwd_output_only_func(dataloader_iter, model):
            batch = next(dataloader_iter)
            extra_arg = {}
            if len(batch) == 3:
                batch = [x.cuda() for x in batch]
                tokens, attention_mask, position_ids = batch
                attention_mask = attention_mask[0:1]
            else:
                (
                    tokens,
                    attention_mask,
                    position_ids,
                    set_inference_key_value_memory,
                    inference_max_sequence_len,
                    additional_tokens,
                    additional_tokens_mask,
                ) = batch
                tokens = tokens.cuda()
                position_ids = position_ids.cuda()
                if attention_mask is not None:
                    attention_mask = attention_mask.cuda()
                    attention_mask = attention_mask[0:1]
                if self.mcore_gpt:
                    # if first step, then clear KV cache, otherwise reuse inference_paarms
                    if set_inference_key_value_memory[0].item():
                        self.inference_params = InferenceParams(
                            max_batch_size=tokens.size(0), max_sequence_length=inference_max_sequence_len[0].item()
                        )
                    extra_arg['inference_params'] = self.inference_params
                else:
                    extra_arg['set_inference_key_value_memory'] = set_inference_key_value_memory[0].item()
                    extra_arg['inference_max_sequence_len'] = inference_max_sequence_len[0].item()
            
            additional_tokens = additional_tokens.cuda()
            additional_tokens_mask = additional_tokens_mask.cuda()

            # put everythin in batch
            batch = {
                'tokens': tokens,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'additional_tokens': additional_tokens,
                'additional_tokens_mask': additional_tokens_mask,
            }
            if self.num_audio_codebooks == 1:
                raise NotImplementedError
            
            forward_args = {
                'input_ids': None,
                'position_ids': None,
                'encoder_input': self._get_encoder_input(batch),
                'attention_mask': batch['attention_mask'],
            }

            forward_args.update(extra_arg)
            output_tensor = model(**forward_args)
            if isinstance(output_tensor, tuple):
                output_tensor = output_tensor[0]

            # Advance inference sequence offset.
            if self.inference_params:
                # if last stage, then (final) output is [b, s, h], otherwise it's [s, b, h]
                if parallel_state.is_pipeline_last_stage():
                    self.inference_params.sequence_len_offset += output_tensor.size(1)
                else:
                    self.inference_params.sequence_len_offset += output_tensor.size(0)

            def id_func(output_tensor):
                return output_tensor, {'logits': output_tensor}

            return output_tensor, id_func

        return fwd_output_only_func    
    
    
    def setup_metric(self, data_cfg):
        metric_name = "exact_string_match"
        if not hasattr(data_cfg, "metric"):
            metric = MetricStringToTorchMetric["exact_string_match"]
        else:
            if not hasattr(data_cfg.metric, "name"):
                raise ValueError("Metric name is not provided in the metric config.")
            if data_cfg.metric.name == "loss":
                return None, "loss"
            if data_cfg.metric.name not in MetricStringToTorchMetric:
                raise KeyError(
                    f"{data_cfg.metric.name} is not supported. List of supported metrics: {MetricStringToTorchMetric.keys()}"
                )
            if data_cfg.metric.name in self._metrics_require_string2category_map:
                if data_cfg.metric.average is None:
                    raise ValueError(
                        f"{data_cfg.metric.name} requires specifying whether you want to compute a micro or macro average. Found None."
                    )
            if (
                data_cfg.metric.get('labels_are_strings', False)
                and data_cfg.metric.name in self._metrics_require_string2category_map
            ):
                if data_cfg.metric.num_classes is None:
                    raise ValueError(
                        "Number of classes is not provided in the metric section within the data config. "
                        f"Please provide the number of classes in the data config to use the {data_cfg.metric.name} metric."
                    )
                if data_cfg.metric.get('class_labels', None) is None or not isinstance(
                    data_cfg.metric.get('class_labels', None), ListConfig
                ):
                    raise ValueError(
                        "Class labels are not provided properly in the metric section witnin the data config. "
                        f"Please provide the class labels as a list of strings in the data config to use the {data_cfg.metric.name} metric."
                    )
                if len(data_cfg.metric.get('class_labels', None)) != data_cfg.metric.num_classes:
                    raise ValueError(
                        f"Number of class labels {len(data_cfg.metric.get('class_labels', None))} does not match `num_classes` : {data_cfg.metric.num_classes}"
                    )

            metric_name = data_cfg.metric.name
            metric = MetricStringToTorchMetric[metric_name]

            if isinstance(data_cfg.file_names, ListConfig):
                if 'rouge' not in data_cfg.metric.name and 'wer' not in data_cfg.metric.name:
                    metric = [
                        metric(average=data_cfg.metric.average, num_classes=data_cfg.metric.num_classes)
                        for _ in range(len(data_cfg.file_names))
                    ]
                else:
                    metric = [metric() for _ in range(len(data_cfg.file_names))]
            else:
                if 'rouge' not in data_cfg.metric.name and 'wer' not in data_cfg.metric.name:
                    metric = [metric(average=data_cfg.metric.average, num_classes=data_cfg.metric.num_classes)]
                else:
                    metric = [metric()]

        return metric, metric_name
    
    # not using this any more - delete
    def predict_step_deprecated(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        # generate either takes list of text prompts or a tuple of (context_ids, context_length)
        if isinstance(batch, list):
            inference_inputs = batch
        else:
            inference_inputs = (batch['contexts'].cuda(), batch['context_lengths'].cuda())
        
        inference_config = self.get_inference_config()
        if inference_config is None:
            return None
        else:
            # need to overwrite some configuration, make it immutable
            inference_config = inference_config.copy()
            compute_logprob = inference_config['compute_logprob']
            if compute_logprob:
                del inference_config['compute_logprob']
                inference_config['inputs'] = inference_inputs
                inference_config['tokens_to_generate'] = 1
                inference_config['all_probs'] = True
                inference_config["add_BOS"] = False
                inference_config['greedy'] = True
                response = generate(self, **inference_config)
                compute_prob_response = get_computeprob_response(self.tokenizer, response, batch)
                return compute_prob_response
            else:
                del inference_config['compute_logprob']
                inference_config['inputs'] = inference_inputs
                response =  generate(self, **inference_config)
                
                # accumulate ground truth and predictions in the form of text
                gt_text = [
                    self.tokenizer.ids_to_text(t.numpy()[l.item() :]) for t, l in zip(batch['tokens'], batch['context_lengths'])
                ] 
                pred_text = [
                    self.tokenizer.ids_to_text(t[l.item() :]) for t, l in zip(response['token_ids'], batch['context_lengths'])
                ]
                
                response['gt_text'] = gt_text
                response['pred_text'] = pred_text
                return response
    

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        inference_config = self.get_inference_config()
        # need to overwrite some configuration, make it immutable
        inference_config = inference_config.copy()
        global_batch_size_per_gpu = batch['tokens'].size(0)
        num_micro_batches_before_decode = get_num_microbatches()

        compute_logprob = inference_config.get('compute_logprob', False)
        if compute_logprob:
            inference_config['inputs'] = batch
            inference_config['tokens_to_generate'] = 1
            inference_config['all_probs'] = True
            inference_config["add_BOS"] = False
            inference_config['greedy'] = True
            response = generate(self, **inference_config)
            response = get_computeprob_response(self.tokenizer, response, batch)
        else:
            # for megatron_gpt_eval.py
            if isinstance(batch, list):
                inference_config['inputs'] = batch
            else:
                # peft_eval.py
                inference_config['inputs'] = (
                    batch['contexts'].cuda(),   # [b s]
                    batch['context_lengths'].cuda(),    # [b]
                    batch['additional_contexts'].cuda(),    # [b, s, nq-1]
                    batch['additional_contexts_mask'].cuda(),   # [b, s, nq-1]
                    )
            response = generate(self, **inference_config)

        app_state = AppState()
        _reconfigure_microbatch_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=global_batch_size_per_gpu * parallel_state.get_data_parallel_world_size(),
            micro_batch_size=global_batch_size_per_gpu // num_micro_batches_before_decode,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )

        return response
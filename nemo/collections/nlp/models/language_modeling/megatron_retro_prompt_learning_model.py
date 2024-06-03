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

import os
import re
from functools import partial
from typing import Any, List, Optional, Union
import itertools
from omegaconf import OmegaConf

import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import open_dict
from pytorch_lightning.trainer.trainer import Trainer
from torch import Tensor

from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
from nemo.collections.nlp.data.language_modeling.megatron.gpt_prompt_learning_dataset import GPTPromptLearningDataset
from nemo.collections.nlp.data.language_modeling.megatron.retro_prompt_learning_dataset import RetroPromptLearningDataset
from nemo.collections.nlp.modules.common.text_generation_strategy import model_inference_strategy_dispatcher
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.models.language_modeling.megatron_retrieval_model import MegatronRetrievalModel
from nemo.collections.nlp.models.language_modeling.megatron_base_prompt_learning_model import (
    MegatronBasePromptLearningModel,
)
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common import (
    PromptEncoder,
    PromptEncoderType,
    VirtualPromptPlaceholderToken,
    VirtualPromptSource,
    VirtualPromptStyle,
)
from nemo.collections.nlp.modules.common.text_generation_utils import generate
from nemo.collections.nlp.modules.common.megatron.utils import build_position_ids
from nemo.collections.nlp.modules.common.text_generation_utils import (
    get_default_length_params,
    get_default_sampling_params,
    megatron_gpt_generate,
    get_computeprob_response,
)
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam, TextGeneration, OutputType
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.utils import AppState, logging
from nemo.collections.nlp.models.language_modeling.megatron_fused_retro import MegatronFusedRetrievalLoraModel, MegatronFusedRetrievalAdapterModel
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_iterator_k_split,
)

try:
    # from apex.transformer import parallel_state, tensor_parallel
    # from apex.transformer.pipeline_parallel.schedules.fwd_bwd_no_pipelining import forward_backward_no_pipelining
    # from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_without_interleaving import (
    #     forward_backward_pipelining_without_interleaving,
    # )
    # from apex.transformer.pipeline_parallel.utils import _reconfigure_microbatch_calculator, get_micro_batch_size
    from apex.transformer.pipeline_parallel.utils import get_micro_batch_size, get_num_microbatches

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


__all__ = ['MegatronRetroPromptLearningModel']


class MegatronRetroPromptLearningModel(MegatronBasePromptLearningModel):
    """
    Model class for prompt-tuning or p-tuning a pretrained Megatron GPT model. 

    Prompt Tuning initalizes virtual prompt embeddings directly from a copy of
    certain token embeddings from the the pretrained GPT model's vocabulary
    and directly tunes these embedding weights. The token embeddings used in 
    initalization are specified by the user in the config file. The model can 
    be prompt-tuned for multiple tasks at once. virtual prompts are stored in a 
    prompt table and can be added or deleted without disrupting virtual prompts 
    for other tasks. 

    P-tuning initializes an LSTM encoder model that generates virtual prompt
    embeddings for every task. Each task shares the same encoder. After ptuning
    is compelete, the learned virtual prompts can be saved to the prompt table
    using add_ptuned_prompts_to_prompt_table(). Thus, if a user wants to add a 
    new virtual prompt via p-tuning, they do not need to retrain on all previous 
    tasks. This gives p-tuning the same task flexiblity as prompt-tuning.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)
        self.init_model(cfg, trainer)

    def init_model(self, cfg: DictConfig, trainer: Trainer):
        self.cfg = cfg
        save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(cfg.get('language_model_path')):
            save_restore_connector.model_extracted_dir = cfg.get('language_model_path')

        if cfg.get('peft', False):
            if cfg.task_templates[0].get('chat_type', True):
                cfg.task_templates[0].prompt_template = "<|VIRTUAL_PROMPT_0|> User: Answer the following question with a short span. {question}\n\nAssistant: The answer is {answer}"
            else:
                cfg.task_templates[0].prompt_template='<|VIRTUAL_PROMPT_0|> {question} {answer}'
            frozen_model_cfg = MegatronRetrievalModel.restore_from(
                cfg.get('language_model_path'),
                trainer=trainer,
                return_config=True,
                save_restore_connector=save_restore_connector,
                strict=False
            )
        else:
            if cfg.task_templates[0].get('chat_type', True):
                cfg.task_templates[0].prompt_template = "User: Answer the following question with a short span. {question}\n\nAssistant: The answer is {answer}"
            else:
                cfg.task_templates[0].prompt_template=' {question} {answer}'
            if self.cfg.virtual_prompt_style == 'no-prompts':
                frozen_model_cfg = MegatronRetrievalModel.restore_from(
                    cfg.get('language_model_path'),
                    trainer=trainer,
                    return_config=True,
                    save_restore_connector=save_restore_connector,
                    strict=False
                )
            else:
                if not cfg.adapter_tuning.get("adapter_key"):
                    frozen_model_cfg = MegatronFusedRetrievalAdapterModel.restore_from(
                        cfg.get('language_model_path'), trainer=trainer, return_config=True, save_restore_connector=save_restore_connector, strict=False
                    )
                else:
                    frozen_model_cfg = MegatronFusedRetrievalLoraModel.restore_from(
                        cfg.get('language_model_path'), trainer=trainer, return_config=True, save_restore_connector=save_restore_connector, strict=False
                    )

                frozen_model_cfg.adapter_tuning = cfg.adapter_tuning

        # cfg.restore_from_path = '/home/aficek/software/playground/retro_convert/gpt3-800m-pretraining-retro-fitting/converted2/mp_rank_00'
        # frozen_model_cfg.tokenizer = cfg.model.tokenizer
        frozen_model_cfg.data = cfg.data
        
        frozen_model_cfg.optim = cfg.optim
        frozen_model_cfg.restore_from_path = cfg.restore_from_path
        # frozen_model_cfg.eval = cfg.model.eval
        frozen_model_cfg.add_position_embedding = cfg.add_position_embedding
        frozen_model_cfg.global_batch_size = cfg.global_batch_size
        frozen_model_cfg.micro_batch_size = cfg.micro_batch_size
        frozen_model_cfg.precision = trainer.precision

        frozen_model_cfg.task_templates = cfg["task_templates"]
        if self.trainer.precision == 'bf16':
            self.autocast_dtype = torch.bfloat16
        elif int(self.trainer.precision) == 32:
            self.autocast_dtype = torch.float
        elif int(self.trainer.precision) == 16:
            self.autocast_dtype = torch.half

        if "shape_file" in frozen_model_cfg:
            frozen_model_cfg.pop("shape_file")

        print(frozen_model_cfg)
        if cfg.get('peft', False) or cfg.virtual_prompt_style == 'no-prompts':
            self.frozen_model = MegatronRetrievalModel.restore_from(
                cfg.get('language_model_path'),
                trainer=trainer,
                save_restore_connector=save_restore_connector,
                override_config_path=frozen_model_cfg,
                strict=False,
            ).to(dtype=self.autocast_dtype)


            self.word_embeddings = self.frozen_model.model.encoder_embedding.word_embeddings
            if hasattr(self.frozen_model.model.encoder_embedding, "position_embeddings"):
                self.pos_embeddings = self.frozen_model.model.encoder_embedding.position_embeddings
            else:
                self.pos_embeddings = None

            OmegaConf.set_struct(self.cfg, False)  # Temporarily disable struct mode
            self.cfg['hidden_size'] = self.frozen_model.cfg.hidden_size
            OmegaConf.set_struct(self.cfg, True)
            self.enable_autocast = True
        else:
            if not cfg.adapter_tuning.get("adapter_key"):
                self.frozen_model = MegatronFusedRetrievalAdapterModel(frozen_model_cfg, trainer)
            else:
                self.frozen_model = MegatronFusedRetrievalLoraModel(frozen_model_cfg, trainer)
            self.pos_embeddings = self.cfg.get('add_position_embedding')

            OmegaConf.set_struct(self.cfg, False)  # Temporarily disable struct mode
            self.cfg['hidden_size'] = self.frozen_model.cfg.hidden_size
            OmegaConf.set_struct(self.cfg, True)
            self.enable_autocast = True

        self.megatron_amp_o2 = self.cfg.get('megatron_amp_O2', False)
        self.pipeline_parallel = self.cfg.get('pipeline_model_parallel_size', 1) > 1
        self.tokenizer = self.frozen_model.tokenizer
        self.hidden_size = self.frozen_model.cfg.hidden_size
        self.existing_tasks = list(self.cfg.get('existing_tasks', []))
        self.new_tasks = list(self.cfg.get('new_tasks', []))
        with open_dict(self.cfg):
            self.cfg.existing_tasks = (
                self.existing_tasks + self.new_tasks
            )  # TODO: for backward compatibility (@adithyare) in general these tasks lists should be depricated

        self.virtual_prompt_style = VirtualPromptStyle(cfg.virtual_prompt_style)
        self.model_type = ModelType.encoder_or_decoder

        if self.pipeline_parallel:
            assert (
                self.cfg.optim.sched.get("min_lr", 0.0) == 0.0
            ), "Minimum lr must be 0.0 when pipeline parallel size is > 1"

        # Load templates for assigning virtual prompt token positions
        self.load_task_templates(self.cfg.task_templates)
        self.padded_vocab_size = self.frozen_model.padded_vocab_size

        # Prepare pseudo token ids for virtual/virtual prompt tokens
        self.pseudo_tokens = get_pseudo_tokens(self.max_virtual_tokens)
        if isinstance(self.tokenizer, SentencePieceTokenizer):
            if not self.tokenizer.legacy:
                self.tokenizer.pad_token = self.tokenizer.ids_to_tokens([self.tokenizer.pad_id])[0]
                self.tokenizer.bos_token = self.tokenizer.ids_to_tokens([self.tokenizer.bos_id])[0]
                self.tokenizer.eos_token = self.tokenizer.ids_to_tokens([self.tokenizer.eos_id])[0]
                self.tokenizer.legacy = True
            self.tokenizer.add_special_tokens(self.pseudo_tokens)
        else:
            self.tokenizer.add_special_tokens({'additional_special_tokens': self.pseudo_tokens})
        self.pseudo_token_ids = self.tokenizer.tokens_to_ids(self.pseudo_tokens)
        self.pseudo_token_ids_start = self.pseudo_token_ids[0] if self.pseudo_token_ids else None
        self.pad_token_id = self.tokenizer.pad_id if self.tokenizer.pad_id is not None else self.tokenizer.unk_id

        # P-Tuning uses an LSTM Encoder to produce virtual token embeddings
        if self.virtual_prompt_style == VirtualPromptStyle.P_TUNING:
            self.virtual_prompt_source = VirtualPromptSource.PROMPT_ENCODER
        elif self.virtual_prompt_style == VirtualPromptStyle.NO_PROMPT:
            self.virtual_prompt_source = VirtualPromptSource.NO_PROMPT
        else:
            raise ValueError(f"\nvirtual prompt style '{cfg.virtual_prompt_style}.'")

        self._reduced_loss_buffer = []
        self._inference_config = None

        # make sure the default pytorch lightning gradient clipping in the basemodel
        self.grad_clip_pl_default = True
        self.lowest_val_loss = None
        self.prompt_encoder = None

    def first_stage_of_pipeline(self):
        # return self.frozen_model.model.pre_process
        if self.cfg.get('peft', False) or self.cfg.virtual_prompt_style == 'no-prompts':
            return self.frozen_model.model.pre_process
        else:
            return self.frozen_model.model.model.pre_process

    def forward(
        self,
        input_ids,
        input_attn_mask,
        retrieved_ids,
        retrieved_attn_mask,
        token_type_ids=None,
        labels=None,
        input_emb=None,
        position_ids=None,
        inference = True,
        set_inference_key_value_memory=False,
        inference_max_sequence_len=None,

    ):
        """
        Special forward method for p-tuning/prompt-tuning pretrained
        GPT style models. Bypasses the vocab token preprocessing done
        in the MegatronGPT class.
        """
        # Get embeddings for text tokens and insert virtual token embeddings
        if self.cfg.get('peft', False) and self.first_stage_of_pipeline() and not (set_inference_key_value_memory==False and inference_max_sequence_len is not None): # for inference, only when predicting the first token should we add virtual embeddings
            # pad strategy 3
            encoder_input = self.make_encoder_input(input_ids, position_ids, inference)
            encoder_input = encoder_input.transpose(0, 1).contiguous()
            if self.cfg.get("sequence_parallel", False):
                encoder_input = tensor_parallel.mappings.scatter_to_sequence_parallel_region(encoder_input)
        else:
            encoder_input = None

        if encoder_input is not None:
            encoder_input = encoder_input.transpose(0, 1).contiguous()
        
        if self.autocast_dtype == torch.float32:
            output = self.frozen_model.model(
                input_ids=input_ids,
                input_attn_mask=input_attn_mask,
                retrieved_ids=retrieved_ids,
                retrieved_attn_mask=retrieved_attn_mask,
                token_type_ids=token_type_ids,
                labels=labels,
                input_emb=encoder_input,
                position_ids=position_ids,
                set_inference_key_value_memory=set_inference_key_value_memory,
                inference_max_sequence_len=inference_max_sequence_len,
            )
        else:
            if self.cfg.get('peft', False) or self.cfg.get('virtual_prompt_style', None) == 'no-prompts':
                output = self.frozen_model.model(
                    input_ids=input_ids,
                    input_attn_mask=input_attn_mask,
                    retrieved_ids=retrieved_ids,
                    retrieved_attn_mask=retrieved_attn_mask,
                    token_type_ids=token_type_ids,
                    labels=labels,
                    input_emb=encoder_input,
                    position_ids=position_ids,
                    set_inference_key_value_memory=set_inference_key_value_memory,
                    inference_max_sequence_len=inference_max_sequence_len,
                )
            else:
                output = self.frozen_model.model(
                    input_ids=input_ids,
                    input_attn_mask=input_attn_mask,
                    retrieved_ids=retrieved_ids,
                    retrieved_attn_mask=retrieved_attn_mask,
                    token_type_ids=token_type_ids,
                    labels=labels,
                    input_emb=encoder_input,
                    position_ids=position_ids)


        return output
        
    def make_encoder_input(self, input_ids, position_ids, inference):
        batch_size, _ = input_ids.shape
        virtual_token_embeds = self.prompt_encoder(batch_size=batch_size, use_cached_reps=inference)
        # if pad strategy 3 for retro, need to find out virtual_token_locations:
        # actually this works for pad strategy 1 & 2 as well?
        discrete_token_ids = input_ids.clone()
        discrete_token_ids[(input_ids >= self.pseudo_token_ids_start)] = self.pad_token_id
        discrete_token_embeds = self.word_embeddings(discrete_token_ids).clone()
        # Find the indicies where virtual tokens should be inserted
        virtual_token_locations = input_ids >= self.pseudo_token_ids_start
        # Create index template specifying where virtual token embeddings should be placed
        _, _, embedding_size = discrete_token_embeds.shape
        virtual_token_index = virtual_token_locations.nonzero().reshape((batch_size, -1, 2))[:, :, 1][:, :, None]
        virtual_token_index = virtual_token_index.expand(
            batch_size, self.prompt_encoder.total_virtual_tokens, embedding_size
        )
        # Make sure discrete_token_embeds and virtual_token_embeds share the same dtype
        discrete_token_embeds = discrete_token_embeds.type(virtual_token_embeds.dtype)
        if self.pos_embeddings:
            position_embeddings = self.pos_embeddings(position_ids)
            discrete_token_embeds = discrete_token_embeds + position_embeddings
        # Insert virtual token embeddings where they belong amoung the discrete token embeddings
        discrete_token_embeds.scatter_(1, virtual_token_index, virtual_token_embeds)
        return discrete_token_embeds
    
    def fwd_bwd_step(self, dataloader_iter, batch_idx, forward_only):
        """
            Dataloader produces a global batch which is turned into a list of microbatches.
            The list of microbatches is then piped through the pipeline using Apex fwd/bwd functions.
        """
        # Get seq length of batch
        batch = next(dataloader_iter)
        _, seq_length = batch[0].shape
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
            dtype=self.autocast_dtype,
            grad_scaler=None, # may need to revert self.trainer.precision_plugin.scaler if self.cfg.precision == 16 else None,
            sequence_parallel=self.cfg.get('sequence_parallel', False),
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
        self.log('global_step', self.trainer.global_step, prog_bar=True, rank_zero_only=True, batch_size=1)
        return loss_mean

    def backward(self, *args, **kwargs):
        """ LightningModule hook to do backward.
            We want this to do nothing since we run backward in the fwd/bwd functions from apex.
            No need to call it here.
        """
        return

    def optimizer_zero_grad(self, *args, **kwargs):
        """ LightningModule hook to zero grad.
            We want this to do nothing as we are zeroing grads during the training_step.
        """
        return

    def validation_step(self, dataloader_iter, batch_idx):
        batch = next(dataloader_iter)
        gbs = self.cfg.get('validation_global_batch_size', self.cfg.global_batch_size)
        self._reconfigure_and_process_inference_batch(batch[0].size(0), gbs)
        loss_mean = self.fwd_bwd_step(itertools.chain([batch]), batch_idx, forward_only=True)
        if loss_mean.item == 0.0:
            loss_mean = []

        return loss_mean

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

    def validation_epoch_end(self, outputs):
        if len(outputs) == 0:
            return
        if parallel_state.is_pipeline_last_stage():
            # only the last pipeline parallel stages return loss
            averaged_loss = torch.stack(outputs).mean()
        else:
            averaged_loss = torch.tensor(0.0).cuda()

        # we can only log on one rank if it is rank zero so we broadcast from last rank
        torch.distributed.broadcast(averaged_loss, get_last_rank())

        self.log('val_loss', averaged_loss, prog_bar=True, rank_zero_only=True)
        logging.info(f'val_loss: {averaged_loss}')

        gbs = self.cfg.global_batch_size
        mbs = self.cfg.micro_batch_size
        self._reconfigure_batch_sizes(gbs, mbs)

    def test_step(self, dataloader_iter, batch_idx):
        return self.validation_step(dataloader_iter, batch_idx)

    def test_epoch_end(self, outputs):
        averaged_loss = average_losses_across_data_parallel_group(outputs)
        logging.info(f'test_loss: {averaged_loss[0]}')

    def setup_training_data(self, training_data_config=None):
        if self.cfg.data.get('train_ds', None):
            max_seq_length = self.frozen_model.cfg.encoder_seq_length
            if "max_seq_length" in self.cfg.data and self.cfg.data.max_seq_length:
                max_seq_length = self.cfg.data.max_seq_length
            self._train_ds, self._train_dl = self.build_virtual_prompt_dataset(
                data=self.cfg.data.train_ds,
                batch_size=self.cfg.global_batch_size,
                max_seq_length=max_seq_length,
                min_seq_length=self.cfg.data.get('min_seq_length', 1),
                add_bos=self.cfg.data.get('add_bos', False),
                add_eos=self.cfg.data.get('add_eos', True),
                for_train=True,
                drop_last=True,
                shuffle=True,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
                cache_data_path=self.cfg.data.get('train_cache_data_path', None),
                load_cache=self.cfg.data.get('load_cache', False),
                num_neighbors=self.cfg.data.neighbors,
                retrieved_doc_len=self.cfg.data.get('retrieved_doc_len', 128),
                chat_type=self.cfg.task_templates[0].chat_type
            )

    def setup_validation_data(self, validation_data_config=None):
        if self.cfg.data.get('validation_ds', None):
            max_seq_length = self.frozen_model.cfg.encoder_seq_length
            if "max_seq_length" in self.cfg.data and self.cfg.data.max_seq_length:
                max_seq_length = self.cfg.data.max_seq_length
            self._validation_ds, self._validation_dl = self.build_virtual_prompt_dataset(
                data=self.cfg.data.validation_ds,
                batch_size=self.cfg.get('validation_global_batch_size', self.cfg.global_batch_size),
                max_seq_length=max_seq_length,
                min_seq_length=self.cfg.data.get('min_seq_length', 1),
                add_bos=self.cfg.data.get('add_bos', False),
                add_eos=self.cfg.data.get('add_eos', True),
                for_train=True,
                drop_last=self.cfg.get('validation_drop_last', True),
                shuffle=False,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
                cache_data_path=self.cfg.data.get('validation_cache_data_path', None),
                load_cache=self.cfg.data.get('load_cache', False),
                num_neighbors=self.cfg.data.neighbors,
                retrieved_doc_len=self.cfg.data.get('retrieved_doc_len', 128),
                chat_type=self.cfg.task_templates[0].chat_type
            )

    def setup_test_data(self, test_data_config=None):
        if self.cfg.data.get('test_ds', None):
            self._test_ds, self._test_dl = self.build_virtual_prompt_dataset(
                data=self.cfg.data.test_ds,
                batch_size=self.cfg.get('validation_global_batch_size', self.cfg.global_batch_size),
                max_seq_length=self.frozen_model.cfg.encoder_seq_length,
                min_seq_length=self.cfg.data.get('min_seq_length', 1),
                add_bos=self.cfg.data.get('add_bos', False),
                add_eos=self.cfg.data.get('add_eos', True),
                for_train=False,
                drop_last=False,
                shuffle=False,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
                cache_data_path=self.cfg.data.get('test_cache_data_path', None),
                load_cache=self.cfg.data.get('load_cache', False),
                num_neighbors=self.cfg.data.neighbors,
                retrieved_doc_len=self.cfg.data.get('retrieved_doc_len', 128),
                chat_type=self.cfg.task_templates[0].chat_type
            )

    def build_virtual_prompt_dataset(
        self,
        data,
        batch_size,
        max_seq_length=2048,
        min_seq_length=1,
        add_bos=False,
        add_eos=False,
        for_train=True,
        drop_last=False,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        tokens_to_generate=None,
        get_dataset_only=False,
        cache_data_path=None,
        load_cache=False,
        num_neighbors=2,
        retrieved_doc_len = 128,
        chat_type = False
    ):
       
        dataset = RetroPromptLearningDataset(
            data=data,
            tokenizer=self.tokenizer,
            virtual_prompt_source=self.virtual_prompt_source,
            task_templates=self.task_templates,
            pseudo_tokens=self.pseudo_tokens,
            pad_token_id=self.tokenizer.eos_id if self.frozen_model.cfg.get('megatron_lm_compatible', False) else self.pad_token_id,
            max_seq_length=max_seq_length,
            min_seq_length=min_seq_length,
            add_bos=add_bos,
            add_eos=add_eos,
            for_train=for_train,
            tokens_to_generate=tokens_to_generate,
            cache_data_path=cache_data_path,  # the cache file
            load_cache=load_cache, # whether to load from the cache if it is available
            seed=1234,
            neighbors=num_neighbors,
            megatron_lm_compatible=self.frozen_model.cfg.get('megatron_lm_compatible', False),
            retrieved_doc_len = retrieved_doc_len,
            chat_type=chat_type
        )

        if get_dataset_only:
            return dataset

        # Make distributed dataloader
        rank = parallel_state.get_data_parallel_rank()
        data_parallel_size = parallel_state.get_data_parallel_world_size()
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=data_parallel_size, rank=rank, shuffle=shuffle, seed=self.cfg.seed
        )

        assert batch_size % data_parallel_size == 0, "Global batch size must be evenly divisible by data parallel size"

        if for_train:
            if self.cfg.get("sequence_parallel", False):
                collate_fn = partial(
                    dataset.collate_fn, tp_workers=parallel_state.get_tensor_model_parallel_world_size()
                )
            else:
                collate_fn = partial(dataset.collate_fn, tp_workers=0)
        else:
            collate_fn = dataset.inference_collate_fn

        dataloader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collate_fn,
            sampler=sampler,
            batch_size=batch_size // data_parallel_size,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True,  # (@adithyare and @eharper) We need this to make spawn=True to work.
        )

        return dataset, dataloader

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.
        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""

        self.frozen_model.model.set_input_tensor(input_tensor)

    def get_forward_output_and_loss_func(self):
        def fwd_output_and_loss_func(dataloader_iter, model):
            batch = next(dataloader_iter)
            batch = [x.cuda(non_blocking=True) for x in batch]

            # for pad strategy 3: input_tokens_id = variable paddings + virtual tokens ids + real tokens + batch padding
            input_tokens_id, input_attn_mask, loss_mask, retrieved_ids, retrieved_attn_mask, labels = batch

            if self.pos_embeddings: #self.cfg.get('add_position_embedding', False):
                input_position_ids = build_position_ids(input_tokens_id)
            else:
                input_position_ids = None
                
            output_tensor = model(
                input_tokens_id,
                input_attn_mask,
                retrieved_ids,
                retrieved_attn_mask,
                labels=labels,
                position_ids=input_position_ids,
                inference=False
            )

            # if isinstance(output_tensor, tuple):
            #     output_tensor, _ = output_tensor

            

            if self.cfg.precision == 16:
                loss_scale = self.trainer.precision_plugin.scaler._scale
                if loss_scale is not None:
                    self.log('loss_scale', loss_scale, batch_size=1)

            def loss_func(output_tensor):
                # loss_mask = loss_mask.float()
                # lm_loss = torch.sum(output_tensor.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()
                lm_loss = self.loss_func(loss_mask, output_tensor)
                reduced_loss = average_losses_across_data_parallel_group([lm_loss])
                # self._reduced_loss_buffer.append(reduced_loss[0])
                
                # loss = self.frozen_model.loss_func(loss_mask, output_tensor)
                # reduced_loss = average_losses_across_data_parallel_group([loss])
                return lm_loss, {'avg': reduced_loss}

            return output_tensor, loss_func

        return fwd_output_and_loss_func
    
    def loss_func(self, loss_mask, output_tensor):
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        # TODO: add nemo version here
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()  # sequence level nll
        return loss
    
    def get_forward_output_only_func(self):
        """
        Used for generate method only.
        """

        def fwd_output_only_func(batch, model):
            extra_arg = {}
            # batch = next(batch)
            (
                tokens,
                attention_mask,
                retrieved,
                retrieved_mask,
                set_inference_key_value_memory,
                inference_max_sequence_len,
                neighbors,
                position_ids,
            ) = next(batch)

            if len(retrieved.shape) == 1:
                retrieved = None
                retrieved_mask = None
            else:
                retrieved = retrieved.cuda()
                retrieved_mask = retrieved_mask.cuda()

            extra_arg['set_inference_key_value_memory'] = set_inference_key_value_memory[0].item()
            extra_arg['inference_max_sequence_len'] = inference_max_sequence_len[0].item()
            # extra_arg['neighbors'] = neighbors[0].item()
            extra_arg['position_ids'] = position_ids

            output_tensor = model(tokens, attention_mask, retrieved, retrieved_mask, **extra_arg)

            def id_func(output_tensor):
                return output_tensor, {'logits': output_tensor}

            return output_tensor, id_func

        return fwd_output_only_func

    def set_inference_config(self, inference_config, retrieval_config):
        self._inference_config = inference_config
        self.inference_strategy = model_inference_strategy_dispatcher(self, **retrieval_config)
        # self.inference_strategy = model_inference_strategy_dispatcher(self)
                                                                      
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        inference_config = self.get_inference_config()
        if inference_config is None:
            return None
        else:
            # need to overwrite some configuration, make it immutable
            inference_config = inference_config.copy()
            compute_logprob = inference_config['compute_logprob']
            if compute_logprob:
                del inference_config['compute_logprob']
                inference_config['inputs'] = batch
                inference_config['tokens_to_generate'] = 1
                inference_config['all_probs'] = True
                inference_config["add_BOS"] = False
                inference_config['greedy'] = True
                response = generate(self, **inference_config, strategy=self.inference_strategy)
                compute_prob_response = get_computeprob_response(self.tokenizer, response, batch)
                return compute_prob_response
            else:
                del inference_config['compute_logprob']
                inference_config['inputs'] = batch
                return generate(self, **inference_config, strategy=self.inference_strategy)

    def generate(
        self,
        inputs: Union[List[str], torch.Tensor, List[dict]],
        length_params: LengthParam,
        sampling_params: SamplingParam = None,
        **args,
    ) -> OutputType:

        # check whether the DDP is initialized
        if parallel_state.is_unitialized():

            def dummy():
                return

            if self.trainer.strategy.launcher is not None:
                self.trainer.strategy.launcher.launch(dummy, trainer=self.trainer)
            self.trainer.strategy.setup_environment()

        # set the default sampling params if it is None.
        # default do greedy sampling
        if sampling_params is None:
            sampling_params = get_default_sampling_params()

        # set the default length params if it is None.
        # default do greedy sampling
        if length_params is None:
            length_params = get_default_length_params()

        return megatron_gpt_generate(self.cuda(), inputs, self.tokenizer, length_params, sampling_params, **args)


    @classmethod
    def list_available_models(cls):
        pass


def get_pseudo_tokens(num_virtual_tokens):
    """
    Takes in an integer and returns a list of strings where each string
    is a numbered virtual token placeholder. If 
    num_virtual_tokens = 3, then this function returns:

    ["<prompt_0>", "<prompt_1>", "<prompt_2>"]

    Args:
        num_virtual_tokens: (int) Number of virtual token strings you want to make

    returns a list of string. 

    """
    pseudo_tokens = [
        VirtualPromptPlaceholderToken.BASE.value + str(i) + VirtualPromptPlaceholderToken.END.value
        for i in range(num_virtual_tokens)
    ]

    return pseudo_tokens


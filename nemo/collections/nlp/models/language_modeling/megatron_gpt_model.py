# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import re
from typing import Any, Dict, Optional

import torch
from apex.transformer import parallel_state, tensor_parallel
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import build_train_valid_test_datasets
from nemo.collections.nlp.models.language_modeling.megatron.gpt_model import GPTModel
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common.megatron.clip_grads import clip_grad_norm_fp32
from nemo.collections.nlp.modules.common.megatron.megatron_init import (
    initialize_model_parallel_for_nemo,
    set_jit_fusion_options,
)
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_ltor_masks_and_position_ids,
)
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.nlp.parts.nlp_overrides import NLPNativeMixedPrecisionPlugin
from nemo.utils import AppState, logging


class MegatronGPTModel(NLPModel):
    """
    Megatron GPT pretraining
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)
        self.cfg = cfg

        if self.cfg.get('use_cpu_initialization', False) is False:
            torch.cuda.set_device(trainer.local_rank)

        # buffer used during train_step for logging average loss over gradient accumulation steps
        self._reduced_loss_buffer = []

        initialize_model_parallel_for_nemo(
            world_size=trainer.world_size,
            global_rank=trainer.global_rank,
            local_rank=trainer.local_rank,
            tensor_model_parallel_size=cfg.get('tensor_model_parallel_size', 1),
            seed=self.cfg.get('seed', 1234),
        )

        if not self.cfg.get('fused_bf16'):
            set_jit_fusion_options()

        self.tokenizer = get_nmt_tokenizer(
            library=self.cfg.tokenizer.library,
            model_name=self.cfg.tokenizer.type,
            tokenizer_model=self.register_artifact("tokenizer_model", self.cfg.tokenizer.model),
            vocab_file=self.register_artifact("vocab_file", self.cfg.tokenizer.vocab_file),
            merges_file=self.register_artifact("merges_file", self.cfg.tokenizer.merge_file),
        )

        vocab_size = self.tokenizer.vocab_size

        padded_vocab_size = self._vocab_size_with_padding(
            orig_vocab_size=vocab_size,
            make_vocab_size_divisible_by=cfg.get('make_vocab_size_divisible_by', 128),
            tensor_model_parallel_size=cfg.get('tensor_model_parallel_size', 1),
        )

        self.model = GPTModel(
            vocab_size=padded_vocab_size,
            hidden_size=cfg.hidden_size,
            max_position_embeddings=cfg.max_position_embeddings,
            num_layers=cfg.num_layers,
            num_attention_heads=cfg.num_attention_heads,
            apply_query_key_layer_scaling=cfg.get('apply_query_key_layer_scaling', True),
            kv_channels=cfg.get('kv_channels', None),
            ffn_hidden_size=cfg.ffn_hidden_size,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=cfg.get('pre_process', True),
            post_process=cfg.get('post_process', True),
            init_method_std=cfg.get('init_method_std', 0.02),
            fp16_lm_cross_entropy=cfg.get('fp16_lm_cross_entropy', False),
            use_cpu_initialization=cfg.get('use_cpu_initialization', False),
            hidden_dropout=cfg.get('hidden_dropout', 0.1),
            fused_fp16=cfg.get('fused_fp16', False),
            fused_bf16=cfg.get('fused_bf16', False),
            fp32_residual_connection=cfg.get('fp32_residual_connection', False),
            activations_checkpoint_method=cfg.get('activations_checkpoint_method', None),
            activations_checkpoint_num_layers=cfg.get('activations_checkpoint_num_layers', 1),
            layernorm_epsilon=cfg.get('layernorm_epsilon', 1e-5),
            onnx_safe=cfg.get('onnx_safe', False),
        )

    def forward(self, tokens, position_ids, attention_mask, labels):
        output_tensor = self.model(tokens, position_ids, attention_mask, labels=labels)
        return output_tensor

    def training_step(self, batch, batch_idx):
        tokens, labels, loss_mask, attention_mask, position_ids = self.process_batch(batch)
        output_tensor = self(tokens, position_ids, attention_mask, labels)
        loss = self.loss_func(loss_mask, output_tensor)
        reduced_loss = average_losses_across_data_parallel_group([loss])

        # cache reduced loss while accumulating gradients
        self._reduced_loss_buffer.append(reduced_loss[0])

        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            # Reduced loss for logging.
            average_reduced_loss = sum(self._reduced_loss_buffer) / len(self._reduced_loss_buffer)
            self.log('reduced_train_loss', average_reduced_loss, prog_bar=True)
            lr = self._optimizer.param_groups[0]['lr']
            self.log('lr', lr)
            self.log('global_step', self.trainer.global_step, prog_bar=True)
            self.log('consumed_samples', self.compute_consumed_samples(self.trainer.global_step), prog_bar=True)
            self._reduced_loss_buffer = []
        return loss

    def validation_step(self, batch, batch_idx):
        tokens, labels, loss_mask, attention_mask, position_ids = self.process_batch(batch)
        output_tensor = self(tokens, position_ids, attention_mask, labels)
        loss = self.loss_func(loss_mask, output_tensor)
        reduced_loss = average_losses_across_data_parallel_group([loss])

        # TODO: add text generation
        # take the first k tokens and then generate text - compare with ground truth (won't look similar in general)
        """
        k = num_context
        n = max_generate_length
        context_tokens = tokens[0:k]
        while k < n:
            output_tensor = self(context_tokens)
            next_token = sample(output_tensor)
            context_tokens.append(next_token)
            k += 1
        """
        return reduced_loss

    def validation_epoch_end(self, outputs):
        averaged_loss = torch.stack(outputs).mean()
        self.log('val_loss', averaged_loss, prog_bar=True)
        self.log('consumed_samples', self.compute_consumed_samples(self.trainer.global_step))

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        averaged_loss = average_losses_across_data_parallel_group(outputs)
        logging.info(f'test_loss: {averaged_loss[0]}')

    def loss_func(self, loss_mask, output_tensor):
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        # TODO: add nemo version here
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()  # sequence level nll
        return loss

    def process_batch(self, batch):

        # Items and their type.
        keys = ['text']
        datatype = torch.int64

        data = batch
        data_b = tensor_parallel.broadcast_data(keys, data, datatype)

        # Unpack.
        tokens_ = data_b['text'].long()
        labels = tokens_[:, 1:].contiguous()
        tokens = tokens_[:, :-1].contiguous()

        # Get the masks and postition ids.
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            self.tokenizer.eos_id,
            self.cfg.data.get('reset_position_ids', False),
            self.cfg.data.get('reset_attention_mask', False),
            self.cfg.data.get('eod_mask_loss', False),
        )

        return tokens, labels, loss_mask, attention_mask, position_ids

    def build_train_valid_test_datasets(self):
        logging.info('Building GPT datasets.')
        global_batch_size = self.trainer.world_size * self.cfg.micro_batch_size / self.cfg.tensor_model_parallel_size
        # Compute trianing micro-batch steps: total_global_batch_steps x grad_acumms_per_global_batch
        max_train_steps = self.trainer.max_steps * self.trainer.accumulate_grad_batches
        eval_iters = (max_train_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches
        test_iters = self.trainer.limit_test_batches

        train_valid_test_num_samples = [
            max_train_steps * global_batch_size,
            eval_iters * global_batch_size,
            test_iters * global_batch_size,
        ]
        self._train_ds, self._validation_ds, self._test_ds = build_train_valid_test_datasets(
            cfg=self.cfg,
            trainer=self.trainer,
            data_prefix=self.cfg.data.data_prefix,
            data_impl=self.cfg.data.data_impl,
            splits_string=self.cfg.data.splits_string,
            train_valid_test_num_samples=train_valid_test_num_samples,
            seq_length=self.cfg.data.seq_length,
            seed=self.cfg.seed,
            skip_warmup=self.cfg.data.get('skip_warmup', True),
        )
        if self._train_ds is not None:
            logging.info(f'Length of train dataset: {len(self._train_ds)}')
        if self._validation_ds is not None:
            logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        if self._test_ds is not None:
            logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building GPT datasets.')
        return self._train_ds, self._validation_ds, self._test_ds

    def build_pretraining_data_loader(self, dataset, consumed_samples):
        """Buld dataloader given an input dataset."""

        if dataset is None:
            return None

        # Megatron sampler
        if hasattr(self.cfg.data, 'dataloader_type') and self.cfg.data.dataloader_type is not None:
            if self.cfg.data.dataloader_type == 'single':
                batch_sampler = MegatronPretrainingSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.micro_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                )
            elif self.cfg.data.dataloader_type == 'cyclic':
                batch_sampler = MegatronPretrainingRandomSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.micro_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                )
            else:
                raise ValueError('cfg.data.dataloader_type must be "single" or "cyclic"')
        else:
            raise ValueError('cfg.data.dataloader_type not found. Must be "single" or "cyclic"')

        # Torch dataloader.
        return torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, num_workers=self.cfg.data.num_workers, pin_memory=True,
        )

    def setup(self, stage=None):
        if stage == 'predict':
            return
        # TODO: consider adding a ModelPT guard to check if model is being restored.
        # allowing restored models to optionally setup datasets
        self.build_train_valid_test_datasets()
        self.setup_training_data(self.cfg.data)
        self.setup_validation_data(self.cfg.data)
        self.setup_test_data(self.cfg.data)

    def setup_training_data(self, cfg):
        if hasattr(self, '_train_ds'):
            resume_checkpoint_path = self.trainer.checkpoint_connector.resume_checkpoint_path
            if resume_checkpoint_path:
                consumed_samples = int(
                    float(re.findall(r"consumed_samples\=([0-9]+.[0-9]+)", resume_checkpoint_path)[0])
                )
            else:
                consumed_samples = 0
            logging.info(
                f'Setting up train dataloader with len(len(self._train_ds)): {len(self._train_ds)} and consumed samples: {consumed_samples}'
            )
            self._train_dl = self.build_pretraining_data_loader(self._train_ds, consumed_samples)

    def setup_validation_data(self, cfg):
        if hasattr(self, '_validation_ds'):
            consumed_samples = 0
            logging.info(
                f'Setting up validation dataloader with len(len(self._validation_ds)): {len(self._validation_ds)} and consumed samples: {consumed_samples}'
            )
            self._validation_dl = self.build_pretraining_data_loader(self._validation_ds, consumed_samples)

    def setup_test_data(self, cfg):
        if hasattr(self, '_test_ds'):
            consumed_samples = 0
            logging.info(
                f'Setting up test dataloader with len(len(self._test_ds)): {len(self._test_ds)} and consumed samples: {consumed_samples}'
            )
            self._test_dl = self.build_pretraining_data_loader(self._test_ds, consumed_samples)

    def compute_consumed_samples(self, global_step):
        app_state = AppState()
        consumed_samples = (
            global_step
            * app_state.data_parallel_size
            * self.cfg.micro_batch_size
            * self.trainer.accumulate_grad_batches
        )
        return int(consumed_samples)

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        """PTL hook that is called after unscaling gradients when using native amp.
           We use gradient clipping implementation from megatron-lm.
        """
        clip_val = self.trainer.gradient_clip_val
        if clip_val is None:
            return

        clip_val = float(clip_val)
        if clip_val <= 0:
            return

        if isinstance(self.trainer.accelerator_connector.precision_plugin, NLPNativeMixedPrecisionPlugin):
            parameters = self.model.parameters()
            clip_grad_norm_fp32(parameters=parameters, max_norm=clip_val)
        else:
            return

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        request = batch
        response = self.complete(request)
        return response

    def complete(self, request: Dict):
        """
            Autoregressively invokes language model in the inference mode
        Args:	
            request: Dictionary with the following fields
                * prompt: a string which text the model should complete.
                * tokens_to_generate: how many tokens to generate while doing prompt completion.
                * stop_after_sentence: (default True) whether to stop generation once sentence end is reached.
        Returns:	
            response: A python dictionary with the following fields
                * prompt: original text of the prompt
                * tokenized_prompt: list of (str) tokens from prompt
                * completion: a python dictionary with the following subfields:
                    * tokens: a list of triples (token, token_id, log_prob) comprising completion
                    * stop reason: either 'eos', 'sentence_end' or 'limit' indicating why generation stopped
                    * text: completion text (as a single string)
                
        """
        response = {}
        self.freeze()
        logsoftmaxlayer = torch.nn.LogSoftmax(dim=-1)
        response['tokenized_prompt'] = request['tokenized_prompt']
        tokens = request['tokens']
        # naive greedy slow loop
        # TODO: add option for BeamSearchDecoder
        response['prompt'] = request['prompt']
        response['completion'] = {}
        response['completion']['stop reason'] = 'limit'
        for i in range(request.get("tokens_to_generate", 64)):
            attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
                data=tokens,
                eod_token=self.tokenizer.eos_id,
                reset_position_ids=self.cfg.get('reset_position_ids', False),
                reset_attention_mask=self.cfg.get('reset_attention_mask', False),
                eod_mask_loss=self.cfg.get('eod_mask_loss', False),
            )
            # No labels during inference. Still need masks to not attend to the right
            output_tensor = self(tokens, position_ids, attention_mask, labels=None)
            output_tensor = tensor_parallel.gather_from_tensor_model_parallel_region(output_tensor)
            log_probs, token_ids = torch.max(logsoftmaxlayer(output_tensor), dim=-1)
            reached_eos = token_ids[0, -1].item() == self.tokenizer.eos_id
            tokens = torch.cat([torch.squeeze(tokens), token_ids[:, -1]])
            response['completion']["tokens"] = list(
                zip(self.tokenizer.ids_to_tokens(tokens), tokens.tolist(), log_probs.tolist()[0])
            )
            completion_text = self.tokenizer.ids_to_text(x[1] for x in response['completion']["tokens"])
            if reached_eos:  # Will it actually ever reach that?
                response['completion']['stop reason'] = 'eos'
                break
            elif request.get("stop_after_sentence", True) and completion_text.endswith(('.', '!', '?')):
                response['completion']['stop reason'] = 'sentence_end'
                break
            tokens = torch.unsqueeze(tokens, 0)
        response['completion']["text"] = self.tokenizer.ids_to_text(x[1] for x in response['completion']["tokens"])
        self.unfreeze()
        return response

    def list_available_models(self):
        return None

    def _vocab_size_with_padding(self, orig_vocab_size, make_vocab_size_divisible_by, tensor_model_parallel_size):
        """Pad vocab size so it is divisible by model parallel size and
        still having GPU friendly size."""

        after = orig_vocab_size
        multiple = make_vocab_size_divisible_by * tensor_model_parallel_size
        while (after % multiple) != 0:
            after += 1
        logging.info(
            f'Padded vocab_size: {after}, original vocab_size: {orig_vocab_size}, dummy tokens: {after - orig_vocab_size}.'
        )
        return after

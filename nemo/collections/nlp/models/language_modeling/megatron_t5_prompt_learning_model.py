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
import itertools
from typing import Any, List

import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import open_dict
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.language_modeling.megatron.t5_prompt_learning_dataset import (
    T5PromptLearningDataset,
    T5Sentinel,
)
from nemo.collections.nlp.models.language_modeling.megatron_base_prompt_learning_model import (
    MegatronPromptLearningBaseModel,
)
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.collections.nlp.modules.common import (
    PromptTable,
    VirtualPromptStyle,
)
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.utils import logging

try:
    from apex.transformer import parallel_state

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


__all__ = ['MegatronT5PromptLearningModel']


class MegatronT5PromptLearningModel(MegatronPromptLearningBaseModel):
    """
    Model class for prompt-tuning or p-tuning a pretrained Megatron T5 model. 

    Prompt Tuning initalizes virtual prompt embeddings directly from a copy of
    certain token embeddings from the the pretrained T5 model's vocabulary
    and directly tunes these embedding weights. The token embeddings used in 
    initalization are specified by the user in the config file. The model can 
    be prompt-tuned for multiple tasks at once. Virtual prompts are stored in a 
    prompt table and can be added or deleted without disrupting virtual prompts 
    for other tasks. 

    P-tuning initializes either an MLP or LSTM encoder model that generates virtual 
    prompt embeddings for every task. Each task shares the same encoder. After ptuning
    is compelete, the learned virtual prompts can be saved to the prompt table
    using add_ptuned_prompts_to_prompt_table(). Thus, if a user wants to add a 
    new virtual prompt via p-tuning, they do not need to retrain on all previous 
    tasks. This gives p-tuning the same task flexiblity as prompt-tuning.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)

        # Encoder and decoder need to have the same hidden size and we check for this in the frozen enc-dec model.
        self.hidden_size = self.frozen_model.cfg.encoder.hidden_size

        if self.frozen_model.enc_dec_model.pre_process and self.virtual_prompt_style in [
            VirtualPromptStyle.P_TUNING,
            VirtualPromptStyle.PROMPT_TUNING,
            VirtualPromptStyle.INFERENCE,
        ]:

            self.word_embeddings = self.frozen_model.enc_dec_model.encoder_embedding.word_embeddings 
            
            # Prompt table stores all task embeddings, p-tuning virtual prompts get added to the table after training
            self.prompt_table = PromptTable(
                existing_tasks=self.existing_tasks,
                task_templates=self.task_templates,
                task_id_num_to_name=self.task_id_num_to_name,
                hidden_size=self.hidden_size,
            )

        self.decoder_seq_length = cfg.get('decoder_seq_length', 40)

    def load_frozen_model(self, cfg, trainer):
        save_restore_connector = NLPSaveRestoreConnector()

        # Load frozen model from unpacked directory 
        if os.path.isdir(cfg.get('language_model_path')):
            save_restore_connector.model_extracted_dir = cfg.get('language_model_path')

        # TODO: Fix this once apex patches FusedScaledMaskedSoftmax.
        # This is a workaround for the fact that `masked_softmax_fusion` has issues with certain input sizes that may be present while finetuning.
        t5_cfg = MegatronT5Model.restore_from(
            cfg.get('language_model_path'), 
            trainer=trainer, 
            return_config=True,
            save_restore_connector=save_restore_connector,
        )

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

        self.frozen_model = MegatronT5Model.restore_from(
            cfg.get('language_model_path'),
            trainer=trainer,
            override_config_path=t5_cfg,
            save_restore_connector=save_restore_connector,
        )

    def state_dict(self, destination=None, prefix=None, keep_vars=False):
        """
        Custom state dict that only contains prompt table and prompt encoder parameters. 
        No frozen model parameters are stored in the state dict. Prompt encoder parameters 
        are only in state dict for intermediate checkpoints saved during training. Final
        nemo checkpoints at the end of training will contain prompt table parameters only. 
        """
      
        if self.frozen_model.enc_dec_model.pre_process:
            super().state_dict(destination, prefix, keep_vars)
        else:
            state_dict_ = {}
        
            return state_dict_

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Custom load state dict method that only loads prompt table and prompt encoder
        parameters. Matching load method for this class' custom state dict method. 
        """
        if self.frozen_model.enc_dec_model.pre_process:
            super().load_state_dict(state_dict, strict)

    def setup_optimizer_param_groups(self):
        """
        ModelPT override. Optimizer will get self._optimizer_param_groups. 
        Only want virtual prompt params to be passed to the optimizer.
        """
        ## Freeze frozen model
        for param in self.frozen_model.parameters():
            param.requires_grad = False

        if self.frozen_model.enc_dec_model.pre_process:
            self.add_virtual_prompt_params_to_param_group()
        else:    
            self._optimizer_param_groups = ({'params': []},)

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.
        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""

        self.frozen_model.enc_dec_model.set_input_tensor(input_tensor)

    def forward(
        self, input_ids, dec_input, enc_mask, dec_mask, position_ids, taskname_ids, labels=None, inference=False,
    ):
        """
        Special forward method for p-tuning/prompt-tuning pretrained
        T5 style models.
        """
        # Get embeddings for text tokens and insert virtual token embeddings
        if self.frozen_model.enc_dec_model.pre_process:
            if inference:
                input_embeds = self.embed_input_inference(input_ids, taskname_ids)
            else:
                input_embeds = self.embed_input_train(input_ids, taskname_ids)

            # TODO: This check needs to be revisited with PP support.
            if hasattr(self.frozen_model.enc_dec_model.encoder_embedding, 'position_embeddings'):
                position_embeddings = self.frozen_model.enc_dec_model.encoder_embedding.position_embeddings(position_ids)
                encoder_input = input_embeds + position_embeddings
            else:
                encoder_input = input_embeds
        else:
            encoder_input = None

        # Call forward on T5 model with preprocessed embeddings
        if self.autocast_dtype == torch.float32:
            output = self.frozen_model.enc_dec_model(
                enc_input_ids=None,
                enc_attn_mask=enc_mask,
                dec_input_ids=dec_input,
                dec_attn_mask=dec_mask,
                token_type_ids=None,
                labels=labels,
                output_enc_hidden_only=False,
                enc_input=encoder_input,
            )
        else:
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                output = self.frozen_model.enc_dec_model(
                    enc_input_ids=None,
                    enc_attn_mask=enc_mask,
                    dec_input_ids=dec_input,
                    dec_attn_mask=dec_mask,
                    token_type_ids=None,
                    labels=labels,
                    output_enc_hidden_only=False,
                    enc_input=encoder_input,
                )

        return output, encoder_input

    def fwd_bwd_step(self, batch, forward_only):
        """
            Dataloader produces a global batch which is turned into a list of microbatches.
            The list of microbatches is then piped through the pipeline using Apex fwd/bwd functions.
        """
        super().fwd_bwd_step(batch, forward_only, disable_autocast=True)

    def get_forward_output_and_loss_func(self):
        def fwd_output_and_loss_func(batch, model):
            batch = [x.cuda(non_blocking=True) for x in batch]
            enc_input, dec_input, labels, loss_mask, enc_mask, dec_mask, position_ids, taskname_ids = batch

            output_tensor, encoder_input = model(
                enc_input, dec_input, enc_mask, dec_mask, position_ids, taskname_ids, labels, inference=False
            )

            def loss_func(output_tensor):
                loss = self.frozen_model.loss_func(loss_mask, output_tensor)
                reduced_loss = average_losses_across_data_parallel_group([loss])
                return loss, {'avg': reduced_loss}

            return output_tensor, loss_func

        return fwd_output_and_loss_func
    
    def setup(self, stage=None):
        if (
            stage == 'predict' or self.virtual_prompt_style == VirtualPromptStyle.INFERENCE
        ) and self.frozen_model.enc_dec_model.pre_process:
            self.freeze_existing_virtual_prompt_params()
            return

        self.setup_test_data()
        if stage == 'test':
            return

        if self.frozen_model.enc_dec_model.pre_process:
            if self.virtual_prompt_style == VirtualPromptStyle.PROMPT_TUNING:
                self.init_new_prompts()
            elif self.virtual_prompt_style == VirtualPromptStyle.P_TUNING:
                self.init_prompt_encoder()

            self.freeze_existing_virtual_prompt_params()

        self.setup_training_data()
        self.setup_validation_data()

    def setup_training_data(self, training_data_config=None):
        if self.cfg.data.get('train_ds', None):
            self._train_ds, self._train_dl = self.build_virtual_prompt_dataset(
                dataset_paths=self.cfg.data.train_ds,
                batch_size=self.cfg.global_batch_size,
                for_train=True,
                drop_last=True,
                shuffle=True,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
            )

    def setup_validation_data(self, validation_data_config=None):
        if self.cfg.data.get('validation_ds', None):
            self._validation_ds, self._validation_dl = self.build_virtual_prompt_dataset(
                dataset_paths=self.cfg.data.validation_ds,
                batch_size=self.cfg.global_batch_size,
                for_train=True,
                drop_last=True,
                shuffle=False,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
            )

    def setup_test_data(self, test_data_config=None):
        if self.cfg.data.get('test_ds', None):
            self._test_ds, self._test_dl = self.build_virtual_prompt_dataset(
                dataset_paths=self.cfg.data.test_ds,
                batch_size=self.cfg.global_batch_size,
                for_train=False,
                drop_last=False,
                shuffle=False,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
            )

    def build_virtual_prompt_dataset(
        self, dataset_paths, batch_size, for_train, drop_last, shuffle, num_workers, pin_memory
    ):
        dataset = T5PromptLearningDataset(
            datasets=dataset_paths,
            tokenizer=self.tokenizer,
            virtual_prompt_source=self.virtual_prompt_source,
            task_templates=self.task_templates,
            pseudo_tokens=self.pseudo_tokens,
            pad_token_id=self.pad_token_id,
            max_seq_length=self.cfg.data.get('max_seq_length', self.frozen_model.cfg.max_position_embeddings),
            min_seq_length=self.cfg.data.get('min_seq_length', 1),
            add_bos=self.cfg.data.get('add_bos', False),
            add_eos=self.cfg.data.get('add_eos', True),
            for_train=for_train,
        )

        rank = parallel_state.get_data_parallel_rank()
        world_size = parallel_state.get_data_parallel_world_size()
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            sampler=sampler,
            batch_size=batch_size // world_size,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        print('build success', len(dataloader), dataset_paths)
        return dataset, dataloader

    def validation_step(self, batch, batch_idx, inference=False):
        outcome = self.inference_step(batch, batch_idx, inference=inference)
        return outcome

    def validation_epoch_end(self, outputs):
        self.inference_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs)

    def on_train_end(self):
        # Save p-tuned prompts to prompt table for inference or future task training
        if self.virtual_prompt_style == VirtualPromptStyle.P_TUNING and self.frozen_model.enc_dec_model.pre_process:
            self.add_ptuned_prompts_to_prompt_table()
            logging.info(f"All p-tuned prompts where moved to the prompt table.")

            # Remove prompt encoder from model
            self.prompt_encoder = None
            logging.info(f"Prompt encoder deleted")

        self.update_config_for_inference_and_save()

    def inference_step(self, batch, batch_idx, inference=False):
        enc_input, dec_input, labels, loss_mask, enc_mask, dec_mask, position_ids, taskname_ids = batch

        mode = self.training
        self.eval()

        input_embeds = self.embed_input_train(enc_input, taskname_ids)

        # TODO: This check needs to be revisited with PP support.
        if hasattr(self.frozen_model.enc_dec_model.encoder_embedding, 'position_embeddings'):
            position_embeddings = self.frozen_model.enc_dec_model.encoder_embedding.position_embeddings(position_ids)
            encoder_input = input_embeds + position_embeddings
        else:
            encoder_input = input_embeds

        loss_mean = self.fwd_bwd_step(batch, batch_idx, forward_only=True)

        predicted_token_ids, log_probs = self.frozen_model.decode(
            tokens_enc=enc_input,
            enc_mask=enc_mask,
            num_tokens_to_generate=self.decoder_seq_length,
            encoder_input=encoder_input,
        )

        processed_inputs, processed_preds, processed_labels = [], [], []
        preds = predicted_token_ids.cpu().numpy().tolist()
        labels = labels.cpu().numpy().tolist()
        enc_inputs = enc_input.cpu().numpy().tolist()

        for i, (enc_input, pred, label) in enumerate(zip(enc_inputs, preds, labels)):
            if self.tokenizer.eos_id in pred:
                idx = pred.index(self.tokenizer.eos_id)
                pred = pred[:idx]

            # Sentencepiece case
            if hasattr(self.tokenizer, 'special_token_to_id'):
                pred = [id for id in pred if id not in self.tokenizer.special_token_to_id.values()]
                label = [id for id in label if id not in self.tokenizer.special_token_to_id.values()]
                enc_input = [id for id in enc_input if id not in self.tokenizer.special_token_to_id.values()]
            # HF Autotokenizer case.
            else:
                pred = [id for id in pred if id not in self.tokenizer.tokenizer.additional_special_tokens_ids]
                label = [id for id in label if id not in self.tokenizer.tokenizer.additional_special_tokens_ids]
                enc_input = [
                    id for id in enc_input if id not in self.tokenizer.tokenizer.additional_special_tokens_ids
                ]

            pred = self.tokenizer.ids_to_text(pred)
            label = self.tokenizer.ids_to_text(label)
            enc_input = self.tokenizer.ids_to_text(enc_input)

            processed_preds.append(pred)
            processed_labels.append(label)
            processed_inputs.append(enc_input)

        self.train(mode=mode)
        return {
            'loss': loss_mean,
            'predicted_token_ids': processed_preds,
            'labels': processed_labels,
            'enc_inputs': processed_inputs,
        }

    def inference_epoch_end(self, outputs):

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

            correct = 0
            for (input, pred, label) in gather_results_dedup:
                if pred == label:
                    correct += 1

            val_acc = correct / len(gather_results_dedup)
            val_acc = torch.tensor(val_acc).cuda()

            logging.info(f'Validation accuracy: {val_acc}')
        else:
            val_acc = torch.tensor(0.0).cuda()

        averaged_loss = torch.stack([item['loss'] for item in outputs]).mean()
        logging.info(f'Validation loss: {averaged_loss}')

        self.log('val_loss', averaged_loss, prog_bar=True, rank_zero_only=True)
        self.log('val_acc', val_acc, prog_bar=True, rank_zero_only=True)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:

        enc_input, dec_input, labels, loss_mask, enc_mask, dec_mask, position_ids, taskname_ids = batch

        input_embeds = self.embed_input_inference(enc_input, taskname_ids)

        # TODO: This check needs to be revisited with PP support.
        if hasattr(self.frozen_model.enc_dec_model.encoder_embedding, 'position_embeddings'):
            position_embeddings = self.frozen_model.enc_dec_model.encoder_embedding.position_embeddings(position_ids)
            encoder_input = input_embeds + position_embeddings
        else:
            encoder_input = input_embeds

        predicted_token_ids, log_probs = self.frozen_model.decode(
            tokens_enc=enc_input,
            enc_mask=enc_mask,
            num_tokens_to_generate=self.decoder_seq_length,
            encoder_input=encoder_input,
        )

        processed_preds = []
        processed_labels = []
        processed_inputs = []

        preds = predicted_token_ids.cpu().numpy().tolist()
        enc_inputs = enc_input.cpu().numpy().tolist()

        if labels is not None:
            labels = labels.cpu().numpy().tolist()
        else:
            labels = [None] * len(preds)

        for i, (enc_input, pred, label) in enumerate(zip(enc_inputs, preds, labels)):
            if self.tokenizer.eos_id in pred:
                idx = pred.index(self.tokenizer.eos_id)
                pred = pred[:idx]

            pred = [
                id
                for id in pred
                if id not in self.tokenizer.tokenizer.additional_special_tokens_ids
                and id not in self.tokenizer.text_to_ids(T5Sentinel.FIRST.value)
            ]  # delete the sentinel token at the beginning of prediction

            pred = self.tokenizer.ids_to_text(pred)
            processed_preds.append(pred)

            enc_input = [
                id for id in enc_input if id not in self.tokenizer.text_to_ids(T5Sentinel.FIRST.value)
            ]  # delete the sentinel token added to the end of input

            input = self.tokenizer.ids_to_text(enc_input)
            processed_inputs.append(input)

            if label:
                label = [
                    id
                    for id in label
                    if id not in self.tokenizer.tokenizer.additional_special_tokens_ids
                    and id not in self.tokenizer.text_to_ids(T5Sentinel.FIRST.value)
                ]  # delete the sentinel token at the beginning of label

                label = self.tokenizer.ids_to_text(label)
            processed_labels.append(label)

        return {
            'enc_input': processed_inputs,
            'predicted_token_ids': processed_preds,
            'log_probs': log_probs,
            'labels': processed_labels,
        }

    def on_predict_epoch_end(self, outputs: List[Any]) -> None:

        gather_results = [None for _ in range(parallel_state.get_data_parallel_world_size())]
        all_preds = list(itertools.chain(*[item['predicted_token_ids'] for item in outputs[0]]))
        all_labels = list(itertools.chain(*[item['labels'] for item in outputs[0]]))
        all_inputs = list(itertools.chain(*[item['enc_input'] for item in outputs[0]]))

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

            results = {'input_prediction_pair': input_prediction_pair, 'acc': acc}
            logging.info(f'Prediction results: {results}')
            logging.info(f'Test finish---------------------------------')

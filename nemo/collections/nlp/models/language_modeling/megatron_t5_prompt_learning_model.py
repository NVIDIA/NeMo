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


from typing import Any, List

import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import open_dict
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.language_modeling.megatron.t5_prompt_learning_dataset import T5PromptLearningDataset
from nemo.collections.nlp.models.language_modeling.megatron_base_prompt_learning_model import (
    MegatronBasePromptLearningModel,
    get_pseudo_tokens,
)
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model
from nemo.collections.nlp.modules.common import PromptTable
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.utils import logging

try:
    from apex.transformer import parallel_state

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


__all__ = ['MegatronT5PromptLearningModel']


class MegatronT5PromptLearningModel(MegatronBasePromptLearningModel):
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
        super().__init__(cfg, trainer)

        self.megatron_amp_o2 = cfg.get('megatron_amp_O2', False)
        # TODO: Fix this once apex patches FusedScaledMaskedSoftmax.
        # This is a workaround for the fact that `masked_softmax_fusion` has issues with certain input sizes that may be present while finetuning.
        t5_cfg = MegatronT5Model.restore_from(cfg.get('language_model_path'), trainer=trainer, return_config=True,)
        OmegaConf.set_struct(t5_cfg, True)
        with open_dict(t5_cfg):
            t5_cfg.masked_softmax_fusion = False
            t5_cfg.megatron_amp_O2 = self.megatron_amp_o2
            # hack to make the _GLOBAL_NUM_MICROBATCHES_CALCULATOR initialize
            t5_cfg.micro_batch_size = cfg.get('micro_batch_size', 4)
            t5_cfg.global_batch_size = cfg.get('global_batch_size', 4)

        self.frozen_model = MegatronT5Model.restore_from(
            cfg.get('language_model_path'), trainer=trainer, override_config_path=t5_cfg,
        )

        # Freeze all T5 model weights for prompt-tuning/p-tuning
        if not cfg.lm_finetune:
            self.frozen_model.freeze()

        self.tokenizer = self.frozen_model.tokenizer
        self.float_type = self.frozen_model.enc_dec_model.enc_dec_model.encoder.model.layers[0].dtype

        self.hidden_size = self.frozen_model.cfg.hidden_size
        self.word_embeddings = self.frozen_model.enc_dec_model.encoder_embedding.word_embeddings

        # Prompt table stores all task embeddings, p-tuning virtual prompts get added to the table after training
        self.prompt_table = PromptTable(
            existing_tasks=self.existing_tasks,
            task_templates=self.task_templates,
            task_id_num_to_name=self.task_id_num_to_name,
            hidden_size=self.hidden_size,
        )

        # Prepare pseudo token ids for virtual/virtual prompt tokens
        self.pseudo_tokens = get_pseudo_tokens(self.max_virtual_tokens)
        self.tokenizer.add_special_tokens({'additional_special_tokens': self.pseudo_tokens})
        self.pseudo_token_ids = self.tokenizer.tokens_to_ids(self.pseudo_tokens)
        self.pseudo_token_ids_start = self.pseudo_token_ids[0]
        self.pad_token_id = self.tokenizer.pad_id if self.tokenizer.pad_id is not None else self.tokenizer.unk_id

        self.decoder_seq_length = cfg.get('decoder_seq_length', 40)

    def forward(
        self, input_ids, dec_input, enc_mask, dec_mask, position_ids, taskname_ids, labels=None, inference=False,
    ):
        """
        Special forward method for p-tuning/prompt-tuning pretrained
        T5 style models.
        """
        # Get embeddings for text tokens and insert virtual token embeddings
        if inference:
            input_embeds = self.embed_input_inference(input_ids, taskname_ids)
        else:
            input_embeds = self.embed_input_train(input_ids, taskname_ids)

        position_embeddings = self.frozen_model.enc_dec_model.encoder_embedding.position_embeddings(position_ids)
        encoder_input = input_embeds + position_embeddings

        # Call forward on T5 model with preprocessed embeddings
        if self.float_type == torch.float32:
            output = self.frozen_model.enc_dec_model(
                enc_input_ids=None,
                enc_attn_mask=enc_mask,
                dec_input_ids=dec_input,
                dec_attn_mask=dec_mask,
                token_type_ids=None,
                labels=labels,
                enc_hidden_states=None,
                output_enc_hidden_only=False,
                enc_input=encoder_input,
            )
        else:
            with torch.autocast(device_type="cuda", dtype=self.float_type):
                output = self.frozen_model.enc_dec_model(
                    enc_input_ids=None,
                    enc_attn_mask=enc_mask,
                    dec_input_ids=dec_input,
                    dec_attn_mask=dec_mask,
                    token_type_ids=None,
                    labels=labels,
                    enc_hidden_states=None,
                    output_enc_hidden_only=False,
                    enc_input=encoder_input,
                )

        return output, encoder_input

    def training_step(self, batch, batch_idx):
        enc_input, dec_input, labels, loss_mask, enc_mask, dec_mask, position_ids, taskname_ids = batch

        output, encoder_input = self.forward(
            enc_input, dec_input, enc_mask, dec_mask, position_ids, taskname_ids, labels, inference=False
        )

        loss = self.frozen_model.loss_func(loss_mask, output)
        self.log('train_loss', loss)

        # Reduced loss for logging.
        reduced_loss = average_losses_across_data_parallel_group([loss])

        # Cache reduced loss while accumulating gradients
        self._reduced_loss_buffer.append(reduced_loss[0])

        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            # Reduced loss for logging.
            average_reduced_loss = sum(self._reduced_loss_buffer) / len(self._reduced_loss_buffer)
            self.log('reduced_train_loss', average_reduced_loss, prog_bar=True)
            lr = self._optimizer.param_groups[0]['lr']
            self.log('lr', lr)
            self.log('global_step', self.trainer.global_step, prog_bar=True)
            self._reduced_loss_buffer = []
        return loss

    def inference_step(self, batch, batch_idx, inference=False):
        enc_input, dec_input, labels, loss_mask, enc_mask, dec_mask, position_ids, taskname_ids = batch

        mode = self.training
        self.eval()
        loss = None
        with torch.no_grad():
            output, encoder_input = self.forward(
                enc_input, dec_input, enc_mask, dec_mask, position_ids, taskname_ids, labels, inference=inference
            )
            loss = self.frozen_model.loss_func(loss_mask, output)

            predicted_token_ids, log_probs = self.frozen_model.decode(
                tokens_enc=enc_input,
                enc_mask=enc_mask,
                num_tokens_to_generate=self.decoder_seq_length,
                encoder_input=encoder_input,
            )

        self.train(mode=mode)
        return {'loss': loss, 'predicted_token_ids': predicted_token_ids, 'labels': labels}

    def inference_epoch_end(self, outputs):
        averaged_loss = None
        losses = [x['loss'] for x in outputs]
        averaged_loss = average_losses_across_data_parallel_group(losses)
        all_preds = []
        all_labels = []
        for item in outputs:
            preds = item['predicted_token_ids'].cpu().numpy().tolist()
            labels = item['labels'].cpu().numpy().tolist()

            for i, (pred, label) in enumerate(zip(preds, labels)):
                if self.tokenizer.eos_id in pred:
                    idx = pred.index(self.tokenizer.eos_id)
                    pred = pred[:idx]

                pred = [id for id in pred if id not in self.tokenizer.tokenizer.additional_special_tokens_ids]
                label = [id for id in label if id not in self.tokenizer.tokenizer.additional_special_tokens_ids]

                pred = self.tokenizer.ids_to_text(pred)
                label = self.tokenizer.ids_to_text(label)

                all_preds.append(pred)
                all_labels.append(label)

        correct = 0
        for pred, label in zip(all_preds, all_labels):
            if pred == label:
                correct += 1
        acc = correct / len(all_preds)

        return averaged_loss[0], all_preds, acc

    def validation_step(self, batch, batch_idx, inference=False):
        outcome = self.inference_step(batch, batch_idx, inference=inference)
        self.log('val_loss', outcome['loss'])
        return outcome

    def validation_epoch_end(self, outputs):
        averaged_loss, val_preds, val_acc = self.inference_epoch_end(outputs)

        # we can only log on one rank if it is rank zero so we broadcast from last rank
        torch.distributed.broadcast(averaged_loss, get_last_rank())
        self.log('val_loss', averaged_loss, prog_bar=True, rank_zero_only=True)

        self.log('val_acc', val_acc, prog_bar=True)
        logging.info(f'Validation loss: {averaged_loss}')
        logging.info(f'Validation accuracy: {val_acc}')

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        averaged_loss = average_losses_across_data_parallel_group(outputs)
        logging.info(f'test_loss: {averaged_loss[0]}')

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
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        print('build success', len(dataloader), dataset_paths)
        return dataset, dataloader

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:

        enc_input, dec_input, labels, loss_mask, enc_mask, dec_mask, position_ids, taskname_ids = batch

        input_embeds = self.embed_input_inference(enc_input, taskname_ids)

        position_embeddings = self.frozen_model.enc_dec_model.encoder_embedding.position_embeddings(position_ids)
        encoder_input = input_embeds + position_embeddings

        predicted_token_ids, log_probs = self.frozen_model.decode(
            tokens_enc=enc_input,
            enc_mask=enc_mask,
            num_tokens_to_generate=self.decoder_seq_length,
            encoder_input=encoder_input,
        )

        return {
            'enc_input': enc_input,
            'predicted_token_ids': predicted_token_ids,
            'log_probs': log_probs,
            'labels': labels,
        }

    def on_predict_epoch_end(self, outputs: List[Any]) -> None:

        all_preds = []
        all_labels = []
        all_inputs = []
        for item in outputs[0]:
            preds = item['predicted_token_ids'].cpu().numpy().tolist()
            enc_inputs = item['enc_input'].cpu().numpy().tolist()
            if item['labels'] is not None:
                labels = item['labels'].cpu().numpy().tolist()
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
                    and id not in self.tokenizer.text_to_ids('<extra_id_0>')
                ]  # delete the <extra_id_0> at the beginning of prediction
                pred = self.tokenizer.ids_to_text(pred)
                all_preds.append(pred)

                enc_input = [
                    id for id in enc_input if id not in self.tokenizer.text_to_ids('<extra_id_0>')
                ]  # delete the <extra_id_0> added to the end of input
                input = self.tokenizer.ids_to_text(enc_input)
                all_inputs.append(input)

                # If labels are given, collect them to calculate the accuracy
                if label is not None:
                    label = [
                        id
                        for id in label
                        if id not in self.tokenizer.tokenizer.additional_special_tokens_ids
                        and id not in self.tokenizer.text_to_ids('<extra_id_0>')
                    ]  # delete the <extra_id_0> at the beginning of label
                    label = self.tokenizer.ids_to_text(label)
                    all_labels.append(label)

        correct, acc = 0, None
        if all_labels:
            for pred, label in zip(all_preds, all_labels):
                if pred == label:
                    correct += 1
            acc = correct / len(all_preds)

        results = {'input_prediction_pair': list(zip(all_inputs, all_preds)), 'acc': acc}
        print(results)

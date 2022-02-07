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

import numpy as np
import torch
from apex.transformer import parallel_state, tensor_parallel
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.t0.multitask_data_manager import (
    DATA_ORG, t0_all_evaldt_names_subset,
    get_data_paths_and_splits
)
from nemo.collections.nlp.data.t0.t0_dataset import T0Dataset
from nemo.collections.common.data.dataset import ConcatDataset
from nemo.collections.nlp.models.language_modeling.megatron_glue_model import MegatronT5FineTuneModel

from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import build_train_valid_test_datasets
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common.megatron.megatron_init import (
    initialize_model_parallel_for_nemo,
    set_jit_fusion_options,
)
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.utils import AppState, logging


class MegatronT0Model(MegatronT5FineTuneModel):
    """
    Megatron t0 multitask fine tuning model
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)
        self.cfg = cfg


    def forward(
        self,
        encoder_input_ids,
        decoder_input_ids,
        encoder_attn_mask,
        decoder_attn_mask,
        encoder_decoder_attn_mask,
        tokentype_ids=None,
        lm_labels=None,
        enc_hidden_states=None,
        output_enc_hidden_only=False,
    ):
        result = self.model(
            encoder_input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            encoder_attn_mask=encoder_attn_mask,
            decoder_attn_mask=decoder_attn_mask,
            encoder_decoder_attn_mask=encoder_decoder_attn_mask,
            tokentype_ids=tokentype_ids,
            lm_labels=lm_labels,
            enc_hidden_states=enc_hidden_states,
            output_enc_hidden_only=output_enc_hidden_only,
        )
        if not output_enc_hidden_only:
            return result[0], result[1]
        else:
            return result

    def training_step(self, batch, batch_idx):
        tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, enc_dec_mask = self.process_batch(batch)

        output_tensor, encoder_hidden_states = self(
            tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask, tokentype_ids=None, lm_labels=labels
        )

        loss = self.model.loss_func(loss_mask, output_tensor)
        self.log('train_loss', loss)
        # Reduced loss for logging. This averages the loss across all workers unlike "loss" above which is specific to a DDP rank.
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
            # TODO: what is compute_consumed_samples?
            self.log('consumed_samples', self.compute_consumed_samples(self.trainer.global_step), prog_bar=True)
            self._reduced_loss_buffer = []

        return loss

    def inference_step(self, batch, batch_idx):
        loss = self.model.validation_step(batch, batch_idx)

        tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, enc_dec_mask = self.process_batch(batch)

        predicted_token_ids, log_probs = self.model.decode(
            tokens_enc=tokens_enc, enc_mask=enc_mask, num_tokens_to_generate=10  #TODO: hardcoded 10 is bad here
        )
        reduced_loss = average_losses_across_data_parallel_group([loss])  # TODO: is this needed?
        return {'loss': reduced_loss, 'predicted_token_ids': predicted_token_ids, 'labels': labels}

    def inference_epoch_end(self, outputs):
        """Uses exact match"""

        losses = [x['loss'] for x in outputs]
        averaged_loss = average_losses_across_data_parallel_group(losses)
        all_preds = []
        all_labels = []
        for item in outputs:
            preds = item['predicted_token_ids'].cpu().numpy().tolist()
            labels = item['labels'].cpu().numpy().tolist()
            for i, (pred, label) in enumerate(zip(preds, labels)):
                if self.model.tokenizer.eos_id in pred:
                    idx = pred.index(self.model.tokenizer.eos_id)
                    pred = pred[:idx]
                pred = self.model.tokenizer.ids_to_text(pred)
                label = self.model.tokenizer.ids_to_text(label)
                all_preds.append(pred)
                all_labels.append(label)

        correct = 0
        for pred, label in zip(all_preds, all_labels):
            if pred == label:
                correct += 1
        acc = correct / len(all_preds)
        return averaged_loss[0], acc

    def validation_step(self, batch, batch_idx):
        return self.inference_step(batch, batch_idx)

    def validation_epoch_end(self, outputs_dict):
        self.log('consumed_samples', self.compute_consumed_samples(self.trainer.global_step))
        for task_name, outputs in outputs_dict.items:
            val_loss, val_acc = self.inference_epoch_end(outputs)
            self.log('val_loss_%s' % task_name, val_loss, prog_bar=True)
            self.log('val_acc_%s' % task_name, val_acc, prog_bar=True)
            logging.info(f'Validation loss for {task_name}: {val_loss}')
            logging.info(f'Validation accuracy for {task_name}: {val_acc}')

    def test_step(self, batch, batch_idx):
        return self.inference_step(batch, batch_idx)

    def test_epoch_end(self, outputs_dict):
        for task_name, outputs in outputs_dict.items:
            test_loss, test_acc = self.inference_epoch_end(outputs)
            self.log('test_loss_%s' % task_name, test_loss, prog_bar=True)
            self.log('test_acc_%s' % task_name, test_acc, prog_bar=True)
            logging.info(f'Test loss for {task_name}: {test_loss}')
            logging.info(f'Test accuracy for {task_name}: {test_acc}')

    def get_dataset_list(self, split, seq_length):
        if split == 'train':
            data_dict = DATA_ORG[self.cfg.data.t0_type]
        else:
            data_dict = t0_all_evaldt_names_subset
        dataset_list = {}
        for dt_name in data_dict.keys():
            logging.info('Dataset name %s.' % dt_name)
            subsets = data_dict[dt_name]
            if not isinstance(subsets, list):
                subsets = [subsets]
            for subset in subsets:
                logging.info('Subset name %s.' % subset)
                file_name = "_%s_%s.jsonl" % (dt_name, "" if subset is None else subset)
                _, data_paths = get_data_paths_and_splits(split, self.cfg.data.file_path, file_name, dt_name)
                for file_path in data_paths:
                    dataset = T0Dataset(
                        file_path,
                        task_name=dt_name,
                        subset=subset,
                        tokenizer=self.model.tokenizer,
                        max_seq_length=seq_length,
                    )
                    #TODO: implement a better task manager
                    dataset_list["%s_%s" % (dt_name, "" if subset is None else subset)] = dataset
        return dataset_list

    def get_sampling_probs(self, dataset_list):
        data_sizes = []
        for dataset in dataset_list:
            data_sizes.append(min(len(dataset), self.cfg.data.max_data_size))
        data_sizes = np.array(data_sizes)
        sampling_probs = data_sizes / np.sum(data_sizes)
        return sampling_probs.tolist()

    def build_train_valid_test_datasets(self):
        # TODO: add cfg.data.t0_type to command args and only allow [t0_train, t0p_train, t0pp_train]:
        logging.info('Building train %s datasets.' % self.cfg.data.t0_type)
        train_data_list = self.get_dataset_list('train', self.cfg.data.train_ds.max_seq_length)
        self._train_ds = ConcatDataset(
            datasets=train_data_list.values(),
            shuffle=True,
            sampling_probabilities=self.get_sampling_probs(train_data_list)
        )
        logging.info('Building validation datasets.')
        self._validation_ds = self.get_dataset_list('validation', self.cfg.data.validation_ds.max_seq_length)
        logging.info('Building test datasets.')
        self._test_ds = self.get_dataset_list('test', self.cfg.data.test_ds.max_seq_length)

        logging.info(f'Length of train dataset: {len(self._train_ds)}')
        logging.info(f'Length of val dataset: {sum([len(dt) for dt in self._validation_ds])}')
        logging.info(f'Length of test dataset: {sum([len(dt) for dt in self._test_ds])}')
        logging.info(f'Finished building T0 datasets.')
        return self._train_ds, self._validation_ds, self._test_ds

    def build_data_loader(self, dataset, batch_size, shuffle, num_workers, pin_memory):
        """Buld dataloader given an input dataset."""

        if dataset is None:
            return None

        # Torch dataloader.
        return torch.utils.data.DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    def setup(self, stage=None):
        """A PTL method to setup the training, validation and test datasets."""
        if stage == 'predict':
            return
        #TODO: megatron_glue_model.py does not have this condition... why?
        if self._train_dl is not None and self._validation_dl is not None:
            return
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
            self._train_dl = self.build_data_loader(
                self._train_ds,
                self.cfg.data.train_ds.batch_size,
                shuffle=True,
                num_workers=self.cfg.data.train_ds.num_workers,
                pin_memory=True,
            )

    def setup_validation_data(self, cfg):
        if hasattr(self, '_validation_ds'):
            consumed_samples = 0
            self._validation_dl = {task_name: self.build_data_loader(
                dataset,
                self.cfg.data.validation_ds.batch_size,
                shuffle=False,
                num_workers=self.cfg.data.validation_ds.num_workers,
                pin_memory=True,
            ) for task_name, dataset in self._validation_ds.items()}

    def setup_test_data(self, cfg):
        if hasattr(self, '_test_ds'):
            consumed_samples = 0
            self._test_dl = {task_name: self.build_data_loader(
                dataset,
                self.cfg.data.test_ds.batch_size,
                shuffle=False,
                num_workers=self.cfg.data.test_ds.num_workers,
                pin_memory=True,
            ) for task_name, dataset in self._test_ds.items()}


    def complete(self, request: Dict):
        #TODO: not sure if I should I keep this
        """
            Autoregressively invokes language model in the inference mode
        Args:	
            request: Dictionary with the following fields
                * prompt: a string which text the model should complete.
                * tokens_to_generate: how many tokens to generate while doing prompt completion.
        Returns:	
            response: A python dictionary with the following fields
                * prompt: original text of the prompt
                * tokenized_prompt: list of (str) tokens from prompt
                * completion: a python dictionary with the following subfields:
                    * tokens: a list of triples (token, token_id, log_prob) comprising completion
                    * text: completion text (as a single string)
                
        """
        response = {}
        self.freeze()
        # naive greedy slow loop
        # TODO: add option for BeamSearchDecoder

        response['prompt'] = request['prompt'][0]
        response['completion'] = {}
        tokens_enc = request['masked_sample']

        response['masked_input'] = ' '.join(self.tokenizer.ids_to_tokens(tokens_enc[0]))
        enc_mask = self.make_inference_attention_mask_3d(tokens_enc, tokens_enc, self.tokenizer.pad_id)
        enc_mask = enc_mask < 0.5

        predicted_tokens_ids, log_probs = self.decode(tokens_enc, enc_mask, int(request['tokens_to_generate']))
        predicted_tokens_ids = predicted_tokens_ids.cpu().numpy()[0].tolist()
        log_probs = log_probs.cpu().numpy()[0].tolist()
        if self.tokenizer.eos_id in predicted_tokens_ids:
            idx = predicted_tokens_ids.index(self.tokenizer.eos_id)
            predicted_tokens_ids = predicted_tokens_ids[:idx]
        else:
            predicted_tokens_ids = [id for id in predicted_tokens_ids if id != self.tokenizer.pad_id]
        predicted_tokens_dec = self.tokenizer.ids_to_tokens(predicted_tokens_ids)
        response['completion']['text'] = self.tokenizer.tokens_to_text(predicted_tokens_dec)
        response['completion']['tokens'] = list(zip(predicted_tokens_ids, predicted_tokens_dec, log_probs))
        self.unfreeze()
        return response

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass

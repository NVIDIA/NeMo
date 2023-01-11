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


import torch
from omegaconf.omegaconf import open_dict
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.core.config import hydra_runner
from nemo.utils.app_state import AppState
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import re
import json

import torch
from tqdm.auto import tqdm

from nemo.collections.nlp.data.language_modeling.megatron.base_prompt_learning_dataset import BasePromptLearningDataset
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import T5Sentinel
from nemo.collections.nlp.modules.common import VirtualPromptSource
from nemo.collections.nlp.modules.common.megatron.utils import build_position_ids
from nemo.utils import logging

try:
    from apex.transformer import parallel_state

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


if not torch.cuda.is_available():
    raise EnvironmentError("GPU is needed for the inference")

class MockRequestDataset(Dataset):
    def __init__(self, request) -> None:
        self.request = request

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.request


class T5ZeroShotPromptLearningDataset(BasePromptLearningDataset):
    """
    The dataset class for prompt-tuning or p-tuning pretrained T5 models.
    """

    def __init__(
        self,
        datasets,
        tokenizer,
        virtual_prompt_source: VirtualPromptSource,
        task_templates: dict,
        pseudo_tokens,
        pad_token_id: str,
        max_seq_length: int,
        min_seq_length: int = 1,
        add_bos: bool = False,
        add_eos: bool = True,
        for_train: bool = True,
        decoder_starts_with_pad: bool = False,
        add_eos_to_decoder_output: bool = True,
        add_sentinel_to_input: bool = True,
        ul2_prompt_token: str = None,
    ):
        # These two variables need to be set before calling super().__init__() because the parent class calls `load_data()` which requires these attributes.
        self.decoder_starts_with_pad = decoder_starts_with_pad
        self.add_eos_to_decoder_output = add_eos_to_decoder_output
        self.add_sentinel_to_input = add_sentinel_to_input
        self.ul2_prompt_token = ul2_prompt_token
        super().__init__(
            datasets=datasets,
            tokenizer=tokenizer,
            virtual_prompt_source=virtual_prompt_source,
            task_templates=task_templates,
            pseudo_tokens=pseudo_tokens,
            pad_token_id=pad_token_id,
            max_seq_length=max_seq_length,
            min_seq_length=min_seq_length,
            add_bos=add_bos,
            add_eos=add_eos,
            for_train=for_train,
        )

    def load_data(self, dataset):
        """
        Loads a dataset by filling in the task templates specified in the config file
        with the information from each training/inference example. Converts all input 
        text into token ids. Also replaces the <|VIRTUAL_PROMPT_#|> placeholders in 
        the task templates with the actual virtual prompt token ids. 

        params:
            dataset: A list of json objects or a dictionary objects each
                     containing the information needed for a training example
        """
        skipped = 0
        for json_line in tqdm(dataset):

            # Read example dict or load the information for a single example from .json file
            if type(json_line) == dict:
                doc = json_line
            else:
                doc = json.loads(json_line)

        
            prompt_template_pre = "Question: \{question\} Context: \{context\} Answer: \{answer\}"
            taskname = "squad"
            prompt_template = prompt_template_pre
            prompt_template_fields = ["question", "context", "answer"]
            truncation_field = "context"
            answer_field = "answer"

            input_example = prompt_template
        
            # Format the input example according to the template
            input_example = self._insert_text_in_template(input_example, prompt_template_fields, doc, answer_field)
            

            # a trick to align with the data format in t5 pretraining
            input_ids = self.tokenizer.text_to_ids(input_example)
            if self.add_sentinel_to_input:
                input_ids = input_ids + self.tokenizer.text_to_ids(T5Sentinel.FIRST.value)

            # Add BOS/EOS to the input of encoder if desired, adds EOS by default
            if self.ul2_prompt_token is not None:
                ul2_prompt_token_id = self.tokenizer.text_to_ids(self.ul2_prompt_token)
                assert len(ul2_prompt_token_id) == 1
                input_ids = ul2_prompt_token_id + input_ids
            if self.add_bos:
                input_ids = [self.tokenizer.bos_id] + input_ids
            if self.add_eos:
                input_ids = input_ids + [self.tokenizer.eos_id]

            # Try to truncate input text to fit into the max sequence length
            if len(input_ids) > self.max_seq_length:
                input_ids = self._truncate_input(truncation_field, input_ids, taskname, doc, 0)

            # get answer ids
            if answer_field in doc.keys():  # training and validation
                answer_text = doc[answer_field]

                if self.decoder_starts_with_pad:
                    answer_text_ids = [self.tokenizer.pad_id]
                else:
                    answer_text_ids = [self.tokenizer.bos_id]
                # a trick to align with the data format in t5 pretraining
                if self.add_sentinel_to_input:
                    answer_text_ids += self.tokenizer.text_to_ids(T5Sentinel.FIRST.value)
                answer_text_ids += self.tokenizer.text_to_ids(answer_text)
                if self.add_eos_to_decoder_output:
                    answer_text_ids += [self.tokenizer.eos_id]
                else:
                    answer_text_ids += self.tokenizer.text_to_ids(T5Sentinel.END.value)

            # Skip example if the final length doesn't fit length requirements even after truncation
            if self.min_seq_length <= len(input_ids) <= self.max_seq_length:
    
                taskname_id = -1
                dec_input = None
                dec_labels = None

                if answer_field in doc.keys():  # training and validation
                    dec_input = answer_text_ids[:-1]
                    dec_labels = answer_text_ids[1:]

                self.examples.append((taskname_id, torch.LongTensor(input_ids), input_example, dec_input, dec_labels))
            else:
                skipped += 1

        logging.info(f'Skipped {skipped} sentences, sequence length too short or too long even after truncation')

    def _insert_text_in_template(self, input_example, prompt_template_fields, doc, answer_field):
        """ Format the input example according to the template """
        for field in prompt_template_fields:
            # discard the last one, {label} / {answer}
            # Or if some fields from the template aren't present, e.g. {answer} during inference
            # just remove that field from the template, leaving the space blank
            if field == answer_field or field not in doc.keys():
                input_example = input_example.replace('\{' + field + '\}', "")
                input_example = input_example.strip()

            else:
                field_text = doc[field]
                input_example = input_example.replace('\{' + field + '\}', field_text)

        return input_example


@hydra_runner(config_path="conf", config_name="megatron_t5_zero_shot")
def main(cfg) -> None:

    # trainer required for restoring model parallel models
    trainer = Trainer(strategy=NLPDDPStrategy(), **cfg.trainer)
    assert (
        cfg.trainer.devices * cfg.trainer.num_nodes
        == cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
    ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"

    app_state = AppState()
    if cfg.tensor_model_parallel_size > 1 or cfg.pipeline_model_parallel_size > 1:
        app_state.model_parallel_size = cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
        (
            app_state.tensor_model_parallel_rank,
            app_state.pipeline_model_parallel_rank,
            app_state.model_parallel_size,
            app_state.data_parallel_size,
            app_state.pipeline_model_parallel_split_rank,
            app_state.virtual_pipeline_model_parallel_rank,
        ) = fake_initialize_model_parallel(
            world_size=app_state.model_parallel_size,
            rank=trainer.global_rank,
            tensor_model_parallel_size_=cfg.tensor_model_parallel_size,
            pipeline_model_parallel_size_=cfg.pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank_=cfg.pipeline_model_parallel_split_rank,
        )

    model_cfg = MegatronT5Model.restore_from(
        restore_path=cfg.language_model_path,
        trainer=trainer,
        save_restore_connector=NLPSaveRestoreConnector(),
        return_config=True,
    )
    
    with open_dict(model_cfg):
        model_cfg.precision = trainer.precision
        model_cfg.micro_batch_size = cfg.data.get('micro_batch_size', 1)
        model_cfg.global_batch_size = cfg.data.get('global_batch_size', 1)

    model = MegatronT5Model.restore_from(
        restore_path=cfg.language_model_path,
        trainer=trainer,
        save_restore_connector=NLPSaveRestoreConnector(),
        override_config_path=model_cfg,
    )

    # check whether the DDP is initialized
    if parallel_state.is_unitialized():

        def dummy():
            return

        if model.trainer.strategy.launcher is not None:
            model.trainer.strategy.launcher.launch(dummy, trainer=model.trainer)
        model.trainer.strategy.setup_environment()

    model.freeze()
    model.training = False


    dataset = T5ZeroShotPromptLearningDataset(
        datasets=cfg.data.test_ds,
        tokenizer=model.tokenizer,
        virtual_prompt_source="",
        task_templates="",
        pseudo_tokens="",
        pad_token_id="",
        max_seq_length=model.cfg.max_position_embeddings,
        min_seq_length=1,
        add_bos=False,
        add_eos=False, #True
        decoder_starts_with_pad=True,
        add_eos_to_decoder_output=True,
        add_sentinel_to_input=False,
        ul2_prompt_token=None,
        for_train="",
    )

    responses = []
    for example in tqdm(dataset.examples):

        taskname_id, input_ids, input_example, dec_input, dec_labels = example

        request = {
            "prompt": input_example,
            "tokens_to_generate": 32,
            "bos_id": model.tokenizer.bos_id,
            "masked_sample": input_ids
        }

        dataset = MockRequestDataset(request)
        request_dl = DataLoader(dataset)

        response = trainer.predict(model, request_dl)

       
        
        responses.append(response[0]['completion']['text'])
        
   
    
    with open(cfg.pred_file_path, "w", encoding="utf-8") as pred_file:
        for response in responses:
            response = response.strip().replace("\n", " ")
            pred_file.write(response + "\n")
    print('test finish---------------------------------')
    

if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter

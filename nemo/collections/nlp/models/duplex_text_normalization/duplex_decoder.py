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

from time import perf_counter
from typing import List, Optional

import nltk
import torch
import wordninja
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq

import nemo.collections.nlp.data.text_normalization.constants as constants
from nemo.collections.nlp.data.text_normalization import TextNormalizationDecoderDataset
from nemo.collections.nlp.models.duplex_text_normalization.utils import is_url
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging
from nemo.utils.decorators.experimental import experimental

nltk.download('punkt')


__all__ = ['DuplexDecoderModel']


@experimental
class DuplexDecoderModel(NLPModel):
    """
    Transformer-based (duplex) decoder model for TN/ITN.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self._tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
        super().__init__(cfg=cfg, trainer=trainer)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.transformer)

    # Training
    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # Apply Transformer
        outputs = self.model(
            input_ids=batch['input_ids'],
            decoder_input_ids=batch['decoder_input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        train_loss = outputs.loss

        lr = self._optimizer.param_groups[0]['lr']
        self.log('train_loss', train_loss)
        self.log('lr', lr, prog_bar=True)
        return {'loss': train_loss, 'lr': lr}

    # Validation and Testing
    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """

        # Apply Transformer
        outputs = self.model(
            input_ids=batch['input_ids'],
            decoder_input_ids=batch['decoder_input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        val_loss = outputs.loss

        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)

        return {
            'val_loss': avg_loss,
        }

    def test_step(self, batch, batch_idx):
        """
        Lightning calls this inside the test loop with the data from the test dataloader
        passed in as `batch`.
        """
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        """
        Called at the end of test to aggregate outputs.
        :param outputs: list of individual outputs of each test step.
        """
        return self.validation_epoch_end(outputs)

    # Functions for inference
    @torch.no_grad()
    def _infer(
        self,
        sents: List[List[str]],
        nb_spans: List[int],
        span_starts: List[List[int]],
        span_ends: List[List[int]],
        inst_directions: List[str],
    ):
        """ Main function for Inference
        Args:
            sents: A list of inputs tokenized by a basic tokenizer (e.g., using nltk.word_tokenize()).
            nb_spans: A list of ints where each int indicates the number of semiotic spans in each input.
            span_starts: A list of lists where each list contains the starting locations of semiotic spans in an input.
            span_ends: A list of lists where each list contains the ending locations of semiotic spans in an input.
            inst_directions: A list of str where each str indicates the direction of the corresponding instance (i.e., INST_BACKWARD for ITN or INST_FORWARD for TN).

        Returns: A list of lists where each list contains the decoded spans for the corresponding input.
        """
        self.eval()

        if sum(nb_spans) == 0:
            return [[]] * len(sents)
        model, tokenizer = self.model, self._tokenizer
        model_max_len = model.config.n_positions
        ctx_size = constants.DECODE_CTX_SIZE
        extra_id_0 = constants.EXTRA_ID_0
        extra_id_1 = constants.EXTRA_ID_1

        # Build all_inputs
        input_centers, input_dirs, all_inputs = [], [], []
        for ix, sent in enumerate(sents):
            cur_inputs = []
            for jx in range(nb_spans[ix]):
                cur_start = span_starts[ix][jx]
                cur_end = span_ends[ix][jx]
                ctx_left = sent[max(0, cur_start - ctx_size) : cur_start]
                ctx_right = sent[cur_end + 1 : cur_end + 1 + ctx_size]
                span_words = sent[cur_start : cur_end + 1]
                span_words_str = ' '.join(span_words)
                if is_url(span_words_str):
                    span_words_str = span_words_str.lower()
                input_centers.append(span_words_str)
                input_dirs.append(inst_directions[ix])
                # Build cur_inputs
                if inst_directions[ix] == constants.INST_BACKWARD:
                    cur_inputs = [constants.ITN_PREFIX]
                if inst_directions[ix] == constants.INST_FORWARD:
                    cur_inputs = [constants.TN_PREFIX]
                cur_inputs += ctx_left
                cur_inputs += [extra_id_0] + span_words_str.split(' ') + [extra_id_1]
                cur_inputs += ctx_right
                all_inputs.append(' '.join(cur_inputs))

        # Apply the decoding model
        batch = tokenizer(all_inputs, padding=True, return_tensors='pt')
        input_ids = batch['input_ids'].to(self.device)
        generated_ids = model.generate(input_ids, max_length=model_max_len)
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Post processing
        generated_texts = self.postprocess_output_spans(input_centers, generated_texts, input_dirs)

        # Prepare final_texts
        final_texts, span_ctx = [], 0
        for nb_span in nb_spans:
            cur_texts = []
            for i in range(nb_span):
                cur_texts.append(generated_texts[span_ctx])
                span_ctx += 1
            final_texts.append(cur_texts)

        return final_texts

    def postprocess_output_spans(self, input_centers, output_spans, input_dirs):
        greek_spokens = list(constants.GREEK_TO_SPOKEN.values())
        for ix, (_input, _output) in enumerate(zip(input_centers, output_spans)):
            # Handle URL
            if is_url(_input):
                output_spans[ix] = ' '.join(wordninja.split(_output))
                continue
            # Greek letters
            if _input in greek_spokens:
                if input_dirs[ix] == constants.INST_FORWARD:
                    output_spans[ix] = _input
                if input_dirs[ix] == constants.INST_BACKWARD:
                    output_spans[ix] = constants.SPOKEN_TO_GREEK[_input]
        return output_spans

    # Functions for processing data
    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        if not train_data_config or not train_data_config.data_path:
            logging.info(
                f"Dataloader config or file_path for the train is missing, so no data loader for train is created!"
            )
            self._train_dl = None
            return
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config, mode="train")

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        if not val_data_config or not val_data_config.data_path:
            logging.info(
                f"Dataloader config or file_path for the validation is missing, so no data loader for validation is created!"
            )
            self._validation_dl = None
            return
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config, mode="val")

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        if not test_data_config or test_data_config.data_path is None:
            logging.info(
                f"Dataloader config or file_path for the test is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config, mode="test")

    def _setup_dataloader_from_config(self, cfg: DictConfig, mode: str):
        tokenizer, model = self._tokenizer, self.model
        start_time = perf_counter()
        logging.info(f'Creating {mode} dataset')
        input_file = cfg.data_path
        dataset = TextNormalizationDecoderDataset(
            input_file,
            tokenizer,
            cfg.mode,
            cfg.get('max_decoder_len', tokenizer.model_max_length),
            cfg.get('decoder_data_augmentation', False),
        )
        data_collator = DataCollatorForSeq2Seq(
            tokenizer, model=model, label_pad_token_id=constants.LABEL_PAD_TOKEN_ID,
        )
        dl = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle, collate_fn=data_collator,
        )
        running_time = perf_counter() - start_time
        logging.info(f'Took {running_time} seconds')
        return dl

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        result = []
        return result

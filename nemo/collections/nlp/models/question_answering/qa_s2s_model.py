# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright 2019 The Google Research Authors.
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

import collections
import json
from typing import Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from torch.cuda.amp import autocast
from transformers import AutoModelForSeq2SeqLM

from nemo.collections.nlp.data.question_answering.data_processor.qa_processing import (
    EVALUATION_MODE,
    INFERENCE_MODE,
    TRAINING_MODE,
    QAProcessor,
)
from nemo.collections.nlp.data.question_answering.dataset.qa_s2s_dataset import S2SQADataset
from nemo.collections.nlp.metrics.qa_metrics import QAMetrics
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.parts.utils_funcs import tensor2list
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.utils import logging


class S2SQAModel(NLPModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        self.cfg = cfg

        if self.cfg.library == "huggingface":
            self.setup_tokenizer(cfg.tokenizer)
        elif self.cfg.library == "megatron":
            # supporting MegatronT5Model in precision = fp16
            t5_cfg = MegatronT5Model.restore_from(
                restore_path=cfg.language_model.lm_checkpoint, trainer=trainer, return_config=True
            )
            # Override the T5 configuration with the one from the config file.
            OmegaConf.set_struct(t5_cfg, True)
            with open_dict(t5_cfg):
                t5_cfg.masked_softmax_fusion = False
                t5_cfg.precision = 16

            language_model = MegatronT5Model.restore_from(
                restore_path=cfg.language_model.lm_checkpoint, trainer=trainer, override_config_path=t5_cfg
            )
            self.tokenizer = language_model.tokenizer

        super().__init__(cfg=cfg, trainer=trainer, no_lm_init=True)

        if self.cfg.library == "huggingface":
            self.language_model = AutoModelForSeq2SeqLM.from_pretrained(cfg.language_model.pretrained_model_name)
            self.language_model.resize_token_embeddings(len(self.tokenizer.tokenizer))
            if self.cfg.language_model.lm_checkpoint:
                self.language_model.load_state_dict(torch.load(self.cfg.language_model.lm_checkpoint))
        elif self.cfg.library == "megatron":
            self.language_model = language_model

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        if not train_data_config or not train_data_config.file:
            logging.info(
                f"Dataloader config or file_path for the train is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return

        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config, mode=TRAINING_MODE)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        if not val_data_config or not val_data_config.file:
            logging.info(
                f"Dataloader config or file_path for the validation is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return

        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config, mode=EVALUATION_MODE)

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        if not test_data_config or test_data_config.file is None:
            logging.info(
                f"Dataloader config or file_path for the test is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return

        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config, mode=EVALUATION_MODE)

    def setup_inference_data(self, input_file, batch_size=1, num_samples=-1, num_workers=2):
        dataloader_cfg = {
            "batch_size": batch_size,
            "file": input_file,
            "shuffle": False,
            "num_samples": num_samples,
            'num_workers': num_workers,
            'pin_memory': False,
            'drop_last': False,
        }
        dataloader_cfg = OmegaConf.create(dataloader_cfg)
        inference_dl = self._setup_dataloader_from_config(cfg=dataloader_cfg, mode=INFERENCE_MODE)

        return inference_dl

    def training_step(self, batch, batch_idx):
        input_ids, input_attn_mask, unique_ids, labels = batch
        loss, _ = self.forward(input_ids, input_attn_mask, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        if self.trainer.testing:
            prefix = 'test'
        else:
            prefix = 'val'

        input_ids, input_attn_mask, unique_ids, labels = batch
        loss, per_sample_perplexity = self.forward(input_ids, input_attn_mask, labels)
        generated_answers = self._generate_candidates(input_ids, input_attn_mask)

        labels[labels == -100] = self.tokenizer.tokenizer.pad_token_id

        return {
            "unique_ids": unique_ids,
            f"{prefix}_loss": loss,
            "per_sample_perplexity": per_sample_perplexity,
            "input": self.tokenizer.tokenizer.batch_decode(input_ids, skip_special_tokens=True),
            "ground_truth_answers": self.tokenizer.tokenizer.batch_decode(labels, skip_special_tokens=True),
            "generated_answers": generated_answers,
        }

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        if self.trainer.testing:
            prefix = 'test'
        else:
            prefix = 'val'

        loss_terms = [x[f"{prefix}_loss"] for x in outputs]
        generated_answers, unique_ids, per_sample_perplexity = self._convert_dict_outputs_to_lists(
            outputs, ["generated_answers", "unique_ids", "per_sample_perplexity"]
        )

        avg_loss = torch.stack(loss_terms).mean()

        eval_dataset = self._test_dl.dataset if self.trainer.testing else self._validation_dl.dataset
        exact_match, f1 = self.evaluate(
            eval_dataset.features, eval_dataset.examples, unique_ids, per_sample_perplexity, generated_answers,
        )

        logging.info(f"{prefix} exact match {exact_match}")
        logging.info(f"{prefix} f1 {f1}")

        self.log(f'{prefix}_loss', avg_loss)
        self.log(f'{prefix}_exact_match', exact_match)
        self.log(f'{prefix}_f1', f1)

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs)

    @typecheck()
    def forward(self, input_ids, input_attn_mask, labels):
        loss, per_sample_perplexity = None, None
        if self.cfg.library == "huggingface":
            with autocast(enabled=False):
                output = self.language_model(input_ids=input_ids, attention_mask=input_attn_mask, labels=labels)
            loss = output['loss']
            lm_logits = output['logits']
            per_sample_perplexity = self._get_per_sample_perplexity(lm_logits, labels)

        elif self.cfg.library == "megatron":
            labels = torch.where(labels != -100, labels, torch.zeros_like(labels))
            output_attn_masks = torch.where(labels > 0, torch.ones_like(labels), torch.zeros_like(labels))
            unmasked_unreduced_loss = self.language_model(
                input_ids, labels[:, :-1], input_attn_mask, output_attn_masks[:, :-1], lm_labels=labels[:, 1:],
            )
            loss = self.language_model.loss_func(output_attn_masks[:, 1:], unmasked_unreduced_loss)
            per_sample_perplexity = torch.exp(unmasked_unreduced_loss)

        return loss, per_sample_perplexity

    @torch.no_grad()
    def inference(
        self, file: str, batch_size: int = 1, num_samples: int = -1, output_prediction_file: Optional[str] = None,
    ):
        all_predictions = []
        mode = self.training
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.cfg.library == "huggingface":
            try:
                # switch model to evaluation mode
                self.eval()
                self.to(device)
                logging_level = logging.get_verbosity()
                logging.set_verbosity(logging.WARNING)

                inference_dl = self.setup_inference_data(file, batch_size=batch_size, num_samples=num_samples)

                outputs = self._inference(inference_dl, device)
                generated_answers, unique_ids, per_sample_perplexity = self._convert_dict_outputs_to_lists(
                    outputs, ["generated_answers", "unique_ids", "per_sample_perplexity"]
                )
                all_predictions = self._get_predictions(
                    inference_dl.dataset.features,
                    inference_dl.dataset.examples,
                    unique_ids,
                    per_sample_perplexity,
                    generated_answers,
                )

                if output_prediction_file:
                    self._dump_predictions_to_file(
                        output_prediction_file, inference_dl.dataset.examples, all_predictions,
                    )

            finally:
                # set mode back to its original value
                self.train(mode=mode)
                logging.set_verbosity(logging_level)

        elif self.cfg.library == 'megatron':
            raise ValueError("Megatron Inference is not supported by S2SQAModel")

        return all_predictions

    def evaluate(
        self, features, examples, unique_ids, per_sample_perplexity, generated_texts,
    ):
        all_predictions = self._get_predictions(
            features, examples, unique_ids, per_sample_perplexity, generated_texts,
        )

        evaluation_results = self._evaluate_predictions(examples, all_predictions)

        return evaluation_results["exact"], evaluation_results["f1"]

    def _setup_dataloader_from_config(self, cfg: DictConfig, mode: str):
        processor = QAProcessor(cfg.file, mode)

        dataset = S2SQADataset(
            data_file=cfg.file,
            processor=processor,
            tokenizer=self.tokenizer,
            keep_doc_spans=self._cfg.dataset.keep_doc_spans,
            doc_stride=self._cfg.dataset.doc_stride,
            max_query_length=self._cfg.dataset.max_query_length,
            max_seq_length=self._cfg.dataset.max_seq_length,
            max_answer_length=self._cfg.dataset.max_answer_length,
            num_samples=cfg.num_samples,
            mode=mode,
            use_cache=self._cfg.dataset.use_cache,
        )

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            collate_fn=dataset.collate_fn,
            drop_last=cfg.drop_last,
            shuffle=cfg.shuffle,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )

        return data_loader

    @torch.no_grad()
    def _get_per_sample_perplexity(self, logits, labels):
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        unreduced_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1),)
        unreduced_loss = unreduced_loss.reshape(labels.shape)
        mask_0 = unreduced_loss != 0
        per_sample_perplexity = torch.exp((unreduced_loss * mask_0).sum(axis=1) / mask_0.sum(axis=1))

        return per_sample_perplexity

    def _get_predictions(
        self, features, examples: List, unique_ids: List[int], per_sample_perplexity: List, generated_texts: List,
    ):

        unique_id_to_pos = {}
        for index, unique_id in enumerate(unique_ids):
            unique_id_to_pos[unique_id] = index

        example_index_to_features = collections.defaultdict(list)
        for feature in features:
            example_index_to_features[feature.example_index].append(feature)

        _PrelimPrediction = collections.namedtuple(
            "PrelimPrediction", ["feature_index", "perplexity", "generated_text"]
        )

        all_predictions = collections.OrderedDict()
        for (example_index, example) in enumerate(examples):

            # finish this loop if we went through all batch examples
            if example_index >= len(unique_ids):
                break

            curr_features = example_index_to_features[example_index]
            prelim_predictions = []
            for (feature_index, feature) in enumerate(curr_features):
                pos = unique_id_to_pos[feature.unique_id]
                curr_perplexity = per_sample_perplexity[pos]
                curr_generated_text = generated_texts[pos]
                prelim_predictions.append(_PrelimPrediction(feature_index, curr_perplexity, curr_generated_text,))

            prelim_predictions = sorted(prelim_predictions, key=lambda x: x.perplexity)
            all_predictions[example.qas_id] = prelim_predictions[0]

        return all_predictions

    def _inference(self, inference_dl, device):
        outputs = []
        for i, batch in enumerate(inference_dl):

            # get predictions
            input_ids, input_attn_mask, unique_ids = batch
            input_ids, input_attn_mask = self._transfer_tensors_to_device([input_ids, input_attn_mask], device,)
            generated_texts = self._generate_candidates(input_ids, input_attn_mask)

            labels = input_ids.clone()
            _, per_sample_perplexity = self.forward(input_ids, input_attn_mask, labels)
            labels[labels == -100] = self.tokenizer.tokenizer.pad_token_id

            outputs.append(
                {
                    "unique_ids": unique_ids,
                    "per_sample_perplexity": per_sample_perplexity,
                    "generated_answers": generated_texts,
                }
            )

        return outputs

    def _generate_candidates(self, input_ids, input_attn_mask):
        num_tokens_to_generate = self.cfg.tokens_to_generate

        if self.cfg.library == "huggingface":
            param_dict = {
                "input_ids": input_ids,
                "attention_mask": input_attn_mask,
                "max_length": num_tokens_to_generate,
            }
            generated_tokens = self.language_model.generate(**param_dict)
            generated_answers = self.tokenizer.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True,)
            generated_answers = [ans.strip() for ans in generated_answers]

        elif self.cfg.library == 'megatron':
            raise ValueError("Megatron Generation is not supported by S2SQAModel")

        return generated_answers

    def _get_raw_scores(self, examples, preds):
        exact_scores = {}
        f1_scores = {}

        for example in examples:
            qas_id = example.qas_id
            gold_answers = [answer["text"] for answer in example.answers if QAMetrics.normalize_answer(answer["text"])]

            if not gold_answers:
                # For unanswerable questions, only correct answer is empty string
                gold_answers = [""]

            if qas_id not in preds:
                logging.warning(f"Missing prediction for {qas_id}")
                continue

            pred = preds[qas_id].generated_text
            exact_scores[qas_id] = max(QAMetrics.get_exact_match(pred, a) for a in gold_answers)
            f1_scores[qas_id] = max(QAMetrics.get_f1(pred, a) for a in gold_answers)

        return exact_scores, f1_scores

    def _evaluate_predictions(
        self, examples, all_predictions: Dict[str, str],
    ):
        exact, f1 = self._get_raw_scores(examples, all_predictions)
        total = len(exact)

        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact.values()) / total),
                ("f1", 100.0 * sum(f1.values()) / total),
                ("total", total),
            ]
        )

    def _dump_predictions_to_file(self, output_file, examples, predictions):
        outputs = {"data": []}
        for ex in examples:
            outputs["data"].append(
                {
                    "id": ex.qas_id,
                    "context": ex.context_text,
                    "question": ex.question_text,
                    "predicted_answer": predictions[ex.qas_id].generated_text,
                    "perplexity": predictions[ex.qas_id].perplexity,
                    "ground_truth_answers": ex.answer_text,
                }
            )

        with open(output_file, "w") as writer:
            writer.write(json.dumps(outputs))

    def _transfer_tensors_to_device(self, tensors: Tuple, device):
        tensors = (tensor.to(device) for tensor in tensors)
        return tensors

    def _convert_dict_outputs_to_lists(self, outputs, keys):
        output_lists = [[] for _ in range(len(keys))]
        for output in outputs:
            for i, key in enumerate(keys):
                if type(output[key]) == torch.Tensor:
                    output_lists[i].extend(tensor2list(output[key]))
                else:
                    output_lists[i].extend(output[key])

        return output_lists

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        result = []
        return result

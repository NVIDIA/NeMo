# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, T5ForConditionalGeneration

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.tts.g2p.data.t5 import T5G2PDataset
from nemo.collections.tts.models.base import G2PModel
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.neural_types import LabelsType, LossType, MaskType, NeuralType, TokenIndex
from nemo.utils import logging

from torchmetrics.functional.text.bleu import _bleu_score_compute, _bleu_score_update
from torchmetrics.text import SacreBLEUScore


__all__ = ['T5G2PModel']


@dataclass
class T5G2PConfig:
    train_ds: Optional[Dict[Any, Any]] = None
    validation_ds: Optional[Dict[Any, Any]] = None
    test_ds: Optional[Dict[Any, Any]] = None
    # Language conditioning parameters
    use_language_conditioning: bool = False
    language_tokens: Optional[Dict[str, str]] = None  # e.g., {"en": "<en>", "es": "<es>", "fr": "<fr>"}
    default_language: str = "en"


class T5G2PModel(G2PModel, Exportable):
    """
    T5-based grapheme-to-phoneme model.
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if self._input_types is None:
            return {
                "input_ids": NeuralType(('B', 'T'), TokenIndex()),
                "attention_mask": NeuralType(('B', 'T'), MaskType(), optional=True),
                "labels": NeuralType(('B', 'T'), LabelsType()),
            }
        return self._input_types

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        if self._output_types is None:
            return {"loss": NeuralType((), LossType())}
        return self._output_types

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self._input_types = None
        self._output_types = None
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_devices

        # Load appropriate tokenizer from HuggingFace
        self.model_name = cfg.model_name
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.max_source_len = cfg.get("max_source_len", self._tokenizer.model_max_length)
        self.max_target_len = cfg.get("max_target_len", self._tokenizer.model_max_length)
        self.do_lower = cfg.get("do_lower", False)
        
        # Language conditioning setup
        self.use_language_conditioning = cfg.get("use_language_conditioning", False)
        self.language_tokens = cfg.get("language_tokens", {"en": "<en>", "es": "<es>", "fr": "<fr>", "de": "<de>"})
        self.default_language = cfg.get("default_language", "en")
        
        if self.use_language_conditioning:
            # Add language tokens to tokenizer if they don't exist
            self._add_language_tokens_to_tokenizer()

        # Ensure passed cfg is compliant with schema
        schema = OmegaConf.structured(T5G2PConfig)
        # ModelPT ensures that cfg is a DictConfig, but do this second check in case ModelPT changes
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        elif not isinstance(cfg, DictConfig):
            raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")
        OmegaConf.merge(cfg, schema)

        super().__init__(cfg, trainer)

        # Load pretrained T5 model from HuggingFace
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        
        # Resize model embeddings if language tokens were added
        if self.use_language_conditioning and hasattr(self, '_resized_embeddings'):
            self.model.resize_token_embeddings(len(self._tokenizer))

    def _add_language_tokens_to_tokenizer(self):
        """Add language tokens to the tokenizer if they don't already exist."""
        if not self.language_tokens:
            return
            
        tokens_to_add = []
        for lang_code, token in self.language_tokens.items():
            if token not in self._tokenizer.get_vocab():
                tokens_to_add.append(token)
        
        if tokens_to_add:
            self._tokenizer.add_tokens(tokens_to_add)
            self._resized_embeddings = True
            logging.info(f"Added language tokens to tokenizer: {tokens_to_add}")
    
    def _add_language_prefix(self, text: str, language: str = None) -> str:
        """Add language prefix to input text."""
        if not self.use_language_conditioning:
            return text
            
        if language is None:
            language = self.default_language
            
        if language not in self.language_tokens:
            logging.warning(f"Language '{language}' not found in language_tokens, using default")
            language = self.default_language
            
        return f"{self.language_tokens[language]} {text}"

    @typecheck()
    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss

    # ===== Training Functions ===== #
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        train_loss = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        self.log('train_loss', train_loss)
        return train_loss

    def on_train_epoch_end(self):
        return super().on_train_epoch_end()

    def _setup_infer_dataloader(self, cfg) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a infer data loader.
        Returns:
            A pytorch DataLoader.
        """

        dataset = T5G2PDataset(
            manifest_filepath=cfg.manifest_filepath,
            tokenizer=self._tokenizer,
            max_source_len=self._tokenizer.model_max_length,
            max_target_len=-1,
            do_lower=self.do_lower,
            grapheme_field=cfg["dataset"].get("grapheme_field", "text_graphemes"),
            phoneme_field=cfg["dataset"].get("phoneme_field", "text"),
            with_labels=False,
            use_language_conditioning=self.use_language_conditioning,
            language_field=cfg["dataset"].get("language_field", "language"),
            language_tokens=self.language_tokens,
            default_language=self.default_language,
        )

        return torch.utils.data.DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            drop_last=False,
        )

    # Functions for inference
    @torch.no_grad()
    def _infer(
        self,
        config: DictConfig,
    ) -> List[int]:
        """
        Runs model inference.

        Args:
            Config: configuration file to set up DataLoader
        Returns:
            all_preds: model predictions
        """
        # store predictions for all queries in a single list
        all_preds = []
        mode = self.training
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # Switch model to evaluation mode
            self.eval()
            self.to(device)

            infer_datalayer = self._setup_infer_dataloader(DictConfig(config))
            for batch in infer_datalayer:
                input_ids, _ = batch
                generated_str, _, _ = self._generate_predictions(input_ids=input_ids.to(device))

                all_preds.extend(generated_str)
                del batch
        finally:
            # set mode back to its original value
            self.train(mode=mode)
        return all_preds

    # ===== Validation Functions ===== #
    def validation_step(self, batch, batch_idx, dataloader_idx=0, split="val"):
        input_ids, attention_mask, labels = batch

        # Get loss from forward step
        val_loss = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        # Get preds from generate function and calculate PER
        labels_str = self._tokenizer.batch_decode(
            # Need to do the following to zero out the -100s (ignore_index).
            torch.ones_like(labels) * ((labels == -100) * 100) + labels,
            skip_special_tokens=True,
        )
        generated_str, _, _ = self._generate_predictions(input_ids=input_ids, model_max_target_len=self.max_target_len)
        
        per = word_error_rate(hypotheses=generated_str, references=labels_str, use_cer=True)

        bleu_calculator = SacreBLEUScore(tokenize="13a", n_gram=4, lowercase=True, smooth="floor")
        target = [[ref] for ref in labels_str]
        bleu_score = bleu_calculator(generated_str, target)
        
        output = {f"{split}_loss": val_loss, 'per': per, 'bleu': bleu_score}

        if split == 'val':
            if isinstance(self.trainer.val_dataloaders, (list, tuple)) and len(self.trainer.val_dataloaders) > 1:
                self.validation_step_outputs[dataloader_idx].append(output)
            else:
                self.validation_step_outputs.append(output)
        else:
            if isinstance(self.trainer.test_dataloaders, (list, tuple)) and len(self.trainer.test_dataloaders) > 1:
                self.test_step_outputs[dataloader_idx].append(output)
            else:
                self.test_step_outputs.append(output)
        return output

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Lightning calls this inside the test loop with the data from the test dataloader
        passed in as `batch`.
        """
        return self.validation_step(batch, batch_idx, dataloader_idx, split="test")

    def multi_validation_epoch_end(self, outputs, dataloader_idx=0, split="val"):
        """
        Called at the end of validation to aggregate outputs (reduces across batches, not workers).
        """
        avg_loss = torch.stack([x[f"{split}_loss"] for x in outputs]).mean()
        self.log(f"{split}_loss", avg_loss, sync_dist=True)

        if split == "test":
            dataloader_name = self._test_names[dataloader_idx].upper()
        else:
            dataloader_name = self._validation_names[dataloader_idx].upper()

        avg_per = sum([x['per'] for x in outputs]) / len(outputs)
        self.log(f"{split}_per", avg_per)

        # to save all PER values for each dataset in WANDB
        self.log(f"{split}_per_{dataloader_name}", avg_per)

        avg_bleu = sum([x['bleu'] for x in outputs]) / len(outputs)
        avg_bleu = float(avg_bleu.cpu().numpy())
        self.log(f"{split}_bleu", avg_bleu)

        # to save all BLEU values for each dataset in WANDB
        self.log(f"{split}_bleu_{dataloader_name}", avg_bleu)

        logging.info(f"PER: {round(avg_per * 100, 2)}%")
        logging.info(f"BLEU: {round(avg_bleu, 2)}")
        logging.info(f"{dataloader_name}, {len(outputs)}examples")
        
        return {'loss': avg_loss, 'per': avg_per, 'bleu': avg_bleu}

    def multi_test_epoch_end(self, outputs, dataloader_idx=0):
        self.multi_validation_epoch_end(outputs, dataloader_idx, split="test")

    @torch.no_grad()
    def _generate_predictions(self, input_ids: torch.Tensor, model_max_target_len: int = 512):
        """
        Generates predictions and converts IDs to text.
        """
        outputs = self.model.generate(
            input_ids, output_scores=True, return_dict_in_generate=True, max_length=model_max_target_len
        )
        generated_ids, sequence_toks_scores = outputs['sequences'], outputs['scores']
        generated_texts = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_texts, generated_ids, sequence_toks_scores

    # ===== Dataset Setup Functions ===== #
    def _setup_dataloader_from_config(self, cfg, name):
        if "dataloader_params" not in cfg or not isinstance(cfg.dataloader_params, DictConfig):
            raise ValueError(f"No dataloader_params for {name}")

        dataset = instantiate(
            cfg.dataset,
            manifest_filepath=cfg.manifest_filepath,
            tokenizer=self._tokenizer,
            max_source_len=self.max_source_len,
            max_target_len=self.max_target_len,
            do_lower=self.do_lower,
            grapheme_field=cfg["dataset"].get("grapheme_field", "text_graphemes"),
            phoneme_field=cfg["dataset"].get("phoneme_field", "text"),
            with_labels=True,
            use_language_conditioning=self.use_language_conditioning,
            language_field=cfg["dataset"].get("language_field", "language"),
            language_tokens=self.language_tokens,
            default_language=self.default_language,
        )

        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params)

    def setup_training_data(self, cfg):
        self._train_dl = self._setup_dataloader_from_config(cfg, name="train")

    def setup_validation_data(self, cfg):
        self._validation_dl = self._setup_dataloader_from_config(cfg, name="validation")

    def setup_test_data(self, cfg):
        self._test_dl = self._setup_dataloader_from_config(cfg, name="test")

    def setup_multiple_validation_data(self, val_data_config: Union[DictConfig, Dict] = None):
        if not val_data_config or val_data_config.manifest_filepath is None:
            self._validation_dl = None
            return
        return super().setup_multiple_validation_data(val_data_config)

    def setup_multiple_test_data(self, test_data_config: Union[DictConfig, Dict] = None):
        if not test_data_config or test_data_config.manifest_filepath is None:
            self._test_dl = None
            return
        return super().setup_multiple_test_data(test_data_config)

    # ===== List Available Models - N/A =====$
    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        return []

    def _prepare_for_export(self, **kwargs):
        super()._prepare_for_export(**kwargs)

        tensor_shape = ('B', 'T')

        # Define input_types and output_types as required by export()
        self._input_types = {
            "input_ids": NeuralType(tensor_shape, TokenIndex()),
        }
        self._output_types = {
            "preds_str": NeuralType(tensor_shape, LabelsType()),
        }

    def _export_teardown(self):
        self._input_types = None
        self._output_types = None

    def input_example(self, max_batch=1, max_dim=44):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        # par = next(self.fastpitch.parameters())
        sentence = "Kupil sem si bicikel in mu zamenjal stol."
        
        # Add language conditioning if enabled
        if self.use_language_conditioning:
            sentence = self._add_language_prefix(sentence, "sl")  # Slovenian example
            
        input_ids = [sentence]
        input_encoding = self._tokenizer(
            input_ids,
            padding='longest',
            max_length=self.max_source_len,
            truncation=True,
            return_tensors='pt',
        )
        return (input_encoding.input_ids,)

    def forward_for_export(self, input_ids):
        outputs = self.model.generate(
            input_ids, output_scores=True, return_dict_in_generate=True, max_length=self.max_source_len
        )
        generated_ids, sequence_toks_scores = outputs['sequences'], outputs['scores']
        return tuple(generated_ids)
    
    def g2p_with_language(self, text: str, language: str = None) -> str:
        """
        Convert grapheme to phoneme with language conditioning.
        
        Args:
            text: Input grapheme text
            language: Language code (e.g., "en", "es", "fr")
            
        Returns:
            Phoneme string
        """
        if self.use_language_conditioning:
            text = self._add_language_prefix(text, language)
        
        input_encoding = self._tokenizer(
            [text],
            padding='longest',
            max_length=self.max_source_len,
            truncation=True,
            return_tensors='pt',
        )
        
        device = next(self.parameters()).device
        input_ids = input_encoding.input_ids.to(device)
        
        with torch.no_grad():
            generated_str, _, _ = self._generate_predictions(
                input_ids=input_ids, 
                model_max_target_len=self.max_target_len
            )
        
        return generated_str[0] if generated_str else ""

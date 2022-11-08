from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict
from pytorch_lightning import Trainer

from nemo.collections.common.data import ConcatDataset
from nemo.collections.common.losses import AggregatorLoss, CrossEntropyLoss
from nemo.collections.common.metrics import GlobalAverageLossMetric
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.nlp.data.token_classification.punct_cap_seg_dataset import (
    InferencePunctCapSegDataset,
    PunctCapSegDataset,
)
from nemo.collections.nlp.metrics.classification_report import ClassificationReport
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.core import PretrainedModelInfo, typecheck
from nemo.core.neural_types import LengthsType, LogitsType, NeuralType, ChannelType
from nemo.utils import logging


class PunctCapSegModel(NLPModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None) -> None:
        super().__init__(cfg=cfg, trainer=trainer)
        # Whether model predicts all analytics in one encoder pass, or multiple passes
        self._multipass: bool = self._cfg.get("multipass", True)
        # Potential sentence boundary detection characters
        self._full_stops = set(self._cfg.get("full_stops", [".", "?", "？", "。", "।", "؟"]))
        # Whether to run `tok.ids_to_text(tok.text_to_ids(text))` on all inputs, to make char-level predictions less
        # error-prone. Needed for SPE when normalization is not identity.
        self._pretokenize: bool = self._cfg.get("pretokenize", False)
        # Should be set to the training DS max length; default to the positional embeddings size
        self._max_length = self._cfg.get("max_length", self.bert_model.config.max_position_embeddings)
        self._freeze_encoder_n_batches = self._cfg.get("freeze_encoder_n_batches", None)

        # Retrieve labels
        self._punct_post_labels: List[str] = self._cfg.punct_post_labels
        self._punct_pre_labels: List[str] = self._cfg.punct_pre_labels
        # Map each label to its integer index
        self._punct_pre_token_to_index: Dict[str, int] = {token: i for i, token in enumerate(self._punct_pre_labels)}
        self._punct_post_token_to_index: Dict[str, int] = {token: i for i, token in enumerate(self._punct_post_labels)}
        # Resolve index of null token
        self._null_punct_token: str = self._cfg.get("null_punct_token", "<NULL>")
        self._null_punct_pre_index: int = self._punct_pre_token_to_index[self._null_punct_token]
        self._null_punct_post_index: int = self._punct_post_token_to_index[self._null_punct_token]

        # Used for loss masking. Should by synchronized with data sets.
        self._ignore_idx: int = self._cfg.get("ignore_idx", -100)

        # Used for making character-level predictions with subwords (predict max_token_len per token)
        # self._max_token_len = max(len(x) for x in self.tokenizer.vocab)
        self._using_sp = hasattr(self.tokenizer.tokenizer, "sp_model")
        if not self._using_sp:
            self._max_token_len = max(len(x) for x in self.tokenizer.vocab)
        else:
            # SentencePiece model - AutoTokenizer doesn't have 'vocab' attr for some SP models
            vocab_size = self.tokenizer.vocab_size
            self._max_token_len = max(len(self.tokenizer.ids_to_tokens([idx])[0]) for idx in range(vocab_size))

        # All logits are shape [B, T, D]
        self._punct_pre_loss: CrossEntropyLoss = CrossEntropyLoss(
            weight=cfg.loss.punct_pre.get("weight"), ignore_index=self._ignore_idx, logits_ndim=4
        )
        self._punct_post_loss: CrossEntropyLoss = CrossEntropyLoss(
            weight=cfg.loss.punct_post.get("weight"), ignore_index=self._ignore_idx, logits_ndim=4
        )
        # TODO why not BCE for a 2-class problem?
        self._seg_loss: CrossEntropyLoss = CrossEntropyLoss(
            weight=cfg.loss.seg.get("weight"), ignore_index=self._ignore_idx, logits_ndim=3
        )
        # For true-casing, we use multi-label classification to predict for each char in a subword
        self._cap_loss: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(cfg.loss.cap["weight"]) if "weight" in cfg.loss.cap else None
        )
        # Weights can be specified in punct{-pre,-post}, cap, seg order.
        self._agg_loss = AggregatorLoss(num_inputs=4, weights=cfg.loss.get("agg_loss_weights"))

        # Punctuation head takes as input encodings and predicts distributions over punct tokens
        self._punct_head_pre: TokenClassifier = TokenClassifier(
            hidden_size=self.hidden_size,
            use_transformer_init=cfg.punct_head_pre.get("use_transformer_init", True),
            num_layers=cfg.punct_head_pre.get("num_layers", 1),
            dropout=cfg.punct_head_pre.get("dropout", 0.1),
            activation="relu",
            log_softmax=False,
            num_classes=len(self._punct_pre_labels) * self._max_token_len,
        )
        self._punct_head_post: TokenClassifier = TokenClassifier(
            hidden_size=self.hidden_size,
            use_transformer_init=cfg.punct_head_post.get("use_transformer_init", True),
            num_layers=cfg.punct_head_post.get("num_layers", 1),
            dropout=cfg.punct_head_post.get("dropout", 0.1),
            activation="relu",
            log_softmax=False,
            num_classes=len(self._punct_post_labels) * self._max_token_len,
        )
        self._seg_head: TokenClassifier = TokenClassifier(
            hidden_size=self.hidden_size,
            use_transformer_init=cfg.seg_head.get("use_transformer_init", True),
            num_layers=cfg.seg_head.get("num_layers", 1),
            dropout=cfg.seg_head.get("dropout", 0.1),
            activation="relu",
            log_softmax=False,
            num_classes=2,
        )
        self._cap_head: TokenClassifier = TokenClassifier(
            hidden_size=self.hidden_size,
            use_transformer_init=cfg.cap_head.get("use_transformer_init", True),
            num_layers=cfg.cap_head.get("num_layers", 1),
            dropout=cfg.cap_head.get("dropout", 0.1),
            activation="relu",
            log_softmax=False,
            num_classes=self._max_token_len,
        )

        # Set each dataset's tokenizer. Model's tokenizer doesn't exist until we initialize BertModule, but datasets are
        # instantiated prior to that.
        if self._train_dl is not None:
            # Train DL has one ConcatDataset
            for dataset in self._train_dl.dataset.datasets:
                dataset.tokenizer = self.tokenizer
        if self._validation_dl is not None:
            # Validation DL is a list of PunctCapSegDataset
            for dataset in self._validation_dl:
                dataset.tokenizer = self.tokenizer

        # Will be populated when dev/test sets are setup
        self._dev_metrics: nn.ModuleList[nn.ModuleDict] = nn.ModuleList()
        if self._validation_dl is not None:
            self._dev_metrics = self._setup_metrics(len(self._validation_dl))
        # module list of module dict
        self._test_metrics: Optional[nn.ModuleList[nn.ModuleDict]] = None

    def setup_multiple_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        self.setup_validation_data(val_data_config)

    def setup_test_data(self, test_data_config: Union[DictConfig, Dict]):
        self._test_dl = self._setup_test_dataloader_from_config(cfg=test_data_config)
        self._test_metrics = self._setup_test_metrics(test_data_config.get("num_thresholds", 1))

    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        self._validation_dl = self._setup_eval_dataloaders_from_config(cfg=val_data_config)
        self._validation_names = [f"val_{dl.dataset.language}" for dl in self._validation_dl]
        # TODO if self._dev_metrics already exists, overwrite it?
        # self._setup_metrics(len(self._validation_dl), self._dev_metrics)

    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        self._train_dl = self._setup_train_dataloader_from_config(cfg=train_data_config)

    def _setup_test_metrics(self, num_thresholds: int = 100) -> nn.ModuleList:
        metrics: nn.ModuleList = nn.ModuleList()
        for _ in range(num_thresholds):
            metrics.append(self._setup_metrics(num_dl=1)[0])
        return metrics

    def _setup_metrics(self, num_dl: int = 1) -> nn.ModuleList:
        """Creates metrics for each data loader. Typically, we have one DL per language.

        Metrics are reported for punctuation (pre- and post-token), true casing, segmentation, and loss.

        Returns:
            A :class:``nn.ModuleList``, with one element per data loader. Each element is another
            :class:``nn.ModuleList`` of metrics for that language.

        """
        module_list: nn.ModuleList = nn.ModuleList()
        for _ in range(num_dl):
            metrics: nn.ModuleDict = nn.ModuleDict(
                {
                    "loss": GlobalAverageLossMetric(dist_sync_on_step=False, take_avg_loss=True),
                    "punct_pre_report": ClassificationReport(
                        num_classes=len(self._punct_pre_labels),
                        label_ids=self._punct_pre_token_to_index,
                        mode="macro",
                        dist_sync_on_step=False,
                    ),
                    "punct_post_report": ClassificationReport(
                        num_classes=len(self._punct_post_labels),
                        label_ids=self._punct_post_token_to_index,
                        mode="macro",
                        dist_sync_on_step=False,
                    ),
                    "cap_report": ClassificationReport(
                        num_classes=2, label_ids={"LOWER": 0, "UPPER": 1}, mode="macro", dist_sync_on_step=False
                    ),
                    "seg_report": ClassificationReport(
                        num_classes=2, label_ids={"NOSTOP": 0, "FULLSTOP": 1}, mode="macro", dist_sync_on_step=False
                    ),
                }
            )
            module_list.append(metrics)
        return module_list

    def _setup_eval_dataloaders_from_config(self, cfg) -> List[torch.utils.data.DataLoader]:
        dataloaders: List[torch.utils.data.DataLoader] = []
        for ds_config in cfg.datasets:
            # Add all common variables, if not set already
            with open_dict(ds_config):
                for k, v in cfg.get("common", {}).items():
                    if k not in ds_config:
                        ds_config[k] = v
            dataset: PunctCapSegDataset = instantiate(ds_config)
            if not isinstance(dataset, PunctCapSegDataset):
                raise ValueError(
                    f"Expected dataset config to instantiate an implementation of 'PunctCapSegDataset' but instead got "
                    f"'{type(dataset)}' from config {ds_config}."
                )
            if hasattr(self, "tokenizer"):
                dataset.tokenizer = self.tokenizer
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                collate_fn=dataset.collate_fn,
                batch_size=cfg.get("batch_size", 128),
                num_workers=cfg.get("num_workers", 8),
                pin_memory=cfg.get("pin_memory", False),
                drop_last=cfg.get("drop_last", False),
            )
            dataloaders.append(dataloader)
        return dataloaders

    def _setup_test_dataloader_from_config(self, cfg) -> List[torch.utils.data.DataLoader]:
        dataset: PunctCapSegDataset = instantiate(cfg.dataset)
        if not isinstance(dataset, PunctCapSegDataset):
            raise ValueError(
                f"Expected dataset config to instantiate an implementation of 'PunctCapSegDataset' but instead got "
                f"'{type(dataset)}' from config {cfg.dataset}."
            )
        if hasattr(self, "tokenizer"):
            dataset.tokenizer = self.tokenizer
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            batch_size=cfg.get("batch_size", 128),
            num_workers=cfg.get("num_workers", 8),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=cfg.get("drop_last", False),
        )
        return dataloader

    def _setup_train_dataloader_from_config(self, cfg) -> List[torch.utils.data.DataLoader]:
        datasets: List[PunctCapSegDataset] = []
        for ds_config in cfg.datasets:
            # Add all common variables, if not set already
            with open_dict(ds_config):
                for k, v in cfg.get("common", {}).items():
                    if k not in ds_config:
                        ds_config[k] = v
            dataset: PunctCapSegDataset = instantiate(ds_config)
            if not isinstance(dataset, PunctCapSegDataset):
                raise ValueError(
                    f"Expected dataset config to instantiate an implementation of 'PunctCapSegDataset' but instead got "
                    f"'{type(dataset)}' from config {ds_config}."
                )
            # If model tokenizer has been set already, assign it
            if hasattr(self, "tokenizer"):
                dataset.tokenizer = self.tokenizer
            datasets.append(dataset)
        # Currently only one type of dataset is implemented; ok to always use a map data set
        dataset: ConcatDataset = ConcatDataset(
            datasets=datasets,
            sampling_technique=cfg.get("sampling_technique", "temperature"),
            sampling_temperature=cfg.get("sampling_temperature", 5),
            sampling_probabilities=cfg.get("sampling_probabilities", None),
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=datasets[0].collate_fn,  # TODO assumption; works for now
            batch_size=cfg.get("batch_size", 32),
            num_workers=cfg.get("num_workers", 8),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=cfg.get("drop_last", False),
        )
        return dataloader

    def on_train_batch_start(self, batch, batch_idx: int, unused: int = 0) -> Optional[int]:
        # If freezing the encoder, freeze before batch 0 and unfreeze before batch N
        if self._freeze_encoder_n_batches is not None:
            if batch_idx == 0:
                logging.info(f"Freezing encoder for {self._freeze_encoder_n_batches} batches")
                self.bert_model.freeze()
            if batch_idx == self._freeze_encoder_n_batches:
                logging.info(f"Unfreeze encoder at batch {batch_idx}")
                self.bert_model.unfreeze()
                # Short circuit future checks
                self._freeze_encoder_n_batches = None
        return None

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"input_ids": NeuralType(("B", "T"), ChannelType()), "lengths": NeuralType(("B",), LengthsType())}

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "punct_pre_logits": NeuralType(("B", "D", "T"), LogitsType()),
            "punct_post_logits": NeuralType(("B", "D", "T"), LogitsType()),
            "cap_logits": NeuralType(("B", "D", "T"), LogitsType()),
            "seg_logits": NeuralType(("B", "D", "T"), LogitsType()),
        }

    def _run_step(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # All inputs and targets are shape [B, T]
        if self._multipass:
            (
                punct_inputs,
                cap_seg_inputs,
                punct_pre_targets,
                punct_post_targets,
                cap_targets,
                seg_targets,
                _,
                _,
            ) = batch
            # Mask sequence mask
            punct_mask = punct_inputs.ne(self.tokenizer.pad_id)
            cap_seg_mask = cap_seg_inputs.ne(self.tokenizer.pad_id)
            # Encoded output is [B, T, D]
            punct_encoded = self.bert_model(input_ids=punct_inputs, attention_mask=punct_mask, token_type_ids=None)
            cap_seg_encoded = self.bert_model(
                input_ids=cap_seg_inputs, attention_mask=cap_seg_mask, token_type_ids=None
            )
            # Megatron will return tuples; the first element is the decoder output.
            if isinstance(punct_encoded, tuple):
                punct_encoded = punct_encoded[0]
                cap_seg_encoded = cap_seg_encoded[0]
            # [B, T, D * max_token_len]
            punct_pre_logits = self._punct_head_pre(hidden_states=punct_encoded)
            punct_post_logits = self._punct_head_post(hidden_states=punct_encoded)
            cap_logits = self._cap_head(hidden_states=cap_seg_encoded)
            # [B, T, 2]
            seg_logits = self._seg_head(hidden_states=cap_seg_encoded)
        else:
            # One pass: all heads uses the same input/encoding and predict in parallel
            inputs, punct_pre_targets, punct_post_targets, cap_targets, seg_targets, _ = batch
            mask = inputs.ne(self.tokenizer.pad_id)
            # Encoded output is [B, T, D]
            encoded = self.bert_model(input_ids=inputs, attention_mask=mask, token_type_ids=None)
            if isinstance(encoded, tuple):
                encoded = encoded[0]
            # [B, T, D * max_token_len]
            punct_pre_logits = self._punct_head_pre(hidden_states=encoded)
            punct_post_logits = self._punct_head_post(hidden_states=encoded)
            cap_logits = self._cap_head(hidden_states=encoded)
            # [B, T, 2]
            seg_logits = self._seg_head(hidden_states=encoded)

        # Unfold the logits to match the targets: [B, T, max_chars_per_token, C]
        punct_pre_logits = punct_pre_logits.view([*punct_pre_logits.shape[:-1], -1, len(self._punct_pre_labels)])
        punct_post_logits = punct_post_logits.view([*punct_post_logits.shape[:-1], -1, len(self._punct_post_labels)])

        # Compute losses
        punct_pre_loss = self._punct_pre_loss(logits=punct_pre_logits, labels=punct_pre_targets)
        punct_post_loss = self._punct_post_loss(logits=punct_post_logits, labels=punct_post_targets)
        seg_loss = self._seg_loss(logits=seg_logits, labels=seg_targets)
        # If all elements are uncased, BCE returns nan. So set to zero if no targets (ja, zh, hi, etc.).
        cap_mask = cap_targets.ne(self._ignore_idx)
        if cap_mask.any():
            cap_loss = self._cap_loss(input=cap_logits[cap_mask], target=cap_targets[cap_mask].float())
        else:
            # Dimensionless 0.0 like cap_logits
            cap_loss = cap_logits.new_zeros(1).squeeze()

        loss = self._agg_loss.forward(loss_1=punct_pre_loss, loss_2=punct_post_loss, loss_3=cap_loss, loss_4=seg_loss)

        return loss, punct_pre_logits, punct_post_logits, cap_logits, seg_logits

    def training_step(self, batch: Tuple[torch.Tensor], batch_idx: int):
        loss, _, _, _, _ = self._run_step(batch)
        lr = self._optimizer.param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True)
        self.log('train_loss', loss)
        return loss

    def _eval_step(self, batch: Tuple, dataloader_idx: int = 0) -> None:
        loss, punct_pre_logits, punct_post_logits, cap_logits, seg_logits = self._run_step(batch)
        if self._multipass:
            (
                _,
                _,  # punct, cap/seg inputs (don't need because we ran step)
                punct_pre_targets,
                punct_post_targets,
                cap_targets,
                seg_targets,
                _,
                _,  # punct, cap/seg lengths (don't need because we use pad_id)
            ) = batch
        else:
            _, punct_pre_targets, punct_post_targets, cap_targets, seg_targets, _ = batch
        # All log probs are [B, T, D]
        punct_pre_preds = punct_pre_logits.argmax(dim=-1)
        punct_post_preds = punct_post_logits.argmax(dim=-1)
        cap_mask = cap_targets.ne(self._ignore_idx)
        cap_preds = cap_logits[cap_mask].sigmoid().gt(0.5)
        seg_mask = seg_targets.ne(self._ignore_idx)
        seg_preds = seg_logits.argmax(dim=-1)

        metrics: nn.ModuleDict = self._dev_metrics[dataloader_idx]
        punct_pre_mask = punct_pre_targets.ne(self._ignore_idx)
        punct_post_mask = punct_post_targets.ne(self._ignore_idx)
        num_targets = punct_pre_mask.sum() + punct_post_mask.sum() + cap_mask.sum() + seg_mask.sum()
        metrics["loss"](loss=loss, num_measurements=num_targets)
        metrics["punct_pre_report"](punct_pre_preds[punct_pre_mask], punct_pre_targets[punct_pre_mask])
        metrics["punct_post_report"](punct_post_preds[punct_post_mask], punct_post_targets[punct_post_mask])
        metrics["cap_report"](cap_preds, cap_targets[cap_mask])
        metrics["seg_report"](seg_preds[seg_mask], seg_targets[seg_mask])

    def _test_step(self, batch: Tuple, dataloader_idx: int = 0) -> None:
        loss, punct_pre_logits, punct_post_logits, cap_logits, seg_logits = self._run_step(batch)
        if self._multipass:
            (
                _,
                _,  # punct, cap/seg inputs (don't need because we ran step)
                punct_pre_targets,
                punct_post_targets,
                cap_targets,
                seg_targets,
                _,
                _,  # punct, cap/seg lengths (don't need because we use pad_id)
            ) = batch
        else:
            _, punct_pre_targets, punct_post_targets, cap_targets, seg_targets, _ = batch
        # Prepare masks
        cap_mask = cap_targets.ne(self._ignore_idx)
        seg_mask = seg_targets.ne(self._ignore_idx)
        punct_pre_mask = punct_pre_targets.ne(self._ignore_idx)
        punct_post_mask = punct_post_targets.ne(self._ignore_idx)
        # Get all probs. All log probs are [B, T, D]
        punct_pre_probs = punct_pre_logits.softmax(dim=-1)
        punct_post_probs = punct_post_logits.softmax(dim=-1)
        seg_probs = seg_logits.softmax(dim=-1)[..., 1]
        cap_probs = cap_logits[cap_mask].sigmoid()  # Setup as a BCE multi-label problem
        # Ignore the punctuation null index, for thresholding purposes
        punct_pre_probs[..., self._null_punct_pre_index] = 1e-3
        punct_post_probs[..., self._null_punct_post_index] = 1e-3

        num_thresholds = len(self._test_metrics)
        # bounds are [0.0, 1.0] inclusive, and N-2 thresholds between bounds. For one, use default 0.5
        if num_thresholds > 1:
            thresholds = torch.arange(num_thresholds) / (num_thresholds - 1)
        else:
            thresholds = [0.5]
        punct_pre_scores, punct_pre_preds = punct_pre_probs.max(dim=-1)
        punct_post_scores, punct_post_preds = punct_post_probs.max(dim=-1)
        for i, threshold in enumerate(thresholds):
            seg_preds = seg_probs.ge(threshold)
            cap_preds = cap_probs.ge(threshold)
            # Predict null if punctuation scores low
            pre_threshold_mask = punct_pre_scores.lt(threshold)
            post_threshold_mask = punct_post_scores.lt(threshold)
            thresholded_pre_preds = punct_pre_preds.masked_fill(pre_threshold_mask, self._null_punct_pre_index)
            thresholded_post_preds = punct_post_preds.masked_fill(post_threshold_mask, self._null_punct_pre_index)
            self._test_metrics[i]["punct_pre_report"](
                thresholded_pre_preds[punct_pre_mask], punct_pre_targets[punct_pre_mask]
            )
            self._test_metrics[i]["punct_post_report"](
                thresholded_post_preds[punct_post_mask], punct_post_targets[punct_post_mask]
            )
            self._test_metrics[i]["cap_report"](cap_preds, cap_targets[cap_mask])
            self._test_metrics[i]["seg_report"](seg_preds[seg_mask], seg_targets[seg_mask])

    def _get_language_for_dl_idx(self, idx: int) -> str:
        ds: PunctCapSegDataset = self._validation_dl[idx].dataset
        language = ds.language
        return language

    def _multi_eval_epoch_end(self, dataloader_idx: int):
        """ Epoch end logic for both validation and test """
        metric_dict: nn.ModuleDict = self._dev_metrics[dataloader_idx]
        # Resolve language for better logging
        language = self._get_language_for_dl_idx(dataloader_idx)

        # Compute, reset, and log the loss for this language
        loss = metric_dict["loss"].compute()
        metric_dict["loss"].reset()
        self.log(f"val_{language}_loss", loss)

        # Compute, reset, and log the precision/recall/f1 for punct/cap/seg for this language
        for analytic in ["punct_pre", "punct_post", "cap", "seg"]:
            precision, recall, f1, report = metric_dict[f"{analytic}_report"].compute()
            metric_dict[f"{analytic}_report"].reset()
            self.log(f"val_{language}_{analytic}_precision", precision)
            self.log(f"val_{language}_{analytic}_recall", recall)
            self.log(f"val_{language}_{analytic}_f1", f1)
            logging.info(f"{analytic} report for '{language}': {report}")

    # TODO re-enable these, in the case of using only one language.
    #   When using multiple data loaders, uncommenting these will break eval.
    #   When using one data loader, commenting these will break eval.
    # def validation_epoch_end(self, outputs) -> None:
    #     # Always use multi implementation and just use index 0.
    #     self.multi_validation_epoch_end(outputs=outputs, dataloader_idx=0)
    #
    def test_epoch_end(self, outputs) -> None:
        num_thresholds = len(self._test_metrics)
        if num_thresholds > 1:
            thresholds = torch.arange(num_thresholds) / (num_thresholds - 1)
        else:
            thresholds = None
        # Compute, reset, and log the precision/recall/f1 for punct/cap/seg for this threshold
        for analytic in ["punct_pre", "punct_post", "cap", "seg"]:
            if thresholds is None:
                precision, recall, f1, report = self._test_metrics[0][f"{analytic}_report"].compute()
                self.log(f"test_{analytic}_precision", precision)
                self.log(f"test_{analytic}_recall", recall)
                self.log(f"test_{analytic}_f1", f1)
                logging.info(f"{analytic} test report: {report}")
            else:
                # TODO messy and ad-hoc way to sweep thresholds
                print(f"Table for {analytic}")
                print(f"threshold\tprecision\trecall\tf1")
                for i, metrics in enumerate(self._test_metrics):
                    precision, recall, f1, report = metrics[f"{analytic}_report"].compute()
                    threshold = thresholds[i]
                    print(f"{threshold:0.2f}\t{precision:0.2f}\t{recall:0.2f}\t{f1:0.2f}")
                    metrics[f"{analytic}_report"].reset()

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0) -> None:
        self._multi_eval_epoch_end(dataloader_idx)

    def validation_step(self, batch: Tuple[torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> None:
        self._eval_step(batch=batch, dataloader_idx=dataloader_idx)

    def test_step(self, batch: Tuple[torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> None:
        self._test_step(batch=batch, dataloader_idx=dataloader_idx)

    def on_validation_epoch_start(self) -> None:
        # For datasets that generate examples on-the-fly, reset the RNG so we get the same examples every
        # epoch for stable evaluation.
        for dl in self._validation_dl:
            if hasattr(dl.dataset, "reseed_rng"):
                dl.dataset.reseed_rng()

    @typecheck()
    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor):
        raise NotImplementedError()

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        return None

    def predict_dataloader(self, config: Union[Dict, DictConfig]):
        dataset: InferencePunctCapSegDataset = InferencePunctCapSegDataset(
            tokenizer=self.tokenizer,
            input_file=config.get("input_file"),
            input_texts=config.get("texts"),
            max_length=config.get("max_length", self._max_length),
            fold_overlap=config.get("fold_overlap", 8),
            pretokenize=config.get("pretokenize", self._pretokenize),
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            batch_size=config.get("batch_size", 16),
            num_workers=config.get("num_workers", 0),
            pin_memory=config.get("pin_memory", False),
            drop_last=config.get("drop_last", False),
        )
        return dataloader

    def _unfold_tensors(
        self,
        folded_tensor: torch.Tensor,
        lengths: torch.Tensor,
        batch_ids: torch.Tensor,
        overlap: int,
        time_dim: int = 1,
    ) -> List[List]:
        # Move everything to CPU
        folded_tensor = folded_tensor.cpu()
        lengths = lengths.cpu()
        batch_ids = batch_ids.cpu()

        unfolded_outputs: List = []
        batch_id_to_indices: DefaultDict[int, List[int]] = defaultdict(list)
        # Sort by original batch index
        for position, batch_id in enumerate(batch_ids.tolist()):
            batch_id_to_indices[batch_id].append(position)
        for batch_index in range(len(batch_id_to_indices)):
            indices = batch_id_to_indices[batch_index]
            unfolded_tensor: Optional[torch.Tensor] = None
            for index in indices:
                length = lengths[index]
                # Strip BOS and EOS here. Reduce time dim by 1 because we extract batch element.
                subsegment_tensor = folded_tensor[index].narrow(time_dim - 1, 1, length - 2)
                if unfolded_tensor is None:
                    unfolded_tensor = subsegment_tensor
                else:
                    unfolded_tensor = torch.cat(
                        (
                            unfolded_tensor.narrow(
                                time_dim - 1, 0, unfolded_tensor.shape[time_dim - 1] - overlap // 2
                            ),
                            subsegment_tensor.narrow(time_dim - 1, overlap // 2, length - 2 - overlap // 2),
                        )
                    )
            # Always return lists, because this function is used after running model
            unfolded_outputs.append(unfolded_tensor.tolist())
        return unfolded_outputs

    def _get_char_cap_preds(
        self, tokens: List[str], probs: List[List[float]], oov_lens: List[int], threshold: float
    ) -> List[int]:
        """Gathers character-level truecase predictions from subword predictions"""
        preds: List[int] = []
        oov_index = 0
        for token_num, token in enumerate(tokens):
            # For unknown tokens, take no action
            if token == self.tokenizer.unk_token:
                token_len = oov_lens[oov_index]
                oov_index += 1
                preds.extend([0] * token_len)
                continue
            start = 0
            if self._using_sp:
                if token.startswith("▁"):
                    start = 1
            elif token.startswith("##"):
                start = 2
            for char_num in range(start, len(token)):
                score = probs[token_num][char_num]
                preds.append(0 if score < threshold else 1)
        return preds

    def _get_char_seg_preds(
        self, tokens: List[str], probs: List[float], oov_lens: List[int], threshold: float, full_stop_preds: List[int]
    ) -> List[int]:
        """Gathers character-level sentence boundary predictions from subword predictions"""
        # We'll do a lot of lookups
        full_stop_preds = set(full_stop_preds)
        preds: List[int] = []
        oov_index = 0
        current_char = 0
        for token_num, token in enumerate(tokens):
            # Find out how many input chars this subtoken consumes
            token_len = len(token)
            if token == self.tokenizer.unk_token:
                token_len = oov_lens[oov_index]
                oov_index += 1
            elif self._using_sp:
                if token.startswith("▁"):
                    token_len = len(token) - 1
            else:
                if token.startswith("##"):
                    token_len = len(token) - 2
            # Advance to the end of this char
            current_char += token_len
            # Note if this char should be a full stop. Only consider positions where we predicted a full stop.
            is_full_stop = current_char - 1 in full_stop_preds or token[-1] in self._full_stops
            if is_full_stop and probs[token_num] >= threshold:
                preds.append(current_char)
        return preds

    def _get_char_punct_preds(
        self,
        tokens: List[str],
        probs: List[List[float]],
        preds: List[List[int]],
        oov_lens: List[int],
        threshold: float,
        is_post: bool,
    ) -> List[str]:
        """Gathers character-level punctuation predictions from subword predictions"""
        char_preds: List[str] = []
        oov_index = 0
        labels = self._punct_post_labels if is_post else self._punct_pre_labels
        for token_num, token in enumerate(tokens):
            # For unknown tokens, take no action
            if token == self.tokenizer.unk_token:
                token_len = oov_lens[oov_index]
                oov_index += 1
                char_preds.extend([self._cfg.null_punct_token] * token_len)
                continue
            start = 0
            if self._using_sp:
                if token.startswith("▁"):
                    start = 1
            elif token.startswith("##"):
                start = 2
            for char_num in range(start, len(token)):
                score = probs[token_num][char_num]
                if score >= threshold:
                    pred = preds[token_num][char_num]
                    label = labels[pred]
                    char_preds.append(label)
                else:
                    char_preds.append(self._cfg.null_punct_token)
        return char_preds

    # TODO just copied from data set
    def _find_oov_lengths(self, input_text: str) -> List[int]:
        if (
            isinstance(self.tokenizer, AutoTokenizer)
            and hasattr(self.tokenizer.tokenizer, "do_basic_tokenize")
            and self.tokenizer.tokenizer.do_basic_tokenize
        ):
            # Special case where tokenize will insert whitespace, and alter word indices
            input_text = " ".join(self.tokenizer.tokenizer.basic_tokenizer.tokenize(input_text))
        tokens = self.tokenizer.text_to_tokens(input_text)
        oov_lengths = []
        words = input_text.split()
        word_num = 0
        for token in tokens:
            if token == self.tokenizer.unk_token:
                oov_lengths.append(len(words[word_num]))
                word_num += 1
            elif self._using_sp:
                if token.startswith("▁"):
                    word_num += 1
            else:
                if not token.startswith("##"):
                    word_num += 1
        return oov_lengths

    @torch.inference_mode()
    def infer(
        self,
        texts: List[str],
        cap_threshold: float = 0.5,
        seg_threshold: float = 0.5,
        punct_threshold: float = 0.5,
        fold_overlap: int = 20,
        batch_size: int = 32,
        max_length: Optional[int] = None,
        pretokenize: Optional[bool] = None,
        do_punctuation: bool = True,
        do_truecasing: bool = True,
        do_segmentation: bool = True,
    ) -> List[List[str]]:
        in_mode = self.training
        self.eval()
        # Default to this model's values
        if max_length is None:
            max_length = self._max_length
        if pretokenize is None:
            pretokenize = self._pretokenize
        dataloader = self.predict_dataloader(
            {
                "texts": texts,
                "max_length": max_length,
                "fold_overlap": fold_overlap,
                "batch_size": batch_size,
                "pretokenize": pretokenize,
            }
        )
        out_texts: List[List[str]] = []
        for batch in dataloader:
            folded_input_ids, folded_batch_indices, lengths, input_strings = batch
            # [B, T, D]
            encoded: torch.Tensor = self.bert_model(
                input_ids=folded_input_ids,
                attention_mask=folded_input_ids.ne(self.tokenizer.pad_id),
                token_type_ids=None,
            )

            if do_truecasing:
                # [B, T, max_token_len]
                cap_logits = self._cap_head(hidden_states=encoded)
                all_cap_scores = cap_logits.sigmoid()
                all_cap_scores = self._unfold_tensors(
                    folded_tensor=all_cap_scores, lengths=lengths, overlap=fold_overlap, batch_ids=folded_batch_indices
                )

            if do_segmentation:
                # [B, T, 2]
                seg_logits = self._seg_head(hidden_states=encoded)
                all_seg_scores = seg_logits.softmax(dim=-1)[..., 1]
                all_seg_scores = self._unfold_tensors(
                    folded_tensor=all_seg_scores, lengths=lengths, overlap=fold_overlap, batch_ids=folded_batch_indices
                )

            if do_punctuation:
                # [B, T, D * max_token_len]
                pre_logits = self._punct_head_pre(hidden_states=encoded)
                post_logits = self._punct_head_post(hidden_states=encoded)
                # Unfold token preds to char preds [B, T, D * max_token_len] -> [B, T, max_token_len, D]
                pre_logits = pre_logits.view([*pre_logits.shape[:-1], -1, len(self._punct_pre_labels)])
                post_logits = post_logits.view([*post_logits.shape[:-1], -1, len(self._punct_post_labels)])
                # Select the highest-scoring value. Set null index to very small value (in case it was 1.0)
                pre_probs = pre_logits.softmax(dim=-1)
                pre_probs[..., self._null_punct_pre_index] = 1e-3
                all_pre_scores, all_pre_preds = pre_probs.max(dim=-1)
                post_probs = post_logits.softmax(dim=-1)
                post_probs[..., self._null_punct_post_index] = 1e-3
                all_post_scores, all_post_preds = post_probs.max(dim=-1)
                all_pre_scores = self._unfold_tensors(
                    folded_tensor=all_pre_scores, lengths=lengths, overlap=fold_overlap, batch_ids=folded_batch_indices
                )
                all_pre_preds = self._unfold_tensors(
                    folded_tensor=all_pre_preds, lengths=lengths, overlap=fold_overlap, batch_ids=folded_batch_indices
                )
                all_post_scores = self._unfold_tensors(
                    folded_tensor=all_post_scores,
                    lengths=lengths,
                    overlap=fold_overlap,
                    batch_ids=folded_batch_indices,
                )
                all_post_preds = self._unfold_tensors(
                    folded_tensor=all_post_preds, lengths=lengths, overlap=fold_overlap, batch_ids=folded_batch_indices
                )

            unfolded_input_ids = self._unfold_tensors(
                folded_tensor=folded_input_ids, lengths=lengths, overlap=fold_overlap, batch_ids=folded_batch_indices
            )
            for batch_idx, ids in enumerate(unfolded_input_ids):
                input_text = input_strings[batch_idx]
                tokens = self.tokenizer.ids_to_tokens(ids)
                oov_lens = self._find_oov_lengths(input_text)
                if do_truecasing:
                    cap_char_preds = self._get_char_cap_preds(
                        tokens=tokens, probs=all_cap_scores[batch_idx], oov_lens=oov_lens, threshold=cap_threshold,
                    )
                if do_punctuation:
                    post_tokens = self._get_char_punct_preds(
                        tokens=tokens,
                        probs=all_post_scores[batch_idx],
                        preds=all_post_preds[batch_idx],
                        oov_lens=oov_lens,
                        threshold=punct_threshold,
                        is_post=True,
                    )
                    pre_tokens = self._get_char_punct_preds(
                        tokens=tokens,
                        probs=all_pre_scores[batch_idx],
                        preds=all_pre_preds[batch_idx],
                        oov_lens=oov_lens,
                        threshold=punct_threshold,
                        is_post=False,
                    )
                if do_segmentation:
                    full_stop_indices = [i for i, token in enumerate(post_tokens) if token in self._full_stops]
                    break_points = self._get_char_seg_preds(
                        tokens=tokens,
                        probs=all_seg_scores[batch_idx],
                        oov_lens=oov_lens,
                        threshold=seg_threshold,
                        full_stop_preds=full_stop_indices,
                    )

                segmented_texts: List[str] = []
                output_chars: List[str] = []
                # All character-level predictions align to non-whitespace inputs
                non_whitespace_index = 0
                for input_char in list(input_text):
                    if input_char == " ":
                        output_chars.append(" ")
                        continue
                    if do_punctuation:
                        # Maybe add punctuation before this char
                        pre_token = pre_tokens[non_whitespace_index]
                        if pre_token != self._cfg.null_punct_token:
                            output_chars.append(pre_token)
                    if do_truecasing:
                        # Append true-cased input char to output
                        if cap_char_preds[non_whitespace_index] == 1:
                            output_chars.append(input_char.upper())
                        else:
                            output_chars.append(input_char.lower())
                    else:
                        # No true-casing; pass through input char
                        output_chars.append(input_char)
                    if do_punctuation:
                        # Maybe add punctuation after this char
                        post_token = post_tokens[non_whitespace_index]
                        if post_token != self._cfg.null_punct_token:
                            output_chars.append(post_token)
                    if do_segmentation:
                        # Maybe split sentence on this char
                        if break_points and break_points[0] == non_whitespace_index + 1:
                            segmented_texts.append("".join(output_chars).strip())
                            output_chars = []
                            del break_points[0]
                    non_whitespace_index += 1
                if output_chars:
                    out_text = "".join(output_chars).strip()
                    segmented_texts.append(out_text)
                out_texts.append(segmented_texts)
        self.train(in_mode)
        return out_texts

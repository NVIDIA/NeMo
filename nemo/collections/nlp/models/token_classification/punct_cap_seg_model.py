
import enum
import itertools
from collections import defaultdict
from typing import Optional, Union, Dict, Tuple, List, Iterable

from hydra.utils import instantiate
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from nemo.collections.common.losses import AggregatorLoss, CrossEntropyLoss
from nemo.core.neural_types import NeuralType, LogitsType, LengthsType, TokenIndex
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.collections.common.data import ConcatDataset
from nemo.core import PretrainedModelInfo, typecheck
from nemo.utils import logging
from nemo.collections.common.metrics import GlobalAverageLossMetric
from nemo.collections.nlp.metrics.classification_report import ClassificationReport
from nemo.collections.nlp.data.token_classification.punct_cap_seg_dataset import (
    PunctCapSegDataset,
    InferencePunctCapSegDataset
)


class Mode(enum.Enum):
    """Value used in many places. Prefer over passing strings."""
    VAL = "val"
    TEST = "test"


class PunctCapSegModel(NLPModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None) -> None:
        super().__init__(cfg=cfg, trainer=trainer)

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

        self._max_token_len = max(len(x) for x in self.tokenizer.vocab)

        # All logits are shape [B, T, D]
        self._punct_pre_loss: CrossEntropyLoss = CrossEntropyLoss(
            weight=cfg.loss.punct_pre.get("weight"),
            ignore_index=self._ignore_idx,
            logits_ndim=4
        )
        self._punct_post_loss: CrossEntropyLoss = CrossEntropyLoss(
            weight=cfg.loss.punct_post.get("weight"),
            ignore_index=self._ignore_idx,
            logits_ndim=4
        )
        # TODO why not BCE for a 2-class problem?
        self._seg_loss: CrossEntropyLoss = CrossEntropyLoss(
            weight=cfg.loss.seg.get("weight"),
            ignore_index=self._ignore_idx,
            logits_ndim=3
        )
        # For true-casing, we use multi-label classification to predict for each char in a subword
        self._cap_loss: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(
            pos_weight=None  # torch.tensor(cfg.loss.cap["weight"]) if "weight" in cfg.loss.cap else None
        )
        # Weights can be specified in punct{-pre,-post}, cap, seg order.
        self._agg_loss = AggregatorLoss(num_inputs=4, weights=cfg.get("agg_loss_weights"))

        # Punctuation head takes as input encodings and predicts distributions over punct tokens
        self._punct_head_pre: TokenClassifier = TokenClassifier(
            hidden_size=self.hidden_size,
            use_transformer_init=cfg.punct_head_pre.get("use_transformer_init", True),
            num_layers=cfg.punct_head_pre.get("num_layers", 1),
            dropout=cfg.punct_head_pre.get("dropout", 0.1),
            activation="relu",
            log_softmax=False,
            num_classes=len(self._punct_pre_labels) * self._max_token_len
        )
        self._punct_head_post: TokenClassifier = TokenClassifier(
            hidden_size=self.hidden_size,
            use_transformer_init=cfg.punct_head_post.get("use_transformer_init", True),
            num_layers=cfg.punct_head_post.get("num_layers", 1),
            dropout=cfg.punct_head_post.get("dropout", 0.1),
            activation="relu",
            log_softmax=False,
            num_classes=len(self._punct_post_labels) * self._max_token_len
        )
        self._seg_head: TokenClassifier = TokenClassifier(
            hidden_size=self.hidden_size,
            use_transformer_init=cfg.seg_head.get("use_transformer_init", True),
            num_layers=cfg.seg_head.get("num_layers", 1),
            dropout=cfg.seg_head.get("dropout", 0.1),
            activation="relu",
            log_softmax=False,
            num_classes=2
        )
        self._cap_head: TokenClassifier = TokenClassifier(
            hidden_size=self.hidden_size,
            use_transformer_init=cfg.cap_head.get("use_transformer_init", True),
            num_layers=cfg.cap_head.get("num_layers", 1),
            dropout=cfg.cap_head.get("dropout", 0.1),
            activation="relu",
            log_softmax=False,
            num_classes=self._max_token_len
        )

        # Set each dataset's tokenizer. Model's tokenizer doesn't exist until we initialize BertModule, but datasets are
        # instantiated prior to the LM.
        if self._train_dl is not None:
            # Train DL has one ConcatDataset
            for dataset in self._train_dl.dataset.datasets:
                dataset.tokenizer = self.tokenizer
        if self._validation_dl is not None:
            # Validation DL is a list of PunctCapSegDataset
            for dataset in self._validation_dl:
                dataset.tokenizer = self.tokenizer

        # Will be populated when dev/test sets are setup
        self._dev_metrics: Iterable[nn.ModuleDict] = nn.ModuleList()
        self._test_metrics: Iterable[nn.ModuleDict] = nn.ModuleList()
        if self._validation_dl is not None:
            self._dev_metrics = self._setup_metrics(len(self._validation_dl))
        if self._test_dl is not None:
            self._test_metrics = self._setup_metrics(len(self._test_dl))

    @property
    def hard_max_length(self):
        """
        Longest sequence length that won't result in `forward()` failure, but the model may be trained on shorter
        lengths.
        """
        return self.bert_model.config.max_position_embeddings

    def setup_multiple_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        self.setup_validation_data(val_data_config)

    def setup_multiple_test_data(self, test_data_config: Union[DictConfig, Dict]):
        self.setup_test_data(test_data_config)

    def setup_test_data(self, test_data_config: Union[DictConfig, Dict]):
        self._test_dl = self._setup_eval_dataloaders_from_config(cfg=test_data_config)
        # self._setup_metrics(len(self._test_dl), self._test_metrics)

    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        self._validation_dl = self._setup_eval_dataloaders_from_config(cfg=val_data_config)
        self._validation_names = [
            f"val_{dl.dataset.language}" for dl in self._validation_dl
        ]
        # TODO if self._dev_metrics already exists, overwrite it?
        # self._setup_metrics(len(self._validation_dl), self._dev_metrics)

    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        self._train_dl = self._setup_train_dataloader_from_config(cfg=train_data_config)

    def _setup_metrics(self, num_dl: int) -> nn.ModuleList:
        """Creates metrics for each data loader. Typically, we have one DL per language.

        Metrics are reported for punctuation (pre- and post-token), true casing, segmentation, and loss.

        Returns:
            A :class:``nn.ModuleList``, with one element per data loader. Each element is another
            :class:``nn.ModuleList`` of metrics for that language.

        """
        module_list: nn.ModuleList = nn.ModuleList()
        for _ in range(num_dl):
            metrics: nn.ModuleDict = nn.ModuleDict({
                "loss": GlobalAverageLossMetric(dist_sync_on_step=False, take_avg_loss=True),
                "punct_pre_report": ClassificationReport(
                    num_classes=len(self._punct_pre_labels),
                    label_ids=self._punct_pre_token_to_index,
                    mode="macro",
                    dist_sync_on_step=False
                ),
                "punct_post_report": ClassificationReport(
                    num_classes=len(self._punct_post_labels),
                    label_ids=self._punct_post_token_to_index,
                    mode="macro",
                    dist_sync_on_step=False
                ),
                "cap_report": ClassificationReport(
                    num_classes=2,
                    label_ids={"LOWER": 0, "UPPER": 1},
                    mode="macro",
                    dist_sync_on_step=False
                ),
                "seg_report": ClassificationReport(
                    num_classes=2,
                    label_ids={"NOSTOP": 0, "FULLSTOP": 1},
                    mode="macro",
                    dist_sync_on_step=False
                )
            })
            module_list.append(metrics)
        return module_list

    def _setup_eval_dataloaders_from_config(self, cfg) -> List[torch.utils.data.DataLoader]:
        dataloaders: List[torch.utils.data.DataLoader] = []
        for ds_config in cfg.datasets:
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

    def _setup_train_dataloader_from_config(self, cfg) -> List[torch.utils.data.DataLoader]:
        datasets: List[PunctCapSegDataset] = []
        for ds_config in cfg.datasets:
            dataset: PunctCapSegDataset = instantiate(ds_config)
            if not isinstance(dataset, PunctCapSegDataset):
                raise ValueError(
                    f"Expected dataset config to instantiate an implementation of 'PunctCapSegDataset' but instead got "
                    f"'{type(dataset)}' from config {ds_config}."
                )
            if hasattr(self, "tokenizer"):
                dataset.tokenizer = self.tokenizer
            datasets.append(dataset)
        dataset: ConcatDataset = ConcatDataset(
            datasets=datasets,
            shuffle=True,
            sampling_technique=cfg.get("sampling_technique", "temperature"),
            sampling_temperature=cfg.get("sampling_temperature", 5),
            sampling_probabilities=cfg.get("sampling_probabilities", None),
            global_rank=self._trainer.global_rank,
            world_size=self._trainer.world_size
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=datasets[0].collate_fn,  # TODO assumption; works for now
            batch_size=cfg.get("batch_size", 32),
            num_workers=cfg.get("num_workers", 8),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=cfg.get("drop_last", False)
        )
        return dataloader

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "input_ids": NeuralType(("B", "T"), TokenIndex()),
            "lengths": NeuralType(("B",), LengthsType())
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "punct_pre_logits": NeuralType(("B", "D", "T"), LogitsType()),
            "punct_post_logits": NeuralType(("B", "D", "T"), LogitsType()),
            "cap_logits": NeuralType(("B", "D", "T"), LogitsType()),
            "seg_logits": NeuralType(("B", "D", "T"), LogitsType())
        }

    def _run_step(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # All inputs and targets are shape [B, T]
        (
            punct_inputs, cap_inputs, seg_inputs,
            punct_pre_targets, punct_post_targets, cap_targets, seg_targets,
            _, _, _
        ) = batch
        # Mask sequence mask
        punct_mask = punct_inputs.ne(self.tokenizer.pad_id)
        cap_mask = cap_inputs.ne(self.tokenizer.pad_id)
        seg_mask = seg_inputs.ne(self.tokenizer.pad_id)

        # Encoded output is [B, T, D]
        punct_encoded = self.bert_model(
            input_ids=punct_inputs,  attention_mask=punct_mask, token_type_ids=torch.zeros_like(punct_inputs)
        )
        cap_encoded = self.bert_model(
            input_ids=cap_inputs, attention_mask=cap_mask, token_type_ids=torch.zeros_like(cap_inputs)
        )
        seg_encoded = self.bert_model(
            input_ids=seg_inputs, attention_mask=seg_mask, token_type_ids=torch.zeros_like(seg_inputs)
        )
        # Megatron will return tuples; the first element is the decoder output.
        if isinstance(punct_encoded, tuple):
            punct_encoded = punct_encoded[0]
            cap_encoded = cap_encoded[0]
            seg_encoded = seg_encoded[0]
        # [B, T, D * max_token_len]
        punct_pre_logits = self._punct_head_pre(hidden_states=punct_encoded)
        punct_post_logits = self._punct_head_post(hidden_states=punct_encoded)
        cap_logits = self._cap_head(hidden_states=cap_encoded)
        # [B, T, max_token_len]
        seg_logits = self._seg_head(hidden_states=seg_encoded)

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

    def _eval_step(self, batch: Tuple, mode: Mode, dataloader_idx: int = 0) -> None:
        loss, punct_pre_logits, punct_post_logits, cap_logits, seg_logits = self._run_step(batch)
        (
            _, _, _,  # punct, cap, seg inputs (don't need because we ran step)
            punct_pre_targets, punct_post_targets, cap_targets, seg_targets,
            _, _, _  # punct, cap, seg lengths (don't need because we use pad_id)
        ) = batch
        # All log probs are [B, T, D]
        punct_pre_mask = punct_pre_targets.ne(self._ignore_idx)
        punct_pre_preds = punct_pre_logits.argmax(dim=-1)
        punct_post_mask = punct_post_targets.ne(self._ignore_idx)
        punct_post_preds = punct_post_logits.argmax(dim=-1)
        cap_mask = cap_targets.ne(self._ignore_idx)
        cap_preds = cap_logits[cap_mask].sigmoid().gt(0.5)
        seg_mask = seg_targets.ne(self._ignore_idx)
        seg_preds = seg_logits.argmax(dim=-1)

        eval_modules: Iterable[nn.ModuleDict] = self._dev_metrics if mode == Mode.VAL else self._test_metrics
        metrics: nn.ModuleDict = eval_modules[dataloader_idx]
        num_targets = punct_pre_mask.sum() + punct_post_mask.sum() + cap_mask.sum() + seg_mask.sum()
        metrics["loss"](loss=loss, num_measurements=num_targets)
        metrics["punct_pre_report"](punct_pre_preds[punct_pre_mask], punct_pre_targets[punct_pre_mask])
        metrics["punct_post_report"](punct_post_preds[punct_post_mask], punct_post_targets[punct_post_mask])
        metrics["cap_report"](cap_preds, cap_targets[cap_mask])
        metrics["seg_report"](seg_preds[seg_mask], seg_targets[seg_mask])

    def _get_language_for_dl_idx(self, idx: int, mode: Mode) -> str:
        dl_list = self._validation_dl if mode == Mode.VAL else self._test_dl
        ds: PunctCapSegDataset = dl_list[idx].dataset
        language = ds.language
        return language

    def _multi_eval_epoch_end(self, mode: Mode, dataloader_idx: int):
        """ Epoch end logic for both validation and test """
        mod_list: Iterable[nn.ModuleDict] = self._dev_metrics if mode == Mode.VAL else self._test_metrics
        metric_dict: nn.ModuleDict = mod_list[dataloader_idx]
        # Resolve language for better logging
        language = self._get_language_for_dl_idx(dataloader_idx, mode)

        # Compute, reset, and log the loss for this language
        loss = metric_dict["loss"].compute()
        metric_dict["loss"].reset()
        self.log(f"{mode.value}_{language}_loss", loss)

        # Compute, reset, and log the precision/recall/f1 for punct/cap/seg for this language
        for analytic in ["punct_pre", "punct_post", "cap", "seg"]:
            precision, recall, f1, report = metric_dict[f"{analytic}_report"].compute()
            metric_dict[f"{analytic}_report"].reset()
            self.log(f"{mode.value}_{language}_{analytic}_precision", precision)
            self.log(f"{mode.value}_{language}_{analytic}_recall", recall)
            self.log(f"{mode.value}_{language}_{analytic}_f1", f1)
            logging.info(f"{analytic} report for '{language}': {report}")

    # TODO re-enable these, in the case of using only one language.
    # def validation_epoch_end(self, outputs) -> None:
    #     # Always use multi implementation and just use index 0.
    #     self.multi_validation_epoch_end(outputs=outputs, dataloader_idx=0)
    #
    # def test_epoch_end(self, outputs) -> None:
    #     # Always use multi implementation and just use index 0.
    #     self.multi_test_epoch_end(outputs=outputs, dataloader_idx=0)

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0) -> None:
        self._multi_eval_epoch_end(Mode.VAL, dataloader_idx)

    def validation_step(self, batch: Tuple[torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> None:
        self._eval_step(batch=batch, mode=Mode.VAL, dataloader_idx=dataloader_idx)

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0) -> None:
        self._multi_eval_epoch_end(Mode.TEST, dataloader_idx)

    def test_step(self, batch: Tuple[torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> None:
        self._eval_step(batch=batch, mode=Mode.TEST, dataloader_idx=dataloader_idx)

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
            max_length=config.get("max_length", self.hard_max_length),
            fold_overlap=config.get("fold_overlap", 16)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            batch_size=config.get("batch_size", 16),
            num_workers=config.get("num_workers", 8),
            pin_memory=config.get("pin_memory", False),
            drop_last=config.get("drop_last", False),
        )
        return dataloader

    def _extract_valid_indices(
            self,
            subtokens: List[str],
            scores: torch.Tensor,
            preds: torch.Tensor
    ) -> Tuple[List, List]:
        """

        Given input tokens of length T and a tensor of shape [T, N], where T is the subtoken sequence length and N is
        the maximum number of chars per subtoken, returns a list of lists, each element

        Args:
            subtokens: List with shape [T] of subtokens
            scores: Tensor with shape [T, N] where N is the longest possible subtoken
            preds: Tensor with shape [T, N] where N is the largest possible subtoken

        """
        output_scores: List[float] = []
        output_preds: List[int] = []
        for token, token_scores, token_preds in zip(subtokens, scores, preds):
            valid_start = 0
            valid_stop = len(token)
            if token.startswith("##"):
                valid_start = 2
            if token == self.tokenizer.unk_token:
                # Utilize only the first prediction array for OOVs TODO assuming OOV is single char
                valid_stop = 1
            for i in range(valid_start, valid_stop):
                output_scores.append(token_scores[i].item())
                output_preds.append(token_preds[i].item())
        return output_scores, output_preds

    def _unfold_tensors(
            self,
            folded_tensor: torch.Tensor,
            lengths: torch.Tensor,
            batch_ids: torch.Tensor,
            overlap: int,
            time_dim: int = 1,
            return_tensors: bool = True
    ) -> Union[List[List], List[torch.Tensor]]:
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
                            unfolded_tensor.narrow(time_dim - 1, 0, unfolded_tensor.shape[time_dim - 1] - overlap//2),
                            subsegment_tensor.narrow(time_dim - 1, overlap//2, length - 2 - overlap//2)
                        )
                    )
            if return_tensors:
                unfolded_outputs.append(unfolded_tensor)
            else:
                unfolded_outputs.append(unfolded_tensor.tolist())
        return unfolded_outputs

    @torch.inference_mode()
    def infer_punctuation(
            self,
            texts: List[str],
            threshold: float = 0.0,
            fold_overlap: int = 16,
            batch_size: int = 32,
            max_length: Optional[int] = None
    ) -> List[str]:
        if max_length is None:
            max_length = self.hard_max_length
        dataloader = self.predict_dataloader(
            {
                "texts": texts,
                "max_length": max_length,
                "fold_overlap": fold_overlap,
                "batch_size": batch_size
            }
        )

        output_texts: List[str] = []
        for batch in dataloader:
            folded_input_ids, folded_batch_indices, lengths, input_strings = batch
            # [B, T, D]
            encoded: torch.Tensor = self.bert_model(
                input_ids=folded_input_ids,
                attention_mask=folded_input_ids.ne(self.tokenizer.pad_id),
                token_type_ids=torch.zeros_like(folded_input_ids)
            )
            # [B, T, D * max_token_len]
            pre_logits = self._punct_head_pre(hidden_states=encoded)
            post_logits = self._punct_head_post(hidden_states=encoded)
            # Unfold token preds to char preds [B, T, D * max_token_len] -> [B, T, max_token_len, D]
            pre_logits = pre_logits.view([*pre_logits.shape[:-1], -1, len(self._punct_pre_labels)])
            post_logits = post_logits.view([*post_logits.shape[:-1], -1, len(self._punct_post_labels)])

            # Select the highest-scoring value
            # [B, T, max_token_len, D] -> [B, T, max_token_len]
            all_pre_scores, all_pre_preds = pre_logits.softmax(dim=-1).max(dim=-1)
            all_post_scores, all_post_preds = post_logits.softmax(dim=-1).max(dim=-1)
            # TODO modify _unfold_tensors to accept list of inputs to reduce these calls
            all_pre_scores = self._unfold_tensors(
                folded_tensor=all_pre_scores,
                lengths=lengths,
                overlap=fold_overlap,
                batch_ids=folded_batch_indices
            )
            all_pre_preds = self._unfold_tensors(
                folded_tensor=all_pre_preds,
                lengths=lengths,
                overlap=fold_overlap,
                batch_ids=folded_batch_indices
            )
            all_post_scores = self._unfold_tensors(
                folded_tensor=all_post_scores,
                lengths=lengths,
                overlap=fold_overlap,
                batch_ids=folded_batch_indices
            )
            all_post_preds = self._unfold_tensors(
                folded_tensor=all_post_preds,
                lengths=lengths,
                overlap=fold_overlap,
                batch_ids=folded_batch_indices
            )
            unfolded_input_ids = self._unfold_tensors(
                folded_tensor=folded_input_ids,
                lengths=lengths,
                overlap=fold_overlap,
                batch_ids=folded_batch_indices
            )

            for batch_idx in range(len(all_pre_scores)):
                # Add punctuation to the input because the HF tokenizer produces non-invertible transformations
                input_chars = list(input_strings[batch_idx])
                pre_scores = all_pre_scores[batch_idx]
                pre_preds = all_pre_preds[batch_idx]
                post_scores = all_post_scores[batch_idx]
                post_preds = all_post_preds[batch_idx]
                # Strip BOS/EOS
                ids = unfolded_input_ids[batch_idx].tolist()
                tokens = self.tokenizer.ids_to_tokens(ids)
                pre_scores, pre_preds = self._extract_valid_indices(tokens, pre_scores, pre_preds)
                post_scores, post_preds = self._extract_valid_indices(tokens, post_scores, post_preds)
                output_char_index = 0
                output_chars: List[str] = []
                input_index = 0
                for i in range(len(post_scores)):
                    # There will be no prediction for spaces
                    if input_chars[input_index] == " ":
                        input_index += 1
                        output_chars.append(" ")
                    output_char_index += 1
                    if pre_preds[i] != self._null_punct_pre_index and pre_scores[i] > threshold:
                        pre_punct = self._punct_pre_labels[pre_preds[i]]
                        output_chars.append(pre_punct)
                    output_chars.append(input_chars[input_index])
                    if post_preds[i] != self._null_punct_post_index and post_scores[i] > threshold:
                        post_punct = self._punct_post_labels[post_preds[i]]
                        output_chars.append(post_punct)
                    input_index += 1
                output_texts.append("".join(output_chars))
        return output_texts

    @torch.inference_mode()
    def infer_segmentation(
            self,
            texts: List[str],
            threshold: float = 0.5,
            batch_size: int = 32,
            fold_overlap: int = 16,
            max_length: Optional[int] = None
    ) -> List[List[str]]:
        if max_length is None:
            max_length = self.hard_max_length
        dataloader = self.predict_dataloader(
            {
                "texts": texts,
                "max_length": max_length,
                "fold_overlap": fold_overlap,
                "batch_size": batch_size
            }
        )

        output_texts: List[List[str]] = []
        for batch in dataloader:
            folded_input_ids, folded_batch_indices, lengths, input_strings = batch
            # [B, T, D]
            encoded: torch.Tensor = self.bert_model(
                input_ids=folded_input_ids,
                attention_mask=folded_input_ids.ne(self.tokenizer.pad_id),
                token_type_ids=torch.zeros_like(folded_input_ids)
            )
            logits = self._seg_head(hidden_states=encoded)
            # Keep p(full_stop)
            all_scores = logits.softmax(dim=-1)[..., 1]
            all_scores = self._unfold_tensors(
                folded_tensor=all_scores,
                lengths=lengths,
                overlap=fold_overlap,
                batch_ids=folded_batch_indices,
                return_tensors=False
            )
            unfolded_input_ids = self._unfold_tensors(
                folded_tensor=folded_input_ids,
                lengths=lengths,
                overlap=fold_overlap,
                batch_ids=folded_batch_indices,
                return_tensors=False
            )
            for batch_idx, scores in enumerate(all_scores):
                ids = unfolded_input_ids[batch_idx]
                tokens = self.tokenizer.ids_to_tokens(ids)
                segmented_texts = []
                start = 0
                stop = 0
                input_text = input_strings[batch_idx]
                for token, token_score in zip(tokens, scores):
                    if input_text[stop] == " ":
                        stop += 1
                    # Determine the input character number at the end of this subtoken
                    valid_num_chars = len(token)
                    if token.startswith("##"):
                        valid_num_chars -= 2
                    if token == self.tokenizer.unk_token:
                        valid_num_chars = 1  # TODO assuming OOV is single char
                    stop += valid_num_chars
                    if token_score > threshold:
                        extracted_text = input_text[start:stop].strip()
                        segmented_texts.append(extracted_text)
                        start = stop
                if stop > start:
                    segmented_texts.append(input_text[start:stop])
                output_texts.append(segmented_texts)
        return output_texts

    @torch.inference_mode()
    def infer_capitalization(
            self,
            texts: List[List[str]],
            threshold: float = 0.5,
            batch_size: int = 32,
            fold_overlap: int = 16,
            max_length: Optional[int] = None
    ) -> List[List[str]]:
        if max_length is None:
            max_length = self.hard_max_length
        # Flatten list of lists into one list
        flat_texts = list(itertools.chain(*texts))
        dataloader = self.predict_dataloader(
            {
                "texts": flat_texts,
                "max_length": max_length,
                "fold_overlap": fold_overlap,
                "batch_size": batch_size
            }
        )

        flat_out_texts: List[str] = []
        for batch in dataloader:
            folded_input_ids, folded_batch_indices, lengths, input_strings = batch
            # [B, T, D]
            encoded: torch.Tensor = self.bert_model(
                input_ids=folded_input_ids,
                attention_mask=folded_input_ids.ne(self.tokenizer.pad_id),
                token_type_ids=torch.zeros_like(folded_input_ids)
            )
            logits = self._cap_head(hidden_states=encoded)
            # [B, T, max_token_len]
            all_probs_upper = logits.sigmoid()
            all_probs_upper = self._unfold_tensors(
                folded_tensor=all_probs_upper,
                lengths=lengths,
                overlap=fold_overlap,
                batch_ids=folded_batch_indices,
                return_tensors=False
            )
            unfolded_input_ids = self._unfold_tensors(
                folded_tensor=folded_input_ids,
                lengths=lengths,
                overlap=fold_overlap,
                batch_ids=folded_batch_indices,
                return_tensors=False
            )

            for batch_idx, probs_upper in enumerate(all_probs_upper):
                ids = unfolded_input_ids[batch_idx]
                tokens = self.tokenizer.ids_to_tokens(ids)
                # Iterate over tokens, tracking which char we are on
                output_chars: List[str] = list(input_strings[batch_idx])
                # Current position in the outputs as we modify them
                output_char_index = 0
                for token_num, token in enumerate(tokens):
                    if output_chars[output_char_index] == " ":
                        output_char_index += 1
                    # Leave OOV unmodified. TODO Assume OOV was a single char
                    if token == self.tokenizer.unk_token:
                        output_char_index += 1
                        continue
                    valid_start = 0
                    valid_stop = len(token)
                    if token.startswith("##"):
                        valid_start = 2
                    for valid_char_num in range(valid_start, valid_stop):
                        score = probs_upper[token_num][valid_char_num]
                        if score > threshold:
                            output_chars[output_char_index] = output_chars[output_char_index].upper()
                        else:
                            output_chars[output_char_index] = output_chars[output_char_index].lower()
                        output_char_index += 1
                output_text = "".join(output_chars)
                flat_out_texts.append(output_text)
        # Un-flatten the texts back into their original lists
        output_texts: List[List[str]] = []
        start = 0
        for i in range(len(texts)):
            stop = start + len(texts[i])
            output_texts.append(flat_out_texts[start:stop])
            start = stop
        return output_texts

    @torch.inference_mode()
    def infer_cap_seg(
            self,
            texts: List[str],
            cap_threshold: float = 0.5,
            seg_threshold: float = 0.5,
            fold_overlap: int = 16,
            batch_size: int = 32,
            max_length: Optional[int] = None
    ) -> List[str]:
        if max_length is None:
            max_length = self.hard_max_length
        dataloader = self.predict_dataloader(
            {
                "texts": texts,
                "max_length": max_length,
                "fold_overlap": fold_overlap,
                "batch_size": batch_size
            }
        )
        out_texts: List[List[str]] = []
        for batch in dataloader:
            folded_input_ids, folded_batch_indices, lengths, input_strings = batch
            # [B, T, D]
            encoded: torch.Tensor = self.bert_model(
                input_ids=folded_input_ids,
                attention_mask=folded_input_ids.ne(self.tokenizer.pad_id),
                token_type_ids=torch.zeros_like(folded_input_ids)
            )

            cap_logits = self._cap_head(hidden_states=encoded)
            seg_logits = self._seg_head(hidden_states=encoded)
            # [B, T, max_token_len]
            all_cap_scores = cap_logits.sigmoid()
            # [B, T, 2]
            all_seg_preds = seg_logits.softmax(dim=-1)[..., 1].gt(seg_threshold)
            all_cap_scores = self._unfold_tensors(
                folded_tensor=all_cap_scores,
                lengths=lengths,
                overlap=fold_overlap,
                batch_ids=folded_batch_indices,
                return_tensors=False
            )
            all_seg_preds = self._unfold_tensors(
                folded_tensor=all_seg_preds,
                lengths=lengths,
                overlap=fold_overlap,
                batch_ids=folded_batch_indices,
                return_tensors=False
            )
            unfolded_input_ids = self._unfold_tensors(
                folded_tensor=folded_input_ids,
                lengths=lengths,
                overlap=fold_overlap,
                batch_ids=folded_batch_indices,
                return_tensors=False
            )

            for batch_idx, ids in enumerate(unfolded_input_ids):
                tokens = self.tokenizer.ids_to_tokens(ids)
                cap_scores = all_cap_scores[batch_idx]
                seg_preds = all_seg_preds[batch_idx]
                # Iterate over tokens, tracking which char we are on
                output_chars: List[str] = list(input_strings[batch_idx])
                # Current position in the outputs as we modify them
                output_char_index = 0
                start = 0
                segmented_texts: List[str] = []
                for token_num, token in enumerate(tokens):
                    if output_chars[output_char_index] == " ":
                        output_char_index += 1
                    # Leave OOV unmodified. TODO Assume OOV was a single char
                    if token == self.tokenizer.unk_token:
                        output_char_index += 1
                        continue
                    valid_start = 0
                    valid_stop = len(token)
                    if token.startswith("##"):
                        valid_start = 2
                    for valid_char_num in range(valid_start, valid_stop):
                        cap_score = cap_scores[token_num][valid_char_num]
                        if cap_score > cap_threshold:
                            output_chars[output_char_index] = output_chars[output_char_index].upper()
                        else:
                            output_chars[output_char_index] = output_chars[output_char_index].lower()
                        output_char_index += 1
                    if seg_preds[token_num]:
                        out_text = "".join(output_chars[start:output_char_index]).strip()
                        segmented_texts.append(out_text)
                        start = output_char_index
                if output_char_index > start:
                    out_text = "".join(output_chars[start:output_char_index]).strip()
                    segmented_texts.append(out_text)
                out_texts.append(segmented_texts)
        return out_texts

    @torch.inference_mode()
    def infer(
            self,
            texts: List[str],
            punct_threshold: float = 0.5,
            seg_threshold: float = 0.5,
            truecase_threshold: float = 0.5,
            batch_size: int = 32,
            fold_overlap: int = 16,
            max_length: Optional[int] = None,
            two_pass: bool = True
    ) -> List[List[str]]:
        """

        Args:
            texts: List of input texts to process.
            batch_size: Sequences per batch.
            fold_overlap: For sequences > `max_length`, re-use this many tokens from the end of each subsegment for the
                beginning of the next.
            max_length: Break the sequences to have at most this many tokens per batch element.
            punct_threshold: Punctuation threshold. If 0.0, use argmax for all non-null predictions.
            seg_threshold: Sentence boundary detection threshold. Split on all subtokens which score above this value.
            truecase_threshold: True-casing threshold. Upper-case all chars that score above this value; lower-case all
                others.
            two_pass: If true, add punctuation after an initial encoding, and re-encode the punctuated text and run
                sentence boundary detection and true-casing in parallel. If false, re-encode the data three times for
                punctuation, segmentation, and true-casing, in that order.
        """
        in_mode = self.training
        self.eval()
        punctuated_texts: List[str] = self.infer_punctuation(
            texts,
            threshold=punct_threshold,
            batch_size=batch_size,
            fold_overlap=fold_overlap,
            max_length=max_length
        )
        if not two_pass:
            segmented_texts: List[List[str]] = self.infer_segmentation(
                punctuated_texts,
                threshold=seg_threshold,
                batch_size=batch_size,
                fold_overlap=fold_overlap,
                max_length=max_length
            )
            output_texts: List[List[str]] = self.infer_capitalization(
                segmented_texts,
                threshold=truecase_threshold,
                batch_size=batch_size,
                fold_overlap=fold_overlap,
                max_length=max_length
            )
        else:
            output_texts: List[List[str]] = self.infer_cap_seg(
                punctuated_texts,
                cap_threshold=truecase_threshold,
                seg_threshold=seg_threshold,
                fold_overlap=fold_overlap,
                batch_size=batch_size,
                max_length=max_length
            )
        self.train(in_mode)
        return output_texts

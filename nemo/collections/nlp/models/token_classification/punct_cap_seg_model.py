
import enum
import itertools
from typing import Optional, Union, Dict, Tuple, List, Iterable

from hydra.utils import instantiate
import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.common.losses import AggregatorLoss, CrossEntropyLoss
from nemo.core.neural_types import NeuralType, LogitsType, LengthsType, TokenIndex
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.collections.common.data import ConcatDataset
from nemo.core import PretrainedModelInfo, typecheck
from nemo.utils import logging
from nemo.collections.common.metrics import GlobalAverageLossMetric
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.nlp.metrics.classification_report import ClassificationReport
from nemo.collections.nlp.data.token_classification.punct_cap_seg_dataset import (
    PunctCapSegDataset,
    CharTokenizerOverlay
)


class Mode(enum.Enum):
    """Value used in many places. Prefer over passing strings."""
    VAL = "val"
    TEST = "test"


class PunctCapSegModel(NLPModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None) -> None:
        super().__init__(cfg=cfg, trainer=trainer)
        # Maybe turn off HF's "convenient" basic tokenization, which messes with punctuation and Chinese chars. This can
        # produce unexpected (and undesirable) results.
        if isinstance(self.tokenizer, AutoTokenizer):
            self.tokenizer.tokenizer.do_basic_tokenize = False
        # Wrap the tokenizer with a tokenizer that always produces chars, for capitalization inputs.
        self._char_tokenizer = CharTokenizerOverlay(self.tokenizer)

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

        # All logits are shape [B, T, D]
        self._punct_pre_loss: CrossEntropyLoss = CrossEntropyLoss(
            weight=cfg.loss.punct_pre.get("weight"),
            ignore_index=self._ignore_idx,
            logits_ndim=3
        )
        self._punct_post_loss: CrossEntropyLoss = CrossEntropyLoss(
            weight=cfg.loss.punct_post.get("weight"),
            ignore_index=self._ignore_idx,
            logits_ndim=3
        )
        self._seg_loss: CrossEntropyLoss = CrossEntropyLoss(
            weight=cfg.loss.seg.get("weight"),
            ignore_index=self._ignore_idx,
            logits_ndim=3
        )
        self._cap_loss: CrossEntropyLoss = CrossEntropyLoss(
            weight=cfg.loss.cap.get("weight"),
            ignore_index=self._ignore_idx,
            logits_ndim=3
        )
        # Weights can be specified in punct-cap-seg order.
        self._agg_loss = AggregatorLoss(num_inputs=4, weights=cfg.get("agg_loss_weights"))

        # Punctuation head takes as input encodings and predicts distributions over punct tokens
        self._punct_head_pre: TokenClassifier = TokenClassifier(
            hidden_size=self.hidden_size,
            use_transformer_init=cfg.punct_head_pre.get("use_transformer_init", True),
            num_layers=cfg.punct_head_pre.get("num_layers", 1),
            dropout=cfg.punct_head_pre.get("dropout", 0.1),
            activation="relu",
            log_softmax=False,
            num_classes=len(self._punct_pre_labels)
        )
        self._punct_head_post: TokenClassifier = TokenClassifier(
            hidden_size=self.hidden_size,
            use_transformer_init=cfg.punct_head_post.get("use_transformer_init", True),
            num_layers=cfg.punct_head_post.get("num_layers", 1),
            dropout=cfg.punct_head_post.get("dropout", 0.1),
            activation="relu",
            log_softmax=False,
            num_classes=len(self._punct_post_labels)
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
            num_classes=2
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
            collate_fn=datasets[0].collate_fn,
            batch_size=cfg.get("batch_size", 128),
            num_workers=cfg.get("num_workers", 8),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=cfg.get("drop_last", False),
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
        # Megatron will return Tuples; make an assumption if we have a Tuple
        if isinstance(punct_encoded, tuple):
            punct_encoded = punct_encoded[0]
            cap_encoded = cap_encoded[0]
            seg_encoded = seg_encoded[0]
        # All logits are shape [B, T, D]
        punct_pre_logits = self._punct_head_pre(hidden_states=punct_encoded)
        punct_post_logits = self._punct_head_post(hidden_states=punct_encoded)
        cap_logits = self._cap_head(hidden_states=cap_encoded)
        seg_logits = self._seg_head(hidden_states=seg_encoded)

        punct_pre_loss = self._punct_pre_loss(logits=punct_pre_logits, labels=punct_pre_targets)
        punct_post_loss = self._punct_post_loss(logits=punct_post_logits, labels=punct_post_targets)
        cap_loss = self._cap_loss(logits=cap_logits, labels=cap_targets)
        seg_loss = self._seg_loss(logits=seg_logits, labels=seg_targets)

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
        cap_preds = cap_logits.argmax(dim=-1)
        seg_mask = seg_targets.ne(self._ignore_idx)
        seg_preds = seg_logits.argmax(dim=-1)

        eval_modules: Iterable[nn.ModuleDict] = self._dev_metrics if mode == Mode.VAL else self._test_metrics
        metrics: nn.ModuleDict = eval_modules[dataloader_idx]
        num_targets = punct_pre_mask.sum() + punct_post_mask.sum() + cap_mask.sum() + seg_mask.sum()
        metrics["loss"](loss=loss, num_measurements=num_targets)
        metrics["punct_pre_report"](punct_pre_preds[punct_pre_mask], punct_pre_targets[punct_pre_mask])
        metrics["punct_post_report"](punct_post_preds[punct_post_mask], punct_post_targets[punct_post_mask])
        metrics["cap_report"](cap_preds[cap_mask], cap_targets[cap_mask])
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
        for t in ["punct_pre", "punct_post", "cap", "seg"]:
            precision, recall, f1, report = metric_dict[f"{t}_report"].compute()
            metric_dict[f"{t}_report"].reset()
            self.log(f"{mode.value}_{language}_{t}_precision", precision)
            self.log(f"{mode.value}_{language}_{t}_recall", recall)
            self.log(f"{mode.value}_{language}_{t}_f1", f1)
            logging.info(f"{t} report for '{language}': {report}")

    # def validation_epoch_end(self, outputs) -> None:
    #     logging.warning("****** single validation epoch end **********8")
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

    def predict_dataloader(self):
        pass

    def infer_punctuation(self, texts: List[str], threshold: float = 0.5) -> List[str]:
        bos = self.tokenizer.bos_id
        eos = self.tokenizer.eos_id
        token_ids_list: List[List[int]] = [
            [bos] + self.tokenizer.text_to_ids(text) + [eos] for text in texts
        ]
        lengths = torch.tensor([len(ids) for ids in token_ids_list], dtype=torch.long)
        batch_size = len(texts)
        max_len = lengths.max().item()
        device = next(self.parameters()).device
        input_ids = torch.full(
            size=(batch_size, max_len), device=device, dtype=torch.long, fill_value=self.tokenizer.pad_id
        )
        for i, length in enumerate(lengths):
            input_ids[i, :length] = torch.tensor(token_ids_list[i])
        # Mask sequence mask
        mask = input_ids.ne(self.tokenizer.pad_id)
        # [B, T, D]
        encoded: torch.Tensor = self.bert_model(
            input_ids=input_ids,  attention_mask=mask, token_type_ids=torch.zeros_like(input_ids)
        )
        pre_logits = self._punct_head_pre(hidden_states=encoded)
        post_logits = self._punct_head_post(hidden_states=encoded)
        pre_probs = pre_logits.softmax(dim=-1)
        post_probs = post_logits.softmax(dim=-1)
        # Zero-out the null index, we don't care.
        pre_probs[..., self._null_punct_pre_index] = -1.
        post_probs[..., self._null_punct_post_index] = -1.
        # Select the highest-scoring value
        pre_scores, pre_preds = pre_probs.max(dim=-1)
        post_scores, post_preds = post_probs.max(dim=-1)
        # Faster to loop over lists than Tensors
        pre_scores = pre_scores.tolist()
        pre_preds = pre_preds.tolist()
        post_scores = post_scores.tolist()
        post_preds = post_preds.tolist()

        output_texts: List[str] = []
        for batch_idx in range(batch_size):
            # Strip BOS/EOS
            seq_length = lengths[batch_idx] - 2
            ids = token_ids_list[batch_idx][1:-1]
            batch_pre_preds = pre_preds[batch_idx][1:seq_length+1]
            batch_post_preds = post_preds[batch_idx][1:seq_length+1]
            batch_pre_scores = pre_scores[batch_idx][1:seq_length+1]
            batch_post_scores = post_scores[batch_idx][1:seq_length+1]
            out_ids = []
            for i in range(len(ids)):
                # Always index +1 to offset for BOS ID
                if batch_pre_scores[i] > threshold:
                    pred_id = self.tokenizer.token_to_id(self._punct_pre_labels[batch_pre_preds[i]])
                    out_ids.append(pred_id)
                out_ids.append(ids[i])
                if batch_post_scores[i] > threshold:
                    pred_id = self.tokenizer.token_to_id(self._punct_post_labels[batch_post_preds[i]])
                    out_ids.append(pred_id)
            # output_texts.append(self.tokenizer.ids_to_text(out_ids))
            output_texts.append(self.tokenizer.tokenizer.decode(out_ids, skip_special_tokens=True))
        return output_texts

    def infer_segmentation(self, texts: List[str], threshold: float = 0.5) -> List[List[str]]:
        bos = self.tokenizer.bos_id
        eos = self.tokenizer.eos_id
        token_ids_list: List[List[int]] = [
            [bos] + self.tokenizer.text_to_ids(text) + [eos] for text in texts
        ]
        lengths = torch.tensor([len(ids) for ids in token_ids_list], dtype=torch.long)
        batch_size = len(texts)
        max_len = lengths.max().item()
        device = next(self.parameters()).device
        input_ids = torch.full(
            size=(batch_size, max_len), device=device, dtype=torch.long, fill_value=self.tokenizer.pad_id
        )
        for i, length in enumerate(lengths):
            input_ids[i, :length] = torch.tensor(token_ids_list[i])
        # Mask sequence mask
        mask = input_ids.ne(self.tokenizer.pad_id)
        # [B, T, D]
        encoded: torch.Tensor = self.bert_model(
            input_ids=input_ids,  attention_mask=mask, token_type_ids=torch.zeros_like(input_ids)
        )
        logits = self._seg_head(hidden_states=encoded)
        probs = logits.softmax(dim=-1)
        # Keep p(full_stop)
        batch_scores = probs[..., 1].tolist()

        output_texts: List[List[str]] = []
        for batch_idx, input_ids in enumerate(input_ids):
            # Strip BOS/EOS
            ids = token_ids_list[batch_idx][1:-1]
            # Extract the scores for this sequence, and strip BOS/EOS
            scores = batch_scores[batch_idx][1:lengths[batch_idx]+1]
            segmented_texts = []
            start = 0
            stop = 0
            for stop, score in enumerate(scores):
                if score > threshold:
                    text = self.tokenizer.tokenizer.decode(ids[start:stop + 1], skip_special_tokens=True)
                    # text = self.tokenizer.ids_to_text(ids[start:stop + 1])
                    segmented_texts.append(text)
                    start = stop + 1
            if stop >= start:
                text = self.tokenizer.tokenizer.decode(ids[start:stop + 1], skip_special_tokens=True)
                # text = self.tokenizer.ids_to_text(ids[start:stop + 1])
                segmented_texts.append(text)
            output_texts.append(segmented_texts)
        return output_texts

    def infer_capitalization(self, texts: List[List[str]], threshold: float = 0.5) -> List[List[str]]:
        bos = self.tokenizer.bos_id
        eos = self.tokenizer.eos_id
        # Flatten list of lists into one list
        flat_texts = list(itertools.chain(*texts))
        token_ids_list: List[List[int]] = [
            [bos] + self._char_tokenizer.text_to_ids(text) + [eos] for text in flat_texts
        ]

        lengths = torch.tensor([len(ids) for ids in token_ids_list], dtype=torch.long)
        batch_size = len(flat_texts)
        max_len = lengths.max().item()
        device = next(self.parameters()).device
        input_ids = torch.full(
            size=(batch_size, max_len), device=device, dtype=torch.long, fill_value=self.tokenizer.pad_id
        )
        for i, length in enumerate(lengths):
            input_ids[i, :length] = torch.tensor(token_ids_list[i])
        # Mask sequence mask
        mask = input_ids.ne(self.tokenizer.pad_id)
        # [B, T, D]
        encoded: torch.Tensor = self.bert_model(
            input_ids=input_ids,  attention_mask=mask, token_type_ids=torch.zeros_like(input_ids)
        )
        logits = self._cap_head(hidden_states=encoded)
        probs = logits.softmax(dim=-1)
        # Keep p(upper)
        batch_scores = probs[..., 1]

        flat_out_texts: List[str] = []
        for batch_idx, ids in enumerate(token_ids_list):
            # Strip BOS/EOS
            ids = ids[1:-1]
            tokens = self._char_tokenizer.ids_to_tokens(ids)
            scores = batch_scores[batch_idx, 1:len(tokens) + 1].tolist()
            processed_tokens: List[str] = []
            for token, token_score in zip(tokens, scores):
                # Don't mess with special tokens; they won't decode correctly
                if token != self.tokenizer.sep_token:
                    token = token.upper() if token_score > threshold else token.lower()
                # TODO thresholding? it always does something. This implies a default action
                processed_tokens.append(token)
            processed_text = self._char_tokenizer.tokens_to_text(processed_tokens)
            flat_out_texts.append(processed_text)
        # Un-flatten the texts back into their original lists
        output_texts: List[List[str]] = []
        start = 0
        for i in range(len(texts)):
            stop = start + len(texts[i])
            output_texts.append(flat_out_texts[start:stop])
            start = stop
        return output_texts

    def infer(
            self,
            texts: List[str],
            punct_threshold: float = 0.5,
            seg_threshold: float = 0.5,
            truecase_threshold: float = 0.5
    ) -> List[List[str]]:
        punctuated_texts: List[str] = self.infer_punctuation(texts, threshold=punct_threshold)
        segmented_texts: List[List[str]] = self.infer_segmentation(punctuated_texts, threshold=seg_threshold)
        capitalized_texts: List[List[str]] = self.infer_capitalization(segmented_texts, threshold=truecase_threshold)
        return capitalized_texts

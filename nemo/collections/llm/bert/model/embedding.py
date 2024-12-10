import sys
from dataclasses import dataclass
from typing import Optional, Callable, Dict

from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.llm.bert.model import BertModel, BertConfig, HuggingFaceBertModel
from nemo.collections.llm.bert.loss import BERTInBatchExclusiveHardNegativesRankingLoss, BERTLossReduction
from nemo.collections.llm.bert.model.base import get_batch_on_this_context_parallel_rank, get_packed_seq_params
from nemo.collections.llm.bert.model.bert import HuggingFaceBertImporter
from nemo.collections.nlp.models.information_retrieval.bert_embedding_model import BertEmbeddingHead
from megatron.core import InferenceParams, parallel_state, tensor_parallel
import torch
from torch import nn
import lightning.pytorch as L
from nemo.lightning import io
from nemo.lightning.pytorch.optim import OptimizerModule

def bert_embedding_data_step(dataloder_iter) -> Dict[str, torch.Tensor]:
    """Setup BERT dataloader batch."""
    batch = next(dataloder_iter)

    _batch: dict
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch

    required_keys = set()
    required_keys.add("attention_mask")
    required_keys.add("token_type_ids")
    # required_keys.add("metadata")
    if parallel_state.is_pipeline_first_stage():
        required_keys.add("input_ids")


    _batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in _batch.items()}
    # slice batch along sequence dimension for context parallelism
    output = get_batch_on_this_context_parallel_rank(_batch)

    return output

def bert_embedding_forward_step(model: L.LightningModule, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    This subsets the batch keys to the ones actually used by forward pass of the model,
    and then calls the model's forward pass. if "cu_seqsens" are defined in the batch,
    then the packed sequence parameters are also passed to the model for forward pass efficiency.
    """
    forward_args = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
    }

    if model.config.num_tokentypes != 0:
        forward_args["tokentype_ids"] = batch["token_type_ids"]

    if "cu_seqlens" in batch:
        forward_args["packed_seq_params"] = get_packed_seq_params(batch)

    return model(**forward_args)

@dataclass
class BertEmbeddingConfig(BertConfig):
    bert_type: str = 'huggingface'
    ce_loss_scale: float = 20
    label_smoothing: float = 0.0
    add_lm_head: bool = False
    bert_binary_head: bool = False
    num_hard_negatives: int = 1
    num_tokentypes: int = 2
    forward_step_fn: Callable = bert_embedding_forward_step
    data_step_fn: Callable = bert_embedding_data_step

@dataclass
class BertEmbeddingLargeConfig(BertEmbeddingConfig):
    """Bert Embedding model follows Bert-large architecture."""
    num_layers: int = 24
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_attention_heads: int = 16

@dataclass
class BertEmbeddingMiniConfig(BertEmbeddingConfig):
    """Bert Embedding model follows Bert-mini (384 hidden size) architecture."""
    num_layers: int = 6
    hidden_size: int = 384
    intermediate_size: int = 1536
    num_attention_heads: int = 12

class BertEmbeddingModel(BertModel):
    """Bert Lightning Module"""

    def __init__(
        self,
        config: BertConfig,
        # TODO: Add transformer_layer_spec when we update mcore
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(config, optim, tokenizer, model_transform)

    def configure_model(self) -> None:
        """Setup the BERT Model based on config definition."""
        if not hasattr(self, "module"):
            self.module = self.config.configure_model(self.tokenizer)
            self.embedding_head = BertEmbeddingHead(
                word_embedding_dimension=self.config.hidden_size,
                pooling_mode_mean_tokens=True,
            )

    def forward(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Call the forward method of the underlying model, and return whatever it outputs."""
        assert "attention_mask" in kwargs, "attention mask is required for BERT Embedding Model."
        output_tensor = self.module(hidden_states_only=True, *args, **kwargs)  # for now just pass through to the underlying model
        embeddings_out = self.embedding_head(output_tensor, kwargs["attention_mask"])
        return embeddings_out

    @property
    def training_loss_reduction(self) -> BERTLossReduction:  # pylint: disable=C0115,C0116
        if not self._training_loss_reduction:
            self._training_loss_reduction = BERTInBatchExclusiveHardNegativesRankingLoss(
                validation_step=False,
                num_hard_negatives=self.config.num_hard_negatives,
                scale=self.config.ce_loss_scale,
                label_smoothing=self.config.label_smoothing,
            )

        return self._training_loss_reduction

    @property
    def validation_loss_reduction(self) -> BERTLossReduction:  # pylint: disable=C0115,C0116
        if not self._validation_loss_reduction:
            self._training_loss_reduction = BERTInBatchExclusiveHardNegativesRankingLoss(
                validation_step=False,
                num_hard_negatives=self.config.num_hard_negatives,
                scale=self.config.ce_loss_scale,
                label_smoothing=self.config.label_smoothing,
            )

        return self._validation_loss_reduction

@io.model_importer(BertEmbeddingModel, "hf")
class BertEmbeddingImporter(HuggingFaceBertImporter):
    def __init__(self, *args, **kwargs):
        if sys.version_info > (3, 11):
            # In Python versions <= 3.11, *Path classes don’t have a __init__ method,
            # and do all their initialization in __new__/ helper methods.
            # Only need to call super().__init__ if version > 3.11
            super().__init__(*args)
        self.type = 'model'

    def init(self) -> BertEmbeddingModel:
        return BertEmbeddingModel(self.config, tokenizer=self.tokenizer)


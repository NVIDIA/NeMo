from dataclasses import dataclass
from typing import TYPE_CHECKING, Union, Callable, Annotated, Optional, Dict, Literal
import lightning.pytorch as L
import nemo.collections.llm.gpt.model.base as GPTBase
from nemo.collections.llm.bert.loss import BERTInBatchExclusiveHardNegativesRankingLoss
from nemo.collections.llm.gpt.model import GPTConfig
from nemo.collections.llm.gpt.model.llama import Llama32Config1B, LlamaConfig, HFLlamaImporter, LlamaModel
from nemo.utils.import_utils import safe_import
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.enums import AttnMaskType
from nemo.lightning import OptimizerModule, io
from nemo.collections.llm.utils import Config
from nemo.lightning.pytorch.utils import dtype_from_hf
import einops
import torch
from torch import nn, Tensor
from megatron.core import parallel_state
import torch.nn.functional as F
if TYPE_CHECKING:
    from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel
    from transformers import LlamaConfig as HFLlamaConfig
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
_, HAVE_TE = safe_import("transformer_engine")

def _local_layer_spec(config: "GPTConfig") -> ModuleSpec:
    gpt_layer_spec = GPTBase.local_layer_spec(config)
    gpt_layer_spec.submodules.self_attention.params['attn_mask_type'] = AttnMaskType.padding
    return gpt_layer_spec

def _transformer_engine_layer_spec(config: "GPTConfig") -> ModuleSpec:
    gpt_layer_spec = GPTBase.transformer_engine_layer_spec(config)
    gpt_layer_spec.submodules.self_attention.params['attn_mask_type'] = AttnMaskType.padding
    return gpt_layer_spec

def get_nv_embedding_layer_spec(config):
    """Customized Layer Spec for NV Embedding Llama Model.
       Bidirectional attention is enabled instead of causal masking.
    """
    if HAVE_TE:
        return _transformer_engine_layer_spec(config)
    else:
        return _local_layer_spec(config)

def nv_embedding_data_step(dataloder_iter) -> Dict[str, torch.Tensor]:
    """Setup NVEmbedding Llama Model dataloader batch."""
    batch = next(dataloder_iter)

    _batch: dict
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch

    required_keys = set()
    required_keys.add("attention_mask")

    if parallel_state.is_pipeline_first_stage():
        required_keys.add("input_ids")
        required_keys.add("position_ids")

    _batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in _batch.items()}
    # slice batch along sequence dimension for context parallelism
    output = GPTBase.get_batch_on_this_context_parallel_rank(_batch)

    return output

def nv_embedding_forward_step(model: L.LightningModule, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    This subsets the batch keys to the ones actually used by forward pass of the model,
    and then calls the model's forward pass. if "cu_seqsens" are defined in the batch,
    then the packed sequence parameters are also passed to the model for forward pass efficiency.
    """
    if "position_ids" not in batch:
        batch["position_ids"] = torch.arange(0, batch["input_ids"].shape[1], dtype=torch.long, device=batch["input_ids"].device)

    forward_args = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "position_ids": batch["position_ids"],
    }

    return model.encode(**forward_args)

@dataclass
class NVEmbedLlama32Config1B(Llama32Config1B):
    """NV Embedding Llama3.2 1B Config"""
    transformer_layer_spec: Union[ModuleSpec, Callable[["GPTConfig"], ModuleSpec]] = get_nv_embedding_layer_spec
    forward_step_fn: Callable = nv_embedding_forward_step
    data_step_fn: Callable = nv_embedding_data_step
    num_hard_negatives: int = 4
    ce_loss_scale: float = 20
    label_smoothing: float = 0.
    global_in_batch_negatives: bool = False
    backprop_type: Literal["local", "global"] = 'local'

    def configure_model(self, tokenizer, pre_process=None, post_process=None) -> "MCoreGPTModel":
        """Configure the NV Embedding Llama3.2 1B Model"""
        model = super().configure_model(tokenizer, pre_process, post_process)
        # post_process need to be overwritten to False after model init because
        # final_layernorm is still needed and it will only be initialized when post_process is True in Mcore.
        # And for forward(), we do not want to run through output_layer thus setting post_process to False.
        model.post_process = False
        return model


def _average_pool(
    last_hidden_states: Tensor,
    attention_mask: Tensor
):
    # [sq, b, h] -> [b, sq, h]
    last_hidden_states = einops.rearrange(last_hidden_states, 's b h -> b s h')
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class NVEmbedLlamaModel(LlamaModel):
    """NV Embedding Llama Model"""
    def __init__(
        self,
        config: Annotated[Optional[LlamaConfig], Config[LlamaConfig]] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(config or LlamaConfig(), optim=optim, tokenizer=tokenizer, model_transform=model_transform)

    def encode(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        decoder_input: Optional[torch.Tensor] = None,
    ):
        """Generate the embedding for the inputs.
           It runs the forward and apply average pooling on the last hidden states of the model.
        """
        if attention_mask.ndim == 2:
            # extend attention mask to [b, 1, 1, sq]
            # Also convert attention mask to binary
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(1) < 0.5
        elif attention_mask.ndim == 4:
            assert attention_mask.shape[1] == 1 and attention_mask.shape[2] == 1, "Attention mask shape incorrect"
            extended_mask = attention_mask
            # Squeeze attention mask to [b, sq] for averaging pooling later

            attention_mask = extended_mask.squeeze() < 0.5
        else:
            raise ValueError("Attention_mask shape incorrect")

        output = self.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=extended_mask,
            decoder_input=decoder_input,
        )
        embeddings = _average_pool(output, attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    @property
    def training_loss_reduction(self) -> BERTInBatchExclusiveHardNegativesRankingLoss:  # pylint: disable=C0115,C0116
        if not self._training_loss_reduction:
            self._training_loss_reduction = BERTInBatchExclusiveHardNegativesRankingLoss(
                validation_step=False,
                num_hard_negatives=self.config.num_hard_negatives,
                scale=self.config.ce_loss_scale,
                label_smoothing=self.config.label_smoothing,
                global_in_batch_negatives=self.config.global_in_batch_negatives,
                backprop_type=self.config.backprop_type,
            )

        return self._training_loss_reduction

    @property
    def validation_loss_reduction(self) -> BERTInBatchExclusiveHardNegativesRankingLoss:  # pylint: disable=C0115,C0116
        if not self._validation_loss_reduction:
            self._validation_loss_reduction = BERTInBatchExclusiveHardNegativesRankingLoss(
                validation_step=True,
                num_hard_negatives=self.config.num_hard_negatives,
                scale=self.config.ce_loss_scale,
                label_smoothing=self.config.label_smoothing,
            )

        return self._validation_loss_reduction

@io.model_importer(NVEmbedLlamaModel, "hf")
class HFNVEmbedLlamaImporter(HFLlamaImporter):
    """HF Importer for NV Embedding Llama Model"""
    def init(self) -> NVEmbedLlamaModel:
        return NVEmbedLlamaModel(self.config, tokenizer=self.tokenizer)

    @property
    def config(self) -> Llama32Config1B:
        # pylint : disable=C0116
        from transformers import LlamaConfig as HFLlamaConfig

        source = HFLlamaConfig.from_pretrained(str(self))

        def make_vocab_size_divisible_by(vocab_size):
            base = 128
            while vocab_size % base != 0:
                base //= 2
            return base

        output = NVEmbedLlama32Config1B(
            num_layers=source.num_hidden_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=source.intermediate_size,
            num_attention_heads=source.num_attention_heads,
            init_method_std=source.initializer_range,
            layernorm_epsilon=source.rms_norm_eps,
            num_query_groups=source.num_key_value_heads,
            rotary_base=source.rope_theta,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by(source.vocab_size),
            share_embeddings_and_output_weights=getattr(source, "tie_word_embeddings", False),
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
        )

        return output


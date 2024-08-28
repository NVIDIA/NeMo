from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Literal, Optional, Union
import copy

import pytorch_lightning as L
import torch
import torch.distributed
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import nn

from nemo.collections.llm import fn
from nemo.lightning import get_vocab_size, io
from nemo.lightning.megatron_parallel import MaskedTokenLossReduction
from nemo.lightning.pytorch.optim import MegatronOptimizerModule, OptimizerModule

HAVE_TE = True
try:
    import transformer_engine
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False

if TYPE_CHECKING:
    from megatron.core.models.T5.t5_model import T5Model as MCoreT5Model

    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


def t5_data_step(dataloader_iter) -> Dict[str, torch.Tensor]:
    from megatron.core import parallel_state

    # Based on: https://github.com/NVIDIA/Megatron-LM/blob/main/pretrain_t5.py#L87
    # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/language_modeling/megatron_t5_model.py#L828-L842

    batch = next(dataloader_iter)

    # # DEBUGGING
    # print("[nemo/collections/llm/t5/model/t5.py] batch: ", batch)

    _batch: dict
    # TODO: to fix for running inferencing
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch

    # convert attention mask values from int to True/False
    _batch['enc_mask'] = _batch['enc_mask'] < 0.5
    _batch['dec_mask'] = _batch['dec_mask'] < 0.5
    _batch['enc_dec_mask'] = _batch['enc_dec_mask'] < 0.5

    required_keys = set()
    required_keys.update(["enc_mask", "dec_mask", "enc_dec_mask"])
    if parallel_state.is_pipeline_first_stage():
        required_keys.update(("text_enc", "text_dec"))
    if parallel_state.is_pipeline_last_stage():
        required_keys.update(("labels", "loss_mask"))
    # if self.get_attention_mask_from_fusion:
    #     required_keys.remove('attention_mask')


    output = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in _batch.items()}

    return output


def t5_forward_step(model, batch) -> torch.Tensor:
    forward_args = {
        "encoder_input_ids": batch["text_enc"],
        "decoder_input_ids": batch["text_dec"],
        "encoder_attn_mask": batch["enc_mask"],
        "decoder_attn_mask": batch["dec_mask"],
        "encoder_decoder_attn_mask": batch["enc_dec_mask"],
        "lm_labels": batch["labels"],
    }

    # # DEBUGGING
    # print("batch (t5_forward_step): ")
    # print(batch)

    return model(**forward_args)


def transformer_engine_layer_spec(encoder_config: "T5Config", decoder_config: "T5Config") -> ModuleSpec:
    from megatron.core.models.T5.t5_spec import (
        get_t5_encoder_with_transformer_engine_block_spec,
        get_t5_decoder_with_transformer_engine_block_spec,
    )

    en_block_spec = get_t5_encoder_with_transformer_engine_block_spec(encoder_config.num_layers)
    de_block_spec = get_t5_decoder_with_transformer_engine_block_spec(decoder_config.num_layers)

    return [en_block_spec, de_block_spec]


def local_layer_spec(encoder_config: "T5Config", decoder_config: "T5Config") -> ModuleSpec:
    from megatron.core.models.T5.t5_spec import (
        get_t5_encoder_with_local_block_spec,
        get_t5_decoder_with_local_block_spec,
    )

    en_block_spec = get_t5_encoder_with_local_block_spec(encoder_config.num_layers)
    de_block_spec = get_t5_decoder_with_local_block_spec(decoder_config.num_layers)

    return [en_block_spec, de_block_spec]



def default_layer_spec(encoder_config: "T5Config", decoder_config: "T5Config") -> ModuleSpec:
    if HAVE_TE:
        return transformer_engine_layer_spec(encoder_config, decoder_config)
    else:
        return local_layer_spec(encoder_config, decoder_config)


@dataclass
class T5Config(TransformerConfig, io.IOMixin):
    # From megatron.core.models.t5.t5_model.T5Model
    encoder_num_layers: int = None
    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = True
    make_vocab_size_divisible_by: int = 128
    position_embedding_type: Literal["learned_absolute", "rope"] = "learned_absolute"
    max_position_embeddings: int = 512
    rotary_percent: float = 1.0
    seq_len_interpolation_factor: Optional[float] = None

    # TODO: Move this to better places?
    get_attention_mask_from_fusion: bool = False

    transformer_layer_spec: Union[ModuleSpec, Callable[["T5Config"], ModuleSpec]] = default_layer_spec
    forward_step_fn: Callable = t5_forward_step
    data_step_fn: Callable = t5_data_step

    def configure_model(self, tokenizer) -> "MCoreT5Model":
        vp_size = self.virtual_pipeline_model_parallel_size
        if vp_size:
            p_size = self.pipeline_model_parallel_size
            assert (
                self.num_layers // p_size
            ) % vp_size == 0, "Make sure the number of model chunks is the same across all pipeline stages."

        from megatron.core import parallel_state
        from megatron.core.models.T5.t5_model import T5Model as MCoreT5Model


        # DEBUGGING
        if torch.distributed.get_rank()==0:
            print("Debugging: matching NeMo 1.0 Transformers config.")
        self.enable_autocast=True
        self.autocast_dtype=torch.bfloat16
        self.deallocate_pipeline_outputs=True
        self.pipeline_model_parallel_split_rank=0
        self.attention_softmax_in_fp32=False
        self.bias_activation_fusion=True
        self.masked_softmax_fusion=True
        self.persist_layer_norm=True
        self.bias_dropout_fusion=True
        self.recompute_num_layers=1
        self.num_moe_experts=1
        self.distribute_saved_activations=False


        encoder_config = copy.deepcopy(self)
        encoder_config.num_layers = self.encoder_num_layers
        # move this check to strategies?
        # if args.pipeline_model_parallel_size > 1:
        #     assert args.encoder_pipeline_model_parallel_size > 0, "Need to know how to shard the encoder & decoder."
        #     encoder_config.pipeline_model_parallel_size = args.encoder_pipeline_model_parallel_size

        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, ModuleSpec):
            transformer_layer_spec = transformer_layer_spec(encoder_config=encoder_config, decoder_config=self)




        # DEBUGGING
        if torch.distributed.get_rank()==0:
            print("config: ", self)
            print("encoder_config: ", encoder_config)
            print("transformer_encoder_layer_spec: ", transformer_layer_spec[0])
            print("transformer_decoder_layer_spec: ", transformer_layer_spec[1])
            print("vocab_size: ", get_vocab_size(self, tokenizer.vocab_size, self.make_vocab_size_divisible_by),)
            print("max_sequence_length: ", self.max_position_embeddings)
            print("pre_process: ", parallel_state.is_pipeline_first_stage())
            print("post_process: ", parallel_state.is_pipeline_last_stage())
            print("fp16_lm_cross_entropy: ", self.fp16_lm_cross_entropy)
            print("parallel_output: ", self.parallel_output)
            print("share_embeddings_and_output_weights: ", self.share_embeddings_and_output_weights)
            print("position_embedding_type: ", self.position_embedding_type)
            print("rotary_percent: ", self.rotary_percent)
            print("seq_len_interpolation_factor: ", self.seq_len_interpolation_factor)


        model = MCoreT5Model(
            config=self,
            encoder_config=encoder_config,
            transformer_encoder_layer_spec=transformer_layer_spec[0],
            transformer_decoder_layer_spec=transformer_layer_spec[1],
            vocab_size=get_vocab_size(self, tokenizer.vocab_size, self.make_vocab_size_divisible_by),
            max_sequence_length=self.max_position_embeddings,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            position_embedding_type=self.position_embedding_type,
            rotary_percent=self.rotary_percent,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
        )

        # # DEBUGGING
        # print("model: ")
        # print(model)
        # for name, param in model.named_parameters():
        #     print("{}: {}".format(name, param.shape))
        # # print(stop_here)        

        return model


class T5Model(L.LightningModule, io.IOMixin, io.ConnectorMixin, fn.FNMixin):
    def __init__(
        self,
        config: T5Config,
        # TODO: Add transformer_layer_spec when we update mcore
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.optim = optim or MegatronOptimizerModule(config=OptimizerConfig(lr=1e-4, use_distributed_optimizer=True))
        self.optim.connect(self)  # This will bind the `configure_optimizers` method
        self.model_transform = model_transform
        self._training_loss_reduction = None
        self._validation_loss_reduction = None

    def configure_model(self) -> None:
        if not hasattr(self, "module"):
            self.module = self.config.configure_model(self.tokenizer)

            # DEBUGGING
            from megatron.core.enums import ModelType
            self.module.model_type = ModelType.encoder_and_decoder

    def forward(
        self,
        encoder_input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        encoder_attn_mask: torch.Tensor,
        decoder_attn_mask: torch.Tensor,
        encoder_decoder_attn_mask: torch.Tensor,
        lm_labels: Optional[torch.Tensor] = None,
        inference_params=None,
    ) -> torch.Tensor:
        output_tensor = self.module(
            encoder_input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            encoder_attn_mask=encoder_attn_mask,
            decoder_attn_mask=decoder_attn_mask,
            encoder_decoder_attn_mask=encoder_decoder_attn_mask,
            lm_labels=lm_labels,
            inference_params=inference_params,
        )

        return output_tensor

    def data_step(self, dataloader_iter) -> Dict[str, torch.Tensor]:
        return self.config.data_step_fn(dataloader_iter)

    def forward_step(self, batch) -> torch.Tensor:
        return self.config.forward_step_fn(self, batch)

    def training_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)
        return self.forward_step(batch)

    def validation_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)

        return self.forward_step(batch)

    @property
    def training_loss_reduction(self) -> MaskedTokenLossReduction:
        if not self._training_loss_reduction:
            self._training_loss_reduction = MaskedTokenLossReduction()

        return self._training_loss_reduction

    @property
    def validation_loss_reduction(self) -> MaskedTokenLossReduction:
        if not self._validation_loss_reduction:
            self._validation_loss_reduction = MaskedTokenLossReduction(validation_step=True)

        return self._validation_loss_reduction

__all__ = [
    "T5Model",
    "T5Config",
    "t5_data_step",
    "t5_forward_step",
    "transformer_engine_layer_spec",
    "local_layer_spec",
]

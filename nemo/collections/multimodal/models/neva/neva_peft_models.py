from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.multimodal.models.neva.neva_model import MegatronNevaModel
from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
    AdapterName,
    InfusedAdapterConfig,
    LoraKQVAdapterConfig,
    MLPInfusedAdapterConfig,
    ParallelLinearAdapterConfig,
    PromptEncoderAdapterConfig,
)
from nemo.core.classes.mixins import adapter_mixins
from nemo.utils import logging, model_utils


class MegatronNevaLoRAModel(MegatronNevaModel):
    """
    MegatronNevaLoRAModel is a model that combines a base model (MegatronNevaModel) with a low-rank adapters.
    The lora adapters will be added in `nemo/collections/nlp/modules/common/megatron/attention.py`
    The implementation is based on Hu et al. nemo/collections/nlp/modules/common/megatron/attention.py

    A single low-rank feedfowrad layer is used in parallel with the KQV projection layer.
    TODO: Add support to also include an option to adda low-rank adapter in the output projection layer.
    """

    def __init__(
        self, cfg: DictConfig, trainer: Trainer,
    ):
        self.peft_name_keys = [
            AdapterName.LORA_KQV_ADAPTER,
        ]
        lora_cfg = cfg.peft.lora_tuning
        if cfg.get("kv_channels", None) is None:
            assert (
                cfg.hidden_size % cfg.num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = cfg.hidden_size // cfg.num_attention_heads
        else:
            kv_channels = cfg.kv_channels
        projection_size = kv_channels * cfg.num_attention_heads

        adapter_cfg = LoraKQVAdapterConfig(
            in_features=cfg.hidden_size,
            out_features=3 * projection_size,
            dim=lora_cfg.adapter_dim,
            norm_position="none",
            norm_type="none",
            activation="identity",
            column_init_method=lora_cfg.get("column_init_method", "normal"),
            row_init_method=lora_cfg.get("row_init_method", "zero"),
            gather_output=False,
            dropout=lora_cfg.adapter_dropout,
        )

        self.name_key_to_cfg = {}
        for k in self.peft_name_keys:
            self.name_key_to_cfg[k] = adapter_cfg

        super().__init__(cfg, trainer)

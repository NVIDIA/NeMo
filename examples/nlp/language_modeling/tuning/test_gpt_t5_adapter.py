import os
import tempfile

import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from pytorch_lightning.trainer.connectors.checkpoint_connector import _CheckpointConnector
from torch.utils.data import DataLoader, Dataset

from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
from nemo.collections.nlp.models.language_modeling.megatron_t5_sft_model import MegatronT5SFTModel
from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
    AdapterName,
    IA3AdapterConfig,
    InfusedAdapterConfig,
    LoraKQVAdapterConfig,
    MLPInfusedAdapterConfig,
    PromptEncoderAdapterConfig,
)
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.core.config import hydra_runner
from nemo.utils import AppState, logging
from nemo.utils.exp_manager import exp_manager

mp.set_start_method("spawn", force=True)


def _modify_config(base_model_cfg, cfg, add_cfg_to_tree=False):
    """
    This function modifies the original gpt/t5 pre-training config (base_model_cfg) with attributes from the finetuning config (cfg).
    The `add_cfg_to_tree` arg adds `cfg` to the top of the yaml tree which is needed for all `hparams.yaml` files when passed as an arg to `load_from_checkpoint()`.
    """
    OmegaConf.set_struct(base_model_cfg, True)
    OmegaConf.resolve(cfg)
    with open_dict(base_model_cfg):
        base_model_cfg.megatron_amp_O2 = cfg.model.get('megatron_amp_O2', False)
        base_model_cfg.micro_batch_size = cfg.model.data.train_ds.micro_batch_size
        base_model_cfg.global_batch_size = cfg.model.data.train_ds.global_batch_size
        base_model_cfg.sequence_parallel = cfg.model.get("sequence_parallel", False)
        base_model_cfg.activations_checkpoint_granularity = cfg.model.get("activations_checkpoint_granularity", None)
        base_model_cfg.activations_checkpoint_num_layers = cfg.model.get("activations_checkpoint_num_layers", None)
        base_model_cfg.activations_checkpoint_method = cfg.model.get("activations_checkpoint_method", None)
        base_model_cfg.data = cfg.model.data
        base_model_cfg.optim = cfg.model.optim
        base_model_cfg.precision = cfg.trainer.precision
        base_model_cfg.answer_only_loss = cfg.model.answer_only_loss
        base_model_cfg.restore_from_path = cfg.model.restore_from_path
        base_model_cfg.resume_from_checkpoint = cfg.model.resume_from_checkpoint
        base_model_cfg.save_nemo_on_validation_end = cfg.model.save_nemo_on_validation_end
        base_model_cfg.gradient_as_bucket_view = cfg.model.gradient_as_bucket_view
        base_model_cfg.hidden_dropout = cfg.model.get('hidden_dropout', 0.0)
        base_model_cfg.attention_dropout = cfg.model.get('attention_dropout', 0.0)
        base_model_cfg.ffn_dropout = cfg.model.ffn_dropout
        base_model_cfg.activations_checkpoint_layers_per_pipeline = None
        base_model_cfg.peft = cfg.model.peft
        peft_cls = MegatronT5SFTModel if "megatron_t5" in cfg.name else MegatronGPTSFTModel
        base_model_cfg.target = f"{peft_cls.__module__}.{peft_cls.__name__}"

        # This is needed when modifying a hparam file directly to load `.ckpt` files.
        # This is not needed to modify the cfg in `.nemo` files.
        if add_cfg_to_tree:
            OmegaConf.resolve(base_model_cfg)
            base_model_cfg.cfg = base_model_cfg

    return base_model_cfg


def validate_checkpoint_loading_args(cfg):
    if cfg.checkpoint_dir is None or not os.path.isdir(cfg.checkpoint_dir):
        raise ValueError(f'Checkpoint directory {cfg.checkpoint_dir} does not exist or is not a directory.')
    if cfg.checkpoint_name is None:
        raise ValueError(f'Checkpoint name {cfg.checkpoint_name} is not valid.')
    if cfg.hparams_file is None or not os.path.isfile(cfg.hparams_file):
        raise ValueError(f'Hparams file {cfg.hparams_file} does not exist or is not a file.')


def generate_lora_config(cfg):
    peft_name_keys = [AdapterName.LORA_KQV_ADAPTER]
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

    return peft_name_keys, adapter_cfg


def generate_ptuning_config(cfg):
    peft_name_keys = [AdapterName.PTUNING_ADAPTER]

    adapter_cfg = PromptEncoderAdapterConfig(
        cfg.peft.p_tuning.virtual_tokens,
        cfg.peft.p_tuning.bottleneck_dim,
        cfg.peft.p_tuning.embedding_dim,
        cfg.peft.p_tuning.init_std,
        cfg.hidden_size,
    )

    return peft_name_keys, adapter_cfg


def generate_ia3_config(cfg):
    peft_name_keys = [AdapterName.KEY_INFUSED, AdapterName.VALUE_INFUSED, AdapterName.MLP_INFUSED]

    mlp_infused_adapter_cfg = MLPInfusedAdapterConfig(
        in_features=cfg.ffn_hidden_size // cfg.tensor_model_parallel_size
    )
    infused_adapter_cfg = InfusedAdapterConfig(in_features=cfg.hidden_size // cfg.tensor_model_parallel_size)

    adapter_cfgs = [infused_adapter_cfg] * 2 + [mlp_infused_adapter_cfg]

    return peft_name_keys, adapter_cfgs


def generate_new_ia3_config(cfg):
    peft_name_keys = AdapterName.IA3_ADAPTER

    adapter_cfgs = IA3AdapterConfig(cfg)

    return peft_name_keys, adapter_cfgs


@hydra_runner(config_path="conf", config_name="megatron_gpt_peft_tuning_config")
def main(cfg) -> None:
    # for T5, use config name `megatron_t5_peft_tuning_config`
    if "megatron_t5" in cfg.name:
        FTModel = MegatronT5SFTModel
    elif "megatron_gpt" in cfg.name:
        FTModel = MegatronGPTSFTModel
    else:
        raise NotImplementedError("Cannot infer model from config")

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    trainer = MegatronTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)

    ##  In this test script, we never resume from checkpoint
    # if cfg.model.resume_from_checkpoint is not None:
    #     trainer.ckpt_path = cfg.model.resume_from_checkpoint
    # logging.info(f'Resuming training from checkpoint: {trainer.ckpt_path}')

    trainer._checkpoint_connector = _CheckpointConnector(trainer)

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision

    if cfg.model.restore_from_path:
        base_model_cfg = FTModel.restore_from(
            restore_path=cfg.model.restore_from_path, trainer=trainer, return_config=True,
        )
        base_model_cfg = _modify_config(base_model_cfg, cfg, add_cfg_to_tree=False)
        model = FTModel.restore_from(
            restore_path=cfg.model.restore_from_path, trainer=trainer, override_config_path=base_model_cfg,
        )

        peft_name_keys, adapter_cfg = generate_new_ia3_config(base_model_cfg)
        model.add_adapters(peft_name_keys, adapter_cfg, layer_selection=[1, 12])
        trainer.fit(model)

        ckpt_dir = os.path.join(trainer._default_root_dir, "checkpoints")
        adapter_path = os.path.join(ckpt_dir, "adapter.ckpt")

        model.save_adapters(adapter_path)

        ckpt_path = os.path.join(ckpt_dir, f"{cfg.name}.nemo")
        model2 = FTModel.restore_from_nemo_with_adapter(
            restore_path=ckpt_path, adapter_names=peft_name_keys, adapter_cfgs=adapter_cfg, trainer=trainer,
        )
        model2_state = str(model2.state_dict())

        model = FTModel.restore_from(
            restore_path=cfg.model.restore_from_path, trainer=trainer, override_config_path=base_model_cfg,
        )
        model.add_adapters(peft_name_keys, adapter_cfg, layer_selection=[1, 12])

        print("Compare state before loading adapter:", str(model.state_dict()) == model2_state)
        model.load_adapters(adapter_path)
        print("Compare state after loading adapter:", str(model.state_dict()) == model2_state)


if __name__ == '__main__':
    main()

import gc
import re
from collections import OrderedDict
from copy import deepcopy
from typing import List, Union

import torch
import torch.multiprocessing as mp
from einops import rearrange, repeat
from megatron.core import parallel_state
from omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector, init_model_parallel
from nemo.collections.nlp.parts.utils_funcs import torch_dtype_from_precision
from nemo.core.config import hydra_runner
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

mp.set_start_method("spawn", force=True)


def modify_config_for_upcycling(gpt_cfg: OmegaConf, cfg: OmegaConf) -> OmegaConf:
    """
    Function that modifies the parallelism configuration of base model to the MoE parallelism config
    so that the weights can be upcycled and loaded correctly on each rank. This makes sure that each
    layer and weight matrix is sharded in the same way across the ranks. Expert parallelism is ignored
    while loading base model weights and is upcycled based on the EP size in the upcycling function.
    Args:
        gpt_cfg: Base model config
        cfg: Config of the model to upcycle to
    Returns:
        Configuration of the base model that is modified to match the specified parallelism config
    """
    OmegaConf.set_struct(gpt_cfg, True)
    with open_dict(gpt_cfg):
        gpt_cfg.tensor_model_parallel_size = cfg.model.get('tensor_model_parallel_size', 1)
        gpt_cfg.pipeline_model_parallel_size = cfg.model.get('pipeline_model_parallel_size', 1)
        gpt_cfg.virtual_pipeline_model_parallel_size = cfg.model.get('virtual_pipeline_model_parallel_size', None)
        gpt_cfg.sequence_parallel = cfg.model.get('sequence_parallel', False)
        gpt_cfg.pipeline_model_parallel_split_rank = cfg.model.get('pipeline_model_parallel_split_rank', 0)
        gpt_cfg.use_tp_pp_dp_mapping = cfg.model.get('use_tp_pp_dp_mapping', False)
        gpt_cfg.context_parallel_size = cfg.model.get('context_parallel_size', 1)
        gpt_cfg.micro_batch_size = cfg.model.get('micro_batch_size')
        gpt_cfg.global_batch_size = cfg.model.get('global_batch_size')
        gpt_cfg.megatron_amp_O2 = cfg.model.get('megatron_amp_O2', True)
        gpt_cfg.use_fp8 = cfg.model.get('fp8', False)
        gpt_cfg.init_mpi_proc_group = cfg.model.get('ub_tp_comm_overlap', False)
        gpt_cfg.seed = cfg.model.get('seed', 1234)
        gpt_cfg.tokenizer = cfg.model.get('tokenizer', gpt_cfg.tokenizer)
        gpt_cfg.precision = cfg.model.get('precision', gpt_cfg.precision)
    return gpt_cfg


def load_state_dict_from_nemo(
    cls, cfg: OmegaConf, save_restore_connector: SaveRestoreConnector, trainer: Trainer
) -> Union[OrderedDict, List[OrderedDict]]:
    """
    Function that loads a model and only returns the state_dict.
    Args:
        cls: Class instance of the base model (MegatronGPTModel)
        cfg: Config of the model to upcycle to
        save_restore_connector: Save restore connector to load a saved .nemo checkpoint (NLPSaveRestoreConnector)
        trainer: Trainer
    Returns:
        Either a state dict of a list of state dict depending on the parallel config
    """
    gpt_cfg = cls.restore_from(
        restore_path=cfg.restore_from_path,
        return_config=True,
        save_restore_connector=save_restore_connector,
    )
    gpt_cfg = modify_config_for_upcycling(gpt_cfg, cfg)
    instance = cls.restore_from(
        restore_path=cfg.restore_from_path,
        trainer=trainer,
        override_config_path=gpt_cfg,
        save_restore_connector=save_restore_connector,
    )
    if isinstance(instance.model, list):
        state_dict = [
            deepcopy(instance.model[i].to(dtype=torch_dtype_from_precision(gpt_cfg.precision)).state_dict())
            for i in range(len(instance.model))
        ]
    else:
        state_dict = deepcopy(instance.model.to(dtype=torch_dtype_from_precision(gpt_cfg.precision)).state_dict())
    del instance

    gc.collect()
    torch.cuda.empty_cache()
    return state_dict


def upcycle_weights_for_moe(cfg: OmegaConf, state_dict: OrderedDict) -> OrderedDict:
    """
    Upcycle base model weights to MoE model weights that can be loaded into a MoE model instance.
    Args:
        cfg: MoE model config. Must specify the following:
            1. grouped_gemm: Whether to use grouped_gemm or not
            2. expert_model_parallel_size: EP size
            3. num_moe_experts: Number of experts
            4. granularity: Granularity when using sparse MoE.
        state_dict: Base model state dict
    Returns:
        state dict of upcycled weights for MoE
    """
    transformer_impl = "grouped_gemm" if cfg.model.get('moe_grouped_gemm', False) else "local"
    num_moe_experts = cfg.model.num_moe_experts
    if cfg.model.expert_model_parallel_size > 0:
        assert (
            num_moe_experts % cfg.model.expert_model_parallel_size == 0
        ), "num_moe_experts must be divisible by expert_model_parallel_size"
        num_moe_experts = num_moe_experts // cfg.model.expert_model_parallel_size

    router_std = cfg.model.get('router_std', 0)
    expert_std = cfg.model.get('expert_std', 0)
    expert_uniform = cfg.model.get('expert_uniform', 0)
    scale_st_w1 = cfg.model.get('scale_st_w1', 1)
    scale_st = cfg.model.get('scale_st', 1)
    granularity = cfg.model.get('granularity', 1)

    router_key_values = []
    new_key_values = []
    old_keys = []

    for k, v in state_dict.items():
        # Turn layer_norm_weight into pre_mlp_layernorm
        m = re.match('^decoder\.layers\.(\d+)\.mlp\.linear_fc1\.layer_norm_weight', k)
        if m:
            new_key = 'decoder.layers.' + m.group(1) + '.pre_mlp_layernorm.weight'
            new_key_values.append((new_key, v.detach().clone()))
            old_keys.append(k)
            continue

        # Turn layer_norm_bias into pre_mlp_layernorm bias
        m = re.match('^decoder\.layers\.(\d+)\.mlp\.linear_fc1\.layer_norm_bias', k)
        if m:
            new_key = 'decoder.layers.' + m.group(1) + '.pre_mlp_layernorm.bias'
            new_key_values.append((new_key, v.detach().clone()))
            old_keys.append(k)
            continue

        # Turn linear_fc1.weight into local_experts.?.linear_fc1.weight
        m = re.match('^decoder\.layers\.(\d+)\.mlp\.linear_fc1.weight', k)
        if m:
            new_key = 'decoder.layers.' + m.group(1) + '.mlp.router.weight'
            # Create a router for each fc1
            print('creating new router', new_key, 'layer', m.group(1))
            router = torch.nn.Linear(v.size(1), cfg.model.num_moe_experts)
            # low init value helps upcycling
            if router_std > 0:
                torch.nn.init.normal_(router.weight, mean=0.0, std=router_std)
            # same router weights across virtual groups
            if granularity > 1:
                router = repeat(router.weight[: num_moe_experts // granularity], 'e h -> (e g) h', g=granularity)
            else:
                router = router.weight

            router_weight = router.to(v)
            router_key_values.append((new_key, router_weight))

            if transformer_impl == 'local':
                for i in range(num_moe_experts):
                    new_key = (
                        'decoder.layers.' + m.group(1) + '.mlp.experts.local_experts.' + str(i) + '.linear_fc1.weight'
                    )  # works with local
                    if expert_uniform != 0:
                        t = v.detach().clone()
                        t += expert_uniform * (torch.rand(t) * 2 - 1)
                        new_key_values.append((new_key, scale_st_w1 * t))
                    elif expert_std != 0:
                        t = v.detach().clone()
                        t += expert_std * torch.randn_like(t)
                        new_key_values.append((new_key, scale_st_w1 * t))
                    else:
                        new_key_values.append((new_key, scale_st_w1 * v.detach().clone()))
            else:
                new_key = 'decoder.layers.' + m.group(1) + '.mlp.experts.weight1'
                if transformer_impl == 'scattermoe':
                    w1 = v.detach().clone()
                    print(w1.shape)
                    w1 = repeat(w1, 'f h -> e f h', e=num_moe_experts // granularity)
                    w1 = rearrange(w1, 'e (f g) h -> (e g) f h', g=granularity).contiguous()
                    print(w1.shape)
                    new_key_values.append((new_key, scale_st_w1 * w1))
                else:
                    w1 = v.detach().clone().t()
                    w1 = repeat(w1, 'h f -> e h f', e=num_moe_experts // granularity)
                    w1 = rearrange(w1, 'e h (f g) -> (e g) h f', g=granularity).contiguous()
                    new_key_values.append((new_key, scale_st_w1 * w1.reshape(v.shape[1], -1).contiguous()))
            old_keys.append(k)
            continue

        # Turn linear_fc2.weight into local_experts.?.linear_fc2.weight
        m = re.match('^decoder\.layers\.(\d+)\.mlp\.linear_fc2.weight', k)
        if m:
            if transformer_impl == 'local':
                for i in range(num_moe_experts):
                    new_key = (
                        'decoder.layers.' + m.group(1) + '.mlp.experts.local_experts.' + str(i) + '.linear_fc2.weight'
                    )  # works with local
                    if expert_uniform != 0:
                        t = v.detach().clone()
                        t += expert_uniform * (torch.rand(t) * 2 - 1)
                        new_key_values.append((new_key, t))
                    elif expert_std != 0:
                        t = v.detach().clone()
                        t += expert_std * torch.randn_like(t)
                        new_key_values.append((new_key, t))
                    else:
                        new_key_values.append((new_key, v.detach().clone()))
            else:
                new_key = 'decoder.layers.' + m.group(1) + '.mlp.experts.weight2'
                if transformer_impl == 'scattermoe':
                    w2 = scale_st * v.detach().clone()
                    print(w2.shape)
                    w2 = repeat(w2, 'h f -> e h f', e=num_moe_experts // granularity)
                    w2 = rearrange(w2, 'e h (f g) -> (e g) h f', g=granularity).contiguous()
                    print(w2.shape)
                    new_key_values.append((new_key, w2))
                else:
                    w2 = scale_st * v.detach().clone().t()
                    w2 = repeat(w2, 'f h -> e f h', e=num_moe_experts // granularity)
                    w2 = rearrange(w2, 'e (f g) h -> (e g) f h', g=granularity).contiguous()
                    new_key_values.append((new_key, w2.reshape(-1, v.shape[0]).contiguous()))

            old_keys.append(k)
            continue

        # Remove the "_extra_state"
        m = re.match('^decoder\.layers\.(\d+)\.mlp\.linear_fc\d._extra_state', k)
        if m:
            old_keys.append(k)
            continue

    for new_key, value in new_key_values:
        state_dict[new_key] = value
    for new_key, value in router_key_values:
        state_dict[new_key] = value
    for old_key in old_keys:
        del state_dict[old_key]

    return state_dict


@hydra_runner(config_path="conf", config_name="megatron_gpt_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    trainer = MegatronTrainerBuilder(cfg).create_trainer()
    save_restore_connector = NLPSaveRestoreConnector()

    # load base model state dict
    state_dict = load_state_dict_from_nemo(
        MegatronGPTModel, cfg, save_restore_connector=save_restore_connector, trainer=trainer
    )
    # destroy the old parallel state to help initialize new state with expert parallelism
    parallel_state.destroy_model_parallel()
    trainer.strategy.setup_environment()

    model_instance = MegatronGPTModel(cfg.model, trainer)
    init_model_parallel(False)

    # upcycle weights to MoE weights
    if isinstance(state_dict, list):
        state_dict = [upcycle_weights_for_moe(cfg=cfg, state_dict=sd) for sd in state_dict]
        state_dict = [save_restore_connector.modify_state_dict(cfg, state_dict=sd) for sd in state_dict]
    else:
        state_dict = upcycle_weights_for_moe(cfg=cfg, state_dict=state_dict)
        state_dict = save_restore_connector.modify_state_dict(cfg, state_dict=state_dict)

    # load new instance with upcycled weights
    if isinstance(model_instance.model, list):
        if not isinstance(state_dict, list):
            state_dict = [state_dict]

        assert len(state_dict) == len(model_instance.model), "Error in upcycling weights"
        for i in range(len(model_instance.model)):
            model_instance.model[i].load_state_dict(state_dict=state_dict[i], strict=True)
    else:
        model_instance.model.load_state_dict(state_dict=state_dict, strict=True)
    logging.info(f"Loaded upcycled model weights from {cfg.restore_from_path} for MoE training.")

    exp_manager(trainer, cfg.exp_manager)
    trainer.fit(model_instance)


if __name__ == "__main__":
    main()

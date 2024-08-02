# import os
import gc
import re
from collections import OrderedDict

# import tempfile
from copy import deepcopy
from typing import Dict

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

# from nemo.utils.app_state import AppState


# import argparse


mp.set_start_method("spawn", force=True)


def modify_state_dict_for_ddp(state_dict):
    ddp_prefix = 'model.module.'
    for k in list(state_dict.keys()):
        state_dict[ddp_prefix + k] = state_dict.pop(k)

    return state_dict


def modify_config_for_upcycling(gpt_cfg: OmegaConf, cfg: OmegaConf) -> OmegaConf:
    OmegaConf.set_struct(gpt_cfg, True)
    with open_dict(gpt_cfg):
        gpt_cfg.tensor_model_parallel_size = cfg.model.get('tensor_model_parallel_size', 1)
        # gpt_cfg.expert_model_parallel_size=cfg.model.expert_model_parallel_size
        gpt_cfg.pipeline_model_parallel_size = cfg.model.get('pipeline_model_parallel_size', 1)
        gpt_cfg.virtual_pipeline_model_parallel_size = cfg.model.get('virtual_pipeline_model_parallel_size', None)
        gpt_cfg.sequence_parallel = cfg.model.get('sequence_parallel', False)
        gpt_cfg.pipeline_model_parallel_split_rank = cfg.model.get('pipeline_model_parallel_split_rank', 0)
        gpt_cfg.use_tp_pp_dp_mapping = cfg.model.get('use_tp_pp_dp_mapping', False)
        gpt_cfg.context_parallel_size = cfg.model.get('context_parallel_size', 1)
        gpt_cfg.micro_batch_size = cfg.model.get('micro_batch_size')
        gpt_cfg.global_batch_size = cfg.model.get('global_batch_size')
        # gpt_cfg.rampup_batch_size = cfg.model.get('rampup_batch_size', None)
        gpt_cfg.megatron_amp_O2 = cfg.model.get('megatron_amp_O2', True)
        gpt_cfg.use_fp8 = cfg.model.get('fp8', False)
        gpt_cfg.init_mpi_proc_group = cfg.model.get('ub_tp_comm_overlap', False)
        gpt_cfg.seed = cfg.model.get('seed', 1234)
        # gpt_cfg.apex_transformer_log_level = cfg.model.get('apex_transformer_log_level', 30)
        gpt_cfg.tokenizer = cfg.model.get('tokenizer', gpt_cfg.tokenizer)
        gpt_cfg.precision = cfg.model.get('precision', gpt_cfg.precision)

    return gpt_cfg


# def print_mem_usage():
#     print()
#     print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
#     print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
#     print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
#     print()


def load_state_dict_from_nemo(
    cls, cfg: OmegaConf, save_restore_connector: SaveRestoreConnector, trainer: Trainer
) -> OrderedDict:
    # cls = MegatronGPTModel
    gpt_cfg = cls.restore_from(
        restore_path=cfg.restore_from_path,
        return_config=True,
        save_restore_connector=save_restore_connector,
    )
    gpt_cfg = modify_config_for_upcycling(gpt_cfg, cfg)
    # trainer = MegatronTrainerBuilder(cfg).create_trainer()
    instance = cls.restore_from(
        restore_path=cfg.restore_from_path,
        trainer=trainer,
        override_config_path=gpt_cfg,
        save_restore_connector=save_restore_connector,
    )
    state_dict = deepcopy(instance.model.to(dtype=torch_dtype_from_precision(gpt_cfg.precision)).state_dict())
    del instance
    gc.collect()
    torch.cuda.empty_cache()
    return state_dict


def upcycle_weights_for_moe(cfg: OmegaConf, state_dict: OrderedDict) -> OrderedDict:
    transformer_impl = "grouped_gemm" if cfg.model.get('moe_grouped_gemm', False) else "local"
    num_moe_experts = cfg.model.num_moe_experts
    if cfg.model.expert_model_parallel_size > 0:
        # print("Using Expert Parallel "+str(cfg.model.expert_model_parallel_size))
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

    # gated_linear_unit = cfg.model.get('gated_linear_unit', False) or cfg.model.get('activation', '').endswith('glu')
    # assert v.shape[0] % cfg.model.ffn_hidden_size == 0, "ffn_hidden_size must either be equal to the original model's ffn_hidden_size for simple MoE or ffn_hidden_size/granularity for fine-grained MoE"
    # granularity = v.shape[0]//cfg.model.ffn_hidden_size

    router_key_values = []
    new_key_values = []
    old_keys = []

    # for k in list(state_dict.keys()):
    # print_mem_usage()
    # try:
    #     device = state_dict[k].device
    #     v = state_dict[k].detach().cpu()
    # except:
    #     device = 'cpu'
    #     v = state_dict[k]

    # if torch.is_tensor(state_dict[k]):
    # device = state_dict[k].device
    # else:
    # device = None

    # v = state_dict[k]

    for k, v in state_dict.items():
        # Turn layer_norm_weight into pre_mlp_layernorm
        m = re.match('^decoder\.layers\.(\d+)\.mlp\.linear_fc1\.layer_norm_weight', k)
        if m:
            # del state_dict[k]
            new_key = 'decoder.layers.' + m.group(1) + '.pre_mlp_layernorm.weight'
            new_key_values.append((new_key, v.detach().clone()))  # .to(device)))
            old_keys.append(k)
            # del v
            continue

        # Turn layer_norm_bias into pre_mlp_layernorm bias
        m = re.match('^decoder\.layers\.(\d+)\.mlp\.linear_fc1\.layer_norm_bias', k)
        if m:
            # del state_dict[k]
            new_key = 'decoder.layers.' + m.group(1) + '.pre_mlp_layernorm.bias'
            new_key_values.append((new_key, v.detach().clone()))  # .to(device)))
            old_keys.append(k)
            # del v
            continue

        # Turn linear_fc1.weight into local_experts.?.linear_fc1.weight
        m = re.match('^decoder\.layers\.(\d+)\.mlp\.linear_fc1.weight', k)
        if m:
            # del state_dict[k]
            # check if granularity is correct based on weight shape and cfg.model.ffn_hidden_size
            # if gated_linear_unit:
            # assert cfg.model.ffn_hidden_size*granularity*2 == v.size(0), "granularity is incorrect based on weight shape and cfg.model.ffn_hidden_size"
            # else:
            # assert cfg.model.ffn_hidden_size*granularity == v.size(0), "granularity is incorrect based on weight shape and cfg.model.ffn_hidden_size"

            new_key = 'decoder.layers.' + m.group(1) + '.mlp.router.weight'
            # Create a router for each fc1
            print('creating new router', new_key, 'layer', m.group(1))
            router = torch.nn.Linear(v.size(1), cfg.model.num_moe_experts)  # num_moe_experts)
            # low init value helps upcycling
            if router_std > 0:
                torch.nn.init.normal_(router.weight, mean=0.0, std=router_std)
            # same router weights across virtual groups
            if granularity > 1:
                router = repeat(router.weight[: num_moe_experts // granularity], 'e h -> (e g) h', g=granularity)
            else:
                router = router.weight

            router_weight = router.to(v)  # .to(device)
            router_key_values.append((new_key, router_weight))

            if transformer_impl == 'local':
                for i in range(num_moe_experts):
                    # new_key = 'decoder.layers.'+m.group(1)+'.mlp.local_experts.'+str(i)+'.linear_fc1.weight'  #works for TE
                    new_key = (
                        'decoder.layers.' + m.group(1) + '.mlp.experts.local_experts.' + str(i) + '.linear_fc1.weight'
                    )  # works with local
                    if expert_uniform != 0:
                        t = v.detach().clone()  # .to(device)
                        t += expert_uniform * (torch.rand(t) * 2 - 1)
                        new_key_values.append((new_key, scale_st_w1 * t))
                    elif expert_std != 0:
                        t = v.detach().clone()  # .to(device)
                        t += expert_std * torch.randn_like(t)
                        new_key_values.append((new_key, scale_st_w1 * t))
                    else:
                        new_key_values.append((new_key, scale_st_w1 * v.detach().clone()))  # .to(device)))
            else:
                new_key = 'decoder.layers.' + m.group(1) + '.mlp.experts.weight1'
                if transformer_impl == 'scattermoe':
                    w1 = v.detach().clone()  # .to(device)
                    print(w1.shape)
                    w1 = repeat(w1, 'f h -> e f h', e=num_moe_experts // granularity)
                    w1 = rearrange(w1, 'e (f g) h -> (e g) f h', g=granularity).contiguous()
                    print(w1.shape)
                    new_key_values.append((new_key, scale_st_w1 * w1))
                else:
                    w1 = v.detach().clone().t()  # .to(device)
                    # print('w1 shape', w1.shape) #torch.Size([6144, 3072])
                    w1 = repeat(w1, 'h f -> e h f', e=num_moe_experts // granularity)
                    w1 = rearrange(w1, 'e h (f g) -> (e g) h f', g=granularity).contiguous()
                    new_key_values.append((new_key, scale_st_w1 * w1.reshape(v.shape[1], -1).contiguous()))
            old_keys.append(k)
            # del v
            continue

        # Turn linear_fc2.weight into local_experts.?.linear_fc2.weight
        m = re.match('^decoder\.layers\.(\d+)\.mlp\.linear_fc2.weight', k)
        if m:
            # del state_dict[k]
            # check if granularity is correct based on weight shape and cfg.model.ffn_hidden_size
            # assert cfg.model.ffn_hidden_size*granularity == v.size(1), "granularity is incorrect based on weight shape and cfg.model.ffn_hidden_size"

            if transformer_impl == 'local':
                for i in range(num_moe_experts):
                    # new_key = 'decoder.layers.'+m.group(1)+'.mlp.local_experts.'+str(i)+'.linear_fc2.weight'  #works for TE
                    new_key = (
                        'decoder.layers.' + m.group(1) + '.mlp.experts.local_experts.' + str(i) + '.linear_fc2.weight'
                    )  # works with local
                    if expert_uniform != 0:
                        t = v.detach().clone()  # .to(device)
                        t += expert_uniform * (torch.rand(t) * 2 - 1)
                        new_key_values.append((new_key, t))
                    elif expert_std != 0:
                        t = v.detach().clone()  # .to(device)
                        t += expert_std * torch.randn_like(t)
                        new_key_values.append((new_key, t))
                    else:
                        new_key_values.append((new_key, v.detach().clone()))  # .to(device)))
            else:
                new_key = 'decoder.layers.' + m.group(1) + '.mlp.experts.weight2'
                if transformer_impl == 'scattermoe':
                    w2 = scale_st * v.detach().clone()  # .to(device)
                    print(w2.shape)
                    w2 = repeat(w2, 'h f -> e h f', e=num_moe_experts // granularity)
                    w2 = rearrange(w2, 'e h (f g) -> (e g) h f', g=granularity).contiguous()
                    print(w2.shape)
                    new_key_values.append((new_key, w2))
                else:
                    w2 = scale_st * v.detach().clone().t()  # .to(device)
                    # print('w2 shape', w2.shape) # torch.Size([3072, 6144])
                    w2 = repeat(w2, 'f h -> e f h', e=num_moe_experts // granularity)
                    w2 = rearrange(w2, 'e (f g) h -> (e g) f h', g=granularity).contiguous()
                    new_key_values.append((new_key, w2.reshape(-1, v.shape[0]).contiguous()))

            old_keys.append(k)
            # del v
            continue

        # Remove the "_extra_state"
        m = re.match('^decoder\.layers\.(\d+)\.mlp\.linear_fc\d._extra_state', k)
        if m:
            # del state_dict[k]
            # del v
            old_keys.append(k)
            continue

        # gc.collect()
        # torch.cuda.empty_cache()

    for new_key, value in new_key_values:
        # print('adding '+new_key)
        state_dict[new_key] = value
    for new_key, value in router_key_values:
        # print('adding '+new_key)
        state_dict[new_key] = value
    for old_key in old_keys:
        # print('removing '+old_key)
        del state_dict[old_key]

    # gc.collect()
    # torch.cuda.empty_cache()

    return state_dict


@hydra_runner(config_path="conf", config_name="megatron_gpt_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    # cfg = OmegaConf.load(args.config_path)
    trainer = MegatronTrainerBuilder(cfg).create_trainer()
    save_restore_connector = NLPSaveRestoreConnector()
    state_dict = load_state_dict_from_nemo(
        MegatronGPTModel, cfg, save_restore_connector=save_restore_connector, trainer=trainer
    )
    # app_state = AppState()
    # print("app state1: ", app_state.expert_model_parallel_size)
    # exit()
    # app_state.__init__()
    parallel_state.destroy_model_parallel()
    trainer.strategy.setup_environment()
    # trainer.strategy.setup_distributed()

    model_instance = MegatronGPTModel(cfg.model, trainer)
    init_model_parallel(False)

    # print("app state2: ", app_state.expert_model_parallel_size)

    state_dict = upcycle_weights_for_moe(cfg=cfg, state_dict=state_dict)
    # if cfg.model.get('expert_model_parallel_size', 1) == 1:
    # state_dict = modify_state_dict_for_ddp(state_dict)

    state_dict = save_restore_connector.modify_state_dict(cfg, state_dict=state_dict)

    # trainer = MegatronTrainerBuilder(cfg).create_trainer()

    # print("HERE!!!!!!!")
    # print("model: ", model)
    # print("model.model: ", model.model)
    # print("model.model[0].state_dict: ", model.model[0].state_dict())
    # exit()
    # model.model[0].load_state_dict(state_dict, strict=True)
    # s1 = set(list(model.model[0].state_dict().keys()))
    # s2 = set(list(state_dict.keys()))
    # print("Intersection: ", s1.intersection(s2))
    # print("Diff s1-s2: ", s1-s2)
    # print("Diff s2-s1: ", s2-s1)
    # exit()
    # print("s1: ", s1)
    # print("s2: ", s2)

    # save_restore_connector.load_instance_with_state_dict(model_instance, state_dict, strict=True)
    if isinstance(model_instance.model, list):
        model_instance.model[0].load_state_dict(state_dict=state_dict, strict=True)
    else:
        model_instance.model.load_state_dict(state_dict=state_dict, strict=True)
    logging.info(f"Loaded upcycled model weights from {cfg.restore_from_path} for MoE training.")

    exp_manager(trainer, cfg.exp_manager)
    trainer.fit(model_instance)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config_path', required=False, default='/lustre/fsw/coreai_dlalgo_llm/avavre/llama3upcycle/config/moe_settings/llama3_8b_bf16_moe_continuepretraining_8ex_1k_hydra.yaml')
    # parser.add_argument('--nemo_model', required=False, default='/lustre/fsw/coreai_dlalgo_llm/avavre/llama3upcycle/models/llama3/8b_pre_trained.nemo', help="Path to .nemo file which needs to be upcycled")
    # args = parser.parse_args()
    # main(args)
    main()

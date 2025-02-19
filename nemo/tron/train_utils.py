import torch
from megatron.core import parallel_state
from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate
from megatron.core.utils import get_data_parallel_group_if_dtensor, to_local_if_dtensor

try:
    from transformer_engine.pytorch.optimizers import multi_tensor_applier, multi_tensor_l2norm
except ImportError:
    try:
        from amp_C import multi_tensor_l2norm
        from apex.multi_tensor_apply import multi_tensor_applier
    except ImportError:

        import warnings

        warnings.warn(
            f'Transformer Engine and Apex are not installed. '
            'Falling back to local implementations of '
            'multi_tensor_applier and multi_tensor_l2norm'
        )

        from megatron.core.utils import local_multi_tensor_applier as multi_tensor_applier
        from megatron.core.utils import local_multi_tensor_l2_norm as multi_tensor_l2norm


def param_is_not_shared(param):
    return not hasattr(param, 'shared') or not param.shared


def calc_params_l2_norm(model, model_config, force_create_fp32_copy=False):
    """Calculate l2 norm of parameters"""
    if not isinstance(model, list):
        model = [model]
    # Seperate moe and dense params
    params_data = []
    moe_params_data = []
    sharded_params_data = []
    data_parallel_group = None

    for model_chunk in model:
        for param in model_chunk.parameters():
            data_parallel_group = get_data_parallel_group_if_dtensor(param, data_parallel_group)
            is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
            if not is_not_tp_duplicate:
                continue
            assert is_not_tp_duplicate
            if not getattr(param, 'allreduce', True):
                # TODO: Implement memory optimization for MoE parameters.
                assert param_is_not_shared(param)
                param = to_local_if_dtensor(param)
                moe_params_data.append(param.data.float() if model_config.bf16 else param.data)
            else:
                if param_is_not_shared(param):
                    param = to_local_if_dtensor(param)
                    if model_config.bf16:
                        if not force_create_fp32_copy and hasattr(param, 'main_param'):
                            if getattr(param, 'main_param_sharded', False):
                                if param.main_param is not None:
                                    sharded_params_data.append(param.main_param)
                            else:
                                params_data.append(param.main_param)
                        else:
                            # Fallback to original logic of making a fp32 copy of the
                            # parameter if `.main_param` attribute is not available.
                            params_data.append(param.data.float())
                    else:
                        params_data.append(param.data)

    # Calculate norm.
    dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
    if len(params_data) > 0:
        norm, _ = multi_tensor_applier(
            multi_tensor_l2norm, dummy_overflow_buf, [params_data], False  # no per-parameter norm.
        )
        norm_2 = norm * norm
    else:
        norm_2 = torch.zeros((1,), dtype=torch.float32, device='cuda')

    if data_parallel_group is not None:
        torch.distributed.all_reduce(norm_2, op=torch.distributed.ReduceOp.SUM, group=data_parallel_group)

    # Add norm contribution from params with sharded main_params. These norms need to be
    # accumulated across the DP group since the main parameters are sharded because
    # of distributed optimizer.
    if len(sharded_params_data) > 0:
        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
        sharded_norm, _ = multi_tensor_applier(
            multi_tensor_l2norm, dummy_overflow_buf, [sharded_params_data], False  # no per-parameter norm.
        )
        sharded_norm_2 = sharded_norm * sharded_norm
        # Sum over all DP groups.
        torch.distributed.all_reduce(
            sharded_norm_2, op=torch.distributed.ReduceOp.SUM, group=parallel_state.get_data_parallel_group()
        )
        norm_2 += sharded_norm_2

    # Sum across all model-parallel GPUs (tensor + pipeline).
    torch.distributed.all_reduce(
        norm_2, op=torch.distributed.ReduceOp.SUM, group=parallel_state.get_model_parallel_group()
    )

    # Add norm contribution from expert layers in MoEs.
    if len(moe_params_data) > 0:
        moe_norm, _ = multi_tensor_applier(
            multi_tensor_l2norm, dummy_overflow_buf, [moe_params_data], False  # no per-parameter norm.
        )
        moe_norm_2 = moe_norm * moe_norm
        # Sum across expert tensor, model and pipeline parallel GPUs.
        torch.distributed.all_reduce(
            moe_norm_2,
            op=torch.distributed.ReduceOp.SUM,
            group=parallel_state.get_expert_tensor_model_pipeline_parallel_group(),
        )
        norm_2 += moe_norm_2

    return norm_2.item() ** 0.5


def reduce_max_stat_across_model_parallel_group(stat: float) -> float:
    """
    Ranks without an optimizer will have no grad_norm or num_zeros_in_grad stats.
    We need to ensure the logging and writer rank has those values.
    This function reduces a stat tensor across the model parallel group.

    We use an all_reduce max since the values have already been summed across optimizer ranks where possible
    """
    if stat is None:
        stat = -1.0
    stat = torch.tensor([stat], dtype=torch.float32, device=torch.cuda.current_device())
    torch.distributed.all_reduce(
        stat, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_model_parallel_group()
    )
    if stat.item() == -1.0:
        return None
    else:
        return stat.item()


def logical_and_across_model_parallel_group(input: bool) -> bool:
    """
    This function gathers a bool value across the model parallel group
    """
    if input is True:
        input = 1
    else:
        input = 0
    input = torch.tensor([input], dtype=torch.int, device=torch.cuda.current_device())
    torch.distributed.all_reduce(
        input, op=torch.distributed.ReduceOp.MIN, group=parallel_state.get_model_parallel_group()
    )
    return bool(input.item())

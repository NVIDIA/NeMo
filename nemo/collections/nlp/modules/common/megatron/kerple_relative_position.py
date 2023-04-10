import math

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults

try:
    from apex.transformer import parallel_state, tensor_parallel
    from apex.transformer.enums import AttnMaskType, AttnType
    from apex.transformer.utils import divide as safe_divide

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

    # fake missing classes with None attributes
    ModelType = AttnMaskType = AttnType = LayerType = ApexGuardDefaults()


def get_kerple_log_params(
    num_attention_heads,
    precision
):

    model_parallel_size = parallel_state.get_tensor_model_parallel_world_size()
    num_heads_per_partition = safe_divide(num_attention_heads, model_parallel_size)

    dtype_dict = {16: torch.float16, 32: torch.float32, 'bf16': torch.bfloat16}

    def get_parameter(scale, init_method):
        if init_method == 'ones':
            return Parameter(torch.ones(
                            num_heads_per_partition,
                            device=torch.cuda.current_device(),
                            dtype=(lambda x, y: x[y])(dtype_dict, precision),
                            )[:,None,None]*scale )
        elif init_method == 'uniform':
            return Parameter(torch.rand(
                            num_heads_per_partition,
                            device=torch.cuda.current_device(),
                            dtype=(lambda x, y: x[y])(dtype_dict, precision),
                            )[:,None,None]*scale )
    
    bias_p = get_parameter(2, 'uniform')
    bias_a = get_parameter(1, 'uniform')

    return torch.concat((bias_p, bias_a))
    

def kerple_log_forward(
    x, relative_position_bias
):
    bias_p, bias_a = torch.split(
        relative_position_bias, relative_position_bias.size(0)//2, dim=0)

    eps = 1e-2

    # [b, np, sq, sk]
    seq_len_q = x.shape[-2]
    seq_len_k = x.shape[-1]
    
    # We may be able to save this and avoid recomputing this every time like in the 
    # reference implementation.
    # Currently kept this way to be compatible with the checkpointed-attn-forward
    # TODO: find a way to avoid recomputing this every time.
    diff = torch.tril(
        torch.arange(seq_len_k, device=x.device).view(seq_len_k, 1).repeat(1, seq_len_k)
        + torch.arange(0, -seq_len_k, -1, device=x.device)
    )
    diff = diff.to(x.dtype)
    
    bias_p.data = bias_p.data.clamp(min=eps)
    bias_a.data = bias_a.data.clamp(min=eps)
    bias = -bias_p*torch.log(1+bias_a*diff) # log kernel
    
    if seq_len_q != seq_len_k:
        # In the train case x has dimensionality [b, np, sq, sk] with sq == sk
        # The number of query tokens is equal to the number of key tokens
        # At inference time with cache in layer_past sq is not equal to sk. sq only contains one token (the last one in the full sequence)
        # In this case we use the appropriate token index of the cache matrix.
        # As the cache matrix could already be bigger from a past inference, not the last token index in the sq sequence is used
        assert (
            seq_len_q == 1
        ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"
        
        if type(bias) != float:
            # seq_len_k - 1 points to the last token index in the current inference batch.
            bias = bias[:, seq_len_k - 1, :].view(bias.shape[0], 1, bias.shape[2])

    return x + bias

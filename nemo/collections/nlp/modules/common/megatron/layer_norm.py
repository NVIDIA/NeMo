from nemo.collections.nlp.modules.common.megatron.fused_layer_norm import get_layer_norm as get_fused_layer_norm
from nemo.collections.nlp.modules.common.megatron.layer_norm_1p import LayerNorm1P, LPLayerNorm
from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults

try:
    from apex.normalization import MixedFusedRMSNorm

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False
    # fake missing classes with None attributes
    ModelType = AttnMaskType = AttnType = LayerType = ApexGuardDefaults()


def get_layer_norm(
    ln_type, hidden_size, eps, persist=False, sequence_parallel=False,
):
    """Get the specified type from many layernorms.

    ln_type: 'layernorm', 'layernorm1p', 'rmsnorm', 'low_precision_layernorm'
    hidden_size: hidden size of the transformer layer
    layernorm_epsilon: epsilon value for layer norm
    persist_layer_norm: whether to use the persistent layer norm kernel
    sequence_parallel: whether to use sequence parallelism
    """
    if ln_type == 'layernorm':
        ln = get_fused_layer_norm(hidden_size=hidden_size, eps=eps, persist_layer_norm=persist,)
    elif ln_type == 'layernorm1p':
        ln = LayerNorm1P(hidden_size=hidden_size, eps=eps, sequence_parallel_enabled=sequence_parallel,)
    elif ln_type == 'low_precision_layernorm':
        ln = LPLayerNorm(normalized_shape=hidden_size, eps=eps)
    elif ln_type == 'rmsnorm':
        ln = MixedFusedRMSNorm(normalized_shape=hidden_size, eps=eps)
    else:
        raise Exception("Unsupported layer norm type: {}".format(ln_type))

    if sequence_parallel:
        # mark sequence parallelism in layer norm parameters
        for param in ln.parameters():
            setattr(param, 'sequence_parallel', True)

    return ln

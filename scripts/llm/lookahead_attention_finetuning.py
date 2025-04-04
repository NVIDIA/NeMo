from functools import partial
from typing import TYPE_CHECKING, List, Optional

import nemo_run as run
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec

from nemo.collections import llm
from nemo.collections.llm.gpt.model.megatron.lookahead_attention import LookAheadAttentionTransformerLayer

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig


def get_gpt_decoder_block_spec_lookahead_attn(
    config: 'TransformerConfig',
    use_transformer_engine: bool = True,
    lookahead_parallel_layers: Optional[List[int]] = None,
):
    spec = get_gpt_decoder_block_spec(config, use_transformer_engine)
    for layer_i in range(len(spec.layer_specs)):
        spec.layer_specs[layer_i].module = LookAheadAttentionTransformerLayer
        spec.layer_specs[layer_i].params = {"lookahead_parallel_layers": lookahead_parallel_layers}
    return spec


if __name__ == "__main__":

    recipe = llm.recipes.deepseek_v2_lite.finetune_recipe(num_nodes=1, num_gpus_per_node=8)
    recipe.model.config.transformer_layer_spec = partial(
        get_gpt_decoder_block_spec_lookahead_attn,
        use_transformer_engine=True,
        lookahead_parallel_layers=[14, 27],
    )

    recipe.trainer.strategy.expert_model_parallel_size = 2
    recipe.trainer.strategy.tensor_model_parallel_size = 2
    recipe.trainer.strategy.sequence_parallel = True

    print(recipe)

    run.run(recipe)

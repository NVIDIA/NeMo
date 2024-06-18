from dataclasses import dataclass, field
from typing import List, Literal

from megatron.core import parallel_state
from nemo.collections.llm.adapter.base import AdapterWrapper, PEFTConfig
from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import ParallelLinearAdapter


class AdapterParallelAdd(AdapterWrapper):
    """ Example: LoRA Adapter """
    def forward(self, x):
        linear_output, bias = self.to_wrap(x)
        if isinstance(linear_output, tuple) and len(linear_output) == 2:
            linear_output, layernorm_output = linear_output
            adapter_output = self.adapter(layernorm_output)
        else:
            adapter_output = self.adapter(x)
        return linear_output + adapter_output, bias


@dataclass
class LoRAConfig(PEFTConfig):
    target_modules: List[str] = field(default_factory=list) #['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2']
    dim: int = 32
    alpha: int = 32
    dropout: float = 0.0
    dropout_position: Literal['pre', 'post'] = 'post'

    def wrap_fn(self, m, name=None, prefix=None):
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        if name in self.target_modules:
            # m.in_features and m.out_features are divided by tp_size already,
            # but in_features and out_features passed to ParallelLinearAdapter are not.
            if name in ['linear_qkv', 'linear_fc1']:
                # Column Parallel Linear
                input_is_parallel = False
                in_features = m.in_features
                out_features = m.out_features * tp_size
            else: # name in ['linear_proj', 'linear_fc2']
                # Row Parallel Linear
                input_is_parallel = True
                in_features = m.in_features * tp_size
                out_features = m.out_features

            print("Adding lora to:", f"{prefix}.{name}", f"{m.in_features}x{m.out_features}")
            adapter = ParallelLinearAdapter(
                in_features,
                out_features,
                self.dim,
                activation='identity',
                norm_position=None,
                norm_type=None,
                column_init_method="normal",
                row_init_method="zero",
                gather_output=False,
                input_is_parallel=input_is_parallel,
                dropout=self.dropout,
                dropout_position=self.dropout_position,
                alpha=self.alpha,
            )
            return AdapterParallelAdd(m, adapter)
        return m
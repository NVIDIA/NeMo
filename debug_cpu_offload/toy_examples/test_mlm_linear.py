import nemo.utils.cpu_offload as cpu_offload
import torch.nn as nn
import torch

from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core import parallel_state
from nemo.collections.nlp.modules.common.megatron.megatron_init import initialize_model_parallel_for_nemo
from nemo.core.optim import MainParamsOptimizerWrapper

torch.manual_seed(1234)

class ToyFFN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, offload_handler):
        super(ToyFFN, self).__init__()
        self.linear1 = ColumnParallelLinear(input_size=input_size, output_size=hidden_size)
        self.ffn_act = nn.GELU()
        self.linear2 = RowParallelLinear(input_size=hidden_size, output_size=output_size)
        self.offload_handler = offload_handler
    
    def forward(self, hidden):
        hidden, _ = self.linear1(hidden)
        hidden = self.ffn_act(hidden)
        hidden, _ = self.linear2(hidden)

        hidden = cpu_offload.group_prefetch_offload_commit(hidden, self.offload_handler)
        return hidden

class ToyModel(nn.Module):
    def __init__(self, num_layer, input_size, hidden_size, output_size, offload_num_layer, ):
        super(ToyModel, self).__init__()

        self.offload_handler = cpu_offload.GroupOffloadHandler(
            offload_num_layer, 
            1, 
            lambda t: not isinstance(t, torch.nn.Parameter),
        )
        
        self.layers = nn.ModuleList(
            [ToyFFN(input_size, hidden_size, output_size, self.offload_handler) for _ in range(num_layer)]
        )

    def forward(self, input):
        hidden = input
        with cpu_offload.CpuOffloadHookWithOffloadHandler(self.offload_handler):
            for l in self.layers:
                hidden = l(hidden)
        return hidden

input_size = 1024
hidden_size = 4096
output_size = 1024
num_layer = 8
batch = 512


# parallel_state.initialize_model_parallel()
initialize_model_parallel_for_nemo(1, 0, 0)

model = ToyModel(num_layer, input_size, hidden_size, output_size, (num_layer - 1)).cuda()

dummy_input = torch.randn(batch, 1, input_size).cuda()


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# optimizer = MainParamsOptimizerWrapper(
#     optimizer,
#     fp32_grad_accum=True,
#     contiguous_grad_bucket=True,
#     async_grad_allreduce=True,
#     grad_div_ar_fusion=False,
# )

optimizer.zero_grad()
for step in range(4):
    logits = model(dummy_input)
    loss = logits.sum()
    loss.backward()

    if step % 2 == 1:
        optimizer.step()
        optimizer.zero_grad()

import torch.nn as nn
import torch

import nemo.utils.cpu_offload_refactored as cpu_offload
class MyLinear(nn.Module):
    def __init__(self, input_dim, output_dim, offload_handler):
        super(MyLinear, self).__init__()
        self.offload_handler = offload_handler
        self.fc = nn.Linear(input_dim, output_dim).cuda()
        self.ln = nn.LayerNorm((output_dim,)).cuda()
    
    def forward(self, hidden):
        hidden = self.fc(hidden)
        with cpu_offload.CpuOffloadHookWithOffloadHandler(self.offload_handler):
            hidden = self.ln(hidden)
        return hidden

class MyMLP(nn.Module):
    def __init__(self, hidden_size, layer_num, offload_num_layers, ):
        super().__init__()

        self.offload_handler = cpu_offload.AsyncDoubleBufferGroupOffloadHandler(offload_num_layers)

        self.layers = nn.ModuleList(
            [MyLinear(hidden_size, hidden_size, self.offload_handler) for _ in range(layer_num)]
        )
    
    def forward(self, x):
        for l in self.layers:
            x = l(x)
            x = cpu_offload.group_prefetch_offload_commit(x, self.offload_handler)
        return x

dummy_input = torch.rand(1024,1024).cuda()
model = MyMLP(1024, 10, 3)

torch.cuda.cudart().cudaProfilerStart()

for step in range(4):
    output = model(dummy_input)
    output.sum().backward()
torch.cuda.cudart().cudaProfilerStop()

mem = torch.cuda.max_memory_allocated()
print(f"Peak memory usage: {mem // 1024 // 1024} MB")

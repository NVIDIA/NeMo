import nemo.utils.cpu_offload as cpu_offload
import torch.nn as nn
import torch

class JitOffloadLinear(nn.Module):
    def __init__(self, input_dim, output_dim, layer_id, offload=False):
        super(JitOffloadLinear, self).__init__()
        self.layer_id = layer_id
        self.fc = nn.Linear(input_dim, output_dim).cuda()
        self.ln = nn.LayerNorm((output_dim,)).cuda()
        self.offload = offload

    def forward(self, x):
        x = self.fc(x)
        # print(f"current memory {torch.cuda.memory_allocated() // 1024 // 1024} MiB")
        if self.offload:
            x = cpu_offload.jit_cpu_offload_saved_tensor(
                self.ln, 
                [x],
                # debug=True
            )
        else:
            x = self.ln(x)
        # x = self.ln(x)
        # print(f"current memory {torch.cuda.memory_allocated() // 1024 // 1024} MiB")
        return x

class JitOffloadMLP(nn.Module):
    def __init__(self, hidden_size, layer_num, offload_layer_num, ):
        super(JitOffloadMLP, self).__init__()

        self.layers = nn.ModuleList(
            [JitOffloadLinear(hidden_size, hidden_size, layer_id, offload=(layer_id<offload_layer_num)) for layer_id in range(layer_num)]
        )
    
    def forward(self, x):
        for l in self.layers:
            x = l(x)
            # print(f"current memory {torch.cuda.memory_allocated() // 1024 // 1024} MiB")
        return x

model = JitOffloadMLP(1024, 10, 0)

dummy_input = torch.rand(1024,1024).cuda()

torch.cuda.cudart().cudaProfilerStart()
torch.cuda.nvtx.range_push("jit_model_it")
for step in range(2):
    output = model(dummy_input)
    output.sum().backward()
torch.cuda.nvtx.range_pop()

torch.cuda.cudart().cudaProfilerStop()

print(f"peak memory {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
torch.cuda.reset_max_memory_allocated()

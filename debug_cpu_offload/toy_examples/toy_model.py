import torch.nn as nn
import torch

class MyLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyLinear, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim).cuda()
        self.ln = nn.LayerNorm((output_dim,)).cuda()
    
    def forward(self, hidden):
        hidden = self.fc(hidden)
        hidden = self.ln(hidden)
        return hidden

class MyMLP(nn.Module):
    def __init__(self, hidden_size, layer_num, ):
        super().__init__()

        self.layers = nn.ModuleList(
            [MyLinear(hidden_size, hidden_size) for _ in range(layer_num)]
        )
    
    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

dummy_input = torch.rand(1024,1024).cuda()
model = MyMLP(1024, 10)

torch.cuda.cudart().cudaProfilerStart()

for step in range(4):
    output = model(dummy_input)
    output.sum().backward()
torch.cuda.cudart().cudaProfilerStop()

mem = torch.cuda.max_memory_allocated()
print(f"Peak memory usage: {mem // 1024 // 1024} MB")
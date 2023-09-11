import torch
import torch.nn as nn

def pack_hook(t):
    print("Packing", t.shape)
    return t
def unpack_hook(t):
    print("Unpacking", t.shape)
    return t

class DummyLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        return torch.matmul(input, weight)
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        return grad_output.matmul(weight), grad_output.t().matmul(input)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight1 = nn.Parameter(torch.randn(1024, 1024))
        self.weight1.a = "aaa"
        self.act = nn.GELU()
    
    def forward(self, x):
        x = DummyLinear.apply(x, self.weight1)
        x = self.act(x)
        return x

model = Model().cuda()
dummy_input = torch.randn(128, 1024).cuda()

for p in model.parameters():
    p.a = torch.randn(128).cuda()

with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    out = model(dummy_input)
out.sum().backward()
print(model.weight1.a)
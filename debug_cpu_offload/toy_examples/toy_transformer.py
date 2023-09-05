import torch.nn as nn
import torch
torch.manual_seed(1234)

class ToyEncoder(nn.Module):
    def __init__(self, layer_id, hidden_size, num_head,):
        super(ToyEncoder, self).__init__()
        self.num_head = num_head
        self.layer_id = layer_id
        self.ln_pre_attn = nn.LayerNorm((hidden_size,))
        self.qkv = nn.Linear(hidden_size, 3*hidden_size, bias=False,)
        self.attn_out = nn.Linear(hidden_size, hidden_size, bias=False,)
        self.ln_pre_mlp = nn.LayerNorm((hidden_size,))
        self.ffn1 = nn.Linear(hidden_size, 4*hidden_size, bias=False,)
        self.mlp_act = nn.GELU()
        self.ffn2 = nn.Linear(4*hidden_size, hidden_size, bias=False,)
    
    def forward(self, hidden):
        s, b, h = hidden.shape
        residual = hidden
        hidden = self.ln_pre_attn(hidden)
        hidden = self.qkv(hidden)

        # [s,b,3h] -> [s,b,3,a, h/a] -> [3,b, a, s, h/a] 
        [q, k, v] = torch.permute(hidden.view(s, b, 3, self.num_head, h//self.num_head), (2,1,3,0,4))
        
        o = nn.functional.scaled_dot_product_attention(q, k, v) # [b, a, s, h/a]
        hidden = torch.permute(o, (2, 0, 1, 3)).view(s, b, h) # -> [s, b, a, h/a] -> [s, b, h]
        hidden = self.attn_out(hidden)
        hidden = residual + hidden

        residual = hidden
        hidden = self.ln_pre_mlp(hidden)
        hidden = self.ffn1(hidden)
        hidden = self.mlp_act(hidden)
        hidden = self.ffn2(hidden)
        hidden = residual + hidden

        print(f"^^^^^ layer {self.layer_id} checksum {hidden.sum()}")
        return hidden

class ToyGPT(nn.Module):
    def __init__(self, num_layer, hidden_size, num_head,):
        super(ToyGPT, self).__init__()
        self.layers = nn.ModuleList(
            [ToyEncoder(i, hidden_size, num_head) for i in range(num_layer)]
        )
    
    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


seq = 8192
batch = 1
hidden_size = 2048
head = 16
layer = 8

input = torch.randn(seq, batch, hidden_size).cuda()


model = ToyGPT(layer, hidden_size, head)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

model = model.cuda()


optimizer.zero_grad()
for step in range(4):
    print(f"===== Step {step} =====")
    logits = model(input)
    logits.sum().backward()
    if step % 2 == 1:
        optimizer.step()
        optimizer.zero_grad()

print(f"peak memory {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
torch.cuda.reset_max_memory_allocated()
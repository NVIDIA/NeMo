import nemo.utils.cpu_offload as cpu_offload
import torch.nn as nn
import torch
torch.manual_seed(1234)

class ToyGroupOffloadEncoder(nn.Module):
    def __init__(self, layer_id, hidden_size, num_head, offload_handler):
        super(ToyGroupOffloadEncoder, self).__init__()
        self.num_head = num_head
        self.layer_id = layer_id
        self.offload_handler = offload_handler
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
        hidden = cpu_offload.offload_saved_tensor_with_handler(
            self.ln_pre_attn,
            [hidden],
            offload_handler=self.offload_handler,
            # debug=True
        )
        hidden = self.qkv(hidden)

        # [s,b,3h] -> [s,b,3,a, h/a] -> [3,b, a, s, h/a] 
        [q, k, v] = torch.permute(hidden.view(s, b, 3, self.num_head, h//self.num_head), (2,1,3,0,4))
        
        o = nn.functional.scaled_dot_product_attention(q, k, v) # [b, a, s, h/a]
        hidden = torch.permute(o, (2, 0, 1, 3)).view(s, b, h) # -> [s, b, a, h/a] -> [s, b, h]
        hidden = self.attn_out(hidden)
        hidden = residual + hidden

        residual = hidden
        hidden = cpu_offload.offload_saved_tensor_with_handler(
            self.ln_pre_mlp,
            [hidden],
            offload_handler=self.offload_handler,
            # debug=True
        )
        hidden = self.ffn1(hidden)
        hidden = cpu_offload.offload_saved_tensor_with_handler(
            self.mlp_act,
            [hidden],
            offload_handler=self.offload_handler,
            # debug=True
        )
        # hidden = self.mlp_act(hidden)
        hidden = self.ffn2(hidden)
        hidden = residual + hidden

        hidden = cpu_offload.group_prefetch_offload_commit(hidden, self.offload_handler)

        # print(f"^^^^^ layer {self.layer_id} checksum {hidden.sum()}")
        return hidden

class ToyGroupOffloadGPT(nn.Module):
    def __init__(self, num_layer, hidden_size, num_head, offload_layer_num, prefetch_num_groups):
        super(ToyGroupOffloadGPT, self).__init__()

        def tensor_need_offloading_checker(tensor):
            return (not isinstance(tensor, torch.nn.Parameter)) 
        
        self.offload_handler = cpu_offload.GroupOffloadHandler(offload_layer_num, prefetch_num_groups, tensor_need_offloading_checker)

        self.layers = nn.ModuleList(
            [ToyGroupOffloadEncoder(i, hidden_size, num_head, self.offload_handler) for i in range(num_layer)]
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


model = ToyGroupOffloadGPT(layer, hidden_size, head, 5, 3).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

torch.cuda.cudart().cudaProfilerStart()
optimizer.zero_grad()
for step in range(4):
    torch.cuda.nvtx.range_push(f"step {step}")

    print(f"===== Step {step} =====")
    logits = model(input)
    loss = logits.sum() 
    print(f"===== loss {loss} =====")
    loss.backward()
    # optimizer.step()
    if step % 2 == 1:
        optimizer.step()
        optimizer.zero_grad()
        
    torch.cuda.nvtx.range_pop()

torch.cuda.cudart().cudaProfilerStop()

print(f"peak memory {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
# torch.cuda.reset_max_memory_allocated()
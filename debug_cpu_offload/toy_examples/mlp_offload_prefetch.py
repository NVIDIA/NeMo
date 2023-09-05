import nemo.utils.cpu_offload as cpu_offload
import torch.nn as nn
import torch

class BucketPrefetchOffloadLinear(nn.Module):
    def __init__(self, input_dim, output_dim, layer_id, offload_handler):
        super(BucketPrefetchOffloadLinear, self).__init__()
        self.layer_id = layer_id
        self.offload_handler = offload_handler
        self.fc = nn.Linear(input_dim, output_dim,).cuda()
        self.ln = nn.LayerNorm((output_dim,)).cuda()
    
    def forward(self, x):
        x = self.fc(x)
        
        # prefetch version
        x = cpu_offload.bucket_prefetch_offload_saved_tensor(
            self.ln, 
            [x],
            bucket_id=self.layer_id,
            offload_handler=self.offload_handler,
            # debug=True
        )
        
        # # jit version
        # x = cpu_offload.jit_cpu_offload_saved_tensor(
        #     self.ln, 
        #     [x],
        #     # debug=True
        # )

        # # no offloading version
        # x = self.ln(x)
        return x

class BucketPrefetchOffloadMLP(nn.Module):
    def __init__(self, hidden_size, layer_num, offload_layer_num, prefetch_num_buckets):
        super(BucketPrefetchOffloadMLP, self).__init__()

        self.offload_handler = cpu_offload.BucketPrefetchOffloadHandler(offload_layer_num, prefetch_num_buckets)
        self.layers = nn.ModuleList(
            [BucketPrefetchOffloadLinear(hidden_size, hidden_size, layer_id, self.offload_handler) for layer_id in range(layer_num)]
        )
    
    def forward(self, x):
        for l in self.layers:
            x = l(x)
            # print(f"current memory {torch.cuda.memory_allocated() // 1024 // 1024} MiB")
        return x


dummy_input = torch.rand(1024,1024).cuda()

model = BucketPrefetchOffloadMLP(1024, 10, 3, 3)


torch.cuda.cudart().cudaProfilerStart()
torch.cuda.nvtx.range_push("bucket prefetch it")

for step in range(2):
    output = model(dummy_input)
    output.sum().backward()

torch.cuda.nvtx.range_pop()
torch.cuda.cudart().cudaProfilerStop()

print(f"peak memory {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
torch.cuda.reset_max_memory_allocated()


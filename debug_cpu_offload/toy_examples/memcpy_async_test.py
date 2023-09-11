import torch

default_stream = torch.cuda.current_stream()
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

tensor1 = torch.rand(8192,1024).cuda()
tensor2 = torch.rand(1024,1024).cuda()

cpu_backup1 = []
for _ in range(10):
    cpu_backup1.append(torch.empty(tensor1.size(), dtype=tensor1.dtype, device='cpu', pin_memory=True))
cpu_backup2 = []
for _ in range(10):
    cpu_backup2.append(torch.empty(tensor2.size(), dtype=tensor2.dtype, device='cpu', pin_memory=True))

with torch.cuda.stream(stream1):
    for i in range(10):
        cpu_backup1[i].copy_(tensor1, non_blocking=True)

with torch.cuda.stream(stream2):
    for i in range(10):
        cpu_backup2[i].copy_(tensor2, non_blocking=True)

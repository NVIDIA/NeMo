import torch
import sys, json
from pathlib import Path

def check_tensors_same(a, b, key=''):
   assert a.dtype == b.dtype, f'Bad dtype {key}\ta= {a.dtype} b= {b.dtype}'
   assert a.shape == b.shape, f'Bad shape {key}\ta= {a.shape} b= {b.shape}'
   print(key, '\t', torch.all(a == b).item())
   assert torch.all(a == b).item(), f'Bad values {key}\na= {a}\nb= {b}'

def dir_iter(root_a, root_b):
   for file in map(lambda x: x.name, Path(root_a).glob('init_ckpt.*')):
      yield f'{root_a}/{file}', f'{root_b}/{file}'

if os.path.isdir(sys.argv[1]):
   assert os.path.isdir(sys.argv[2])
   files_iter = dir_iter(sys.argv[1], sys.argv[2])
elif os.path.isfile(sys.argv[1]):
   assert os.path.isfile(sys.argv[2])
   files_iter = [[sys.argv[1]], [sys.argv[2]]]
else:
   raise ValueError("Bad path?")

for file1, file2 in files_iter:
   print(file1)
   a = torch.load(file1, map_location=torch.device('cpu'))
   b = torch.load(file2, map_location=torch.device('cpu'))
   for key in a.keys():
      if type(a[key]) is not torch.Tensor:
         continue
      check_tensors_same(a[key], b[key], f'{file}:{key}')
   for key in b.keys():
      if type(b[key]) is not torch.Tensor or key in a:
         continue
      check_tensors_same(a[key], b[key], f'{file}:{key}')

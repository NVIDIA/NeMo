import math

import re
import numpy as np


# N = 8192
N = 16384
# The case of 0 / N is special, we want to simplify it to 0 / 2 instead of 0 / 1
numerator = np.arange(1, N // 8 + 1)
gcd = np.gcd(numerator, N)
num = numerator // gcd
denom = N // gcd
lut_vals = ['T_2_0'] + [f'T_{d}_{n}' for n, d in zip(num, denom)]
lut_string = f"static const __device__ float2 lut_mine_sp_8_{N}[{N // 8 + 1}] = {{\n  {','.join(lut_vals)}\n}};"
print(lut_string)

# Only define new values if it's not already in the cuFFTDx lookup table
cufftdx_lut_filename = 'mathdx/22.02/include/cufftdx/include/database/lut_defines_0.hpp.inc'
matches = set()
reg = re.compile(f'^#define T_{N}_([0-9]+) ')
with open(cufftdx_lut_filename, 'r') as f:
    for line in f:
        if (match := reg.match(line)) is not None:
            matches.add(int(match[1]))

numerator = np.arange(1, N // 8 + 1, 2)
angle = -2 * math.pi * numerator.astype(np.float64) / N
cos, sin = np.cos(angle), np.sin(angle)
defs = [f'#define T_{N}_{n} {{{c:.40f},{s:.40f}}}' for n, c, s in zip(numerator, cos, sin) if n not in matches]
def_string = '\n'.join(defs)
print(def_string)

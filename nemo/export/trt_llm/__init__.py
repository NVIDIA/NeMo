# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from mpi4py import MPI

# Pre load MPI libs to avoid tensorrt_llm importing failures.
print(f"Loaded mpi lib {MPI.__file__} successfully")

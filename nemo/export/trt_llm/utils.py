from typing import Optional

import tensorrt_llm


def is_rank(rank: Optional[int]) -> bool:
    """
    Check if the current MPI rank matches the specified rank(s).

    Args:
        rank (Optional[int]): The rank to check against.

    Returns:
        bool: True if the current rank matches the specified rank or if rank is None.
    """
    current_rank = tensorrt_llm.mpi_rank()
    if rank is None:
        return True
    if isinstance(rank, int):
        return current_rank == rank
    raise ValueError(f"Invalid rank argument: {rank}.")

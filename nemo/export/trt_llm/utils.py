from typing import List, Tuple, Union

import tensorrt_llm


def is_rank(rank: Union[int, List[int], Tuple[int], None]) -> bool:
    """
    Check if the current MPI rank matches the specified rank(s).

    Args:
        rank (Union[int, List[int], Tuple[int], None]): The rank or ranks to check against.

    Returns:
        bool: True if the current rank matches the specified rank(s) or if rank is None.
    """
    current_rank = tensorrt_llm.mpi_rank()
    if rank is None:
        return True
    if isinstance(rank, int):
        return current_rank == rank
    if isinstance(rank, (list, tuple)):
        return current_rank in rank
    else:
        raise ValueError(f"Invalid rank argument: {rank}.")

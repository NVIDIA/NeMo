import abc


class WithOptionalCudaGraphs(abc.ABC):
    """Abstract interface for """

    @abc.abstractmethod
    def disable_cuda_graphs(self):
        """Disable (maybe temporary) CUDA graphs"""
        raise NotImplementedError

    @abc.abstractmethod
    def maybe_enable_cuda_graphs(self):
        """Enable CUDA graphs if all conditions met"""
        raise NotImplementedError

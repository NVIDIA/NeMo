import torch
from lightning import Callback, LightningModule, Trainer


class NsightProfiling(Callback):
    """
    A simple Callback enabling Nsight Systems (nsys) profiling without a dependency on Megatron Core
    (unlike nemo.lightning.pytorch.callbacks.nsys.NsysCallback).

    It can be enabled from YAML configuration as follows:

    .. code-block:: yaml

        trainer:
          ...
          callbacks:
            - _target_: nemo.collections.speechlm2.parts.nsight.NsightProfiling
              start_step: 5
              end_step: 10
              gen_shape: true
              nvtx_ranges: true
    """

    def __init__(
        self,
        begin_step: int,
        end_step: int,
        gen_shape: bool = False,
        nvtx_ranges: bool = False,
    ) -> None:
        super().__init__()
        self.begin_step = begin_step
        self.end_step = end_step
        self.gen_shape = gen_shape
        self.nvtx_ranges = nvtx_ranges

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx: int) -> None:
        if batch_idx == self.begin_step:
            print("STARTING NSIGHT PROFILING")
            torch.cuda.profiler.cudart().cudaProfilerStart()
            if self.nvtx_ranges:
                if self.gen_shape:
                    torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()
                else:
                    torch.autograd.profiler.emit_nvtx().__enter__()

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx: int) -> None:
        if batch_idx == self.end_step:
            print("STOPPING NSIGHT PROFILING")
            torch.cuda.profiler.cudart().cudaProfilerStop()
            if self.nvtx_ranges:
                torch.autograd.profiler.emit_nvtx().__exit__(None, None, None)

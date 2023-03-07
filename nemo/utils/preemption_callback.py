import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from nemo.collections.common.callbacks.nemomodelcheckpoint import NeMoModelCheckpoint
import signal  
import torch
import sys

class PreemptionCallback(Callback):

    def __init__(self, device, checkpoint_callback, sig=signal.SIGTERM):
        self.sig = sig
        self.device = device
        self.checkpoint_callback = checkpoint_callback

    @property
    def interrupted(self):
        interrupted = torch.tensor(self._interrupted).int().to(self.device)
        torch.distributed.broadcast(interrupted, 0)
        interrupted = bool(interrupted.item())
        return interrupted

    def on_train_start(self, trainer, pl_module):
        self._interrupted = False
        self.released = False
        self.original_handler = signal.getsignal(self.sig)

        def master_handler(signum, frame):
            self.release()
            self._interrupted = True

        def ignoring_handler(signum, frame):
            self.release()

        self.private_rank = torch.distributed.get_rank()
        if self.private_rank == 0:
            signal.signal(self.sig, master_handler)
        else:
            signal.signal(self.sig, ignoring_handler)

        return self

    def on_train_end(self, trainer, pl_module):
        self.release()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int):
        # check if the job was preempted
        # NOTE: "timeout_handler.interrupted" is a property which triggers a
        # distributed broadcast of "_interrupted" flag from rank 0 to all other
        # ranks, to avoid performance overheads it's best to store the result in
        # a regular local variable
        interrupted = self.interrupted
        if interrupted:
            print("Received SIGTERM, exiting")
            monitor_candidates = self.checkpoint_callback._monitor_candidates(trainer)
            self.checkpoint_callback._save_last_checkpoint(trainer, monitor_candidates)
            sys.exit(0)

    def release(self):
        if self.released:
            return False

        signal.signal(self.sig, self.original_handler)
        self.released = True
        return True

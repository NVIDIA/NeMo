# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import signal  
import torch
import sys
from pytorch_lightning.callbacks import Callback
from nemo.utils import logging

class PreemptionCallback(Callback):
    """
    PreemptionCallback class creates a callback that checks for preemption during training at the end of every step.
    Upon preemption the callback provides function to gracefully exit the training immediately and also saves the state of training (to be able to start from the 
    same step without wasting any compute while resuming the next time).

    PreemptionCallback is always enabled.
    """

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

        if pl_module.device.type == 'cuda':
            assert torch.distributed.is_available() and torch.distributed.is_initialized(), "Preemption requires torch distributed to be initialized"
        else:
            logging.info("Preemption is supported only on GPUs")

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
        #interrupted = self.interrupted(pl_module.device)
        interrupted = self.interrupted()
        if interrupted:
            logging.info("Received SIGTERM, exiting")
            monitor_candidates = self.checkpoint_callback._monitor_candidates(trainer)
            self.checkpoint_callback._save_last_checkpoint(trainer, monitor_candidates)
            sys.exit(0)
            
    def release(self):
        if self.released:
            return False

        signal.signal(self.sig, self.original_handler)
        self.released = True
        return True

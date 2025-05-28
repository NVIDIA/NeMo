# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any

import lightning.pytorch as pl
import torch
import wandb


class OptimizerMonitor(pl.Callback):
    """
    Computes and logs the L2 norm of gradients as well as any optimizer-specific metrics implemented in the optimizer's `report_per_parameter_metrics` method.

    L2 norms are calculated after the reduction of gradients across GPUs. This function iterates over the parameters of
    the model and may cause a reduction in throughput while training large models. In order to ensure the
    correctness of the norm, this function should be called after gradient unscaling in cases where gradients are scaled.

    Example:
        import nemo_run as run
        from nemo.lightning.pytorch.callbacks import OptimizerMonitor

        recipe.trainer.callbacks.append(
            run.Config(OptimizerMonitor)
        )

    The metrics are logged by the :class:`.Logger` to the following keys as described below. `grad_l2_norm` and `layer_grad_l2_norm` are
    logged in addition to metrics logged by the optimizer's `report_per_parameter_metrics` method. For convenience we have listed
    the metrics logged by DecoupledAdamW below.

    +-----------------------------------------------+-----------------------------------------------------+
    | Key                                           | Logged data                                         |
    +===============================================+=====================================================+
    |                                               | L2 norm of the gradients of all parameters in       |
    | ``l2_norm/grad/global``                       | the model on the :attr:`.Event.AFTER_TRAIN_BATCH`   |
    |                                               | event.                                              |
    +-----------------------------------------------+-----------------------------------------------------+
    |                                               | Layer-wise L2 norms                                 |
    | ``l2_norm/grad/LAYER_NAME``                   |                                                     |
    |                                               |                                                     |
    +-----------------------------------------------+-----------------------------------------------------+
    |                                               | Layer-wise L2 norms of Adam first moment after      |
    | ``l2_norm/moment/LAYER_NAME``                 |  calling optimizer step.                            |
    |                                               |                                                     |
    +-----------------------------------------------+-----------------------------------------------------+
    |                                               | Layer-wise L2 norms of parameter weights            |
    | ``l2_norm/param/LAYER_NAME``                  |                                                     |
    |                                               |                                                     |
    +-----------------------------------------------+-----------------------------------------------------+
    |                                               | Layer-wise L2 norms of the step                     |
    | ``l2_norm/update/LAYER_NAME``                 |                                                     |
    |                                               |                                                     |
    +-----------------------------------------------+-----------------------------------------------------+
    """

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: pl.utilities.types.STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if trainer.global_step % trainer.log_every_n_steps == 0:
            norm = 0.0
            optimizer_metrics = {}

            for name, p in pl_module.named_parameters():
                if p.main_grad is not None and p.requires_grad:

                    # Always log grad norm as a default metric if it's not specified
                    if f'l2_norm/grad/{name}' not in optimizer_metrics:
                        param_grad_norm = torch.linalg.vector_norm(p.main_grad)
                        optimizer_metrics[f'l2_norm/grad/{name}'] = param_grad_norm

            for metric in optimizer_metrics:
                if metric.startswith('l2_norm/grad'):
                    norm += optimizer_metrics[metric] ** 2

            optimizer_metrics['l2_norm/grad/global'] = norm**0.5

            for metric in optimizer_metrics:
                if isinstance(optimizer_metrics[metric], torch.Tensor):
                    optimizer_metrics[metric] = optimizer_metrics[metric].item()

            for metric, value in optimizer_metrics.items():
                self.log(metric, value)

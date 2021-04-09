# ! /usr/bin/python
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

# Copyright 2018-2019, Mingkun Huang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import operator
from dataclasses import dataclass
from typing import Optional

import torch

from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types import LabelsType, LengthsType, LogprobsType, LossType, NeuralType
from nemo.utils import logging, model_utils

try:
    import warprnnt_pytorch as warprnnt

    WARP_RNNT_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    WARP_RNNT_AVAILABLE = False

try:
    from nemo.collections.asr.parts.numba.rnnt_loss import RNNTLossNumba

    NUMBA_RNNT_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    NUMBA_RNNT_AVAILABLE = False


WARP_RNNT_INSTALLATION_MESSAGE = (
    "Could not import `warprnnt_pytorch`.\n"
    "Please visit https://github.com/HawkAaron/warp-transducer "
    "and follow the steps in the readme to build and install the "
    "pytorch bindings for RNNT Loss, or use the provided docker "
    "container that supports RNN-T loss."
)

NUMBA_INSTALLATION_MESSAGE = (
    "Could not import `numba`.\n"
    "To use transducer loss, please install numba in one of the following ways."
    "1) If using conda, simply install `conda install -c numba numba`\n"
    "2) If using pip, `pip install --upgrade --ignore-installed numba`\n"
    "followed by `export NUMBAPRO_LIBDEVICE='/usr/local/cuda/nvvm/libdevice/'` and \n"
    "`export NUMBAPRO_NVVM='/usr/local/cuda/nvvm/lib64/libnvvm.so'`"
)


@dataclass
class RNNTLossConfig:
    loss_name: str
    lib_name: str
    is_available: bool = False
    installation_msg: str = ""
    min_version: Optional[str] = None


# Resolved list of available RNNT losses
RNNT_LOSS_RESOLVER = {
    "warprnnt": RNNTLossConfig(
        loss_name="warprnnt",
        lib_name="warprnnt_pytorch",
        is_available=WARP_RNNT_AVAILABLE,
        installation_msg=WARP_RNNT_INSTALLATION_MESSAGE,
    ),
    "warprnnt_numba": RNNTLossConfig(
        loss_name="warprnnt_numba",
        lib_name="numba",
        min_version='0.53.0',
        is_available=NUMBA_RNNT_AVAILABLE,
        installation_msg=NUMBA_INSTALLATION_MESSAGE,
    ),
}

RNNT_LOSS_RESOLVER['default'] = RNNT_LOSS_RESOLVER['warprnnt_numba']


def _warn_unused_additional_kwargs(loss_name, kwargs):
    if len(kwargs) > 0:
        logging.warning(
            f"Loss function `{loss_name}` was provided with following additional kwargs,\n"
            f"however they were ignored as it is unused.\n"
            f"{kwargs}"
        )


def resolve_rnnt_default_loss_name() -> str:
    return RNNT_LOSS_RESOLVER['default'].loss_name


def resolve_rnnt_loss(loss_name: str, blank_idx: int, loss_kwargs: dict = None) -> torch.nn.Module:
    loss_function_names = list(RNNT_LOSS_RESOLVER.keys())

    if loss_name not in loss_function_names:
        raise ValueError(
            f"Provided `loss_name` {loss_name} not in list of available RNNT losses \n" f"{loss_function_names}"
        )

    all_available_losses = {name: config for name, config in RNNT_LOSS_RESOLVER.items() if config.is_available}

    loss_config = RNNT_LOSS_RESOLVER[loss_name]  # type: RNNTLossConfig

    # Re-raise import error with installation message
    if not loss_config.is_available:
        msg = (
            f"Installed RNNT losses are : {list(all_available_losses.keys())}.\n"
            f"****************************************************************\n"
            f"To install the selected loss function, please follow the steps below:\n"
            f"{loss_config.installation_msg}"
        )
        raise ImportError(msg)

    # Library version check
    if loss_config.min_version is not None:
        ver_matched, msg = model_utils.check_lib_version(
            loss_config.lib_name, checked_version=loss_config.min_version, operator=operator.ge
        )

        if ver_matched is False:
            msg = (
                f"{msg}\n"
                f"****************************************************************\n"
                f"To update the selected loss function, please follow the steps below:\n"
                f"{loss_config.installation_msg}"
            )
            raise RuntimeError(msg)

    # Resolve loss functions sequentially
    loss_kwargs = {} if loss_kwargs is None else loss_kwargs

    # Get actual loss name for `default`
    if loss_name == 'default':
        loss_name = loss_config.loss_name

    """
    Resolve RNNT loss functions
    """
    if loss_name == 'warprnnt':
        loss_func = warprnnt.RNNTLoss(blank=blank_idx, reduction='none')
        _warn_unused_additional_kwargs(loss_name, loss_kwargs)

    elif loss_name == 'warprnnt_numba':
        loss_func = RNNTLossNumba(blank=blank_idx, reduction='none')
        _warn_unused_additional_kwargs(loss_name, loss_kwargs)

    else:
        raise ValueError(
            f"Invalid value of `loss_name`: {loss_name}. Allowed loss names are :" f"{loss_function_names}"
        )

    return loss_func


class RNNTLoss(Loss):
    @property
    def input_types(self):
        """Input types definitions for CTCLoss.
        """
        return {
            "log_probs": NeuralType(('B', 'T', 'D', 'D'), LogprobsType()),
            "targets": NeuralType(('B', 'T'), LabelsType()),
            "input_lengths": NeuralType(tuple('B'), LengthsType()),
            "target_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Output types definitions for CTCLoss.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, num_classes, reduction: str = 'mean_batch', loss_name: str = "default", loss_kwargs=None):
        """
        RNN-T Loss function based on https://github.com/HawkAaron/warp-transducer.
        Optionally, can utilize a numba implementation of the same loss without having to compile the loss,
        albiet there is a small speed penalty for JIT numba compile.

        Note:
            Requires the pytorch bindings to be installed prior to calling this class.

        Warning:
            In the case that GPU memory is exhausted in order to compute RNNTLoss, it might cause
            a core dump at the cuda level with the following error message.

            ```
                ...
                costs = costs.to(acts.device)
            RuntimeError: CUDA error: an illegal memory access was encountered
            terminate called after throwing an instance of 'c10::Error'
            ```

            Please kill all remaining python processes after this point, and use a smaller batch size
            for train, validation and test sets so that CUDA memory is not exhausted.

        Args:
            num_classes: Number of target classes for the joint network to predict.
                (Excluding the RNN-T blank token).

            reduction: Type of reduction to perform on loss. Possibly values are `mean`, `sum` or None.
                None will return a torch vector comprising the individual loss values of the batch.

            loss_name: String that is resolved into an RNNT loss function. Available list of losses
                is ininitialized in `RNNT_LOSS_RESOLVER` dictionary.

            loss_kwargs: Optional Dict of (str, value) pairs that are passed to the instantiated loss
                function.
        """
        super(RNNTLoss, self).__init__()

        if reduction not in [None, 'mean', 'sum', 'mean_batch']:
            raise ValueError('`reduction` must be one of [mean, sum, mean_batch]')

        self._blank = num_classes
        self.reduction = reduction
        self._loss = resolve_rnnt_loss(loss_name, blank_idx=self._blank, loss_kwargs=loss_kwargs)

    @typecheck()
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # Cast to int 32
        targets = targets.int()
        input_lengths = input_lengths.int()
        target_lengths = target_lengths.int()

        max_logit_len = input_lengths.max()
        max_targets_len = target_lengths.max()

        # Force cast joint to float32
        # TODO: Remove once Numba supports FP16
        if log_probs.dtype != torch.float32:
            logits_orig = log_probs
            log_probs = log_probs.float()
            del logits_orig  # save memory *before* computing the loss

        # Ensure that shape mismatch does not occur due to padding
        # Due to padding and subsequent downsampling, it may be possible that
        # max sequence length computed does not match the actual max sequence length
        # of the log_probs tensor, therefore we increment the input_lengths by the difference.
        # This difference is generally small.
        if log_probs.shape[1] != max_logit_len:
            log_probs = log_probs.narrow(dim=1, start=0, length=max_logit_len).contiguous()

        # Reduce transcript length to correct alignment if additional padding was applied.
        # Transcript: [B, L] -> [B, L']; If L' < L
        if targets.shape[1] != max_targets_len:
            targets = targets.narrow(dim=1, start=0, length=max_targets_len)

        # Loss reduction can be dynamic, so set it prior to call
        if self.reduction != 'mean_batch':
            self._loss.reduction = self.reduction

        # Compute RNNT loss
        loss = self._loss(acts=log_probs, labels=targets, act_lens=input_lengths, label_lens=target_lengths)

        # Loss reduction can be dynamic, so reset it after call
        if self.reduction != 'mean_batch':
            self._loss.reduction = 'none'

        # Loss reduction only for mean_batch mode
        if self.reduction == 'mean_batch':
            loss = torch.mean(loss)

        # del new variables that may have been created
        del (
            log_probs,
            targets,
            input_lengths,
            target_lengths,
        )

        return loss

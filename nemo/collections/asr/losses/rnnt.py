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
from omegaconf import DictConfig, OmegaConf

from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types import LabelsType, LengthsType, LogprobsType, LossType, NeuralType
from nemo.core.utils.numba_utils import NUMBA_INSTALLATION_MESSAGE
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
    "pytorch": RNNTLossConfig(
        loss_name="pytorch",
        lib_name="pytorch",
        min_version='0.0',
        is_available=True,
        installation_msg="it just works but slow",
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


def resolve_rnnt_loss(loss_name: str, blank_idx: int, big_blank_idx: int, huge_blank_idx, blank_duration: int, loss_kwargs: dict = None, sigma: float = 0.0) -> torch.nn.Module:
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

    if isinstance(loss_kwargs, DictConfig):
        loss_kwargs = OmegaConf.to_container(loss_kwargs, resolve=True)

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
        fastemit_lambda = loss_kwargs.pop('fastemit_lambda', 0.0)
        clamp = loss_kwargs.pop('clamp', -1.0)
        loss_func = RNNTLossNumba(blank=blank_idx, big_blank=big_blank_idx, huge_blank=huge_blank_idx, blank_duration=blank_duration, reduction='none', fastemit_lambda=fastemit_lambda, clamp=clamp, sigma=sigma)
        _warn_unused_additional_kwargs(loss_name, loss_kwargs)

    elif loss_name == 'pytorch':
        loss_func = RNNTLossPytorch(blank=blank_idx, big_blank=big_blank_idx, huge_blank=huge_blank_idx, reduction='none')

    else:
        raise ValueError(
            f"Invalid value of `loss_name`: {loss_name}. Allowed loss names are :" f"{loss_function_names}"
        )

    return loss_func

class RNNTLossPytorch(Loss):
    @property
    def input_types(self):
        """Input types definitions for CTCLoss.
        """
        return {
            "acts": NeuralType(('B', 'T', 'T', 'D'), LogprobsType()),
            "labels": NeuralType(('B', 'T'), LabelsType()),
            "act_lens": NeuralType(tuple('B'), LengthsType()),
            "label_lens": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Output types definitions for CTCLoss.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, blank, blank_duration, reduction):
        super().__init__()
        self.blank = blank
        self.big_blank = 1 + blank
        self.huge_blank = 2 + blank
        self.blank_duration = blank_duration
        self.reduction = reduction

    @typecheck()
    def forward(self, acts, labels, act_lens, label_lens):
        acts = torch.log_softmax(acts, -1)
        forward_logprob = self.compute_forward_prob(acts, labels, act_lens, label_lens)
        return -forward_logprob

    def compute_forward_prob(self, acts, labels, act_lens, label_lens):
        B, T, U, _ = acts.shape

        log_alpha = torch.zeros(B, T, U)
        log_alpha = log_alpha.cuda()
        for t in range(T):
            for u in range(U):
                if u == 0:
                    if t == 0:
                        log_alpha[:, t, u] = 0.0
                    else:
                        tmp = log_alpha[:, t-1, u] + acts[:, t-1, 0, self.blank] 
                        if t >= self.blank_duration:
                            tt = log_alpha[:, t-self.blank_duration, u] + acts[:, t-self.blank_duration, 0, self.big_blank]
                            tmp2 = torch.logsumexp(torch.stack([tmp, tt]), dim=0)
#                            log_alpha[:, t, u] = torch.logsumexp(torch.stack([tmp, tt]), dim=0)
                            if t >= 2 * self.blank_duration:
                                tt = log_alpha[:, t-2 * self.blank_duration, u] + acts[:, t-2*self.blank_duration, 0, self.huge_blank]
                                log_alpha[:, t, u] = torch.logsumexp(torch.stack([tmp2, tt]), dim=0)
                            else:
                                log_alpha[:, t, u] = tmp2

                        else:
                            log_alpha[:, t, u] = tmp
                            
                else:
                    if t == 0:
                        gathered = torch.gather(acts[:, t, u-1], dim=1, index=labels[:,u-1].view(-1,1).type(torch.int64) ).reshape(-1)
                        log_alpha[:, t, u] = log_alpha[:, t,u-1] + gathered.cuda()
                    else:
                        tmp = torch.logsumexp(torch.stack([
                            log_alpha[:, t-1, u] + acts[:, t-1, u, self.blank],
                            log_alpha[:, t, u-1] + torch.gather(acts[:, t, u-1], dim=1, index=labels[:,u-1].view(-1,1).type(torch.int64) ).reshape(-1)
                        ]), dim=0)
                        if t >= self.blank_duration:
                            tmp2 = torch.logsumexp(torch.stack([
                                tmp,
                                log_alpha[:, t-self.blank_duration, u] + acts[:, t-self.blank_duration, u, self.big_blank],
                            ]), dim=0)
                            if t >= 2 * self.blank_duration:
                                log_alpha[:, t, u] = torch.logsumexp(torch.stack([
                                    tmp2,
                                    log_alpha[:, t-2*self.blank_duration, u] + acts[:, t-2*self.blank_duration, u, self.huge_blank],
                                ]), dim=0)
                            else:
                                log_alpha[:, t, u] = tmp2
                        else:
                            log_alpha[:, t, u] = tmp

        log_probs = []
        for b in range(B):
            tt = log_alpha[b, act_lens[b]-1, label_lens[b]] + acts[b, act_lens[b]-1, label_lens[b], self.blank]
            if act_lens[b] >= self.blank_duration:
                jj = log_alpha[b, act_lens[b]-self.blank_duration, label_lens[b]] + acts[b, act_lens[b]-self.blank_duration, label_lens[b], self.big_blank]
                tt = torch.logsumexp(torch.stack([
                      tt, jj
                ]), dim=0)
                if act_lens[b] >= 2 * self.blank_duration:
                    kk = log_alpha[b, act_lens[b]-2 * self.blank_duration, label_lens[b]] + acts[b, act_lens[b]-2 * self.blank_duration, label_lens[b], self.huge_blank]
                    to_append = torch.logsumexp(torch.stack([
                          tt, kk
                    ]), dim=0)
                else:
                    to_append = tt
                
                log_probs.append(to_append)
            else:
                log_probs.append(tt)
        log_prob = torch.stack(log_probs) 

        return log_prob

class RNNTLoss(Loss):
    @property
    def input_types(self):
        """Input types definitions for CTCLoss.
        """
        return {
            "log_probs": NeuralType(('B', 'T', 'T', 'D'), LogprobsType()),
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

    def __init__(self, num_classes, blank_duration, reduction: str = 'mean_batch', loss_name: str = "default", loss_kwargs=None, sigma: float = 0.0):
        """
        RNN-T Loss function based on https://github.com/HawkAaron/warp-transducer.
        Optionally, can utilize a numba implementation of the same loss without having to compile the loss,
        albiet there is a small speed penalty for JIT numba compile.

        Note:
            Requires Numba 0.53.0 or later to be installed to use this loss function.

        Losses can be selected via the config, and optionally be passed keyword arguments as follows.

        Examples:
            .. code-block:: yaml

                model:  # RNNT Model config
                    ...
                    loss:
                        loss_name: "warprnnt_numba"
                        warprnnt_numba_kwargs:
                            fastemit_lambda: 0.0

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
        self._big_blank = num_classes + 1
        self._huge_blank = num_classes + 2
        self._sigma = sigma
        self._blank_duration = blank_duration
        self.reduction = reduction
        self._loss = resolve_rnnt_loss(loss_name, blank_idx=self._blank, big_blank_idx=self._big_blank, huge_blank_idx=self._huge_blank, blank_duration=self._blank_duration, loss_kwargs=loss_kwargs, sigma=sigma)

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
            targets = targets.narrow(dim=1, start=0, length=max_targets_len).contiguous()

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

if __name__ == "__main__":
#    B, T, U, V = 2, 32, 16, 256
    B, T, U, V = 3, 11, 7, 5 
#    B, T, U, V = 3, 3, 3, 3

    duration=2

    Loss = RNNTLossPytorch(V - 3, blank_duration=duration, reduction='mean')
    Loss2 = RNNTLoss(V - 3, blank_duration=duration, reduction='mean_batch', loss_name='warprnnt_numba')

    for t in range(1000):
        acts = torch.rand([B, T, U, V]) - 0.5
        acts = torch.nn.Parameter(acts * 5, requires_grad=True)

        labels = torch.randint(low=0, high=V, size=[B, U])
        act_lens = torch.randint(low=1, high=T + 1, size=[B])
        label_lens = torch.randint(low=1, high=U + 1, size=[B]) - 1
        act_lens[0] = T
        label_lens[0] = U - 1
        logits = acts
        
        logits = logits.cuda()
        labels = labels.cuda()
        act_lens = act_lens.cuda()
        label_lens = label_lens.cuda()

        labels = labels.contiguous()

        loss = Loss(acts=logits, labels=labels, act_lens=act_lens, label_lens=label_lens)
        loss = torch.mean(loss)
        loss.backward()
        grad1 = torch.clone(acts.grad)
        acts.grad *= 0.0

        loss2 = Loss2(log_probs=logits, targets=labels, input_lengths=act_lens, target_lengths=label_lens)
        
        loss2.backward()

        print("loss diff", float(loss - loss2))
        print("grad norm diff per element", float(torch.norm(acts.grad - grad1) / (B * T * U * V)))
#        print("they are")
#        print(acts.grad)
#        print(grad1)


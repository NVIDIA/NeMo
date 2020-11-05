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

# Copyright 2018-2020 William Falcon
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

import os
import pickle
import sys
from functools import partial
from typing import Callable, Optional

import numpy as np
import pytest
import torch
from pytorch_lightning.metrics import Metric
from scipy.stats import entropy
from torch.distributions.utils import logits_to_probs
from torch.multiprocessing import Pool, set_start_method

from nemo.collections.common.metrics import Perplexity


NUM_PROCESSES = 2
NUM_BATCHES = 10
BATCH_SIZE = 16
NUM_CLASSES = 5
EXTRA_DIM = 3
THRESHOLD = 0.5


def setup_ddp(rank, world_size):
    """ Setup ddp enviroment """
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ['MASTER_PORT'] = '8088'

    if torch.distributed.is_available() and sys.platform not in ['win32', 'cygwin']:
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)


def _class_test(
    rank: int,
    worldsize: int,
    preds: torch.Tensor,
    target: torch.Tensor,
    metric_class: Metric,
    sk_metric: Callable,
    dist_sync_on_step: bool,
    metric_args: dict = {},
    check_dist_sync_on_step: bool = True,
    check_batch: bool = True,
    atol: float = 1e-8,
):
    """ Utility function doing the actual comparison between lightning class metric
        and reference metric.
        Args:
            rank: rank of current process
            worldsize: number of processes
            preds: torch tensor with predictions
            target: torch tensor with targets
            metric_class: lightning metric class that should be tested
            sk_metric: callable function that is used for comparison
            dist_sync_on_step: bool, if true will synchronize metric state across
                processes at each ``forward()``
            metric_args: dict with additional arguments used for class initialization
            check_dist_sync_on_step: bool, if true will check if the metric is also correctly
                calculated per batch per device (and not just at the end)
            check_batch: bool, if true will check if the metric is also correctly
                calculated across devices for each batch (and not just at the end)
    """
    # Instanciate lightning metric
    metric = metric_class(compute_on_step=True, dist_sync_on_step=dist_sync_on_step, **metric_args)

    # verify metrics work after being loaded from pickled state
    pickled_metric = pickle.dumps(metric)
    metric = pickle.loads(pickled_metric)

    for i in range(rank, NUM_BATCHES, worldsize):
        batch_result = metric(preds[i], target[i])

        if metric.dist_sync_on_step:
            if rank == 0:
                ddp_preds = torch.stack([preds[i + r] for r in range(worldsize)])
                ddp_target = torch.stack([target[i + r] for r in range(worldsize)])
                sk_batch_result = sk_metric(ddp_preds, ddp_target)
                # assert for dist_sync_on_step
                if check_dist_sync_on_step:
                    assert np.allclose(batch_result.numpy(), sk_batch_result, atol=atol)
        else:
            sk_batch_result = sk_metric(preds[i], target[i])
            # assert for batch
            if check_batch:
                assert np.allclose(batch_result.numpy(), sk_batch_result, atol=atol)

    # check on all batches on all ranks
    result = metric.compute()
    assert isinstance(result, torch.Tensor)

    total_preds = torch.stack([preds[i] for i in range(NUM_BATCHES)])
    total_target = torch.stack([target[i] for i in range(NUM_BATCHES)])
    sk_result = sk_metric(total_preds, total_target)

    # assert after aggregation
    assert np.allclose(result.numpy(), sk_result, atol=atol)


def _functional_test(
    preds: torch.Tensor,
    target: torch.Tensor,
    metric_functional: Callable,
    sk_metric: Callable,
    metric_args: dict = {},
    atol: float = 1e-8,
):
    """ Utility function doing the actual comparison between lightning functional metric
        and reference metric.
        Args:
            preds: torch tensor with predictions
            target: torch tensor with targets
            metric_functional: lightning metric functional that should be tested
            sk_metric: callable function that is used for comparison
            metric_args: dict with additional arguments used for class initialization
    """
    metric = partial(metric_functional, **metric_args)

    for i in range(NUM_BATCHES):
        lightning_result = metric(preds[i], target[i])
        sk_result = sk_metric(preds[i], target[i])

        # assert its the same
        assert np.allclose(lightning_result.numpy(), sk_result, atol=atol)


class MetricTester:
    """ Class used for efficiently run alot of parametrized tests in ddp mode.
        Makes sure that ddp is only setup once and that pool of processes are
        used for all tests.
        All tests should subclass from this and implement a new method called
            `test_metric_name`
        where the method `self.run_metric_test` is called inside.
    """

    atol = 1e-8

    def setup_class(self):
        """ Setup the metric class. This will spawn the pool of workers that are
            used for metric testing and setup_ddp
        """
        try:
            set_start_method('spawn')
        except RuntimeError:
            pass
        self.poolSize = NUM_PROCESSES
        self.pool = Pool(processes=self.poolSize)
        self.pool.starmap(setup_ddp, [(rank, self.poolSize) for rank in range(self.poolSize)])

    def teardown_class(self):
        """ Close pool of workers """
        self.pool.close()
        self.pool.join()

    def run_functional_metric_test(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        metric_functional: Callable,
        sk_metric: Callable,
        metric_args: dict = {},
    ):
        """ Main method that should be used for testing functions. Call this inside
            testing method
            Args:
                preds: torch tensor with predictions
                target: torch tensor with targets
                metric_functional: lightning metric class that should be tested
                sk_metric: callable function that is used for comparison
                metric_args: dict with additional arguments used for class initialization
        """
        _functional_test(
            preds=preds,
            target=target,
            metric_functional=metric_functional,
            sk_metric=sk_metric,
            metric_args=metric_args,
            atol=self.atol,
        )

    def run_class_metric_test(
        self,
        ddp: bool,
        preds: torch.Tensor,
        target: torch.Tensor,
        metric_class: Metric,
        sk_metric: Callable,
        dist_sync_on_step: bool,
        metric_args: dict = {},
        check_dist_sync_on_step: bool = True,
        check_batch: bool = True,
    ):
        """ Main method that should be used for testing class. Call this inside testing
            methods.
            Args:
                ddp: bool, if running in ddp mode or not
                preds: torch tensor with predictions
                target: torch tensor with targets
                metric_class: lightning metric class that should be tested
                sk_metric: callable function that is used for comparison
                dist_sync_on_step: bool, if true will synchronize metric state across
                    processes at each ``forward()``
                metric_args: dict with additional arguments used for class initialization
                check_dist_sync_on_step: bool, if true will check if the metric is also correctly
                    calculated per batch per device (and not just at the end)
                check_batch: bool, if true will check if the metric is also correctly
                    calculated across devices for each batch (and not just at the end)
        """
        if ddp:
            if sys.platform == "win32":
                pytest.skip("DDP not supported on windows")

            self.pool.starmap(
                partial(
                    _class_test,
                    preds=preds,
                    target=target,
                    metric_class=metric_class,
                    sk_metric=sk_metric,
                    dist_sync_on_step=dist_sync_on_step,
                    metric_args=metric_args,
                    check_dist_sync_on_step=check_dist_sync_on_step,
                    check_batch=check_batch,
                    atol=self.atol,
                ),
                [(rank, self.poolSize) for rank in range(self.poolSize)],
            )
        else:
            _class_test(
                0,
                1,
                preds=preds,
                target=target,
                metric_class=metric_class,
                sk_metric=sk_metric,
                dist_sync_on_step=dist_sync_on_step,
                metric_args=metric_args,
                check_dist_sync_on_step=check_dist_sync_on_step,
                check_batch=check_batch,
                atol=self.atol,
            )


def reference_perplexity_func(probs):
    ent = entropy(probs, axis=-1)
    ppl = np.exp(ent)
    return ppl.mean()


def _perplexity_class_test(
    rank: int,
    worldsize: int,
    probs: Optional[torch.Tensor],
    logits: Optional[torch.Tensor],
    dist_sync_on_step: bool,
    metric_args: dict = {},
    check_dist_sync_on_step: bool = True,
    check_batch: bool = True,
    atol: float = 1e-8,
):
    """ Utility function doing the actual comparison between lightning class metric
        and reference metric.
        Args:
            rank: rank of current process
            worldsize: number of processes
            probs: torch tensor with probabilities
            logits: torch tensor with logits. The function checks ``probs`` and ``logits are mutually exclusive for
                ``Perplexity`` metric.
            dist_sync_on_step: bool, if true will synchronize metric state across
                processes at each ``forward()``
            metric_args: dict with additional arguments used for class initialization
            check_dist_sync_on_step: bool, if true will check if the metric is also correctly
                calculated per batch per device (and not just at the end)
            check_batch: bool, if true will check if the metric is also correctly
                calculated across devices for each batch (and not just at the end)
    """
    # Instanciate lightning metric
    perplexity = Perplexity(compute_on_step=True, dist_sync_on_step=dist_sync_on_step, **metric_args)
    if (probs is None) == (logits is None):
        with pytest.raises(ValueError):
            perplexity(probs, logits)
        return

    # verify perplexity works after being loaded from pickled state
    pickled_metric = pickle.dumps(perplexity)
    perplexity = pickle.loads(pickled_metric)

    for i in range(rank, NUM_BATCHES, worldsize):
        batch_result = perplexity(None if probs is None else probs[i], None if logits is None else logits[i])

        if perplexity.dist_sync_on_step:
            if rank == 0:
                if probs is not None:
                    ddp_probs = torch.stack([probs[i + r] for r in range(worldsize)])
                else:
                    ddp_logits = torch.stack([logits[i + r] for r in range(worldsize)])
                    ddp_probs = logits_to_probs(ddp_logits, is_binary=False)
                sk_batch_result = reference_perplexity_func(ddp_probs)
                # assert for dist_sync_on_step
                if check_dist_sync_on_step:
                    assert np.allclose(batch_result.numpy(), sk_batch_result, atol=atol)
        else:
            if probs is None:
                p = logits_to_probs(logits[i], is_binary=False)
            else:
                p = probs[i]
            sk_batch_result = reference_perplexity_func(p)
            # assert for batch
            if check_batch:
                assert np.allclose(batch_result.numpy(), sk_batch_result, atol=atol)

    assert (probs is None) != (logits is None)
    # check on all batches on all ranks
    result = perplexity.compute()
    assert isinstance(result, torch.Tensor)

    if probs is None:
        probs = logits_to_probs(logits, is_binary=False)
    sk_result = reference_perplexity_func(probs)

    # assert after aggregation
    assert np.allclose(result.numpy(), sk_result, atol=atol)


class PerplexityTester(MetricTester):
    def run_class_perplexity_test(
        self,
        ddp: bool,
        probs: Optional[torch.Tensor],
        logits: Optional[torch.Tensor],
        dist_sync_on_step: bool,
        metric_args: dict = {},
        check_dist_sync_on_step: bool = True,
        check_batch: bool = True,
    ):
        """ Main method that should be used for testing class. Call this inside testing
            methods.
            Args:
                ddp: bool, if running in ddp mode or not
                probs: torch tensor with probabilities.
                logits: torch tensor with logits. This test checks that probs and logits are mutually exclusive for
                    ``Perplexity`` metric.
                dist_sync_on_step: bool, if true will synchronize metric state across
                    processes at each ``forward()``
                metric_args: dict with additional arguments used for class initialization
                check_dist_sync_on_step: bool, if true will check if the metric is also correctly
                    calculated per batch per device (and not just at the end)
                check_batch: bool, if true will check if the metric is also correctly
                    calculated across devices for each batch (and not just at the end)
        """
        if ddp:
            if sys.platform == "win32":
                pytest.skip("DDP not supported on windows")

            self.pool.starmap(
                partial(
                    _perplexity_class_test,
                    probs=probs,
                    logits=logits,
                    dist_sync_on_step=dist_sync_on_step,
                    metric_args=metric_args,
                    check_dist_sync_on_step=check_dist_sync_on_step,
                    check_batch=check_batch,
                    atol=self.atol,
                ),
                [(rank, self.poolSize) for rank in range(self.poolSize)],
            )
        else:
            _perplexity_class_test(
                0,
                1,
                probs=probs,
                logits=logits,
                dist_sync_on_step=dist_sync_on_step,
                metric_args=metric_args,
                check_dist_sync_on_step=check_dist_sync_on_step,
                check_batch=check_batch,
                atol=self.atol,
            )

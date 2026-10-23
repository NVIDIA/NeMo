# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""CompositeLoss: separately normalized loss terms with scales, and their normalization across ranks."""

import os
import socket
import sys

import pytest
import torch
import torch.multiprocessing as mp

from nemo.collections.common.losses import CompositeLoss, LossTerm


def _terms():
    return {
        "ce": LossTerm(torch.tensor(12.0), torch.tensor(4.0)),
        "ctc": LossTerm(torch.tensor(30.0), torch.tensor(5.0)),
    }


class TestLossTerm:
    @pytest.mark.unit
    def test_from_elements_counts_the_valid_elements(self):
        losses = torch.tensor([[1.0, 2.0, 100.0], [3.0, 100.0, 100.0]])
        mask = torch.tensor([[True, True, False], [True, False, False]])
        term = LossTerm.from_elements(losses, mask)
        assert term.summed.item() == 6.0 and term.count.item() == 3.0
        whole = LossTerm.from_elements(losses)
        assert whole.summed.item() == 306.0 and whole.count.item() == 6.0

    @pytest.mark.unit
    def test_from_sequences_counts_the_lengths(self):
        term = LossTerm.from_sequences(torch.tensor([4.0, 6.0]), torch.tensor([3, 7]))
        assert term.summed.item() == 10.0 and term.count.item() == 10.0


class TestCompositeLoss:
    @pytest.mark.unit
    def test_total_is_the_scaled_sum_of_the_means(self):
        loss = CompositeLoss(scales={"ctc": 0.5})
        total, reported = loss(_terms())
        assert total.item() == pytest.approx(12.0 / 4.0 + 0.5 * 30.0 / 5.0)
        assert reported["ce"].item() == 3.0 and reported["ctc"].item() == 6.0
        assert not reported["ce"].requires_grad

    @pytest.mark.unit
    def test_scale_zero_reports_without_training(self):
        terms = _terms()
        terms["ce"] = LossTerm(terms["ce"].summed.clone().requires_grad_(), terms["ce"].count)
        total, reported = CompositeLoss(scales={"ce": 0.0})(terms)
        assert total.item() == 6.0 and reported["ce"].item() == 3.0
        total.backward()
        assert terms["ce"].summed.grad.item() == 0.0

    @pytest.mark.unit
    def test_unknown_scale_names_and_normalizations_are_rejected(self):
        with pytest.raises(ValueError, match="not provided"):
            CompositeLoss(scales={"rnnt": 1.0})(_terms())
        with pytest.raises(ValueError, match="normalization"):
            CompositeLoss(normalization="mean")
        with pytest.raises(ValueError, match="at least one"):
            CompositeLoss()({})

    @pytest.mark.unit
    def test_empty_term_contributes_nothing(self):
        total, reported = CompositeLoss()({"ce": LossTerm(torch.tensor(0.0), torch.tensor(0.0))})
        assert total.item() == 0.0 and reported["ce"].item() == 0.0

    @pytest.mark.unit
    def test_rank_normalizations_differ_with_unequal_counts(self):
        """The mean of per-rank means is not the global mean; both forms are derived by hand here."""
        rank_terms = [
            {"ce": LossTerm(torch.tensor(10.0), torch.tensor(2.0))},
            {"ce": LossTerm(torch.tensor(10.0), torch.tensor(8.0))},
        ]
        per_rank = [CompositeLoss()(terms)[0].item() for terms in rank_terms]
        assert sum(per_rank) / 2 == pytest.approx((10.0 / 2.0 + 10.0 / 8.0) / 2)
        global_mean = (10.0 + 10.0) / (2.0 + 8.0)
        assert sum(per_rank) / 2 != pytest.approx(global_mean)

    @pytest.mark.unit
    @pytest.mark.skipif(sys.platform in ("win32", "cygwin"), reason="needs the gloo backend")
    def test_global_normalization_across_ranks(self):
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        world_size = 2
        port = _free_port()
        processes = [
            ctx.Process(target=_global_worker, args=(rank, world_size, port, queue)) for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        results = {}
        for _ in processes:
            status, rank, payload = queue.get(timeout=120)
            assert status == "ok", payload
            results[rank] = payload
        for process in processes:
            process.join(timeout=30)
        # rank 0: sum 10 over 2 elements; rank 1: sum 10 over 8 elements; global mean 20 / 10 = 2
        for rank, expected_local in ((0, world_size * 10.0 / 10.0), (1, world_size * 10.0 / 10.0)):
            total, reported = results[rank]
            assert total == pytest.approx(expected_local)
            assert reported == pytest.approx(2.0)
        # the data-parallel average of the per-rank losses is the global mean
        assert (results[0][0] + results[1][0]) / world_size == pytest.approx(2.0)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _global_worker(rank: int, world_size: int, port: int, queue) -> None:
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        torch.distributed.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        count = 2.0 if rank == 0 else 8.0
        terms = {"ce": LossTerm(torch.tensor(10.0), torch.tensor(count))}
        total, reported = CompositeLoss(normalization="global")(terms)
        queue.put(("ok", rank, (total.item(), reported["ce"].item())))
    except Exception as exc:  # pragma: no cover - reported to the test process
        queue.put(("err", rank, repr(exc)))
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

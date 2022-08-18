from copy import deepcopy
from typing import Any
from unittest import mock

import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT

from nemo.collections.common.callbacks import EMA
from nemo.collections.common.callbacks.ema import apex_available
from nemo.collections.cv.models import MNISTLeNet5, MNISTLeNet5Config


class TestEMAConfig:
    @pytest.mark.unit
    def test_ema_value(self):
        with pytest.raises(MisconfigurationException, match="between 0 and 1"):
            EMA(ema=2)

    @pytest.mark.unit
    @pytest.mark.run_only_on('GPU')
    @pytest.mark.skipif(not apex_available, reason="apex is not installed")
    def test_ema_cuda(self):
        lenet5 = MNISTLeNet5(MNISTLeNet5Config())
        lenet5.setup_training_data()
        lenet5.setup_optimization()

        trainer = Trainer(
            max_epochs=1,
            fast_dev_run=True,
            logger=False,
            enable_model_summary=False,
            accelerator='cpu',
            devices=1,
            callbacks=EMA(ema=0.999),
        )
        with pytest.raises(MisconfigurationException, match="Apex EMA Callback only works with CUDA"):
            trainer.fit(model=lenet5)

    @mock.patch('nemo.collections.common.callbacks.ema.apex_available', False)
    def test_ema_apex_unavailable(self):
        with pytest.raises(MisconfigurationException, match="EMA requires Apex to be installed"):
            EMA(ema=0.999)


@pytest.mark.parametrize("precision", [32, 16, "bf16"])
@pytest.mark.parametrize("accumulate_grad_batches", [1, 2])
@pytest.mark.run_only_on('GPU')
@pytest.mark.skipif(not apex_available, reason="apex is not installed")
class TestEMATrain:
    @pytest.mark.unit
    def test_mnist_run(self, test_data_dir, precision, accumulate_grad_batches):
        lenet5 = MNISTLeNet5(MNISTLeNet5Config())
        lenet5.setup_training_data()
        lenet5.setup_optimization()

        trainer = Trainer(
            max_epochs=1,
            precision=precision,
            limit_train_batches=10,
            logger=False,
            accumulate_grad_batches=accumulate_grad_batches,
            enable_model_summary=False,
            accelerator='gpu',
            devices=1,
            callbacks=[EMA(ema=0.999), EMAAssertCallback()],
        )
        trainer.fit(model=lenet5)


class EMAAssertCallback(Callback):
    def __init__(self):
        self.before_calc_ema_weights = None

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        model_weights = list(pl_module.state_dict().values())
        ema_callback = [x for x in trainer.callbacks if isinstance(x, EMA)][0]
        for x, y in zip(model_weights, ema_callback._ema_model_weights):
            assert torch.allclose(x, y)

    def on_train_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        ema_callback = [x for x in trainer.callbacks if isinstance(x, EMA)][0]
        if trainer.global_step == ema_callback._cur_step:
            return
        # saved for manual calculation of ema to compare against implementation
        self.before_calc_ema_weights = deepcopy(ema_callback._ema_model_weights)

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        ema_callback = [x for x in trainer.callbacks if isinstance(x, EMA)][0]
        if trainer.global_step == ema_callback._cur_step:
            return
        ema = ema_callback.ema
        expected_ema_weights = []
        for orig_weight, ema_weight in zip(list(pl_module.state_dict().values()), self.before_calc_ema_weights):
            expected_ema_weight = orig_weight * (1 - ema) + ema_weight * ema
            expected_ema_weights.append(expected_ema_weight)

        for actual_ema_weight, expected_ema_weight in zip(ema_callback._ema_model_weights, expected_ema_weights):
            assert torch.allclose(actual_ema_weight, expected_ema_weight)

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        ema_callback = [x for x in trainer.callbacks if isinstance(x, EMA)][0]
        if ema_callback.ema_initialized:
            for ema_weights, module_weights in zip(ema_callback._ema_model_weights, pl_module.state_dict().values()):
                torch.allclose(ema_weights, module_weights)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        ema_callback = [x for x in trainer.callbacks if isinstance(x, EMA)][0]
        model_weights = list(pl_module.state_dict().values())
        if ema_callback.ema_initialized:
            # todo (sean): shouldn't use the weights buffer to check original weights
            orig_model_weights = ema_callback._weights_buffer
            for orig_weights, module_weights in zip(orig_model_weights, model_weights):
                # original weights are stored on cpu to reduce mem overhead
                torch.allclose(orig_weights, module_weights.cpu())

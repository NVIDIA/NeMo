from copy import deepcopy
from typing import Any

import pytest
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

import os

import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningModule, Trainer

from nemo.collections.common.callbacks import EMA


@pytest.mark.parametrize("devices", [1, 2])
@pytest.mark.parametrize("precision", [32, 16, "bf16"])
@pytest.mark.parametrize("accumulate_grad_batches", [1, 2])
class TestEMACallback:
    @pytest.mark.unit
    def test_apex_ema_callback(self, test_data_dir, devices, precision, accumulate_grad_batches):
        train_data = DataLoader(RandomDataset(32, 64), batch_size=2)
        val_data = DataLoader(RandomDataset(32, 64), batch_size=2)
        test_data = DataLoader(RandomDataset(32, 64), batch_size=2)

        # replace boring model with a ModelPT class from NeMo
        model = BoringModel()
        trainer = Trainer(
            default_root_dir=os.getcwd(),
            max_epochs=5,
            precision=precision,
            accumulate_grad_batches=accumulate_grad_batches,
            enable_model_summary=False,
            accelerator='gpu',
            devices=devices,
            callbacks=[EMA(ema=0.999), TestCallback()],
        )
        trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)
        trainer.test(model, dataloaders=test_data)


class TestCallback(Callback):
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


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)

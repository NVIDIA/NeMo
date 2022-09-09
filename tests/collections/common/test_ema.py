import os.path
from copy import deepcopy
from typing import Any, Dict, Union
from unittest import mock

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT

from nemo.collections.common.callbacks import EMA
from nemo.collections.common.callbacks.ema import apex_available
from nemo.core import ModelPT


class OnesDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_len):
        super().__init__()
        self.__dataset_len = dataset_len

    def __getitem__(self, *args):
        return torch.ones(2)

    def __len__(self):
        return self.__dataset_len


class ExampleModel(ModelPT):
    def __init__(self, *args, **kwargs):
        cfg = OmegaConf.structured({})
        super().__init__(cfg)
        self.l1 = torch.nn.modules.Linear(in_features=2, out_features=1)

    def train_dataloader(self):
        dataset = OnesDataset(16)
        return torch.utils.data.DataLoader(dataset, batch_size=2)

    def val_dataloader(self):
        dataset = OnesDataset(10)
        return torch.utils.data.DataLoader(dataset, batch_size=2)

    def forward(self, batch):
        output = self.l1(batch)
        return torch.nn.functional.l1_loss(output, torch.zeros(output.size()).to(output.device))

    def validation_step(self, batch, batch_idx):
        return self(batch)

    def training_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.1)

    def list_available_models(self):
        pass

    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        pass

    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        pass

    def validation_epoch_end(self, loss):
        self.log("val_loss", torch.stack(loss).mean())


class TestEMAConfig:
    @pytest.mark.unit
    def test_ema_value(self):
        with pytest.raises(MisconfigurationException, match="between 0 and 1"):
            EMA(ema=2)

    @pytest.mark.unit
    @pytest.mark.run_only_on('GPU')
    @pytest.mark.skipif(not apex_available, reason="apex is not installed")
    def test_ema_cuda(self):
        model = ExampleModel()

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
            trainer.fit(model=model)

    @mock.patch('nemo.collections.common.callbacks.ema.apex_available', False)
    def test_ema_apex_unavailable(self):
        with pytest.raises(MisconfigurationException, match="EMA requires Apex to be installed"):
            EMA(ema=0.999)

    @pytest.mark.unit
    @pytest.mark.run_only_on('GPU')
    @pytest.mark.skipif(not apex_available, reason="apex is not installed")
    def test_ema_saved_state(self, tmpdir):
        """Test to ensure that when we re-load the EMA callback, it loads the state correctly"""
        checkpoint_dir = os.path.join(tmpdir, 'checkpoints')
        ema_callback = EMA(ema=0.999)

        class TerminateCallback(Callback):
            def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
                self.saved_ema_weights = ema_callback._ema_model_weights
                self.pl_module_weights = list(pl_module.state_dict().values())
                raise SystemExit

        model = ExampleModel()
        terminate_callback = TerminateCallback()

        trainer = Trainer(
            default_root_dir=checkpoint_dir,
            max_epochs=2,
            limit_val_batches=0,
            limit_train_batches=16,
            logger=False,
            enable_model_summary=False,
            accelerator='gpu',
            devices=1,
            callbacks=[
                ema_callback,
                ModelCheckpoint(dirpath=checkpoint_dir, every_n_train_steps=8, save_top_k=-1, verbose=True),
                terminate_callback,
            ],
        )
        with pytest.raises(SystemExit):
            trainer.fit(model=model)
        resume_path = os.path.join(checkpoint_dir, 'epoch=0-step=16.ckpt')

        ema_callback = EMA(ema=0.999)

        model = ExampleModel()

        class CheckStateCallback(Callback):
            def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
                weights = list(pl_module.state_dict().values())
                for x, y in zip(weights, terminate_callback.pl_module_weights):
                    assert torch.allclose(x, y)
                for x, y in zip(ema_callback._ema_model_weights, terminate_callback.saved_ema_weights):
                    assert torch.allclose(x, y)
                assert ema_callback._cur_step == 16

        trainer = Trainer(
            default_root_dir=checkpoint_dir,
            max_epochs=2,
            limit_val_batches=0,
            limit_train_batches=16,
            logger=False,
            enable_model_summary=False,
            accelerator='gpu',
            devices=1,
            callbacks=[ema_callback, CheckStateCallback()],
        )
        trainer.fit(model, ckpt_path=resume_path)


@pytest.mark.parametrize("precision", [16, "bf16", 32])
@pytest.mark.parametrize("accumulate_grad_batches", [1, 2])
@pytest.mark.run_only_on('GPU')
@pytest.mark.skipif(not apex_available, reason="apex is not installed")
class TestEMATrain:
    @pytest.mark.unit
    def test_example_run(self, test_data_dir, precision, accumulate_grad_batches):
        pl.seed_everything(1234)

        model = ExampleModel()

        trainer = Trainer(
            max_epochs=1,
            precision=precision,
            limit_train_batches=10,
            limit_val_batches=10,
            logger=False,
            accumulate_grad_batches=accumulate_grad_batches,
            num_sanity_val_steps=0,
            enable_model_summary=False,
            accelerator='gpu',
            devices=1,
            callbacks=[EMA(ema=0.999), EMAAssertCallback()],
        )
        trainer.fit(model=model, val_dataloaders=model.train_dataloader())


class EMAAssertCallback(Callback):
    def __init__(self):
        self._before_calc_ema_weights = None

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        model_weights = list(pl_module.state_dict().values())
        ema_callback = [x for x in trainer.callbacks if isinstance(x, EMA)][0]
        for x, y in zip(model_weights, ema_callback._ema_model_weights):
            assert torch.allclose(x, y)

    def on_train_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        ema_callback = [x for x in trainer.callbacks if isinstance(x, EMA)][0]
        # saved for manual calculation of ema to compare against implementation
        self._before_calc_ema_weights = deepcopy(ema_callback._ema_model_weights)

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        if (batch_idx + 1) % trainer.accumulate_grad_batches != 0:
            # skip assertion as ema weights are not updated.
            return
        ema_callback = [x for x in trainer.callbacks if isinstance(x, EMA)][0]
        ema = ema_callback.ema
        expected_ema_weights = []
        for orig_weight, ema_weight in zip(list(pl_module.state_dict().values()), self._before_calc_ema_weights):
            expected_ema_weight = orig_weight * (1 - ema) + ema_weight * ema
            expected_ema_weights.append(expected_ema_weight)

        for actual_ema_weight, expected_ema_weight in zip(ema_callback._ema_model_weights, expected_ema_weights):
            assert torch.allclose(actual_ema_weight, expected_ema_weight)

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        ema_callback = [x for x in trainer.callbacks if isinstance(x, EMA)][0]
        # todo (sean): shouldn't use the weights buffer to check original weights
        self._original_weights = list(x.detach().clone() for x in ema_callback._weights_buffer)
        if ema_callback.ema_initialized:
            for ema_weights, module_weights in zip(ema_callback._ema_model_weights, pl_module.state_dict().values()):
                torch.allclose(ema_weights, module_weights)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        ema_callback = [x for x in trainer.callbacks if isinstance(x, EMA)][0]
        model_weights = list(pl_module.state_dict().values())
        if ema_callback.ema_initialized:
            for orig_weights, module_weights in zip(self._original_weights, model_weights):
                # original weights are stored on cpu to reduce mem overhead
                torch.allclose(orig_weights, module_weights.cpu())

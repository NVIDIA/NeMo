import os
import pytest
import torch
import pytorch_lightning as pl
import nemo.lightning as nl

from pathlib import Path
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from nemo.lightning.io.mixin import IOMixin

class RandomDataset(pl.LightningDataModule):
    def __init__(self, size, length):
        super().__init__()
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dataset = RandomDataset(32, 16)
        return torch.utils.data.DataLoader(dataset, batch_size=2)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        dataset = RandomDataset(32, 16)
        return torch.utils.data.DataLoader(dataset, batch_size=2)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        dataset = RandomDataset(32, 16)
        dl = torch.utils.data.DataLoader(dataset, batch_size=2)
        #self._test_names = ['test_{}_'.format(idx) for idx in range(len(dl))]
        return dl

class ExampleModel(pl.LightningModule, IOMixin):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.l1 = torch.nn.modules.Linear(in_features=32, out_features=32)
        self.bn = torch.nn.BatchNorm1d(32)
        self.validation_step_outputs = []

    def forward(self, batch):
        return self.l1(self.bn(batch)).sum()

    def train_dataloader(self):
        dataset = RandomDataset(32, 16)
        return torch.utils.data.DataLoader(dataset, batch_size=2)

    def val_dataloader(self):
        dataset = RandomDataset(32, 16)
        return torch.utils.data.DataLoader(dataset, batch_size=2)

    def test_dataloader(self):
        dataset = RandomDataset(32, 16)
        dl = torch.utils.data.DataLoader(dataset, batch_size=2)
        self._test_names = ['test_{}_'.format(idx) for idx in range(len(dl))]
        return dl

    def training_step(self, batch, batch_idx):
        return self(batch)

    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        self.validation_step_outputs.append(loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self(batch)
        self.test_step_outputs.append(loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-3)

    def on_validation_epoch_end(self):
        self.log("val_loss", torch.stack(self.validation_step_outputs).mean())
        self.validation_step_outputs.clear()  # free memory

class TestModelCheckpoint:

    @pytest.mark.unit
    def test_link_ckpt(self, tmpdir):
        """Test to ensure that we always keep top_k checkpoints, even after resuming."""
        tmp_path = tmpdir / "link_ckpt_test"
        model = ExampleModel()

        data = RandomDataset(32, 64)
        save_top_k = 3

        nemo_logger = nl.NeMoLogger(
            log_dir=tmp_path,
            use_datetime_version=False,
        )

        strategy = nl.MegatronStrategy(
            ckpt_async_save=True,
            replace_progress_bar=False
        )

        trainer = nl.Trainer(
            max_epochs=5,
            devices=1,
            val_check_interval=5,
            callbacks=nl.ModelCheckpoint(
                monitor="val_loss",
                save_top_k=3,
                save_on_train_epoch_end=True,
                save_context_on_train_end=False,
                filename=f'{{step}}-{{epoch}}-{{val_loss}}-{{consumed_samples}}',
                save_last="link",
            )
        )
        nemo_logger.setup(trainer)
        trainer.fit(model, data)

        checkpoint_dir = Path(tmp_path / "default" / "checkpoints")
        dist_checkpoints = [d for d in list(checkpoint_dir.glob("*")) if d.is_dir()]
        last_checkpoints = [d for d in dist_checkpoints if d.match("*last")]        
        final_ckpt = sorted(last_checkpoints, key=lambda pth: pth.lstat().st_mtime, reverse=True)[0]
        assert os.path.islink(final_ckpt)

        link = final_ckpt.resolve()
        assert str(final_ckpt).replace("-last", "") == str(link)


import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pytorch_lightning.callbacks import ModelCheckpoint as PTLModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from nemo import lightning as nl
from nemo.constants import NEMO_ENV_VARNAME_VERSION
from nemo.utils.exp_manager import NotFoundError


class TestNeMoLogger:
    @pytest.fixture
    def trainer(self):
        return nl.Trainer(accelerator="cpu")

    def test_loggers(self):
        trainer = nl.Trainer(accelerator="cpu")
        logger = nl.NeMoLogger(
            update_logger_directory=True,
            wandb=WandbLogger(save_dir="test", offline=True),
        )

        logger.setup(trainer)
        assert logger.tensorboard is None
        assert len(logger.extra_loggers) == 0
        assert len(trainer.loggers) == 2
        assert isinstance(trainer.loggers[1], WandbLogger)
        assert str(trainer.loggers[1].save_dir).endswith("nemo_experiments")
        assert trainer.loggers[1]._name == "default"

    def test_explicit_log_dir(self, trainer):
        explicit_dir = "explicit_test_dir"
        logger = nl.NeMoLogger(name="test", explicit_log_dir=explicit_dir)

        with patch("nemo.utils.exp_manager.check_explicit_log_dir") as mock_check:
            logger.setup(trainer)
            mock_check.assert_called_once_with(trainer, explicit_dir, None, "test", None)

    def test_default_log_dir(self, trainer):

        if os.environ.get(NEMO_ENV_VARNAME_VERSION, None) is not None:
            del os.environ[NEMO_ENV_VARNAME_VERSION]
        logger = nl.NeMoLogger(use_datetime_version=False)
        app_state = logger.setup(trainer)
        assert app_state.log_dir == Path(Path.cwd() / "nemo_experiments" / "default")

    def test_custom_version(self, trainer):
        custom_version = "v1.0"
        logger = nl.NeMoLogger(name="test", version=custom_version, use_datetime_version=False)

        app_state = logger.setup(trainer)
        assert app_state.version == custom_version

    def test_file_logging_setup(self, trainer):
        logger = nl.NeMoLogger(name="test")

        with patch("nemo.lightning.nemo_logger.logging.add_file_handler") as mock_add_handler:
            logger.setup(trainer)
            mock_add_handler.assert_called_once()

    def test_model_checkpoint_setup(self, trainer):
        ckpt = PTLModelCheckpoint(dirpath="test_ckpt", filename="test-{epoch:02d}-{val_loss:.2f}")
        logger = nl.NeMoLogger(name="test", ckpt=ckpt)

        logger.setup(trainer)
        assert any(isinstance(cb, PTLModelCheckpoint) for cb in trainer.callbacks)
        ptl_ckpt = next(cb for cb in trainer.callbacks if isinstance(cb, PTLModelCheckpoint))
        assert str(ptl_ckpt.dirpath).endswith("test_ckpt")
        assert ptl_ckpt.filename == "test-{epoch:02d}-{val_loss:.2f}"

    def test_resume(self, trainer, tmp_path):
        """Tests the resume capabilities of exp_manager"""

        if os.environ.get(NEMO_ENV_VARNAME_VERSION, None) is not None:
            del os.environ[NEMO_ENV_VARNAME_VERSION]

        # Error because explicit_log_dir does not exist
        ### TODO: make "model" arg optional
        with pytest.raises(NotFoundError):
            nl.AutoResume(
                dirpath=str(tmp_path / "test_resume"),
                resume_if_exists=True,
            ).setup(None, trainer)

        # Error because checkpoints folder does not exist
        with pytest.raises(NotFoundError):
            nl.AutoResume(
                dirpath=str(tmp_path / "test_resume" / "does_not_exist"),
                path="does_not_exist",
                resume_if_exists=True,
            ).setup(None, trainer)

        # No error because we tell autoresume to ignore notfounderror
        nl.AutoResume(
            dirpath=str(tmp_path / "test_resume" / "does_not_exist"),
            resume_if_exists=True,
            resume_ignore_no_checkpoint=True,
        ).setup(None, trainer)

        path = Path(tmp_path / "test_resume" / "default" / "version_0" / "checkpoints").mkdir(parents=True)
        # Error because checkpoints do not exist in folder
        with pytest.raises(NotFoundError):
            nl.AutoResume(
                dirpath=path,
                resume_if_exists=True,
            ).setup(None, trainer)

        Path(tmp_path / "test_resume" / "default" / "version_0" / "checkpoints" / "mymodel--end").mkdir()
        # Error because *end.ckpt is in folder indicating that training has already finished
        with pytest.raises(ValueError):
            nl.AutoResume(
                dirpath=Path(tmp_path / "test_resume" / "default" / "version_0" / "checkpoints"),
                resume_if_exists=True,
            ).setup(None, trainer)

        Path(tmp_path / "test_resume" / "default" / "version_0" / "checkpoints" / "mymodel--end").rmdir()
        Path(tmp_path / "test_resume" / "default" / "version_0" / "checkpoints" / "mymodel--last").mkdir()
        Path(tmp_path / "test_resume" / "default" / "version_0" / "checkpoints" / "mymodel2--last").mkdir()
        # Error because multiple *last.ckpt is in folder. If more than one, don't know which to restore
        with pytest.raises(ValueError):
            nl.AutoResume(
                dirpath=Path(tmp_path / "test_resume" / "default" / "version_0" / "checkpoints"),
                resume_if_exists=True,
            ).setup(None, trainer)

        # Finally succeed
        logger = nl.NeMoLogger(
            name="default",
            dir=str(tmp_path) + "/test_resume",
            version="version_0",
            use_datetime_version=False,
        )
        logger.setup(trainer)
        Path(tmp_path / "test_resume" / "default" / "version_0" / "checkpoints" / "mymodel2--last").rmdir()
        nl.AutoResume(
            resume_if_exists=True,
        ).setup(None, trainer)
        checkpoint = Path(tmp_path / "test_resume" / "default" / "version_0" / "checkpoints" / "mymodel--last")
        assert Path(trainer.ckpt_path).resolve() == checkpoint.resolve()

        trainer = nl.Trainer(accelerator="cpu", logger=False)
        # Check that model loads from `dirpath` and not <log_dir>/checkpoints
        dirpath_log_dir = Path(tmp_path / "test_resume" / "dirpath_test" / "logs")
        dirpath_log_dir.mkdir(parents=True)
        dirpath_checkpoint_dir = Path(tmp_path / "test_resume" / "dirpath_test" / "ckpts")
        dirpath_checkpoint = Path(dirpath_checkpoint_dir / "mymodel--last")
        dirpath_checkpoint.mkdir(parents=True)
        logger = nl.NeMoLogger(
            name="default",
            explicit_log_dir=dirpath_log_dir,
        )
        logger.setup(trainer)
        nl.AutoResume(
            resume_if_exists=True,
            dirpath=str(dirpath_checkpoint_dir),
        ).setup(None, trainer)
        assert Path(trainer.ckpt_path).resolve() == dirpath_checkpoint.resolve()

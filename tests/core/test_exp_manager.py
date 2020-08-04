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

import re
import pytest
import shutil
from pathlib import Path

import pytorch_lightning as pl
from omegaconf.errors import OmegaConfBaseException

from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager, LoggerMisconfigurationError, CheckpointMisconfigurationError


@pytest.fixture
def cleanup_local_folder():
    # Asserts in fixture are not recommended, but I'd rather stop users from deleting expensive training runs
    assert not Path("./lightning_logs").exists()
    assert not Path("./NeMo_experiments").exists()

    yield

    if Path("./lightning_logs").exists():
        shutil.rmtree('./lightning_logs')
    if Path("./NeMo_experiments").exists():
        shutil.rmtree('./NeMo_experiments')


class TestExpManager:
    @pytest.mark.unit
    def test_omegaconf(self):
        """Ensure omegaconf raises an error when an unexcepted argument is passed"""
        with pytest.raises(OmegaConfBaseException):
            exp_manager(None, {"unused": 1})

    @pytest.mark.unit
    def test_trainer_default_logger(self, cleanup_local_folder, tmp_path):
        """ Test that a trainer with logger errors out with a number of arguments. Test that it works with
        create_tensorboard_logger set to False
        """
        test_trainer = pl.Trainer()  # Should create logger and modelcheckpoint

        with pytest.raises(LoggerMisconfigurationError):  # Fails because exp_manager defaults to trainer
            exp_manager(test_trainer, {"root_dir": str(tmp_path)})
        with pytest.raises(LoggerMisconfigurationError):  # Fails because exp_manager defaults to trainer
            exp_manager(test_trainer, {"explicit_log_dir": str(tmp_path)})
        with pytest.raises(LoggerMisconfigurationError):  # Fails because exp_manager defaults to trainer
            exp_manager(test_trainer, {"resume": True})

        # Check that exp_manager uses trainer.logger
        log_dir = exp_manager(test_trainer, {"create_tensorboard_logger": False, "create_checkpoint_callback": False})
        assert log_dir.resolve() == Path("./lightning_logs/0").resolve()
        assert Path("./lightning_logs").exists()
        assert Path("./lightning_logs/0").exists()

    @pytest.mark.unit
    def test_checkpoint_configurations(self, cleanup_local_folder):
        """ Test that trainer creating modelcheckpoint and asking exp_manager to do it too results in errors, but
        is error free if only one is asked to do so.
        """
        disable_tb_logger = {"create_tensorboard_logger": False}
        test_trainer = pl.Trainer()  # Should create logger and modelcheckpoint
        with pytest.raises(CheckpointMisconfigurationError):  # Fails because both try to create modelcheckpoint
            exp_manager(test_trainer, disable_tb_logger)

        # Should success without error
        exp_manager(test_trainer, {"create_checkpoint_callback": False, "create_tensorboard_logger": False})

        test_trainer_2 = pl.Trainer(checkpoint_callback=False)
        exp_manager(test_trainer_2, disable_tb_logger)  # Should success without error

    @pytest.mark.unit
    def test_default_log_dir(self, cleanup_local_folder):
        """Check the default of ./NeMo_experiments/default/datetime works as intended"""
        test_trainer = pl.Trainer(checkpoint_callback=False, logger=False)

        log_dir = exp_manager(test_trainer, {"create_tensorboard_logger": False, "create_checkpoint_callback": False})
        assert (log_dir / "..").resolve() == Path("./NeMo_experiments/default/").resolve()
        assert Path("./NeMo_experiments").exists()
        assert Path("./NeMo_experiments/default/").exists()
        sub_dirs = [x for x in Path("./NeMo_experiments/default/").iterdir() if x.is_dir()]
        assert len(sub_dirs) == 1
        assert re.match(r"[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}", sub_dirs[0].name)

    @pytest.mark.unit
    def test_log_dir_overrides(self, tmp_path):
        """Check a variety of trainer options with exp_manager"""
        test_trainer = pl.Trainer(checkpoint_callback=False, logger=False)

        log_dir = exp_manager(test_trainer, {"explicit_log_dir": str(tmp_path / "test_log_dir_overrides")})
        assert log_dir.resolve() == (tmp_path / "test_log_dir_overrides").resolve()
        assert Path(tmp_path).exists()
        assert Path(tmp_path / "test_log_dir_overrides").exists()

    # Test log dir creation
    # Test logging
    # Test find_last_checkpoint

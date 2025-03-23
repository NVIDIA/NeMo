# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from pathlib import Path

import pytest
import torch
from unittest.mock import patch

from multistorageclient.types import MSC_PROTOCOL
from nemo.utils.callbacks.msc_checkpoint_io import MSCCheckpointIO


@pytest.fixture
def msc_dirpath():
    return f"{MSC_PROTOCOL}test/path"


@pytest.fixture
def checkpoint_io(msc_dirpath):
    return MSCCheckpointIO(dirpath=msc_dirpath)


@pytest.fixture
def mock_msc():
    with patch('nemo.utils.callbacks.msc_checkpoint_io.msc') as mock:
        yield mock


@pytest.fixture
def sample_checkpoint():
    return {
        'state_dict': {
            'layer.weight': torch.tensor([1.0, 2.0, 3.0]),
            'layer.bias': torch.tensor([0.1, 0.2, 0.3])
        }
    }


@pytest.fixture
def mock_nemo_logger(mocker):
    """Mock NeMo's logger."""
    return mocker.patch('nemo.utils.logging.warning')


class TestMSCCheckpointIO:
    def test_init_with_non_msc_dir_path(self):
        """Test initialization with non-MSC path should raise AssertionError."""
        with pytest.raises(AssertionError):
            MSCCheckpointIO(dirpath="invalid/path")

    def test_init_with_msc_dir_path(self, msc_dirpath):
        """Test initialization with valid MSC path."""
        checkpoint_io = MSCCheckpointIO(dirpath=msc_dirpath)
        assert isinstance(checkpoint_io, MSCCheckpointIO)

    def test_save_checkpoint(self, checkpoint_io, mock_msc, sample_checkpoint):
        """Test saving checkpoint to MSC storage."""
        save_path = f"{MSC_PROTOCOL}test/path/checkpoint.ckpt"
        
        # Call save_checkpoint
        checkpoint_io.save_checkpoint(sample_checkpoint, save_path)
        
        # Verify msc.torch.save was called
        mock_msc.torch.save.assert_called_once_with(sample_checkpoint, save_path)
    
    def test_save_checkpoint_with_storage_options(self, checkpoint_io, mock_msc, sample_checkpoint, mock_nemo_logger):
        """Test save_checkpoint with storage_options logs warning."""
        save_path = f"{MSC_PROTOCOL}test/path/checkpoint.ckpt"
        storage_options = {'option': 'value'}

        # Save checkpoint with storage_options
        checkpoint_io.save_checkpoint(sample_checkpoint, save_path, storage_options=storage_options)

        # Verify the warning was logged
        mock_nemo_logger.assert_called_once_with(
            f"MSCCheckpointIO does not support storage_options, but storage_options={storage_options} was provided. Ignoring given storage_options"
        )

        # Verify the checkpoint was still saved despite the warning
        mock_msc.torch.save.assert_called_once_with(sample_checkpoint, save_path)

    def test_load_checkpoint(self, checkpoint_io, mock_msc, sample_checkpoint):
        """Test loading checkpoint from MSC storage."""
        load_path = f"{MSC_PROTOCOL}test/path/checkpoint.ckpt"
        
        # Setup mock to return the sample checkpoint
        mock_msc.torch.load.return_value = sample_checkpoint
        
        # Load checkpoint
        loaded_checkpoint = checkpoint_io.load_checkpoint(load_path)
        
        # Verify msc.torch.load was called
        mock_msc.torch.load.assert_called_once()
        call_args = mock_msc.torch.load.call_args
        assert call_args[0][0] == load_path  # First arg should be the path
        
        # Verify the loaded checkpoint matches the original
        assert loaded_checkpoint == sample_checkpoint

    def test_remove_checkpoint(self, checkpoint_io, mock_msc):
        """Test removing checkpoint from MSC storage."""
        # This is a no-op in the current implementation
        remove_path = f"{MSC_PROTOCOL}test/path/checkpoint.ckpt"
        
        # Call remove_checkpoint
        checkpoint_io.remove_checkpoint(remove_path)
        
        # Currently this is a no-op, so we're just verifying it doesn't raise an exception
        # Once implemented, we would add assertions to check the appropriate MSC method is called

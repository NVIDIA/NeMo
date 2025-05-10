import unittest
from unittest.mock import MagicMock, patch

import pytorch_lightning as pl

from nemo.utils.callbacks.dist_ckpt_io import AsyncFinalizableCheckpointIO, AsyncFinalizerCallback


class TestAsyncFinalizerCallback(unittest.TestCase):
    def setUp(self):
        self.mock_trainer = MagicMock(spec=pl.Trainer)
        self.mock_checkpoint_io = MagicMock(spec=AsyncFinalizableCheckpointIO)
        self.mock_async_calls_queue = MagicMock()

        self.mock_checkpoint_io.async_calls_queue = self.mock_async_calls_queue
        self.mock_trainer.strategy.checkpoint_io = self.mock_checkpoint_io

    def test_init(self):
        callback = AsyncFinalizerCallback()
        self.assertIsNone(callback.max_num_unfinalized_calls)

        callback = AsyncFinalizerCallback(max_num_unfinalized_calls=5)
        self.assertEqual(callback.max_num_unfinalized_calls, 5)

    def test_on_train_batch_end_no_blocking(self):
        callback = AsyncFinalizerCallback()
        callback.on_train_batch_end(self.mock_trainer)

        self.mock_checkpoint_io.maybe_finalize_save_checkpoint.assert_called_once_with(blocking=False)

    def test_on_train_batch_end_with_blocking(self):
        callback = AsyncFinalizerCallback(max_num_unfinalized_calls=3)
        self.mock_async_calls_queue.get_num_unfinalized_calls.return_value = 5

        with patch("nemo.utils.logging.info") as mock_log:
            callback.on_train_batch_end(self.mock_trainer)

            mock_log.assert_called_once()

        self.mock_checkpoint_io.maybe_finalize_save_checkpoint.assert_called_once_with(blocking=True)

    def test_on_train_epoch_end(self):
        callback = AsyncFinalizerCallback()
        callback.on_train_epoch_end(self.mock_trainer)

        self.mock_checkpoint_io.maybe_finalize_save_checkpoint.assert_called_once_with(blocking=False)

    def test_on_train_end_with_pending_checkpoints(self):
        callback = AsyncFinalizerCallback()
        self.mock_async_calls_queue.get_num_unfinalized_calls.return_value = 2

        with patch("nemo.utils.logging.info") as mock_log:
            callback.on_train_end(self.mock_trainer)

            mock_log.assert_called_once()

        self.mock_checkpoint_io.maybe_finalize_save_checkpoint.assert_called_once_with(blocking=True)

    def test_on_train_end_no_pending_checkpoints(self):
        callback = AsyncFinalizerCallback()
        self.mock_async_calls_queue.get_num_unfinalized_calls.return_value = 0

        with patch("nemo.utils.logging.info") as mock_log:
            callback.on_train_end(self.mock_trainer)

            mock_log.assert_not_called()

        self.mock_checkpoint_io.maybe_finalize_save_checkpoint.assert_called_once_with(blocking=True)

    def test_invalid_checkpoint_io(self):
        callback = AsyncFinalizerCallback()
        self.mock_trainer.strategy.checkpoint_io = MagicMock()  # Not an AsyncFinalizableCheckpointIO

        with self.assertRaises(ValueError):
            callback.on_train_batch_end(self.mock_trainer)

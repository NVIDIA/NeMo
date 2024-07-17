import os
from pathlib import Path
import pytest
import types
import pytorch_lightning as pl
from pytorch_lightning.demos.boring_classes import RandomDataset
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Optional, Tuple

from nemo.collections import llm
import nemo.lightning as nl
from nemo.lightning.io.pl import MegatronCheckpointIO
from nemo.lightning.pytorch.callbacks import ModelCheckpoint
from nemo.utils.callbacks.dist_ckpt_io import (
    AsyncFinalizableCheckpointIO,
    AsyncFinalizerCallback,
    DistributedCheckpointIO,
)
from nemo.utils.callbacks.dist_ckpt_io import AsyncFinalizableCheckpointIO

try:
    from megatron.core.dist_checkpointing import ShardedTensor

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False

def _get_nlp_strategy_without_optimizer_state():
    strategy = nl.MegatronStrategy(
        enable_nemo_ckpt_io=False,
    )
    # this ensures optimizer sharded state creation is skipped
    strategy.optimizer_sharded_state_dict = types.MethodType(
        lambda self, unsharded_optim_state: unsharded_optim_state, strategy
    )
    return strategy

def _get_last_checkpoint_dir(model: pl.LightningModule, suffix: str = '') -> Path:
    return f'epoch={model.trainer.current_epoch - 1}-step={model.trainer.max_steps - 1}{suffix}'

def get_model_and_data():
    seq_length = 128
    data = llm.MockDataModule(seq_length=seq_length, global_batch_size=32)

    config = llm.GPTConfig(
        num_layers=2,
        hidden_size=64,
        ffn_hidden_size=256,
        num_attention_heads=4,
        seq_length=seq_length,
    )
    return llm.GPTModel(config, tokenizer=data.tokenizer), data

class TestDistCkptIO:
    @pytest.fixture
    def model_and_data(self):
        seq_length = 128
        data = llm.MockDataModule(seq_length=seq_length, global_batch_size=32)

        config = llm.GPTConfig(
            num_layers=2,
            hidden_size=64,
            ffn_hidden_size=256,
            num_attention_heads=4,
            seq_length=seq_length,
        )
        return llm.GPTModel(config, tokenizer=data.tokenizer), data

    @pytest.mark.run_only_on('GPU')
    def test_dist_ckpt_io_called_for_mcore_models(self, model_and_data, tmp_path):

        model, data = model_and_data

        strategy = _get_nlp_strategy_without_optimizer_state()
        
        trainer = nl.Trainer(
            devices=2,
            accelerator="gpu",
            strategy=strategy,
            enable_checkpointing=True,
            max_steps=2,
            default_root_dir=str(tmp_path),
        )

        trainer.fit(model, data)

        assert isinstance(trainer.strategy.checkpoint_io, MegatronCheckpointIO)
        # Ckpt path doesn't contain the .ckpt suffix
        ## TODO: make more generic
        ## why does this path include "lightning logs" while original test does not?
        ckpts = os.listdir(Path(tmp_path / "lightning_logs/version_0/checkpoints" ))
        assert len(ckpts) == 1 ## can do other things with this. Assert the right number of checkpoints are present
        ckpt = ckpts[0]
        assert str(ckpt)==_get_last_checkpoint_dir(model)

    @pytest.mark.run_only_on('GPU')
    def test_async_save_produces_same_checkpoints_as_sync(self, tmp_path):

        model, data = get_model_and_data()

        sync_ckpt_dir = tmp_path / 'sync_checkpoints'
        async_ckpt_dir = tmp_path / 'async_checkpoints'

        sync_checkpoint_io = MegatronCheckpointIO('torch_dist')
        async_checkpoint_io = AsyncFinalizableCheckpointIO(MegatronCheckpointIO('torch_dist', async_save=True))

        # dummy_trainer just to initialize NCCL
        dummy_trainer = pl.Trainer(
            logger=False,
            max_steps=2,
            strategy=_get_nlp_strategy_without_optimizer_state(),
        )
        dummy_trainer.fit(model, data)
        strategy = _get_nlp_strategy_without_optimizer_state()
        tmp_path = strategy.broadcast(tmp_path)

        ## reset the model and data and train with sync checkpointing
        model, data = get_model_and_data()
        sync_test_trainer = pl.Trainer(
            enable_checkpointing=True,
            logger=False,
            max_steps=2,
            strategy=_get_nlp_strategy_without_optimizer_state(),
            plugins=[sync_checkpoint_io],
            default_root_dir=str(sync_ckpt_dir),
        )
        sync_test_trainer.fit(model, data)

        ## reset the model and data and train with sync checkpointing
        model, data = get_model_and_data()
        async_test_trainer = pl.Trainer(
            enable_checkpointing=True,
            logger=False,
            max_steps=2,
            strategy=_get_nlp_strategy_without_optimizer_state(),
            plugins=[async_checkpoint_io], ## error is not specific to async checkpointing
            callbacks=AsyncFinalizerCallback(),
            default_root_dir=str(async_ckpt_dir),
        )
        async_test_trainer.fit(model, data)

        checkpoint = {'sharded_state_dict': model.sharded_state_dict()}

        sync_state_dict = sync_checkpoint_io.load_checkpoint(
            Path(f"{sync_ckpt_dir}/checkpoints/{_get_last_checkpoint_dir(model)}"), sharded_state_dict=checkpoint
        )

        async_state_dict = async_checkpoint_io.load_checkpoint(
            Path(f"{async_ckpt_dir}/checkpoints/{_get_last_checkpoint_dir(model)}"), sharded_state_dict=checkpoint
        )

        ## one of the keys is a _io.BytesIO object
        for k in sync_state_dict['sharded_state_dict'].keys():
            if isinstance(sync_state_dict['sharded_state_dict'][k], torch.Tensor):
                assert(torch.all(sync_state_dict['sharded_state_dict'][k] == async_state_dict['sharded_state_dict'][k]))

    def test_sharded_strategies(self):

        model_checkpoint = nl.ModelCheckpoint(async_save=True)

        strategy = nl.MegatronStrategy(
            enable_nemo_ckpt_io=False,
            save_ckpt_format='torch_dist',
            ckpt_parallel_save=True,
            ckpt_load_directly_on_device=False,
        )
        trainer = nl.Trainer(
            accelerator = 'cpu',
            callbacks = [model_checkpoint],
            strategy = strategy,
        )
        strategy.trainer = trainer

        assert (isinstance(strategy.checkpoint_io, AsyncFinalizableCheckpointIO))
        assert (isinstance(strategy.checkpoint_io._checkpoint_io, MegatronCheckpointIO))

        base_checkpoint_io = strategy.checkpoint_io._checkpoint_io

        assert(base_checkpoint_io.save_ckpt_format=='torch_dist')
        assert(base_checkpoint_io.parallel_save)
        assert(base_checkpoint_io.load_directly_on_device==False)




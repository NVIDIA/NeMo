import pytest
from unittest.mock import Mock
import torch
from torch.optim import SGD
import lightning.pytorch as pl
from nemo.lightning.pytorch.optim.base import LRSchedulerModule
from nemo.lightning.pytorch.optim.pytorch import PytorchOptimizerModule

class DummyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 2)
    
    def forward(self, x):
        return self.layer(x)

@pytest.fixture
def dummy_model():
    return DummyModel()

@pytest.fixture
def optimizer_fn():
    return Mock(side_effect=lambda params: SGD(params, lr=0.01, weight_decay=0.1))

@pytest.fixture
def lr_scheduler():
    return Mock(spec=LRSchedulerModule)

@pytest.fixture
def optimizer_module(optimizer_fn, lr_scheduler):
    return PytorchOptimizerModule(optimizer_fn, lr_scheduler)

def test_optimizer_module_initialization(optimizer_module, optimizer_fn, lr_scheduler):
    assert optimizer_module.optimizer_fn == optimizer_fn
    assert optimizer_module.lr_scheduler == lr_scheduler
    assert callable(optimizer_module.no_weight_decay_cond)
    assert optimizer_module.lr_mult == 1.0

def test_optimizer_creation(dummy_model, optimizer_module):
    optimizer = optimizer_module.optimizers(dummy_model)
    assert isinstance(optimizer, list)
    assert len(optimizer) > 0
    assert isinstance(optimizer[0], torch.optim.Optimizer)

def test_connect_method(dummy_model, optimizer_module):
    dummy_model.connect_optim_builder = Mock()
    optimizer_module.connect(dummy_model)
    dummy_model.connect_optim_builder.assert_called_once_with(optimizer_module)

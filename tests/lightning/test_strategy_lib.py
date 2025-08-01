# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from unittest.mock import ANY, MagicMock, patch

import torch
from torch import nn

from nemo.lightning import MegatronStrategy, _strategy_lib  # , DataConfig


class Identity(nn.Identity):
    def __init__(self):
        super().__init__()


class WithCopy(nn.Identity):
    def copy(self):
        return WithCopy()


def test_set_model_parallel_attributes() -> None:
    strategy = MegatronStrategy(
        pipeline_model_parallel_size=2,
        expert_model_parallel_size=2,
        sequence_parallel=False,
        pipeline_dtype=torch.float32,
    )
    from megatron.core.transformer.transformer_config import TransformerConfig

    class DummyModel:
        def __init__(self):
            self.config = TransformerConfig(
                hidden_size=128, num_attention_heads=2, num_layers=2, num_moe_experts=2, add_bias_linear=False
            )

        def configure_model(self):
            pass

    model = DummyModel()
    assert model.config.pipeline_model_parallel_size != 2
    assert model.config.expert_model_parallel_size != 2
    assert model.config.pipeline_dtype != torch.float32
    _strategy_lib.set_model_parallel_attributes(model, strategy.parallelism)
    assert model.config.pipeline_model_parallel_size == 2
    assert model.config.expert_model_parallel_size == 2
    assert model.config.sequence_parallel == False
    assert model.config.pipeline_dtype == torch.float32


def test_init_parallel_ranks() -> None:
    from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
    from megatron.core.parallel_state import destroy_model_parallel

    from nemo.utils import AppState

    app_state = AppState()

    app_state.tensor_model_parallel_size = 2
    app_state.pipeline_model_parallel_size = 3
    app_state.context_parallel_size = 2
    app_state.expert_model_parallel_size = 2
    app_state.global_rank = 1
    app_state.local_rank = 0

    mock_parallel_config = MagicMock()
    mock_parallel_config.tensor_model_parallel_size = 2
    mock_parallel_config.pipeline_model_parallel_size = 3
    mock_parallel_config.virtual_pipeline_model_parallel_size = 4
    mock_parallel_config.context_parallel_size = 2
    mock_parallel_config.expert_model_parallel_size = 2
    mock_parallel_config.expert_tensor_parallel_size = None
    mock_parallel_config.tp_comm_overlap = False
    mock_parallel_config.use_te_rng_tracker = False

    _strategy_lib.init_parallel_ranks(
        world_size=24,
        global_rank=1,
        local_rank=0,
        parallel_config=mock_parallel_config,
        seed=1234,
        fp8=False,
    )
    expected_app_state = {
        "world_size": 24,
        "global_rank": 1,
        "local_rank": 0,
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 3,
        "virtual_pipeline_model_parallel_size": 4,
        "context_parallel_size": 2,
        "expert_model_parallel_size": 2,
        "use_fp8": False,
        "init_mpi_proc_group": False,
    }
    for k, v in expected_app_state.items():
        assert hasattr(app_state, k), f"Expected to find {k} in AppState"
        app_attr = getattr(app_state, k)
        assert app_attr == v, f"{k} in AppState is incorrect, Expected: {v} Actual: {app_attr}"

    destroy_model_parallel()
    destroy_num_microbatches_calculator()


@patch('torch.distributed.is_initialized', return_value=True)
@patch('megatron.core.parallel_state')
def test_init_model_parallel(mock_mpu, *args):
    from nemo.utils import AppState

    app_state = AppState()
    app_state.model_parallel_size = 1
    app_state.tensor_model_parallel_size = 2
    app_state.pipeline_model_parallel_size = 1
    app_state.pipeline_model_parallel_comm_backend = None
    app_state.context_parallel_size = 2
    app_state.expert_model_parallel_size = 2
    app_state.expert_tensor_parallel_size = 1
    app_state.expert_tensor_parallel_rank = 0
    app_state.init_mpi_proc_group = False
    app_state.tensor_model_parallel_rank = 2
    app_state.pipeline_model_parallel_rank = 0

    _mpu_tp_2(mock_mpu)
    _strategy_lib.init_model_parallel(nn.Identity())

    mock_mpu.initialize_model_parallel.assert_called_once_with(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        pipeline_model_parallel_comm_backend=None,
        context_parallel_size=2,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        use_sharp=False,
        order="tp-cp-ep-dp-pp",
        num_distributed_optimizer_instances=1,
        nccl_communicator_config_path=None,
        create_gloo_process_groups=True,
    )


@patch('torch.distributed.is_initialized', return_value=True)
@patch('megatron.core.parallel_state')
def test_init_model_parallel_with_tp_pp_dp(mock_mpu, *args):
    from nemo.utils import AppState

    app_state = AppState()
    app_state.model_parallel_size = 1
    app_state.tensor_model_parallel_size = 2
    app_state.pipeline_model_parallel_size = 1
    app_state.pipeline_model_parallel_comm_backend = None
    app_state.context_parallel_size = 2
    app_state.expert_model_parallel_size = 2
    app_state.expert_tensor_parallel_size = 1
    app_state.expert_tensor_parallel_rank = 0
    app_state.init_mpi_proc_group = False
    app_state.tensor_model_parallel_rank = 2
    app_state.pipeline_model_parallel_rank = 0

    app_state.use_tp_pp_dp_mapping = True

    _mpu_tp_2(mock_mpu)
    _strategy_lib.init_model_parallel(nn.Identity())

    mock_mpu.initialize_model_parallel.assert_called_once_with(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        pipeline_model_parallel_comm_backend=None,
        context_parallel_size=2,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        use_sharp=False,
        order="tp-cp-ep-pp-dp",
        num_distributed_optimizer_instances=1,
        nccl_communicator_config_path=None,
        create_gloo_process_groups=True,
    )


# TODO @chcui uncomment after fabric API is merged
# @patch('nemo.lightning._strategy_lib.DataLoader', return_value=MagicMock())
# @patch('megatron.core.parallel_state')
# def test_process_dataloader(mock_mpu, mock_dataloader) -> None:
#     mock_dataloader_instance = MagicMock()
#     mock_dataloader_instance.dataset = [1, 2, 3]
#     mock_dataloader_instance.num_workers = 4
#     mock_dataloader_instance.pin_memory = True
#     mock_dataloader_instance.persistent_workers = False
#
#     data_config = DataConfig(256)
#     data_config.micro_batch_size = 2
#     data_config.global_batch_size = 6
#     data_config.rampup_batch_size = 3
#
#     mock_mpu.get_data_parallel_rank.return_value = 0
#     mock_mpu.get_data_parallel_world_size.return_value = 1
#
#     out = _strategy_lib.process_dataloader(mock_dataloader_instance, data_config)
#     assert isinstance(out.batch_sampler, MagicMock)
#     mock_dataloader.assert_called_once_with(
#         mock_dataloader_instance.dataset,
#         batch_sampler=ANY,
#         num_workers=4,
#         pin_memory=True,
#         persistent_workers=False,
#         collate_fn=ANY
#     )


# @patch('nemo.lightning._strategy_lib.init_parallel_ranks')
# @patch('megatron.core.parallel_state')
# def test_setup_megatron_parallel_with_trainer(mock_mpu, mock_init_parallel_ranks) -> None:
#     _mpu_tp_2(mock_mpu)
#     mock_trainer = MagicMock(spec=pl.Trainer)
#     mock_trainer.strategy = MegatronStrategy(
#         ModelParallelConfig(tensor_model_parallel_size=2),
#         DataConfig(256),
#     )
#     mock_trainer.world_size = 2
#     mock_trainer.local_rank = 0
#     mock_trainer.global_rank = 1

#     result = _strategy_lib.setup_megatron_parallel(mock_trainer, nn.Identity())
#     mock_init_parallel_ranks.assert_called_once()
#     assert isinstance(result, LightningMegatronParallel)
#     assert len(result) == 1

#     # Test with function
#     assert len(_strategy_lib.setup_megatron_parallel(mock_trainer, lambda: nn.Identity())) == 1


# @patch('nemo.lightning._strategy_lib.init_parallel_ranks')
# @patch('megatron.core.parallel_state')
# def test_setup_megatron_parallel_virtual_pipelining(mock_mpu, mock_init_parallel_ranks) -> None:
#     vp_size = 4
#     _mpu_tp_2(mock_mpu)
#     mock_mpu.get_pipeline_model_parallel_world_size.return_value = 4
#     mock_trainer = MagicMock(spec=pl.Trainer)
#     mock_trainer.strategy = MegatronStrategy(
#         ModelParallelConfig(
#             virtual_pipeline_model_parallel_size=vp_size,
#             tensor_model_parallel_size=2,
#         ),
#         DataConfig(256),
#     )
#     mock_trainer.world_size = 8
#     mock_trainer.local_rank = 0
#     mock_trainer.global_rank = 1

#     result = _strategy_lib.setup_megatron_parallel(mock_trainer, Identity())
#     mock_init_parallel_ranks.assert_called_once()
#     assert len(result) == vp_size

#     # Test with function
#     assert len(_strategy_lib.setup_megatron_parallel(mock_trainer, lambda: nn.Identity())) == vp_size

#     # Test with a module with a copy method
#     assert len(_strategy_lib.setup_megatron_parallel(mock_trainer, WithCopy())) == vp_size

#     with pytest.raises(
#         ValueError,
#         match="Model does not have a copy method. Please implement this or " +
#         "pass in a function that returns the model"
#     ):
#         _strategy_lib.setup_megatron_parallel(mock_trainer, nn.Identity())


# @patch('nemo.lightning._strategy_lib.init_parallel_ranks')
# @patch('megatron.core.parallel_state')
# def test_setup_megatron_parallel_with_fabric(mock_mpu, mock_init_parallel_ranks) -> None:
#     _mpu_tp_2(mock_mpu)
#     mock_trainer = MagicMock(spec=fl.Fabric)
#     mock_trainer.strategy = FabricMegatronStrategy(
#         ModelParallelConfig(tensor_model_parallel_size=2),
#         DataConfig(256),
#     )
#     mock_trainer.world_size = 2
#     mock_trainer.local_rank = 0
#     mock_trainer.global_rank = 1

#     result = _strategy_lib.setup_megatron_parallel(mock_trainer, nn.Identity())

#     mock_init_parallel_ranks.assert_called_once()
#     assert isinstance(result, MegatronParallel)
#     assert len(result) == 1


# @patch('nemo.lightning._strategy_lib.init_parallel_ranks')
# @patch('megatron.core.parallel_state')
# def test_setup_megatron_parallel_with_strategy(mock_mpu, mock_init_parallel_ranks) -> None:
#     _mpu_tp_2(mock_mpu)
#     mock_trainer = MagicMock(spec=FabricMegatronStrategy)
#     mock_trainer.configure_mock(
#         parallelism=ModelParallelConfig(tensor_model_parallel_size=2),
#         data_config=DataConfig(256),
#         world_size=2,
#         local_rank=0,
#         global_rank=1
#     )

#     result = _strategy_lib.setup_megatron_parallel(mock_trainer, nn.Identity())

#     mock_init_parallel_ranks.assert_called_once()
#     assert isinstance(result, MegatronParallel)
#     assert len(result) == 1


def _mpu_tp_2(mock_mpu) -> None:
    mock_mpu.get_tensor_model_parallel_rank.return_value = 2
    mock_mpu.get_pipeline_model_parallel_rank.return_value = 0
    mock_mpu.get_pipeline_model_parallel_world_size.return_value = 1
    mock_mpu.get_pipeline_model_parallel_group.return_value = 0
    mock_mpu.get_tensor_model_parallel_group.return_value = 1
    mock_mpu.get_expert_tensor_parallel_rank.return_value = 0

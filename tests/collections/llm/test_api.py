import nemo_run as run
import pytest
import torch
from megatron.core.distributed import DistributedDataParallelConfig

from nemo import lightning as nl
from nemo.collections.llm.api import _validate_config
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.model.llama import Llama3Config8B, LlamaModel
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed


class TestValidateConfig:

    def reset_configs(self):
        model = run.Config(LlamaModel, config=run.Config(Llama3Config8B))
        data = run.Config(MockDataModule, seq_length=2048, global_batch_size=512, micro_batch_size=1)
        trainer = run.Config(
            nl.Trainer,
            accelerator="gpu",
            accumulate_grad_batches=1,
            callbacks=None,
            devices=8,
            limit_test_batches=50,
            limit_val_batches=32,
            log_every_n_steps=10,
            max_steps=1168251,
            num_nodes=1,
            plugins=bf16_mixed(),
            strategy=run.Config(
                nl.MegatronStrategy,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                pipeline_dtype=None,
                virtual_pipeline_model_parallel_size=None,
                context_parallel_size=1,
                sequence_parallel=1,
                gradient_as_bucket_view=True,
                ckpt_async_save=True,
                ckpt_parallel_load=True,
                ddp=run.Config(
                    DistributedDataParallelConfig,
                    check_for_nan_in_grad=True,
                    grad_reduce_in_fp32=True,
                    overlap_grad_reduce=True,
                    overlap_param_gather=True,
                    average_in_collective=True,
                ),
            ),
            use_distributed_sampler=False,
            val_check_interval=2000,
        )
        return model, data, trainer

    def test_model_validation(self):
        model, data, trainer = self.reset_configs()
        _validate_config(model, data, trainer)

        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            model.config.seq_length = 0
            _validate_config(model, data, trainer)

        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            model.config.num_layers = 0
            _validate_config(model, data, trainer)

        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            model.config.hidden_size = 0
            _validate_config(model, data, trainer)

        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            model.config.num_attention_heads = 0
            _validate_config(model, data, trainer)

        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            model.config.ffn_hidden_size = 0
            _validate_config(model, data, trainer)

    def test_data_validation(self):
        model, data, trainer = self.reset_configs()
        _validate_config(model, data, trainer)

        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            data.micro_batch_size = 0
            _validate_config(model, data, trainer)

        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            data.global_batch_size = 0
            _validate_config(model, data, trainer)

        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            data.seq_length = 0
            _validate_config(model, data, trainer)

        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            data.micro_batch_size = 3
            data.global_batch_size = 128
            _validate_config(model, data, trainer)

    def test_trainer_validatiopn(self):
        model, data, trainer = self.reset_configs()
        _validate_config(model, data, trainer)

        # DP validation
        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            trainer.strategy.tensor_model_parallel_size = 8
            trainer.strategy.pipeline_model_parallel_size = 2
            _validate_config(model, data, trainer)

        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            trainer.strategy.tensor_model_parallel_size = 3
            trainer.strategy.pipeline_model_parallel_size = 2
            _validate_config(model, data, trainer)

        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            data.global_batch_size = 3
            data.micro_batch_size = 1
            trainer.strategy.tensor_model_parallel_size = 2
            trainer.strategy.pipeline_model_parallel_size = 2
            _validate_config(model, data, trainer)

        # TP/SP validation
        model, data, trainer = self.reset_configs()
        trainer.strategy.tensor_model_parallel_size = 1
        trainer.strategy.sequence_parallel = True
        _validate_config(model, data, trainer)
        assert trainer.strategy.sequence_parallel == False

        # PP/VP validation
        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            trainer.strategy.pipeline_model_parallel_size = 2
            trainer.strategy.pipeline_dtype = None
            _validate_config(model, data, trainer)

        model, data, trainer = self.reset_configs()
        trainer.strategy.pipeline_model_parallel_size = 1
        trainer.strategy.virtual_pipeline_model_parallel_size = 2
        trainer.strategy.pipeline_dtype = torch.bfloat16
        _validate_config(model, data, trainer)
        assert trainer.strategy.virtual_pipeline_model_parallel_size == None
        assert trainer.strategy.pipeline_dtype == None

        # CP validation
        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            model.config.seq_length = 5
            trainer.strategy.context_parallel_size = 2
            _validate_config(model, data, trainer)

        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            model.config.seq_length = 2
            trainer.strategy.context_parallel_size = 2
            _validate_config(model, data, trainer)

        # EP validation
        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            model.config.num_moe_experts = None
            trainer.strategy.expert_model_parallel_size = 2
            _validate_config(model, data, trainer)

        with pytest.raises(AssertionError):
            model, data, trainer = self.reset_configs()
            model.config.num_moe_experts = 3
            trainer.strategy.expert_model_parallel_size = 2
            _validate_config(model, data, trainer)

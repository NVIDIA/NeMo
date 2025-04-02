from typing import TypeVar
from pathlib import Path
import os
import gc
import shutil
import torch
import time
from torch import nn
import contextlib
from lightning.pytorch import LightningModule

import nemo.lightning as nl
from nemo.common.ckpt.registry import detect_checkpoint_type
from nemo.common.ckpt.resolver import get_checkpoint
from nemo.common.plan.plan import Plan
from nemo.common.ckpt.model import register_parser

from nemo.common.ckpt.convert.model_converter import ModelConverter
from nemo.common.ckpt.convert.state_converter import StateConverter
from nemo.common.ckpt.impl.megatron import (
    temporary_single_process_group, 
    InitMegatronLightningEnv, 
    InitMegatronModel, 
    MegatronSaveModel
)
from nemo.common.plan.model_context import ModelContext
from nemo.common.ckpt.model import PreTrainedModel

ModelT = TypeVar("ModelT", bound=nn.Module)
InputModuleT = TypeVar("InputModuleT", bound=nn.Module)
TargetModuleT = TypeVar("TargetModuleT", bound=nn.Module)


@register_parser("megatron")
class MegatronModelConverter(ModelConverter):
    def setup(
        self, import_path, path_resolver: str | None = None, overwrite: bool = False
    ):
        self.import_path = import_path
        self.path_resolver = path_resolver
        self.ckpt_type = detect_checkpoint_type(
            import_path, path_resolver=path_resolver
        )
        if self.ckpt_type.fs.protocol == "hf":
            self.import_path = f"hf://{import_path}"

        if not self.output_path:
            if self.ckpt_type.fs.protocol == "hf":
                nemo_home = os.getenv("NEMO_HOME", str(Path.home() / ".cache" / "nemo"))
                self.output_path = str(
                    Path(nemo_home) / self.name / self.ckpt_type.fs_path
                )
            else:
                full_path = get_checkpoint(import_path, path_resolver)
                self.output_path = str(full_path / "converted" / self.name)

        if overwrite and os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)

        # Create source and target models
        self.source = ModelContext(self.import_path, path_resolver=path_resolver)
        # Get importer info
        _source = self.source()
        self._source_class, self._target_class, self.context_converter = (
            self.get_context_converter(str(self.ckpt_type), _source)
        )
        self.source.model_class = self._source_class
        if str(self.ckpt_type) != "megatron":
            self.target = ModelContext(self.output_path, model_class=self._target_class)

        self._converting.add(self.import_path)

    @property
    def source_class(self):
        """Get the source model class."""
        return self._source_class

    @property
    def model_class(self):
        """Get the target model class."""
        return self._target_class
    
    @property
    def state_convert(self) -> Plan:
        return self.get_state_converter(str(self.ckpt_type), self.source())()

    def execute(self) -> LightningModule:
        """Execute the import plan.

        Returns:
            The imported model
        """

        # if hasattr(self, "convert_state"):
        #     self.convert_state()

        out = self.context_converter(self.source())
        out.plan = self._parent

        return out

    def loader(self, env: Plan, peft: Plan | None = None):
        """Get the loader plan for this import.

        Returns:
            A LoadMegatronModel plan
        """

        from nemo.common.plan.env import LightningEnv

        if isinstance(env, LightningEnv) and not env.can_load:
            return InitMegatronLightningEnv(env)
        
        convert = None
        if not os.path.exists(self.output_path):
            convert = MegatronStateConverter(
                self.get_state_converter(str(self.ckpt_type), self.source())(),
                self.import_path,
                to="megatron", 
                path_resolver=self.path_resolver, 
                output_path=self.output_path,
                trainer=getattr(env, "trainer", None)
            )

        return InitMegatronModel(
            self.output_path, 
            env, 
            self.path_resolver,
            convert_state=convert
        )

    def extra_repr(self) -> str:
        return f"context_converter={self.context_converter.__module__}.{self.context_converter.__name__}"


class MegatronStateConvertPlan(Plan, primary_worker_only="sync"):
    def __init__(self, converter: StateConverter, use_single_process_group=True, model_cls: type[ModelT] = None):
        self.converter = converter
        self.saver = MegatronSaveModel()
        self.use_single_process_group = use_single_process_group
        self.model_cls = model_cls or PreTrainedModel
    
    def create_source_module(self, input_path) -> InputModuleT:
        raise NotImplementedError("Please implement")

    def create_target_module(self, input_path) -> TargetModuleT:
        with self.context:
            target_module = self.model_cls(input_path.replace("hf://", ""), parse="megatron", env="megatron_meta")

        return target_module

    def execute(self, input_path: str, output_path: str) -> ModelT:
        source_module = self.create_source_module(input_path)
        target_module = self.create_target_module(input_path)

        self.converter(source_module, target_module._forward_module.pipeline.module)
        with self.context:
            self.saver(target_module, output_path)

    @property
    def context(self):
        if self.use_single_process_group:
            return temporary_single_process_group()
        else:
            return contextlib.nullcontext()


class MegatronStateConverter(Plan):
    def __init__(
        self,
        converter: Plan,
        path: str,
        to: str,
        path_resolver: str | None = None,
        output_path: str | None = None,
        trainer: nl.Trainer | None = None,
    ):
        super().__init__()
        self.converter = converter
        self.path = path.replace("hf://", "")
        self.to = to
        self.path_resolver = path_resolver
        self.output_path = output_path
        self.trainer = trainer

    def execute(self, model: ModelT) -> ModelT:
        # self._move_to_meta(model)

        self.converter(self.path, self.output_path)
        time.sleep(10)

        # megatron_parallel = self.trainer.strategy.megatron_parallel
        # sharded_state = megatron_parallel.sharded_state_dict()        

        # with temporary_single_process_group():
        #     temp_config = replace(model.config, use_cpu_initialization=True)
        #     new_model = temp_config.configure_model(model.tokenizer)
        #     rich.print(new_model.state_dict())

        # with temporary_single_process_group():
        #     from auto_model import AutoModel

        #     new_model = AutoModel(self.path.replace("hf://", ""), convert="megatron", setup="megatron_meta")
            
        #     rich.print(sharded_state["module.decoder.layers.15.mlp.linear_fc2.weight"])
        #     rich.print(new_model._forward_module.sharded_state_dict()["module.decoder.layers.15.mlp.linear_fc2.weight"])
        
        # import pdb; pdb.set_trace()

    def _move_to_meta(self, model: ModelT):
        model.to(device="meta")

        for module in model.modules():
            if hasattr(module, 'fp8_weight') and hasattr(module.fp8_weight, 'clear_cache'):
                module.fp8_weight.clear_cache()

        gc.collect()
        torch.cuda.empty_cache()



if __name__ == "__main__":
    _imp = MegatronModelConverter("meta-llama/Llama-3.1-8B")
    print(_imp)
    llama_model = _imp()
    # print(llama_model.config)

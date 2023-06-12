# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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


import torch
import torch.multiprocessing as mp
from megatron.core import parallel_state
from omegaconf import OmegaConf
from omegaconf.omegaconf import open_dict
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_t5_adapter_model import (
    MegatronT5AdapterLearningModel,
    MegatronT5LoraModel,
    MegatronT5InfusedAdapterModel,
)
from nemo.collections.nlp.models.language_modeling.megatron_t5_prompt_learning_model import (
    MegatronT5PromptLearningModel,
)
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.core.config import hydra_runner
from nemo.utils.app_state import AppState
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model



mp.set_start_method("spawn", force=True)
"""
 

"""

def _get_peft_scheme(cfg):
    if cfg.peft.peft_scheme == "adapter":
        peft_cls = MegatronT5AdapterLearningModel
    elif cfg.peft.peft_scheme == "ia3":
        peft_cls = MegatronT5InfusedAdapterModel
    elif cfg.peft.peft_scheme == "ptuning":
        peft_cls = MegatronT5PromptLearningModel
    elif cfg.peft.peft_scheme == "lora":
        peft_cls = MegatronT5LoraModel
    else:
        raise RuntimeError("Invalid Peft scheme")
    return peft_cls


@hydra_runner(config_path="conf", config_name="megatron_t5_peft_eval_config")
def main(cfg) -> None:
    
    # trainer required for restoring model parallel models
    trainer = Trainer(strategy=NLPDDPStrategy(), **cfg.trainer)

    if (
        cfg.tensor_model_parallel_size < 0
        or cfg.pipeline_model_parallel_size < 0
        or cfg.get('pipeline_model_parallel_split_rank', -1) < 0
    ):
        model_config = MegatronT5Model.restore_from(
            restore_path=cfg.language_model_path, trainer=trainer, return_config=True,
        )

        with open_dict(cfg):
            cfg.tensor_model_parallel_size = model_config.get('tensor_model_parallel_size', 1)
            cfg.pipeline_model_parallel_size = model_config.get('pipeline_model_parallel_size', 1)
            cfg.pipeline_model_parallel_split_rank = model_config.get('pipeline_model_parallel_split_rank', 0)

    app_state = AppState()
    if cfg.tensor_model_parallel_size > 1 or cfg.pipeline_model_parallel_size > 1:
        app_state.model_parallel_size = cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
        (
            app_state.tensor_model_parallel_rank,
            app_state.pipeline_model_parallel_rank,
            app_state.model_parallel_size,
            app_state.data_parallel_size,
            app_state.pipeline_model_parallel_split_rank,
            app_state.virtual_pipeline_model_parallel_rank,
        ) = fake_initialize_model_parallel(
            world_size=app_state.model_parallel_size,
            rank=trainer.global_rank,
            tensor_model_parallel_size_=cfg.tensor_model_parallel_size,
            pipeline_model_parallel_size_=cfg.pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank_=cfg.pipeline_model_parallel_split_rank,
        )

    # Load an adapter model,  must be provided in config
    if cfg.get("peft_model_file", None) is not None and cfg.get("language_model_path", None) is not None:
        # Update frozen T5 model path in case it has changed
        peft_cls = _get_peft_scheme(cfg)
        adapter_tuning_cfg = peft_cls.restore_from(
            cfg.peft_model_file, trainer=trainer, return_config=True
        )
        with open_dict(adapter_tuning_cfg):
            adapter_tuning_cfg.language_model_path = cfg.language_model_path
            adapter_tuning_cfg.pretrained_language_model_path = cfg.language_model_path
            adapter_tuning_cfg.micro_batch_size = cfg.data.micro_batch_size
            adapter_tuning_cfg.global_batch_size = cfg.data.global_batch_size

        # Now load prompt learning model with frozen T5 model base
        model = peft_cls.restore_from(
            restore_path=cfg.peft_model_file, trainer=trainer, override_config_path=adapter_tuning_cfg
        )

    # Or load regular T5 model
    else:
        raise NotImplementedError(
            "This script is meant for inference from an Infused Adapter Tuned T5 Model, config should contain an adapter_model_file and a language_model_path"
        )

    # check whether the DDP is initialized
    if parallel_state.is_unitialized():

        def dummy():
            return

        if trainer.strategy.launcher is not None:
            trainer.strategy.launcher.launch(dummy, trainer=trainer)
        trainer.strategy.setup_environment()

    model.freeze()

    # Have to turn off activations_checkpoint_method for inference
    try:
        model.model.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass

    try:
        model.frozen_model.model.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass

    test_ds, test_dl = model.build_virtual_prompt_dataset(
        dataset_paths=cfg.data.test_ds,
        batch_size=cfg.data.global_batch_size,
        for_train=False,
        drop_last=False,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    config = OmegaConf.to_container(cfg.inference)
    model.set_inference_config(config)
    response = trainer.predict(model, test_dl)
    print("***************************")
    if cfg.pred_file_path is not None:
        with open(cfg.pred_file_path, "w", encoding="utf-8") as f:
            for batch in response:
                for inp, pred in zip(batch['input_text'], batch['preds_text']):
                    inp = ' '.join(inp.split('\n'))
                    pred = ' '.join(pred.split('\n'))
                    f.write(f'{inp} {pred}\n')
        print("predictions saved to {}".format(cfg.pred_file_path))
    else:
        print(response)
    print("***************************")



if __name__ == "__main__":
    main()

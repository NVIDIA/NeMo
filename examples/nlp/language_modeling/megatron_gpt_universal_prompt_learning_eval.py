# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import threading
from functools import partial

import torch
from apex.transformer import parallel_state
from omegaconf import OmegaConf
from omegaconf.omegaconf import open_dict
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_universal_prompt_model import (
    MegatronGPTUniversalPromptLearningModel,
)
from nemo.collections.nlp.modules.common.megatron_web_server import get_demo
from nemo.collections.nlp.modules.common.text_generation_server import MegatronServer
from nemo.collections.nlp.modules.common.text_generation_utils import generate
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.core.config import hydra_runner

try:
    from apex.transformer import parallel_state

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

"""
"""


def build_data_loader(dataset, data_cfg, batch_size):
    """Buld dataloader given an input dataset."""
    collate_fun = dataset.collate_fn
    collate_fn = partial(collate_fun, tp_workers=0)

    # Make distributed dataloader
    rank = parallel_state.get_data_parallel_rank()
    data_parallel_size = parallel_state.get_data_parallel_world_size()
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=data_parallel_size, rank=rank, shuffle=False,
    )
    # Torch dataloader.
    return torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=data_cfg.drop_last,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        collate_fn=collate_fn,
    )


@hydra_runner(config_path="conf", config_name="megatron_gpt_universal_prompt_learning_inference")
def main(cfg) -> None:
    if not torch.cuda.is_available():
        raise EnvironmentError("GPU is needed for the inference")

    # trainer required for restoring model parallel models
    trainer = Trainer(strategy=NLPDDPStrategy(), **cfg.trainer)
    # assert (
    #     cfg.trainer.devices * cfg.trainer.num_nodes
    #     == cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
    # ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"

    # Update frozen GPT model path if it is given in case it has changed
    prompt_learning_cfg = MegatronGPTUniversalPromptLearningModel.restore_from(
        cfg.prompt_encoder_model_file, trainer=trainer, return_config=True,
    )
    if cfg.get("gpt_model_file"):
        with open_dict(prompt_learning_cfg):
            prompt_learning_cfg.language_model_path = cfg.gpt_model_file
            prompt_learning_cfg.sequence_parallel = False
            prompt_learning_cfg.activations_checkpoint_granularity = None
            prompt_learning_cfg.activations_checkpoint_method = None
            prompt_learning_cfg.micro_batch_size = cfg.batch_size
            prompt_learning_cfg.global_batch_size = cfg.global_batch_size

    # Load prompt tuned model, virtual_prompt_model_file must be provided in config
    # Now load prompt learning model with frozen gpt model base
    model = MegatronGPTUniversalPromptLearningModel.restore_from(
        restore_path=cfg.prompt_encoder_model_file, trainer=trainer, override_config_path=prompt_learning_cfg,
    )
    model.freeze()

    # Have to turn off activations_checkpoint_method for inference
    try:
        model.frozen_model.model.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass

    # Check whether the DDP is initialized
    if parallel_state.is_unitialized():

        def placeholder():
            return

        if model.trainer.strategy.launcher is not None:
            model.trainer.strategy.launcher.launch(placeholder, trainer=model.trainer)
        model.trainer.strategy.setup_environment()

    if cfg.data.test_ds.file_names:
        eval_ds = model.build_dataset(data_cfg=cfg.data.test_ds, is_train=False,)
        eval_dls = []
        for dataset in eval_ds:
            eval_dl = build_data_loader(dataset=dataset, data_cfg=cfg.data.test_ds, batch_size=cfg.batch_size,)
            eval_dls.append(eval_dl)

        config = OmegaConf.to_container(cfg.inference)
        model.set_inference_config(config, cfg.data.test_ds)
        response = trainer.predict(model, eval_dls)

        print("***************************")
        # print(response)
        print("***************************")

    # Third method of running text generation, use inference server
    if cfg.server:
        if parallel_state.is_pipeline_first_stage() and parallel_state.get_tensor_model_parallel_rank() == 0:
            if cfg.web_server:
                thread = threading.Thread(target=get_demo, daemon=True, args=(cfg.share, cfg.username, cfg.password))
                thread.start()
            server = MegatronServer(model.cuda())
            server.run("0.0.0.0", port=cfg.port)

        while True:
            choice = torch.cuda.LongTensor(1)
            torch.distributed.broadcast(choice, 0)
            if choice[0].item() == 0:
                generate(model.cuda())


if __name__ == '__main__':
    main()

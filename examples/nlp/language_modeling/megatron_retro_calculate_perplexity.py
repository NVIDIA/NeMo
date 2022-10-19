# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import os
import threading

import torch
#from examples.nlp.language_modeling.megatron_gpt_eval import RequestDataSet
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.timer import Timer

from nemo.collections.nlp.models.language_modeling.megatron_retrieval_model import MegatronRetrievalModel
# from nemo.collections.nlp.modules.common.megatron_web_server import get_demo
#from nemo.collections.nlp.modules.common.text_generation_server import MegatronServer
#from nemo.collections.nlp.modules.common.text_generation_utils import generate
#from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from torch.utils.data import BatchSampler

from nemo.core.config import hydra_runner

try:
    from apex.transformer import parallel_state

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

"""
This is the script to run RETRO Model text generation.

Usage:
    Assume the model has TP=1, PP=1
    run greedy inference from a nemo file:
        python megatron_retro_eval.py \
            trainer.devices=1 \
            trainer.num_nodes=1 \
            trainer.accelerator=gpu \
            trainer.precision=16 \
            inference.tokens_to_generate=128 \
            inference.greedy=True \
            retro_model_file=path_to_retro_nemo_file \
            tensor_model_parallel_size=1 \
            pipeline_model_parallel_size=1 \
            retrieval_service.faiss_devices='0' \
            retrieval_service.faiss_index=path_to_faiss_index \
            retrieval_service.retrieval_index=path_to_retrieval_dataset \
            retrieval_service.neighbors=20
"""


@hydra_runner(config_path="conf", config_name="megatron_retro_perplexity")
def main(cfg) -> None:
    trainer = Trainer(strategy=NLPDDPStrategy(), **cfg.trainer)

    model_path = cfg.retro_model_file

    save_restore_connector = NLPSaveRestoreConnector()

    if os.path.isdir(model_path):
        save_restore_connector.model_extracted_dir = model_path

    model_cfg = MegatronRetrievalModel.restore_from(
        model_path, trainer=trainer, return_config=True, save_restore_connector=save_restore_connector,
    )

    with open_dict(model_cfg):
        model_cfg.precision = trainer.precision
        model_cfg.micro_batch_size = cfg.micro_batch_size
        model_cfg.data = cfg.data

    model = MegatronRetrievalModel.restore_from(
        model_path, trainer=trainer, save_restore_connector=save_restore_connector, override_config_path=model_cfg,
    )

    # Override timer callback to a stateless one
    for idx, callback in enumerate(trainer.callbacks):
        if isinstance(callback, Timer):
            trainer.callbacks[idx] = StatelessTimer(cfg.trainer.max_time,)

    # check whether the DDP is initialized
    if parallel_state.is_unitialized():
        def dummy():
            return

        if trainer.strategy.launcher is not None:
            trainer.strategy.launcher.launch(dummy, trainer=trainer)
        trainer.strategy.setup_environment()
    

    def build_pretraining_data_loader(dataset):
        """Buld dataloader given an input dataset."""

        # Make distributed dataloader
        rank = parallel_state.get_data_parallel_rank()
        data_parallel_size = parallel_state.get_data_parallel_world_size()
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=data_parallel_size, rank=rank, shuffle=False,
        )
        # Torch dataloader.
        return torch.utils.data.DataLoader(
            dataset, sampler=sampler, batch_size=cfg.micro_batch_size, drop_last=False, num_workers=model.cfg.data.num_workers, pin_memory=True,
        )

    model.build_train_valid_test_datasets()
    test_dl = build_pretraining_data_loader(model._test_ds)
    trainer.test(model, test_dl)


if __name__ == '__main__':
    main()

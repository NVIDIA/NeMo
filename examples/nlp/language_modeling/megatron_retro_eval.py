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

import torch
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.plugins.environments.torchelastic_environment import TorchElasticEnvironment
from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector

from nemo.collections.nlp.models.language_modeling.megatron_retrieval_model import MegatronRetrievalModel
from nemo.collections.nlp.modules.common.megatron.retrieval_service import FaissRetrievalService
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import StatelessTimer, exp_manager


@hydra_runner(config_path="conf", config_name="megatron_retro_inference")
def main(cfg) -> None:
    trainer = Trainer(strategy=NLPDDPStrategy(), **cfg.trainer)

    model_path = cfg.retro_model_file

    # model = MegatronRetrievalModel.restore_from(restore_path=model_path, trainer=trainer)
    save_restore_connector = NLPSaveRestoreConnector()
    if model_path:
        save_restore_connector.model_extracted_dir = model_path

    model_cfg = MegatronRetrievalModel.restore_from(
        model_path, trainer=trainer, return_config=True, save_restore_connector=save_restore_connector,
    )

    # Need to overwrite some params in frozen model's config before restoring
    with open_dict(model_cfg):
        pass
        # model_cfg.data.data_prefix = cfg.model.data.data_prefix
        # model_cfg.data.knn_index = cfg.model.data.knn_index
        # model_cfg.data.retrieval_prefix = cfg.model.data.retrieval_prefix
        # model_cfg.data.index_mapping_dir = cfg.model.data.index_mapping_dir

    if trainer.precision == 32:
        autocast_dtype = torch.float
    elif trainer.precision == 16:
        autocast_dtype = torch.half
    elif trainer.precision == 'bf16':
        autocast_dtype = torch.bfloat16
    else:
        raise ValueError('precision must be in [32, 16, "bf16"]')

    model = MegatronRetrievalModel.restore_from(
        model_path, trainer=trainer, save_restore_connector=save_restore_connector, override_config_path=model_cfg,
    )

    length_params: LengthParam = {
        "max_length": cfg.inference.tokens_to_generate,
        "min_length": cfg.inference.min_tokens_to_generate,
    }

    sampling_params: SamplingParam = {
        "use_greedy": cfg.inference.greedy,
        "temperature": cfg.inference.temperature,
        "top_k": cfg.inference.top_k,
        "top_p": cfg.inference.top_p,
        "repetition_penalty": cfg.inference.repetition_penalty,
        "add_BOS": cfg.inference.add_BOS,
        "all_probs": cfg.inference.all_probs,
        "compute_logprob": cfg.inference.compute_logprob,
    }

    #     service = FaissRetrievalService(faiss_index=cfg.faiss_index,
    #                                     faiss_devices=cfg.faiss_devices,
    #                                     nprobe=cfg.nprobe,
    #                                     retrieval_index=cfg.retrieval_index,
    #                                     tokenizer=model.tokenizer,
    #                                     sentence_bert=cfg.sentence_bert,
    #                                     sentence_bert_batch=cfg.sentence_bert_batch
    #                                     )
    # service = FaissRetrievalService(tokenizer=model.tokenizer, **cfg.retrieval_service)

    # q = service.get_knn('Massachusetts taxpayers are slated to receive hundreds of dollars in direct relief starting this November', 4)
    # print(q)

    # # First method of running text generation, call model.generate method
    response = model.generate(
        inputs=OmegaConf.to_container(cfg.prompts),
        length_params=length_params,
        sampling_params=sampling_params,
        **cfg.retrieval_service,
    )

    print("***************************")
    print(response)
    print("***************************")


if __name__ == '__main__':
    main()

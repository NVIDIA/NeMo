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
from apex.transformer import parallel_state
from omegaconf import OmegaConf
import os
from omegaconf.omegaconf import open_dict
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_fused_retro import MegatronFusedRetrievalAdapterModel
from nemo.collections.nlp.data.language_modeling.megatron.retro_fine_tune_dataset import RetroQAFineTuneDataset

from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector

from nemo.core.config import hydra_runner

try:
    from apex.transformer import parallel_state, tensor_parallel
    from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_without_interleaving import (
        forward_backward_pipelining_without_interleaving,
    )
    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

mp.set_start_method("spawn", force=True)

@hydra_runner(config_path="conf", config_name="megatron_retro_adapter_inference")
def main(cfg) -> None:
    if not torch.cuda.is_available():
        raise EnvironmentError("GPU is needed for the inference")

    # trainer required for restoring model parallel models
    trainer = Trainer(strategy=NLPDDPStrategy(), **cfg.trainer)
    assert (
        cfg.trainer.devices * cfg.trainer.num_nodes
        == cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
    ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"

    save_restore_connector = NLPSaveRestoreConnector()
    if os.path.isdir(cfg.model.restore_path):
        save_restore_connector.model_extracted_dir = cfg.model.restore_path

    model_cfg = MegatronFusedRetrievalAdapterModel.restore_from(
        cfg.model.restore_path, trainer=trainer, return_config=True, save_restore_connector=save_restore_connector,
    )

    with open_dict(model_cfg):
        model_cfg.precision = trainer.precision
        model_cfg.eval = True
        if "shape_file" in model_cfg:
            model_cfg.pop("shape_file")
        model_cfg.restore_from_path = cfg.model.original_model

    model = MegatronFusedRetrievalAdapterModel.restore_from(
        cfg.model.restore_path,
        trainer=trainer,
        save_restore_connector=save_restore_connector,
        override_config_path=model_cfg,
        strict=False,
    )

    # Check whether the DDP is initialized
    if parallel_state.is_unitialized():
        def placeholder():
            return

        if model.trainer.strategy.launcher is not None:
            model.trainer.strategy.launcher.launch(placeholder, trainer=model.trainer)
        model.trainer.strategy.setup_environment()

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

    max_input_length = model.cfg.encoder_seq_length - length_params["max_length"]

    _, dataloader = model.model.build_virtual_prompt_dataset(
        data=cfg.data_paths,
        batch_size=cfg.inference.batch_size,
        max_seq_length=max_input_length,
        min_seq_length=model.cfg.data.get('min_seq_length', 1),
        add_bos=sampling_params["add_BOS"],
        add_eos=False,
        for_train=False,
        tokens_to_generate=length_params["max_length"],
        drop_last=False,
        shuffle=False,
        num_workers=cfg.get("num_workers", 1),
        num_neighbors=int(cfg.retrieval_service.neighbors)+1,
        retrieved_doc_len=cfg.retrieval_service.retrieved_doc_len
    )

    config = OmegaConf.to_container(cfg.inference)
    retrieval_service = OmegaConf.to_container(cfg.retrieval_service)
    model.set_inference_config(config, retrieval_service)

    response = trainer.predict(model, dataloader)

    print("***************************")
    with open(cfg.pred_file_path, "w", encoding="utf-8") as pred_file:
        for i in range(len(response)):
            for sent in response[i]["sentences"]:
                sent = sent.strip()
                sent = sent.replace("\n", " ")
                pred_file.write(sent + "\n")
    print(f"Inference Complete, prediction file saved at {cfg.pred_file_path}")
    print("***************************")


if __name__ == '__main__':
    main()
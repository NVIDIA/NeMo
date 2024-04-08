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

from examples.nlp.language_modeling.megatron_gpt_eval import RequestDataSet
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from nemo.collections.nlp.models.language_modeling.megatron_retrieval_model import MegatronRetrievalModel
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from scripts.metric_calculation.compute_rouge import load_preds, load_ref, calculate_rouge
from scripts.metric_calculation.squad_metric_calc import f1_score, exact_match_score
from nemo.core.config import hydra_runner
import torch

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

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
            tensor_model_parallel_size=-1 \
            pipeline_model_parallel_size=-1 \
            retrieval_service.faiss_devices='0' \
            retrieval_service.faiss_index=path_to_faiss_index \
            retrieval_service.retrieval_index=path_to_retrieval_dataset \
            retrieval_service.neighbors=20
"""


@hydra_runner(config_path="conf", config_name="megatron_retro_inference")
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
        model_cfg.sequence_parallel = False
        model_cfg.activations_checkpoint_granularity = None
        model_cfg.activations_checkpoint_method = None

    if (
        cfg.tensor_model_parallel_size < 0
        or cfg.pipeline_model_parallel_size < 0
        or cfg.get('pipeline_model_parallel_split_rank', -1) < 0
    ):
        with open_dict(cfg):
            cfg.tensor_model_parallel_size = model_cfg.get('tensor_model_parallel_size', 1)
            cfg.pipeline_model_parallel_size = model_cfg.get('pipeline_model_parallel_size', 1)
            cfg.pipeline_model_parallel_split_rank = model_cfg.get('pipeline_model_parallel_split_rank', 0)

    model_cfg.task_templates = cfg["task_templates"]
    autocast_dtype = torch.float
    model = MegatronRetrievalModel.restore_from(
        model_path, trainer=trainer, save_restore_connector=save_restore_connector, override_config_path=model_cfg, strict=False
    ).to(dtype=autocast_dtype)

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

    if parallel_state.is_unitialized():

        def placeholder():
            return

        if model.trainer.strategy.launcher is not None:
            model.trainer.strategy.launcher.launch(placeholder, trainer=model.trainer)
        model.trainer.strategy.setup_environment()

    config = OmegaConf.to_container(cfg.inference)
    retrieval_service = OmegaConf.to_container(cfg.retrieval_service)
    model.set_inference_config(config, retrieval_service)
    model.enable_autocast = True

    _, request_dl = model.build_virtual_prompt_dataset(
        data=cfg.data_paths,
        batch_size=cfg.inference.batch_size,
        max_seq_length=model.cfg.encoder_seq_length - length_params["max_length"],
        min_seq_length=model.cfg.data.get('min_seq_length', 1),
        add_bos=sampling_params["add_BOS"],
        add_eos=False,
        for_train=False,
        tokens_to_generate=length_params["max_length"],
        drop_last=False,
        shuffle=False,
        num_workers=cfg.get("num_workers", 1),
        num_neighbors=int(cfg.retrieval_service.neighbors),
        retrieved_doc_len=cfg.retrieval_service.retrieved_doc_len,
        add_top_1=cfg.retrieval_service.get("add_top_1", True),
        chat_type=cfg.task_templates.get("chat_type", False)
    )


    if not cfg.use_predict_method:
        # First method of running text generation, call model.generate method
        response = model.generate(
            inputs=OmegaConf.to_container(cfg.prompts),
            length_params=length_params,
            sampling_params=sampling_params,
            strategy=model.inference_strategy,
        )
    else:
        # Second method of running text generation, call trainer.predict
        # ds = RequestDataSet(OmegaConf.to_container(cfg.prompts))
        # request_dl = DataLoader(dataset=ds, batch_size=cfg.inference_batch_size)
        response = trainer.predict(model, request_dl)

    print("***************************")
    # print(response)
    print("***************************")


    sentences = [sent for item in response for sent in item["sentences"]]
    with open("/results/temp_output.jsonl", "w") as pred_file:
        for sent in sentences:
            sent = sent.strip()
            sent = sent.replace("\n", " ")
            sent = sent.replace("\r", " ")
            pred_file.write(sent + "\n")

    output_lns = load_preds("/results/temp_output.jsonl", "Answer:")
    reference_lns = load_ref(cfg.data_paths, "answer")

    assert len(output_lns) == len(reference_lns)
    print("Calculating Rouge")

    rouge_scores = calculate_rouge(output_lns=output_lns, reference_lns=reference_lns)
    print("rouge: ", rouge_scores)

if __name__ == '__main__':
    main()

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

import asyncio
import datetime
import json
import os
import threading
from functools import partial

import torch
from omegaconf import OmegaConf, open_dict
from pytorch_lightning.trainer.trainer import Trainer
from torch.utils.data import DataLoader, Dataset

from nemo.collections.nlp.data.question_answering.input_example.qa_input_example import QAExample
from nemo.collections.nlp.metrics.qa_metrics import QAMetrics
from nemo.collections.nlp.models.language_modeling.megatron_retro_model import MegatronRetroModel
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.modules.common.text_generation_server import MegatronServer
from nemo.collections.nlp.modules.common.text_generation_utils import generate
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.collections.nlp.parts.nlp_overrides import CustomProgressBar, NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.core.config import hydra_runner
from nemo.utils.app_state import AppState
from nemo.utils.model_utils import inject_model_parallel_rank

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

"""
This is the script to run Retro text generation.

Usage:
    Assume the model has TP=1, PP=1 in the following use cases.
    a. run greedy inference from a nemo file:
        python megatron_retro_eval.py \
            gpt_model_file=PATH_TO_MODEL \
            inference.greedy=True \
            inference.add_BOS=True \
            trainer.devices=1 \
            trainer.num_nodes=1 \
            tensor_model_parallel_size=-1 \
            pipeline_model_parallel_size=-1 \
            inference.retro_inference.retro_num_neighbors=2 \
            qa_file_path= ""\
            pred_file_path = ""\


        ```
"""

if not torch.cuda.is_available():
    raise EnvironmentError("GPU is needed for the inference")


class RequestDataSet(Dataset):
    def __init__(self, sentences, neighbors):
        super().__init__()
        self.sentences = sentences
        self.neighbors = neighbors

    def __len__(self,):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {'prompts': self.sentences[idx], 'neighbors': self.neighbors[idx]}


def remove_padded_prompts(response, nb_paddings):
    result = {}
    for k, v in response.items():
        if v != None and (type(v) is list or type(v) is torch.Tensor):
            v = v[:-nb_paddings]
        result[k] = v
    return result


def process_qasample(sample, retro_num_neighbors=2, ft_neighbours=5):
    # process prompt
    question = sample['question']
    if not question.endswith("?"):
        question = question + "?"
    processed_prompt = "Question: {} Answer: The answer is".format(question)

    # process neighbors
    neighbors = sample['ctxs']
    neighbors = ["title: " + ctx["title"] + ", source: " + ctx["text"] for ctx in neighbors]
    processed_neighbors = neighbors[:retro_num_neighbors]

    # # concate neighbors to prompt
    if ft_neighbours > 0:
        contexts = "\n\n".join(neighbors[:ft_neighbours]) + "\n\n"
        processed_prompt = contexts + processed_prompt

    return processed_prompt, processed_neighbors


def process_qaresponse(response):
    prediction = response.split("The answer is")[1]
    # truncate text
    prediction = prediction.split(".")[0]
    prediction = prediction.split("\n")[0]
    prediction = prediction.split("\n\n")[0]
    return prediction


@hydra_runner(config_path="conf", config_name="megatron_retro_qatask")
def main(cfg) -> None:

    # trainer required for restoring model parallel models
    trainer = Trainer(
        strategy=NLPDDPStrategy(timeout=datetime.timedelta(seconds=18000)),
        **cfg.trainer,
        callbacks=[CustomProgressBar()],
    )

    if cfg.retro_model_file is not None:
        if (
            cfg.tensor_model_parallel_size < 0
            or cfg.pipeline_model_parallel_size < 0
            or cfg.get('pipeline_model_parallel_split_rank', -1) < 0
        ):
            save_restore_connector = NLPSaveRestoreConnector()
            if os.path.isdir(cfg.retro_model_file):
                save_restore_connector.model_extracted_dir = cfg.retro_model_file
            model_config = MegatronRetroModel.restore_from(
                restore_path=cfg.retro_model_file,
                trainer=trainer,
                return_config=True,
                save_restore_connector=save_restore_connector,
            )

            # with dist checkpointing we don't need to set this
            if not model_config.get('mcore_gpt', False):
                with open_dict(cfg):
                    cfg.tensor_model_parallel_size = model_config.get('tensor_model_parallel_size', 1)
                    cfg.pipeline_model_parallel_size = model_config.get('pipeline_model_parallel_size', 1)
                    cfg.pipeline_model_parallel_split_rank = model_config.get('pipeline_model_parallel_split_rank', 0)

    assert (
        cfg.trainer.devices * cfg.trainer.num_nodes
        == cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
    ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"

    if cfg.retro_model_file:
        save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(cfg.retro_model_file):
            save_restore_connector.model_extracted_dir = cfg.retro_model_file

        pretrained_cfg = MegatronRetroModel.restore_from(
            restore_path=cfg.retro_model_file,
            trainer=trainer,
            return_config=True,
            save_restore_connector=save_restore_connector,
        )

        OmegaConf.set_struct(pretrained_cfg, True)
        with open_dict(pretrained_cfg):
            pretrained_cfg.sequence_parallel = False
            pretrained_cfg.activations_checkpoint_granularity = None
            pretrained_cfg.activations_checkpoint_method = None
            pretrained_cfg.precision = trainer.precision
            pretrained_cfg["use_flash_attention"] = cfg.inference.get("use_flash_attention", False)
            if pretrained_cfg.get('mcore_gpt', False):
                # with dist checkpointing we can use the model parallel config specified by the user
                pretrained_cfg.tensor_model_parallel_size = cfg.tensor_model_parallel_size
                pretrained_cfg.pipeline_model_parallel_size = cfg.pipeline_model_parallel_size
            if trainer.precision == "16":
                pretrained_cfg.megatron_amp_O2 = False
            elif trainer.precision in ['bf16', 'bf16-mixed'] and cfg.get('megatron_amp_O2', False):
                pretrained_cfg.megatron_amp_O2 = True
        model = MegatronRetroModel.restore_from(
            restore_path=cfg.retro_model_file,
            trainer=trainer,
            override_config_path=pretrained_cfg,
            save_restore_connector=save_restore_connector,
            map_location=f'cuda:{trainer.local_rank}',  # map_location is needed for converted models
        )
    elif cfg.checkpoint_dir:
        app_state = AppState()
        if cfg.tensor_model_parallel_size > 1 or cfg.pipeline_model_parallel_size > 1:
            app_state.model_parallel_size = cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
            app_state.tensor_model_parallel_size = cfg.tensor_model_parallel_size
            app_state.pipeline_model_parallel_size = cfg.pipeline_model_parallel_size
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
        checkpoint_path = os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name)
        # checkpoint_path is a dir in case of distributed checkpointing
        if not os.path.isdir(checkpoint_path):
            # legacy checkpoint needs model parallel rank injection
            checkpoint_path = inject_model_parallel_rank(os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name))
        model = MegatronRetroModel.load_from_checkpoint(
            checkpoint_path, hparams_file=cfg.hparams_file, trainer=trainer
        )

    else:
        raise ValueError("need at least a nemo file or checkpoint dir")

    model.freeze()

    # Have to turn off activations_checkpoint_method for inference
    try:
        model.model.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass

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
        "end_strings": cfg.inference.end_strings,
    }

    # Reading QA data files
    qa_samples = []
    with open(cfg.qa_file_path, 'r', encoding='utf-8') as f:
        qa_samples = json.load(f)

    # Processing prompts and neighbors
    prompts = []
    neighbors = []
    ground_truths = []
    for sample in qa_samples:
        processed_prompt, processed_neighbors = process_qasample(
            sample, cfg.inference.retro_inference.retro_num_neighbors, cfg.inference.retro_inference.ft_neighbours
        )
        prompts.append(processed_prompt)
        neighbors.append(processed_neighbors)
        ground_truths.append(
            sample['answers'][0]
        )  # Boxin only takes the first value of sample['answers'] (https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/blob/boxin/instructretro-internal-test/tools/retro/text_generation/evaluate.py?ref_type=heads#L85)

    # DEBUGGING
    n = 20
    prompts = prompts[:n]
    neighbors = neighbors[:n]
    ground_truths = ground_truths[:n]
    print("prompts[0]: ", prompts[0])

    # Running prediction
    bs = 1
    ds = RequestDataSet(prompts, neighbors)
    request_dl = DataLoader(dataset=ds, batch_size=bs)
    config = OmegaConf.to_container(cfg.inference)
    model.set_inference_config(config)
    response = trainer.predict(model, request_dl)

    # Generating answers
    print("***************************")
    with open(cfg.pred_file_path, "w", encoding="utf-8") as pred_file:
        for i in range(len(response)):
            for sent in response[i]["sentences"]:
                sent = sent.strip()
                sent = sent.replace("\n", " ")
                pred_file.write(sent + "\n")
            for neighbor in neighbors[i]:
                neighbor = neighbor.replace("\n", " ")
                neighbor = "Neighbor: " + neighbor
                pred_file.write(neighbor + "\n")
            pred_file.write("---------\n")
    print(f"Inference Complete, prediction file saved at {cfg.pred_file_path}")
    print("***************************")

    # Compute metrics
    predictions = [process_qaresponse(response[i]["sentences"][0]) for i in range(len(response))]
    formatted_ground_truths = []
    formatted_predictions = []
    for i in range(len(predictions)):  # formatting to use NeMo's QAMetrics methods
        question_id = i
        qaexample = QAExample(
            qas_id=question_id,
            answers=[{'text': ground_truths[i]}],
            question_text="",
            context_text="",
            context_id="",
            answer_text="",
            start_position_character="",
            title="",
        )
        formatted_ground_truths.append(qaexample)
        formatted_predictions.append(predictions[i])
    eval_results = QAMetrics.evaluate_predictions(formatted_ground_truths, formatted_predictions)
    print("Eval_results: ", eval_results)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter

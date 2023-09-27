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

import json
import os

import torch
from constants import TASKS
from omegaconf import OmegaConf, open_dict
from prepare_truncated_data import process_data
from pytorch_lightning.trainer.trainer import Trainer
from torch.utils.data import DataLoader, Dataset

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.core.config import hydra_runner
from nemo.utils.app_state import AppState
from nemo.utils.model_utils import inject_model_parallel_rank

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

"""
This is the script to Zero-Scrolls evaluation with base GPT models (FA is enabled).

Supported tasks:
    ['book_sum_sort', 'gov_report', 'narrative_qa', 'qasper', 'qmsum', 'summ_screen_fd', 'quality', 'squality', 'musique', 'space_digest']

See NeMo/examples/nlp/language_modeling/constants.py for the exact subset used for each task. We use 'test' if labeled data is available, otherwise 'validation'.
`DATA_DIR` - is the path to the folder with subsets for all scrolls tasks.

Usage:
    python ${NEMO_DIR}/examples/nlp/language_modeling/megatron_gpt_eval_zero_scrolls.py \
        gpt_model_file=${NEMO_FILE} \
        server=False \
        tensor_model_parallel_size=${TP} \
        pipeline_model_parallel_size=${PP} \
        trainer.devices=${TP} \
        trainer.num_nodes=1 \
        trainer.precision=bf16 \
        inference.output_file=${PREDICTIONS}.jsonl \
        inference.task=${TASK} \
        inference.data_dir=${DATA_DIR} \
        inference.batch_size=1 \
        inference.max_seq_length=${MAX_SEQ_LENGTH}

The output of this script is a jsonl file with original fields and the generated output stored at 'pred' field.

To evaluate the generated predictions, run (https://gitlab-master.nvidia.com/yangzhang/llm_long_context):
    python llm_long_context/gpt_zero-shot/zero_scrolls/metrics_zero_scrolls.py \
        --predictions_file=${PREDICTIONS}.jsonl \
        --task=${TASK}
"""

if not torch.cuda.is_available():
    raise EnvironmentError("GPU is needed for the inference")


class RequestDataSet(Dataset):
    def __init__(self, sentences):
        super().__init__()
        self.sentences = sentences

    def __len__(self,):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


def remove_padded_prompts(response, nb_paddings):
    result = {}
    for k, v in response.items():
        if v != None and (type(v) is list or type(v) is torch.Tensor):
            v = v[:-nb_paddings]
        result[k] = v
    return result


@hydra_runner(config_path="conf", config_name="megatron_gpt_inference_zero_scrolls")
def main(cfg) -> None:

    if cfg.inference.task not in TASKS:
        raise NotImplementedError(f"{cfg.inference.task} not implemented. Choose from {TASKS.keys()}")
    cfg.inference.tokens_to_generate = TASKS[cfg.inference.task]["tokens_to_generate"]

    # trainer required for restoring model parallel models
    trainer = Trainer(strategy=NLPDDPStrategy(), **cfg.trainer)

    if (
        cfg.tensor_model_parallel_size < 0
        or cfg.pipeline_model_parallel_size < 0
        or cfg.get('pipeline_model_parallel_split_rank', -1) < 0
    ):
        save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(cfg.gpt_model_file):
            save_restore_connector.model_extracted_dir = cfg.gpt_model_file
        model_config = MegatronGPTModel.restore_from(
            restore_path=cfg.gpt_model_file,
            trainer=trainer,
            return_config=True,
            save_restore_connector=save_restore_connector,
        )

        with open_dict(cfg):
            cfg.tensor_model_parallel_size = model_config.get('tensor_model_parallel_size', 1)
            cfg.pipeline_model_parallel_size = model_config.get('pipeline_model_parallel_size', 1)
            cfg.pipeline_model_parallel_split_rank = model_config.get('pipeline_model_parallel_split_rank', 0)

    assert (
        cfg.trainer.devices * cfg.trainer.num_nodes
        == cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
    ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"

    if cfg.gpt_model_file:
        save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(cfg.gpt_model_file):
            save_restore_connector.model_extracted_dir = cfg.gpt_model_file

        pretrained_cfg = MegatronGPTModel.restore_from(
            restore_path=cfg.gpt_model_file,
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
            if cfg.get('seq_len_interpolation_factor', None) is not None:
                try:
                    pretrained_cfg.seq_len_interpolation_factor = cfg.seq_len_interpolation_factor
                except:
                    pretrained_cfg['seq_len_interpolation_factor'] = cfg.seq_len_interpolation_factor
            if trainer.precision == "16":
                pretrained_cfg.megatron_amp_O2 = False
            if cfg.inference.get("use_flash_attention", None) is not None:
                try:
                    pretrained_cfg.use_flash_attention = cfg.inference.use_flash_attention
                except:
                    pretrained_cfg["use_flash_attention"] = cfg.inference.use_flash_attention

            if cfg.inference.get("apply_query_key_layer_scaling", None) is not None:
                pretrained_cfg.apply_query_key_layer_scaling = cfg.inference.apply_query_key_layer_scaling

            if cfg.get("model", None) is not None and cfg.model.get("encoder", None) is not None:
                for k, v in cfg.model.encoder.items():
                    pretrained_cfg[k] = v

            pretrained_cfg.apply_query_key_layer_scaling = False
        model = MegatronGPTModel.restore_from(
            restore_path=cfg.gpt_model_file,
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
        checkpoint_path = inject_model_parallel_rank(os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name))
        model = MegatronGPTModel.load_from_checkpoint(checkpoint_path, hparams_file=cfg.hparams_file, trainer=trainer)
    else:
        raise ValueError("need at least a nemo file or checkpoint dir")

    print(f'\n{OmegaConf.to_yaml(model._cfg)}')
    model.freeze()

    # Have to turn off activations_checkpoint_method for inference
    try:
        model.model.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass

    fp8_enabled = hasattr(model.cfg, "fp8") and (model.cfg.fp8 == True)
    if fp8_enabled:
        nb_paddings = 0

    print("Processing data...")
    original_lines, truncated_input = process_data(
        model.tokenizer,
        prompt=cfg.chatbot_config.prompt if cfg.chat else None,
        task=cfg.inference.task,
        max_seq_length=cfg.inference.max_seq_length,
        data_dir=cfg.inference.data_dir,
        n_jobs=cfg.inference.n_jobs,
        remove_newline_tab=cfg.inference.remove_newline_tab,
    )

    print("Running inference...")
    bs = cfg.inference.batch_size
    ds = RequestDataSet(truncated_input)
    request_dl = DataLoader(dataset=ds, batch_size=bs)
    config = OmegaConf.to_container(cfg.inference)
    model.set_inference_config(config)
    response = trainer.predict(model, request_dl)

    if fp8_enabled:
        response[-1] = remove_padded_prompts(response[-1], nb_paddings)

    if model.global_rank == 0:
        print("***************************")
        if cfg.inference.output_file is not None:
            idx = 0
            with open(cfg.inference.output_file, "w", encoding="utf-8") as f:
                for batch in response:
                    batch_sentences = [s for s in batch['sentences']]
                    for s in batch_sentences:
                        cur_line = original_lines[idx]
                        cur_line['pred'] = s
                        f.write(json.dumps(cur_line) + '\n')
                        idx += 1
            print("predictions saved to {}".format(cfg.inference.output_file))
        else:
            print(response)
    print("***************************")


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter

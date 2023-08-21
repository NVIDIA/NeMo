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

from nemo.collections.nlp.models.language_modeling.megatron_fused_retro import MegatronFusedRetrievalLoraModel
from nemo.collections.nlp.data.language_modeling.megatron.retro_fine_tune_dataset import RetroQAFineTuneDataset

from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector

from nemo.core.config import hydra_runner

try:
    from apex.transformer import parallel_state, tensor_parallel
    from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_without_interleaving import (
        forward_backward_pipelining_without_interleaving,
    )
    from apex.transformer.pipeline_parallel.schedules.fwd_bwd_no_pipelining import forward_backward_no_pipelining

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

mp.set_start_method("spawn", force=True)

"""
This is the script to run GPT text generation.
    a. run greedy inference from a p-tuned/prompt-tuned model's nemo file:
        python megatron_gpt_prompt_learning_eval.py \
            virtual_prompt_model_file=PATH_TO_NEMO_PROMPT_LEARNING_MODEL_FILE \
            gpt_model_file=PATH_TO_FROZEN_GPT_MODEL_FILE \
            inference.greedy=True \
            inference.add_BOS=False \
            trainer.devices=1 \
            trainer.num_nodes=1 \
            tensor_model_parallel_size=1 \
            pipeline_model_parallel_size=1 \
            pred_file_path=PATH_WHERE_PRED_TEXT_FILE_WILL_BE_SAVED \
            data_paths=[path/to/dataset1.jsonl, path/to/dataset2.jsonl]

        virtual_prompt_model_file should be a path to a .nemo file saved after p-tuning/prompt tuning and model file
        is still the path to the gpt model's .nemo file.         

        data_paths should be a list of .json or .jsonl files containing json objects similar to the ones 
        used during prompt learning. They should have keys that match the fields specified in the prompt template.
        Fields can be dropped from the prompt dict and their corresponding section of the prompt template will 
        be automatically removed. 

        For example, say the prompt template during p-tuning/prompt-tuning looked like:

        '<|VIRTUAL_PROMPT_0|> Context: {context} Question: {question} Answer: {answer}'

        but you don't want to include the answer field during inference. Just don't 
        include the answer field in the prompt dict like below:

        {"taskname": "squad", "context": "some paragraph", "question": "question related to paragraph"}
        {"taskname": "squad", "context": "another paragraph", "question": "a different question related to paragraph"}

        And the dataset class will automatically format your input to have the form:

        [
            '<|VIRTUAL_PROMPT_0|> Context: some paragraph Question: question related to paragraph Answer:',
            '<|VIRTUAL_PROMPT_0|> Context: another paragraph Question: a different question related to paragraph Answer:'
        ]

        Similarly for other senarios, just add virtual_prompt_model_file=PATH_TO_NEMO_PROMPT_LEARNING_MODEL_FILE if you're using a 
        p-tuned/prompt-tuned model. 
"""


@hydra_runner(config_path="conf", config_name="megatron_retro_lora_inference")
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

    model_cfg = MegatronFusedRetrievalLoraModel.restore_from(
        cfg.model.restore_path, trainer=trainer, return_config=True, save_restore_connector=save_restore_connector,
    )

    with open_dict(model_cfg):
        # model_cfg.restore_from_path = "/workspaces/research/downloaded/megatron_retro_converted.nemo"
        # work around for the fused softmax bug
        # model_cfg.masked_softmax_fusion = False
        model_cfg.precision = trainer.precision
        model_cfg.eval = True
        if "shape_file" in model_cfg:
            model_cfg.pop("shape_file")
        model_cfg.restore_from_path = cfg.model.original_model
        # model_cfg.restore_from_path = model

    model = MegatronFusedRetrievalLoraModel.restore_from(
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

    global_batch_size = trainer.world_size * cfg.micro_batch_size // cfg.tensor_model_parallel_size
    test_iters = int(trainer.limit_test_batches)

    # test_ds = RetroQAFineTuneDataset(
    #     cfg.test_ds.get('file_name'),
    #     model.tokenizer,
    #     cfg.test_ds.get('answer_only_loss'),
    #     model.tokenizer.pad_id,
    #     cfg.test_ds.get('seq_length'),
    #     cfg.test_ds.get('add_bos'),
    #     cfg.test_ds.get('add_eos'),
    #     test_iters * global_batch_size,
    #     cfg.test_ds.get('seed'),
    #     cfg.test_ds.get('neighbors'),
    # )
    # test_dl = model.build_pretraining_data_loader(test_ds, 0)

    # # Have to turn off activations_checkpoint_method for inference
    # try:
    #     model.frozen_model.model.language_model.encoder.activations_checkpoint_method = None
    # except AttributeError:
    #     pass



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

    # model.set_inference_config(config)

    # response = model.generate(
    #     inputs=OmegaConf.to_container(cfg.prompts),
    #     length_params=length_params,
    #     sampling_params=sampling_params,
    #     strategy=model.inference_strategy,
    # )
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
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from pathlib import Path

import torch
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from nemo.collections.multimodal.speech_llm.models.modular_models import ModularAudioGPTModel
from nemo.collections.multimodal.speech_llm.parts.utils.data_utils import write_predictions_to_file
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.collections.nlp.parts.peft_config import PEFT_CONFIG_MAP
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.model_utils import inject_model_parallel_rank

mp.set_start_method("spawn", force=True)

"""
This is the script to run inference with a ModularAudioGPTModel.

If you want to evaluate an ModularAudioGPTModel:

MEGATRON_CKPT=/path/to/megatron-llm.nemo
ALM_DIR=/path/to/nemo_experiments/job_name
ALM_YAML=$ALM_DIR/version_0/hparams.yaml
ALM_CKPT="$ALM_DIR/checkpoints/AudioGPT--validation_wer\=0.5-step\=103-epoch\=0-last.ckpt"

VAL_MANIFESTS="[/data/libri-test-other.json,/data/MCV_7.1_test.json,/data/wsj-test.json]"
VAL_NAMES="[ls-test-other,mcv7.1-test,wsj-test]"

HYDRA_FULL_ERROR=1 \
NVTE_MASKED_SOFTMAX_FUSION=0 \
NVTE_FLASH_ATTN=0 \
NVTE_FUSED_ATTN=0 \
CUDA_VISIBLE_DEVICES=0 python modular_audio_gpt_eval.py \
    model.restore_from_path=$MEGATRON_CKPT \
    model.peft.restore_from_path=$ALM_CKPT \
    model.peft.restore_from_hparams_path=$ALM_YAML \
    model.data.test_ds.manifest_filepath=$VAL_MANIFESTS \
    model.data.test_ds.names=$VAL_NAMES \
    model.data.test_ds.global_batch_size=8 \
	model.data.test_ds.micro_batch_size=8 \
	model.data.test_ds.tokens_to_generate=256 \
    ++inference.greedy=False \
    ++inference.top_k=50 \
    ++inference.top_p=0.95 \
    ++inference.temperature=0.4 \
    ++inference.repetition_penalty=1.2 \
    ++model.data.test_ds.output_dir=${ALM_DIR}
"""


@hydra_runner(config_path="conf", config_name="modular_audio_gpt_config_eval")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")
    assert cfg.model.restore_from_path is not None

    trainer = MegatronTrainerBuilder(cfg).create_trainer()

    if cfg.model.from_pretrained:
        logging.info(f"Loading model from cloud: {cfg.model.restore_from_path}")
        model_cfg = ModularAudioGPTModel.from_pretrained(
            cfg.model.restore_from_path, trainer=trainer, return_config=True
        )
        model_cfg = ModularAudioGPTModel.merge_inference_cfg(cfg, trainer, model_cfg)
        model = ModularAudioGPTModel.from_pretrained(
            cfg.model.restore_from_path, trainer=trainer, override_config_path=model_cfg
        )
    else:
        model_cfg = ModularAudioGPTModel.merge_inference_cfg(cfg, trainer)
        model = ModularAudioGPTModel.restore_from(
            restore_path=cfg.model.restore_from_path, trainer=trainer, override_config_path=model_cfg,
        )
        if cfg.model.peft.restore_from_path:
            if '\\' in cfg.model.peft.restore_from_path:
                cfg.model.peft.restore_from_path = cfg.model.peft.restore_from_path.replace('\\', '')
            if "peft" in model_cfg:
                peft_cfg_cls = PEFT_CONFIG_MAP[model_cfg.peft.peft_scheme]
                model.load_adapters(cfg.model.peft.restore_from_path, peft_cfg_cls(model_cfg))
            else:
                model.load_state_dict(torch.load(cfg.model.peft.restore_from_path), strict=False)
        elif cfg.model.peft.restore_from_ckpt.checkpoint_dir and cfg.model.peft.restore_from_ckpt.checkpoint_name:
            checkpoint_path = os.path.join(
                cfg.model.peft.restore_from_ckpt.checkpoint_dir, cfg.model.peft.restore_from_ckpt.checkpoint_name
            )
            # checkpoint_path is a dir in case of distributed checkpointing
            if not os.path.isdir(checkpoint_path):
                # legacy checkpoint needs model parallel rank injection
                checkpoint_path = inject_model_parallel_rank(
                    os.path.join(
                        cfg.model.peft.restore_from_ckpt.checkpoint_dir,
                        cfg.model.peft.restore_from_ckpt.checkpoint_name,
                    )
                )
                if "peft" in model_cfg:
                    peft_cfg_cls = PEFT_CONFIG_MAP[cfg.model.peft.peft_scheme]
                    model.load_adapters(checkpoint_path, peft_cfgs=peft_cfg_cls(model_cfg))
                else:
                    model.load_state_dict(torch.load(checkpoint_path), strict=False)
            else:
                raise NotImplementedError("distributed checkpointing of PEFT weights is not supported")

        if model_cfg.freeze_audio_encoder:
            model = ModularAudioGPTModel.load_audio_encoder_for_inference(cfg, model_cfg, model)

    model.freeze()
    if cfg.get("save_as_nemo", None):
        model.save_to(cfg.save_as_nemo)
        logging.info(f"Model saved to {Path(cfg.save_as_nemo).absolute()}")

    if not cfg.model.get('use_flash_attention', False):
        cfg.inference.compute_attention_mask = True
    config = OmegaConf.to_container(cfg.inference, resolve=True)
    model.set_inference_config(config)
    model.freeze()

    if cfg.evaluate_metric:
        trainer.test(model)
        exit(0)

    test_loaders = model.get_test_dataloader(model_cfg.data.test_ds)
    predictions = trainer.predict(model, test_loaders)
    write_predictions_to_file(cfg, predictions, model)


if __name__ == "__main__":
    main()

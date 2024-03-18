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


import json
from pathlib import Path

import torch
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict

from nemo.collections.multimodal.speechllm.models.speechllm_models import ModularAudioGPTModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.collections.nlp.parts.peft_config import PEFT_CONFIG_MAP
from nemo.core.config import hydra_runner
from nemo.utils import logging

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


def get_model_cfg(cfg, trainer, pretrained_model_cfg=None):
    if pretrained_model_cfg:
        model_cfg = pretrained_model_cfg
    elif cfg.model.peft.restore_from_path:
        if cfg.model.peft.restore_from_path.endswith(".nemo"):
            model_cfg = ModularAudioGPTModel.restore_from(
                restore_path=cfg.model.peft.restore_from_path, trainer=trainer, return_config=True,
            )
        elif cfg.model.peft.restore_from_hparams_path:  # not a .nemo model we expect a hparams.yaml file
            model_cfg = OmegaConf.to_container(OmegaConf.load(cfg.model.peft.restore_from_hparams_path).cfg)
            model_cfg = OmegaConf.create(model_cfg)
            # extract dict inside cfg key and convert it to DictConfig
            # this allows interpolation to work the same way as config from the .restore_from method
        else:
            raise RuntimeError("This script requires a .nemo peft model or path to hparams.yaml (and a ckpt path).")
    else:
        model_cfg = MegatronGPTSFTModel.restore_from(
            restore_path=cfg.model.restore_from_path, trainer=trainer, return_config=True,
        )

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(model_cfg):
        # update the model config of the trained model with params we want to set at inference time.
        model_cfg.data.test_ds = cfg.model.data.test_ds
        model_cfg.activations_checkpoint_granularity = None
        model_cfg.activations_checkpoint_method = None
        if model_cfg.get("use_flash_attention", False):
            model_cfg.use_flash_attention = cfg.model.use_flash_attention
        if cfg.model.get("seq_len_interpolation_factor", None) is not None:
            model_cfg["seq_len_interpolation_factor"] = cfg.model.seq_len_interpolation_factor

        # fix for old checkpoints that don't have certain keys
        if "peft" in model_cfg:
            peft_key = f"{model_cfg.peft.peft_scheme}_tuning"
            if "weight_tying" not in model_cfg.peft.get(peft_key):
                model_cfg.peft[peft_key]["weight_tying"] = False
            if "layer_selection" not in model_cfg.peft.get(peft_key):
                model_cfg.peft[peft_key]["layer_selection"] = None
            if "position_embedding_strategy" not in model_cfg.peft.get(peft_key):
                model_cfg.peft[peft_key]["position_embedding_strategy"] = None

    return model_cfg


def load_audio_models(cfg, model_cfg, model):
    with open_dict(cfg):
        if (
            model_cfg.get("pretrained_audio_model", None) is not None
            and cfg.model.get("pretrained_audio_model", None) is None
        ):
            logging.info(
                f"model.pretrained_audio_model not found in config, setting it to {model_cfg.pretrained_audio_model} from loaded checkpoint."
            )
            cfg.model.pretrained_audio_model = model_cfg.pretrained_audio_model
        cfg.model.perception = model_cfg.perception

    audio_model, _ = ModularAudioGPTModel.get_audio_encoder_models_and_configs(cfg)
    speaker_model, _ = ModularAudioGPTModel.get_speaker_model_and_config(cfg)
    model = ModularAudioGPTModel.load_pretrained_audio_weights(cfg, model, audio_model, speaker_model)
    return model


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
        model_cfg = get_model_cfg(cfg, trainer, model_cfg)
        model = ModularAudioGPTModel.from_pretrained(
            cfg.model.restore_from_path, trainer=trainer, override_config_path=model_cfg
        )
    else:
        model_cfg = get_model_cfg(cfg, trainer)
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
        if model_cfg.freeze_audio_encoder:
            model = load_audio_models(cfg, model_cfg, model)

    if cfg.get("save_as_nemo", None):
        model.save_to(cfg.save_as_nemo)
        logging.info(f"Model saved to {Path(cfg.save_as_nemo).absolute()}")

    config = OmegaConf.to_container(cfg.inference, resolve=True)
    model.set_inference_config(config)
    model.freeze()

    if cfg.get("save_as_nemo", None):
        model.save_to(cfg.save_as_nemo)
        logging.info(f"Model saved to {Path(cfg.save_as_nemo).absolute()}")

    if cfg.evaluate_metric:
        trainer.test(model)
        exit(0)

    test_loaders = model.get_test_dataloader(model_cfg.data.test_ds)
    response = trainer.predict(model, test_loaders)

    if model.global_rank == 0:
        print("***************************")
        metadata_set = set()
        if cfg.inference.outfile_path is not None:
            with open(cfg.inference.outfile_path, "w", encoding="utf-8") as f:
                for batch in response:
                    batch_sentences = [s for s in batch['sentences']]
                    batch_tokens = [s for s in batch['tokens']]
                    batch_contexts = [s for s in batch['inputst']]
                    batch_labels = [s for s in batch['labels']]
                    batch_preds = [s for s in batch['preds']]
                    batch_metadata = [s for s in batch['metadata']]
                    if cfg.inference.compute_logprob:
                        batch_logprob = [s.tolist() for s in batch['logprob']]
                        for s, t, l in zip(batch_sentences, batch_tokens, batch_logprob):
                            if cfg.inference.get("verbose", False):
                                d = {
                                    'sentence': s,
                                    'tokens_with_logprobs': ', '.join([f"{_t} {_l:.4f}" for _t, _l in zip(t, l)]),
                                }
                                f.write(json.dumps(d, sort_keys=True, indent=2) + '\n')
                    else:
                        for i in range(len(batch_sentences)):
                            key = str(batch_metadata[i])
                            if key in metadata_set:
                                continue
                            metadata_set.add(key)
                            d = {
                                'metadata': batch_metadata[i],
                                'context': batch_contexts[i],
                                'label': batch_labels[i],
                                'prediction': batch_preds[i],
                                'sentence': batch_sentences[i],
                            }
                            f.write(json.dumps(d) + '\n')
            print("predictions saved to {}".format(cfg.inference.outfile_path))
        else:
            print(response)
    print("***************************")


if __name__ == "__main__":
    main()

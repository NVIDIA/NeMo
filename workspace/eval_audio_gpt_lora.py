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


import json
import os

import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from torch.utils.data import DataLoader

from nemo.collections.multimodal.models.speechllm_models import ModularizedAudioGPTModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_peft_models import MegatronGPTPEFTModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
    PEFTSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging

mp.set_start_method("spawn", force=True)
"""
This is the script to run inference with a PEFT model or an SFT Model.

If you want to evaluate an SFT .nemo file:

python examples/nlp/language_modeling/tuning/megatron_gpt_peft_eval.py \
	model.restore_from_path=<path_to_sft_nemo_file> \
	model.peft.restore_from_path=null \
	trainer.devices=1 model.data.test_ds.file_names=\[<path_to_test_jsonl_file1>, <path_to_test_jsonl_file2>] \
	model.data.test_ds.names=\['name_for_test_file1', 'name_for_test_file2'] \  # this is not the filename just some identifier
	model.data.test_ds.global_batch_size=4 \  # or some other value
	model.data.test_ds.micro_batch_size=4 \
	model.data.test_ds.tokens_to_generate=30 \
	inference.greedy=True \
	inference.outfile_path=\'<path_to_jsonl_output_file>'  

If you want to evaluate a PEFT Model, you should provide a base GPT model and a PEFT model .nemo file

python examples/nlp/language_modeling/tuning/megatron_gpt_peft_eval.py \
	model.restore_from_path=<path_to_sft_nemo_file> \
	model.peft.restore_from_path=<path_to_peft_nemo_file> \ # this will be created if you use `megatron_gpt_peft_tuning.py`
	trainer.devices=1 model.data.test_ds.file_names=\[<path_to_test_jsonl_file1>, <path_to_test_jsonl_file2>] \
	model.data.test_ds.names=\['name_for_test_file1', 'name_for_test_file2'] \  # this is not the filename just some identifier
	model.data.test_ds.global_batch_size=4 \  # or some other value
	model.data.test_ds.micro_batch_size=4 \
	model.data.test_ds.tokens_to_generate=30 \
	inference.greedy=True \
	inference.outfile_path=\'<path_to_jsonl_output_file>'  

"""


@hydra_runner(config_path="../examples/multimodel/conf/speechllm", config_name="modularized_speech_gpt_config_eval")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")
    assert cfg.model.restore_from_path is not None
    megatron_amp_o2 = cfg.model.get("megatron_amp_O2", False)
    with_distributed_adam = False

    plugins = []
    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=True,  # we don't use DDP for async grad allreduce
        gradient_as_bucket_view=cfg.model.gradient_as_bucket_view,
        find_unused_parameters=False,
    )
    if cfg.trainer.precision in [16, "bf16"]:
        scaler = None
        if cfg.trainer.precision == 16:
            scaler = GradScaler(
                init_scale=cfg.model.get("native_amp_init_scale", 2 ** 32),
                growth_interval=cfg.model.get("native_amp_growth_interval", 1000),
                hysteresis=cfg.model.get("hysteresis", 2),
                enabled=False
                if cfg.model.pipeline_model_parallel_size > 1
                else True,  # turn off the grad scale for pipeline parallel LM model
            )
        if megatron_amp_o2 and not with_distributed_adam:
            plugins.append(MegatronHalfPrecisionPlugin(precision=cfg.trainer.precision, device="cuda", scaler=scaler))
        else:
            plugins.append(PipelineMixedPrecisionPlugin(precision=cfg.trainer.precision, device="cuda", scaler=scaler))

    if cfg.get("cluster_type", None) == "BCP":
        plugins.append(TorchElasticEnvironment())

    trainer = Trainer(plugins=plugins, strategy=strategy, **cfg.trainer)
    if cfg.model.peft.restore_from_path:
        if cfg.model.peft.restore_from_path.endswith(".nemo"):
            peft_model_cfg = MegatronGPTPEFTModel.restore_from(
                restore_path=cfg.model.peft.restore_from_path, trainer=trainer, return_config=True,
            )
        elif cfg.model.peft.restore_from_hparams_path:  # not a .nemo model we expect a hparams.yaml file
            peft_model_cfg = OmegaConf.to_container(OmegaConf.load(cfg.model.peft.restore_from_hparams_path).cfg)
            peft_model_cfg = OmegaConf.create(peft_model_cfg)
            # extract dict inside cfg key and convert it to DictConfig
            # this allows interpolation to work the same way as config from the .restore_from method
        else:
            raise RuntimeError("This script requires a .nemo peft model or path to hparams.yaml (and a ckpt path).")
    else:
        peft_model_cfg = MegatronGPTSFTModel.restore_from(
            restore_path=cfg.model.restore_from_path, trainer=trainer, return_config=True,
        )

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(peft_model_cfg):
        # update the model config of the trained model with params we want to set at inference time.
        peft_model_cfg.precision = cfg.trainer.precision
        peft_model_cfg.data.test_ds = cfg.model.data.test_ds
        peft_model_cfg.activations_checkpoint_granularity = None
        peft_model_cfg.activations_checkpoint_method = None
        if peft_model_cfg.get("use_flash_attention", False):
            peft_model_cfg.use_flash_attention = cfg.model.use_flash_attention
        if cfg.model.get("seq_len_interpolation_factor", None) is not None:
            peft_model_cfg["seq_len_interpolation_factor"] = cfg.model.seq_len_interpolation_factor

    if cfg.model.peft.restore_from_path:
        if '\\' in cfg.model.peft.restore_from_path:
            cfg.model.peft.restore_from_path = cfg.model.peft.restore_from_path.replace('\\', '')

        if cfg.model.peft.restore_from_path.endswith(".nemo"):
            save_restore_connector = PEFTSaveRestoreConnector(
                peft_model_nemo_path=cfg.model.peft.restore_from_path, peft_model_ckpt_path=None,
            )
        else:
            # attempting to load a ckpt peft model.
            if cfg.model.peft.restore_from_ckpt_name:
                ckpt_name = cfg.model.peft.restore_from_ckpt_name
            else:
                ckpt_name = "model_weights.ckpt"
            save_restore_connector = PEFTSaveRestoreConnector(
                peft_model_nemo_path=None,
                peft_model_ckpt_path=cfg.model.peft.restore_from_path,
                peft_model_ckpt_name=ckpt_name,
            )
    else:
        save_restore_connector = NLPSaveRestoreConnector()

    if os.path.isdir(cfg.model.restore_from_path):
        save_restore_connector.model_extracted_dir = cfg.model.restore_from_path

    model = ModularizedAudioGPTModel.restore_from(
        restore_path=cfg.model.restore_from_path,
        trainer=trainer,
        override_config_path=peft_model_cfg,
        save_restore_connector=save_restore_connector,
    )
    config = OmegaConf.to_container(cfg.inference, resolve=True)
    model.set_inference_config(config)
    model.freeze()

    if cfg.evaluate_metric:
        trainer.test(model)
        exit(0)

    _test_ds = model._build_dataset(peft_model_cfg.data.test_ds, is_train=False)
    if isinstance(_test_ds, list):
        _test_ds = _test_ds[0]

    request_dl = DataLoader(
        dataset=_test_ds, batch_size=peft_model_cfg.data.test_ds.global_batch_size, collate_fn=_test_ds.collate_fn,
    )
    response = trainer.predict(model, request_dl)

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

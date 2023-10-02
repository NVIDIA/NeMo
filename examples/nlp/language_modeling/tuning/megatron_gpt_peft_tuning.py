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

import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronLMPPTrainerBuilder
from nemo.collections.nlp.parts.peft_config import PEFT_CONFIG_MAP

from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

mp.set_start_method("spawn", force=True)

"""
This is the script to finetuning a GPT Model with any PEFT method.
A base GPT Model is required as a starting point. This script will then insert
Adapters into each Transformer layer and will train/update only these adapters
during training. The base GPT Model weights will remain frozen.

During training this script will only save the newly trained Adapter weights
in checkpoints. At the end of training a .nemo file of Adapter weights will 
be saved.

Usage:
    Assuming the base model is a 125m GPT Model, with TP=1, PP=1:
    a. run a training run for a base gpt nemo file:
        python megatron_gpt_peft_tuning.py \
            "model.data.train_ds.file_names=[PATH TO TRAINING JSONL FILE]",
            "model.data.train_ds.concat_sampling_probabilities=[SAMPLING VAL]",
            "model.data.validation_ds.file_names=[PATH TO VALIDATION JSONL FILE]",
            "model.data.validation_ds.names=[NAME FOR METRIC LOGGING]",
            model.restore_from_path="PATH TO BASE GPT MODEL .nemo FILE"
            model.peft.peft_scheme='lora'  # lora, ptuning, adapter, ia3, or none for full fineutning
            name="NAME OF TRAINING RUN"
            exp_manager.exp_dir="DIR TO SAVE CHECKPOINTS and .nemo FILE",
Please see lora.ipynb for a step-by-step guide.
"""


def _modify_config(gpt_cfg, cfg, add_cfg_to_tree=False):
    """
    This function modifies the original gpt pre-training config (gpt_cfg) with attributes from the finetuning config (cfg).
    The `add_cfg_to_tree` arg adds `cfg` to the top of the yaml tree which is needed for all `hparams.yaml` files when passed as an arg to `load_from_checkpoint()`.
    """
    OmegaConf.set_struct(gpt_cfg, True)
    OmegaConf.resolve(cfg)
    with open_dict(gpt_cfg):
        gpt_cfg.megatron_amp_O2 = cfg.model.get('megatron_amp_O2', False)
        gpt_cfg.micro_batch_size = cfg.model.data.train_ds.micro_batch_size
        gpt_cfg.global_batch_size = cfg.model.data.train_ds.global_batch_size
        gpt_cfg.sequence_parallel = cfg.model.get("sequence_parallel", False)
        gpt_cfg.activations_checkpoint_granularity = cfg.model.get("activations_checkpoint_granularity", None)
        gpt_cfg.activations_checkpoint_num_layers = cfg.model.get("activations_checkpoint_num_layers", None)
        gpt_cfg.activations_checkpoint_method = cfg.model.get("activations_checkpoint_method", None)
        gpt_cfg.activations_checkpoint_layers_per_pipeline = cfg.model.get(
            "activations_checkpoint_layers_per_pipeline", None
        )
        gpt_cfg.data = cfg.model.data
        gpt_cfg.optim = cfg.model.optim
        gpt_cfg.precision = cfg.trainer.precision
        gpt_cfg.answer_only_loss = cfg.model.answer_only_loss
        gpt_cfg.restore_from_path = cfg.model.restore_from_path
        gpt_cfg.resume_from_checkpoint = cfg.model.resume_from_checkpoint
        gpt_cfg.save_nemo_on_validation_end = cfg.model.save_nemo_on_validation_end
        gpt_cfg.gradient_as_bucket_view = cfg.model.gradient_as_bucket_view
        gpt_cfg.hidden_dropout = cfg.model.get('hidden_dropout', 0.0)
        gpt_cfg.attention_dropout = cfg.model.get('attention_dropout', 0.0)
        gpt_cfg.ffn_dropout = cfg.model.ffn_dropout
        gpt_cfg.peft = cfg.model.peft
        peft_cls = _get_peft_scheme(cfg.model)
        gpt_cfg.target = f"{peft_cls.__module__}.{peft_cls.__name__}"
        gpt_cfg.tensor_model_parallel_size = cfg.model.get('tensor_model_parallel_size', 1)
        gpt_cfg.pipeline_model_parallel_size = cfg.model.get('pipeline_model_parallel_size', 1)
        gpt_cfg.pipeline_model_parallel_split_rank = cfg.model.get('pipeline_model_parallel_split_rank', 0)

        # This is needed when modifying a hparam file directly to load `.ckpt` files.
        # This is not needed to modify the cfg in `.nemo` files.
        if add_cfg_to_tree:
            OmegaConf.resolve(gpt_cfg)
            gpt_cfg.cfg = gpt_cfg

    return gpt_cfg


def _get_peft_scheme(cfg):
    if cfg.peft.peft_scheme == "adapter":
        if cfg.peft.adapter_tuning.weight_tying:
            peft_cls = MegatronGPTAdapterModelWeightTying
        else:
            peft_cls = MegatronGPTAdapterModel
    elif cfg.peft.peft_scheme == "ia3":
        peft_cls = MegatronGPTIA3Model
    elif cfg.peft.peft_scheme == "ptuning":
        peft_cls = MegatronGPTPTuningModel
    elif cfg.peft.peft_scheme == "adapter_and_ptuning":
        peft_cls = MegatronGPTAdapterPTuningModel
    elif cfg.peft.peft_scheme == "lora":
        if cfg.peft.lora_tuning.weight_tying:
            peft_cls = MegatronGPTLoRAModelWeightTying
        else:
            peft_cls = MegatronGPTLoRAModel
    else:
        raise RuntimeError("Invalid Peft scheme")
    return peft_cls


def load_from_checkpoint_dir(cls, cfg, trainer, modify_confg_fn):
    app_state = AppState()
    if cfg.model.tensor_model_parallel_size > 1 or cfg.model.pipeline_model_parallel_size > 1:
        app_state.model_parallel_size = cfg.model.tensor_model_parallel_size * cfg.model.pipeline_model_parallel_size
        app_state.tensor_model_parallel_size = cfg.model.tensor_model_parallel_size
        app_state.pipeline_model_parallel_size = cfg.model.pipeline_model_parallel_size
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
            tensor_model_parallel_size_=cfg.model.tensor_model_parallel_size,
            pipeline_model_parallel_size_=cfg.model.pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank_=cfg.model.pipeline_model_parallel_split_rank,
        )
    checkpoint_path = inject_model_parallel_rank(
        os.path.join(cfg.model.pretrained_checkpoint.checkpoint_dir, cfg.model.pretrained_checkpoint.checkpoint_name)
    )
    hparams_file = OmegaConf.load(cfg.model.pretrained_checkpoint.hparams_file)
    gpt_cfg = modify_confg_fn(hparams_file.cfg, cfg, add_cfg_to_tree=True)
    with tempfile.NamedTemporaryFile(suffix='.yaml') as f:
        OmegaConf.save(config=gpt_cfg, f=f.name)
        model = cls.load_from_checkpoint(checkpoint_path=checkpoint_path, trainer=trainer, hparams_file=f.name,)
        return model


def validate_checkpoint_loading_args(cfg):
    if cfg.checkpoint_dir is None or not os.path.isdir(cfg.checkpoint_dir):
        raise ValueError(f'Checkpoint directory {cfg.checkpoint_dir} does not exist or is not a directory.')
    if cfg.checkpoint_name is None:
        raise ValueError(f'Checkpoint name {cfg.checkpoint_name} is not valid.')
    if cfg.hparams_file is None or not os.path.isfile(cfg.hparams_file):
        raise ValueError(f'Hparams file {cfg.hparams_file} does not exist or is not a file.')


@hydra_runner(config_path="conf", config_name="megatron_gpt_peft_tuning_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    trainer = MegatronLMPPTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)

    model_cfg = MegatronGPTSFTModel.merge_cfg_with(cfg.model.restore_from_path, cfg)
    model = MegatronGPTSFTModel.restore_from(cfg.model.restore_from_path, model_cfg, trainer=trainer)
    peft_cfg_cls = PEFT_CONFIG_MAP[cfg.model.peft.peft_scheme]

    if cfg.model.peft.restore_from_path is not None:
        # initialize peft weights from a checkpoint instead of randomly
        # This is not the same as resume training because optimizer states are not restored.
        logging.info("PEFT Weights will be loaded from", cfg.model.peft.restore_from_path)
        model.load_adapters(cfg.model.peft.restore_from_path, peft_cfg_cls(model_cfg))
    elif peft_cfg_cls is not None:
        logging.info("Adding adapter weights to the model for PEFT")
        model.add_adapter(peft_cfg_cls(model_cfg))
    else:
        logging.info(f"Running full finetuning since no peft scheme is given.\n{model.summarize()}")

    trainer.fit(model)


if __name__ == '__main__':
    main()

# # Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# import torch.multiprocessing as mp
# from omegaconf.omegaconf import OmegaConf, open_dict

# from nemo.collections.nlp.models.information_retrieval.megatron_bert_embedding_model import MegatronBertEmbeddingModel
# from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronBertTrainerBuilder
# from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
# from nemo.core.config import hydra_runner
# from nemo.utils import logging
# from nemo.utils.exp_manager import exp_manager


# @hydra_runner(config_path="conf", config_name="megatron_bert_embedding_config")
# def main(cfg) -> None:
#     if cfg.model.data.dataloader_type != "LDDL":
#         mp.set_start_method("spawn", force=True)

#     logging.info("\n\n************** Experiment configuration ***********")
#     logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

#     trainer = MegatronBertTrainerBuilder(cfg).create_trainer()
#     exp_manager(trainer, cfg.exp_manager)

#     model_cfg = MegatronBertEmbeddingModel.merge_cfg_with(cfg.restore_from_path, cfg)

#     assert (
#         model_cfg.micro_batch_size * cfg.trainer.devices == model_cfg.global_batch_size
#     ), "Gradiant accumulation is not supported for contrastive learning yet"

#     OmegaConf.set_struct(model_cfg, True)
#     with open_dict(model_cfg):
#         model_cfg.precision = trainer.precision

#     logging.info(f"Loading model from {cfg.restore_from_path}")
#     model = MegatronBertEmbeddingModel.restore_from(
#         restore_path=cfg.restore_from_path,
#         trainer=trainer,
#         save_restore_connector=NLPSaveRestoreConnector(),
#         override_config_path=model_cfg,
#         strict=True,
#     )

#     trainer.fit(model)


# if __name__ == '__main__':
#     main()



# # ================================================== #
# #!/bin/bash

# PROJECT= # wandb project name
# NAME= # wandb run name
# export WANDB_API_KEY= # your_wandb_key

# NUM_DEVICES=1 # number of gpus to train on
# CONFIG_PATH="/NeMo/examples/nlp/information_retrieval/conf/"
# CONFIG_NAME="megatron_bert_embedding_config"
# PATH_TO_NEMO_MODEL= # Path to conveted nemo model from hf
# TRAIN_DATASET_PATH= # Path to json dataset 
# VALIDATION_DATASET_PATH= # Path to validation dataset 
# SAVE_DIR= # where the checkpoint and logs are saved
# mkdir -p $SAVE_DIR
# export NVTE_FLASH_ATTN=0
# export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
# export NVTE_FUSED_ATTN=0

# python NeMo/examples/nlp/information_retrieval/megatron_bert_embedding_finetuning.py \
# --config-path=${CONFIG_PATH} \
# --config-name=${CONFIG_NAME} \
# restore_from_path=${PATH_TO_NEMO_MODEL} \
# trainer.devices=${NUM_DEVICES} \
# trainer.max_steps=10000 \
# trainer.val_check_interval=100 \
# trainer.max_epochs=1 \
# +trainer.num_sanity_val_steps=0 \
# model.mcore_bert=True \
# model.post_process=False \
# model.global_batch_size=8 \ # should be NUM_DEVICES * model.micro_batch_size
# model.micro_batch_size=8 \
# model.optim.lr=0.000005 \
# model.optim.sched.min_lr=0.00000001 \
# model.optim.sched.warmup_steps=100 \
# model.encoder_seq_length=512 \
# model.tokenizer.library="huggingface" \
# model.tokenizer.type="intfloat/e5-large-unsupervised" \
# model.data.data_train=${TRAIN_DATASET_PATH} \
# model.data.data_validation=${VALIDATION_DATASET_PATH} \
# model.data.hard_negatives_to_train=4 \
# exp_manager.explicit_log_dir=${SAVE_DIR} \
# exp_manager.create_wandb_logger=True \
# exp_manager.resume_if_exists=True \
# exp_manager.wandb_logger_kwargs.name=${NAME} \
# exp_manager.wandb_logger_kwargs.project=${PROJECT}




from nemo.collections.nlp.models.information_retrieval.megatron_bert_embedding_model import MegatronBertEmbeddingModel
from pytorch_lightning.trainer.trainer import Trainer
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids

# load checkpoint
trainer_config = {
    "devices": 1,
    "num_nodes": 1,
    "accelerator": "gpu",
    "logger": False,
    "precision": 'bf16-mixed'
}
trainer = Trainer(strategy=NLPDDPStrategy(), **trainer_config)
model_file = '/lustre/fsw/coreai_dlalgo_genai/ataghibakhsh/checkpoints/bert_nemo.nemo'
model_cfg = MegatronBertEmbeddingModel.restore_from(restore_path=model_file, trainer=trainer, return_config=True)
model_cfg.micro_batch_size = 1
model_cfg.global_batch_size = 8
model = MegatronBertEmbeddingModel.restore_from(restore_path=model_file, trainer=trainer, override_config_path=model_cfg, strict=True)

# sample data
dummy_inputs = model.input_example(max_batch=8, max_dim=256)[0]

# forward
output = model.forward(
    input_ids = dummy_inputs['input_ids'],
    attention_mask = dummy_inputs['attention_mask'],
    token_type_ids = dummy_inputs['token_type_ids']
)
print("output.shape: ", output.shape)
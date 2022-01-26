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

from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


"""
Can currently only prompt tune on one task at a time, but can
run inference with multiple soft-prompts/tasks within a batch.

Datasets should be formatted with in a json file like:
{"prompt_tag": <tag1>, "text": <text1>}
{"prompt_tag": <tag1>, "text": <text2>}
{"prompt_tag": <tag2>, "text": <text3>}

Example Usage for first prompt tuning task:

EXPR_NAME='winogrande_prompt_tuning'
RESTORE_PATH='megatron_gpt.nemo'
GPUS=1
MAX_STEPS=4800
PROMPT_LENGTH=20

echo "Prompt tuning starting"
python megatron_gpt_prompt_tuning.py \
        --config-name=megatron_gpt_config \
        trainer.gpus=$GPUS \
        trainer.max_steps=$MAX_STEPS \
        restore_from_path=$RESTORE_PATH \
        exp_manager.name=$EXPR_NAME \
        exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
        +model.use_soft_prompts=True \
        +model.num_prompt_tokens=$PROMPT_LENGTH \
        +model.existing_prompt_tags=[] \
        +model.new_prompt_tags=['Winogrande'] \
        +model.new_prompt_init_text=['disambiguate pronoun noun names pick correct name fill blank'] \
        +model.new_prompt_init_methods=['text'] \
        model.data.data_prefix=None \
        +model.data.train_ds='winogrande_prompt_tuning_train.jsonl' \
        +model.data.valid_ds='winogrande_prompt_tuning_val.jsonl' \
        +model.data.batch_size=32 \
        model.optim.lr=2e-3 \
        model.optim.sched.min_lr=2e-6 \
        model.optim.sched.warmup_steps=320 \
        model.optim.sched.constant_steps=2240 \
        model.encoder_seq_length=2048


Example Usage for second prompt tuning task:
Be sure to update model.exsiting_prompt_tags with tags from previous prompt tuning session
and to use the .nemo file saved at the end of the last prompt tuning session

EXPR_NAME='rte_prompt_tuning'
RESTORE_PATH='winograde_megatron_gpt.nemo'
GPUS=1
MAX_STEPS=780
PROMPT_LENGTH=20
VAL_CHECK_INTERVAL=50

echo "Prompt tuning starting"
python megatron_gpt_prompt_tuning.py \
        --config-name=megatron_gpt_config \
        trainer.gpus=$GPUS \
        trainer.max_steps=$MAX_STEPS \
        trainer.val_check_interval=$VAL_CHECK_INTERVAL \
        restore_from_path=$RESTORE_PATH \
        exp_manager.name=$EXPR_NAME \
        exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
        +model.use_soft_prompts=True \
        +model.num_prompt_tokens=$PROMPT_LENGTH \
        +model.existing_prompt_tags=['Winogrande'] \
        +model.new_prompt_tags=['RTE'] \
        +model.new_prompt_init_text=['entailment cause relationship imply label text'] \
        +model.new_prompt_init_methods=['text'] \
        model.data.data_prefix=None \
        +model.data.train_ds='RTE_prompt_tuning_train.jsonl' \
        +model.data.valid_ds='RTE_prompt_tuning_val.jsonl' \
        +model.data.batch_size=32 \
        model.optim.lr=2e-4 \
        model.optim.sched.min_lr=2e-6 \
        model.optim.sched.warmup_steps=78 \
        model.optim.sched.constant_steps=545 \
        model.encoder_seq_length=2048


Example Usage for third prompt tuning task:
Be sure to update model.exsiting_prompt_tags with tags from previous prompt tuning sessions
and to use the .nemo file saved at the end of the last prompt tuning sessions

EXPR_NAME='boolq_prompt_tune'
GPUS=1
MAX_STEPS=2950
PROMPT_LENGTH=20
RESTORE_PATH='rte_winogrande_megatron_gpt.nemo'

echo "Prompt tuning starting"
python megatron_gpt_prompt_tuning.py \
        --config-name=megatron_gpt_config \
        trainer.gpus=$GPUS \
        trainer.max_steps=$MAX_STEPS \
        exp_manager.name=$EXPR_NAME \
        exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
        restore_from_path=$RESTORE_PATH \
        +model.use_soft_prompts=True \
        +model.num_prompt_tokens=$PROMPT_LENGTH \
        +model.existing_prompt_tags=['Winogrande, RTE'] \
        +model.new_prompt_tags=['BoolQ'] \
        +model.new_prompt_init_text=['true false question answer reading comprehension'] \
        +model.new_prompt_init_methods=['text'] \
        +model.calc_loss_on_answer_only=False \
        model.data.data_prefix=None \
        +model.data.train_ds='boolq_prompt_tuning_train.jsonl' \
        +model.data.valid_ds='boolq_prompt_tuning_val.jsonl' \
        +model.data.batch_size=32 \
        model.optim.lr=2e-4 \
        model.optim.sched.min_lr=2e-6 \
        model.optim.sched.warmup_steps=295 \
        model.optim.sched.constant_steps=2063 \
        model.encoder_seq_length=2048

"""


@hydra_runner(config_path="conf", config_name="megatron_gpt_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    plugins = [NLPDDPPlugin(num_nodes=cfg.trainer.num_nodes)]

    trainer = Trainer(plugins=plugins, **cfg.trainer)

    exp_manager(trainer, cfg.exp_manager)

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision

    model = MegatronGPTModel.restore_from(cfg.restore_from_path, cfg.model, trainer=trainer)

    # Init all new prompts
    for idx, tag in enumerate(cfg.model.new_prompt_tags):
        init_method = cfg.model.new_prompt_init_methods[idx]

        if init_method == "text":
            init_text = cfg.model.new_prompt_init_text[idx]
            model.init_prompt_from_text(tag, init_text)

        elif init_method == 'random':
            model.init_prompt_from_random(tag)

        else:
            logging.info(f'\n Soft prompt init method {init_method} is not recognized, please use text or random')

    logging.info(f'\nCurrent soft prompts include {model.get_prompt_table()}')
    trainer.fit(model)


if __name__ == '__main__':
    main()

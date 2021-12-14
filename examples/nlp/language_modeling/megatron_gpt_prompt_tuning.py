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
Example Usage:

GPUS=1
MAX_STEPS=3000
PROMPT_LENGTH=10
RESTORE_PATH='megatron_gpt.nemo'

echo "Prompt tuning starting"
python megatron_gpt_prompt_tuning.py \
        --config-name=megatron_gpt_config \
        trainer.gpus=$GPUS \
        trainer.max_steps=$MAX_STEPS \
        restore_from_path=$RESTORE_PATH \
        +model.use_soft_prompts=True \
        +model.prompt_length=$PROMPT_LENGTH \
        +model.new_prompt_tags=['NER-Yes-No, NER-Complete'] \
        +model.new_prompt_init_text=['named entities yes no, None'] \
        +model.new_prompt_init_methods=['text, random'] \
        model.data.data_prefix=None \
        +model.data.train_ds='prompt_tuning_ner_train.json' \
        +model.data.valid_ds='prompt_tuning_ner_val.json' \
        +model.data.test_ds='prompt_tuning_ner_test.json' \
        +model.data.batch_size=32 \
        model.optim.lr=2e-3 \
        model.optim.sched.min_lr=2e-6 \
        model.optim.sched.warmup_steps=200 \
        model.optim.sched.constant_steps=1000 \
        model.encoder_seq_length=2048

"""


@hydra_runner(config_path="conf", config_name="megatron_gpt_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    plugins = [NLPDDPPlugin(num_nodes=cfg.trainer.num_nodes)]

    trainer = Trainer(plugins=plugins, **cfg.trainer)
    exp_manager(trainer, cfg.exp_manager)
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

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


from omegaconf import DictConfig, OmegaConf

from nemo import lightning as nl
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm.api import _setup, finetune
from nemo.collections.speechlm.data.audio_to_text_module import AudioToTextDataModule
from nemo.collections.speechlm.models.speech_to_text_llm_model import SpeechToTextLLM, SpeechToTextLLMConfig
from nemo.collections.speechlm.modules.asr_module import ASRModuleConfig
from nemo.collections.speechlm.modules.modality_adapter import ModalityAdapterConfig
from nemo.collections.speechlm.utils import SpeechToTextLLMPEFT, get_object_list_from_config
from nemo.core.classes.common import Serialization, typecheck
from nemo.lightning.pytorch.callbacks import PreemptionCallback
from nemo.utils import logging
from nemo.utils.exp_manager import StatelessTimer, TimingCallback


def speech_to_text_llm_train(cfg: DictConfig):
    """Train the model using provided config."""
    typecheck.set_typecheck_enabled(enabled=False)  # disable typechecks from NeMo 1.x
    cfg = OmegaConf.to_container(cfg, resolve=True)

    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    # 1. build the model
    tokenizer = AutoTokenizer(cfg['model']['llm']['pretrained_model'])
    model_config = SpeechToTextLLMConfig(
        language_model_class=cfg['model']['llm']['_target_'],
        language_model_config=Serialization.from_config_dict(cfg['model']['llm']['config']),
        speech_model_config=ASRModuleConfig(**cfg['model']['speech_encoder']),
        modality_adapter_config=ModalityAdapterConfig(**cfg['model']['modality_adapter']),
        language_model_from_pretrained=cfg['model']['llm']['pretrained_model'],
        freeze_language_model=cfg['model']['freeze_language_model'],
        freeze_speech_model=cfg['model']['freeze_speech_model'],
        freeze_modality_adapter=cfg['model']['freeze_modality_adapter'],
        data_config=cfg['data']['common'],
    )

    model = SpeechToTextLLM(config=model_config, tokenizer=tokenizer)

    # 2. build dataset
    data = AudioToTextDataModule(cfg['data'], tokenizer=tokenizer)

    # 3. setup the optimizer
    optim = Serialization.from_config_dict(cfg['optim'])

    # 4. setup trainer
    callbacks = get_object_list_from_config(cfg['callbacks'])
    if cfg.get('max_time_per_run', None):
        if cfg['strategy'].get('ckpt_async_save', True):
            raise ValueError(
                f"`strategy.ckpt_async_save` must be `False` to save ckpt when `max_time_per_run` is set,",
                f"got {cfg['strategy']['ckpt_async_save']}. `max_time_per_run` will not work in this case!",
            )
        else:
            # ckpt_async_save must be False to save ckpt when training is interrupted by max_time_per_run
            logging.info(f"Setting max_time_per_run={cfg['max_time_per_run']} for the training job.")
            callbacks.append(StatelessTimer(cfg['max_time_per_run']))
    else:
        callbacks.append(PreemptionCallback())
    callbacks.append(TimingCallback())

    trainer = nl.Trainer(
        strategy=Serialization.from_config_dict(cfg['strategy']),
        plugins=get_object_list_from_config(cfg['plugins']),
        callbacks=callbacks,
        **cfg['trainer'],
    )

    # 5. setup PEFT
    peft = None
    if cfg['model'].get('peft', None):
        peft = SpeechToTextLLMPEFT(peft=Serialization.from_config_dict(cfg['model']['peft']))

    # 6. setup logger and auto-resume
    resume = Serialization.from_config_dict(cfg['resume'])
    logger = Serialization.from_config_dict(cfg['logger'])

    # 7. train the model
    finetune(
        model=model,
        data=data,
        trainer=trainer,
        optim=optim,
        log=logger,
        peft=peft,
        resume=resume,
    )
    return logger.log_dir


def speech_to_text_llm_validate(cfg: DictConfig):
    """Validate the model using provided config, groundtruth required."""
    typecheck.set_typecheck_enabled(enabled=False)  # disable typechecks from NeMo 1.x
    cfg = OmegaConf.to_container(cfg, resolve=True)
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    # 1. build the model
    tokenizer = AutoTokenizer(cfg['model']['llm']['pretrained_model'])
    model_config = SpeechToTextLLMConfig(
        language_model_class=cfg['model']['llm']['_target_'],
        language_model_config=Serialization.from_config_dict(cfg['model']['llm']['config']),
        speech_model_config=ASRModuleConfig(**cfg['model']['speech_encoder']),
        modality_adapter_config=ModalityAdapterConfig(**cfg['model']['modality_adapter']),
        language_model_from_pretrained=cfg['model']['llm']['pretrained_model'],
        freeze_language_model=cfg['model']['freeze_language_model'],
        freeze_speech_model=cfg['model']['freeze_speech_model'],
        freeze_modality_adapter=cfg['model']['freeze_modality_adapter'],
    )

    model = SpeechToTextLLM(config=model_config, tokenizer=tokenizer)

    # 2. build dataset
    data = AudioToTextDataModule(cfg['data'], tokenizer=tokenizer)

    # 3. setup the optimizer
    optim = Serialization.from_config_dict(cfg['optim'])

    # 4. setup trainer
    trainer = nl.Trainer(
        strategy=Serialization.from_config_dict(cfg['strategy']),
        plugins=get_object_list_from_config(cfg['plugins']),
        callbacks=get_object_list_from_config(cfg['callbacks']),
        **cfg['trainer'],
    )

    # 5. setup PEFT
    peft = None
    if cfg['model'].get('peft', None):
        peft = SpeechToTextLLMPEFT(peft=Serialization.from_config_dict(cfg['model']['peft']))

    # 6. setup logger and auto-resume
    resume = Serialization.from_config_dict(cfg['resume'])
    logger = Serialization.from_config_dict(cfg['logger'])

    # 7. run the inference
    app_state = _setup(
        model=model,
        data=data,
        trainer=trainer,
        log=logger,
        resume=resume,
        optim=optim,
        tokenizer=tokenizer,
        model_transform=peft,
    )

    trainer.validate(model, datamodule=data)

    return app_state.log_dir


def speech_to_text_llm_generate(cfg: DictConfig):
    """Running inference using provided config without groundtruth."""
    # TODO: implement this based on llm.generate()
    raise NotImplementedError("This function is not implemented yet.")

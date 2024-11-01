from copy import deepcopy

import torch
from megatron.core import dist_checkpointing
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.llm.api import _setup
from nemo.collections.llm.gpt.model.base import GPTModel
from nemo.collections.speechlm.data.audio_to_text_data import AudioToTextDataModule
from nemo.collections.speechlm.models.speech_to_text_llm_model import SpeechToTextLLM, SpeechToTextLLMConfig
from nemo.collections.speechlm.modules.asr_module import ASRModuleConfig
from nemo.collections.speechlm.modules.modality_adapter import ModalityAdapterConfig
from nemo.collections.speechlm.utils import SpeechToTextLLMPEFT, get_object_list_from_config
from nemo.core.classes.common import Serialization, typecheck
from nemo.core.config import hydra_runner
from nemo.utils import logging, model_utils


def speech_to_text_llm_train(cfg: DictConfig):
    typecheck.set_typecheck_enabled(enabled=False)  # disable typechecks from NeMo 1.x
    cfg = OmegaConf.to_container(cfg, resolve=True)
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    # 1. build the model

    ## retain common settings in data config
    data_config = deepcopy(cfg['data'])
    for key in ['train_ds', 'validation_ds', 'test_ds']:
        if key in data_config:
            data_config.pop(key)

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
        data_config=data_config,
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

    # 7. train the model
    llm.finetune(
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

    raise NotImplementedError("This function is not implemented yet.")

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

    trainer.predict(model, datamodule=data)

    return app_state.log_dir

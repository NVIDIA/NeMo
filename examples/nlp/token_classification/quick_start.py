import os
import pytorch_lightning as pl
from omegaconf import OmegaConf

import nemo.collections.nlp as nemo_nlp
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

@hydra_runner(config_path="conf", config_name="quick_start_config")
def main(cfg) -> None:
    # Setup logging
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    # Setup trainer
    trainer = pl.Trainer(**cfg.trainer)

    # Setup model
    model = nemo_nlp.models.TokenClassificationModel(cfg=cfg.model, trainer=trainer)

    # Setup data
    train_data = nemo_nlp.data.TokenClassificationDataLayer(
        tokenizer=model.tokenizer,
        dataset=cfg.model.train_ds.dataset,
        max_seq_length=cfg.model.train_ds.max_seq_length,
        batch_size=cfg.model.train_ds.batch_size,
        shuffle=True,
    )

    val_data = nemo_nlp.data.TokenClassificationDataLayer(
        tokenizer=model.tokenizer,
        dataset=cfg.model.validation_ds.dataset,
        max_seq_length=cfg.model.validation_ds.max_seq_length,
        batch_size=cfg.model.validation_ds.batch_size,
        shuffle=False,
    )

    # Setup callbacks
    callbacks = []
    if cfg.model.train_ds.use_cache:
        callbacks.append(nemo_nlp.callbacks.TokenClassificationDataCallback())

    # Add OneLogger callback if available
    try:
        from nemo.lightning import OneLoggerNeMoCallback
        from nemo.utils.meta_info_manager import MetaInfoManager
        
        one_logger_cb = OneLoggerNeMoCallback(
            callback_config=MetaInfoManager(cfg).get_metadata(),
            perf_tag="nemo_test",
            session_tag="quick_start",
            log_every_n_iterations=10,
        )
        callbacks.append(one_logger_cb)
        logging.info("Added OneLogger callback for telemetry")
    except ImportError:
        logging.warning("OneLogger dependencies not found. Skipping telemetry setup.")

    # Train
    trainer.fit(model)

if __name__ == "__main__":
    main() 
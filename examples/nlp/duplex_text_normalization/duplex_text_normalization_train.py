from omegaconf import DictConfig, OmegaConf
from utils import TAGGER_MODEL, DECODER_MODEL, initialize_model_and_trainer

from nemo.utils import logging
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager

@hydra_runner(config_path="conf", config_name="duplex_tn_config")
def main(cfg: DictConfig) -> None:
    logging.info(f'Config Params: {OmegaConf.to_yaml(cfg)}')

    # Train the tagger
    if cfg.tagger_model.do_training:
        logging.info("================================================================================================")
        logging.info('Starting training tagger...')
        tagger_trainer, tagger_model = initialize_model_and_trainer(cfg, TAGGER_MODEL, True)
        exp_manager(tagger_trainer, cfg.get('tagger_exp_manager', None))
        tagger_trainer.fit(tagger_model)
        if cfg.tagger_model.nemo_path:
            tagger_model.save_to(cfg.tagger_model.nemo_path)
        logging.info('Training finished!')

    # Train the decoder
    if cfg.decoder_model.do_training:
        logging.info("================================================================================================")
        logging.info('Starting training decoder...')
        decoder_trainer, decoder_model = initialize_model_and_trainer(cfg, DECODER_MODEL, True)
        exp_manager(decoder_trainer, cfg.get('decoder_exp_manager', None))
        decoder_trainer.fit(decoder_model)
        if cfg.decoder_model.nemo_path:
            decoder_model.save_to(cfg.decoder_model.nemo_path)
        logging.info('Training finished!')

if __name__ == '__main__':
    main()

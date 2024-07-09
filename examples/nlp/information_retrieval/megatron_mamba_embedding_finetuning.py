import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict

from nemo.collections.nlp.models.information_retrieval.megatron_mamba_embedding_model import (
    MegatronMambaEmbeddingModel,
)
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector

from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

mp.set_start_method("spawn", force=True)


@hydra_runner(config_path="conf", config_name="megatron_bert_embedding_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    trainer = MegatronTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)

    logging.info(f"Loading model from {cfg.restore_from_path}")
    model = MegatronMambaEmbeddingModel.restore_from(
        restore_path=cfg.restore_from_path,
        trainer=trainer,
        save_restore_connector=NLPSaveRestoreConnector(),
        override_config_path=cfg,
        strict=True,
    )

    trainer.fit(model)


if __name__ == '__main__':
    main()

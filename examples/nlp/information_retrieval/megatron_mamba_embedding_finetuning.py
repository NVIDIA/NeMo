import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from nemo.collections.nlp.models.information_retrieval.megatron_mamba_embedding_model import (
    MegatronMambaEmbeddingModel,
)
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronLMPPTrainerBuilder
from nemo.core.config import hydra_runner
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

mp.set_start_method("spawn", force=True)


@hydra_runner(config_path="conf", config_name="megatron_mamba_embedding_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    precision = cfg.trainer.precision
    trainer = MegatronLMPPTrainerBuilder(cfg).create_trainer()

    # Restore the precision value after Trainer is built.
    cfg.trainer.precision = precision
    exp_manager(trainer, cfg.exp_manager)

    model_cfg = MegatronMambaEmbeddingModel.merge_cfg_with(cfg.restore_from_path, cfg)
    assert (
        model_cfg.micro_batch_size * cfg.trainer.devices * cfg.trainer.num_nodes / model_cfg.tensor_model_parallel_size == \
            model_cfg.global_batch_size
    ), f"Gradiant accumulation is not supported for contrastive learning yet."

    logging.info(f"Loading model from {cfg.restore_from_path}")
    model = MegatronMambaEmbeddingModel.restore_from(
        restore_path=cfg.restore_from_path, trainer=trainer, override_config_path=model_cfg, strict=True
    )
    model._save_restore_connector = SaveRestoreConnector()

    trainer.fit(model)


if __name__ == '__main__':
    main()

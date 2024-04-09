import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict

from nemo.collections.nlp.models.language_modeling.megatron_griffin_sft_model import MegatronGriffinSFTModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronLMPPTrainerBuilder
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy



@hydra_runner(config_path="conf", config_name="megatron_griffin_finetuning_config")
def main(cfg) -> None:

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    precision = cfg.trainer.precision
    trainer = MegatronLMPPTrainerBuilder(cfg).create_trainer()
    trainer.num_sanity_val_steps=0 
    # Restore the precision value after Trainer is built.
    cfg.trainer.precision = precision
    exp_manager(trainer, cfg.exp_manager)

    model_cfg = MegatronGriffinSFTModel.merge_cfg_with(cfg.model.restore_from_path, cfg)
    model = MegatronGriffinSFTModel.restore_from(cfg.model.restore_from_path, model_cfg, trainer=trainer)
    # model = MegatronGriffinSFTModel(cfg.model, trainer)
    
    trainer.fit(model)


if __name__ == '__main__':
    main()

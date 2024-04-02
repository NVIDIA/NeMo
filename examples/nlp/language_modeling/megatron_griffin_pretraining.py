import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict

from nemo.collections.nlp.models.language_modeling.megatron_griffin_model import MegatronGriffinModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from pytorch_lightning.trainer.trainer import Trainer
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy



@hydra_runner(config_path="conf", config_name="megatron_griffin_config")
def main(cfg) -> None:
    if cfg.model.data.dataloader_type != "LDDL":
        mp.set_start_method("spawn", force=True)

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    trainer = Trainer(
    strategy=NLPDDPStrategy(),
    devices=-1,
    accelerator="gpu",
    num_nodes=1,
    precision="bf16",
    logger=False,
    enable_checkpointing=False,
    use_distributed_sampler=False,
)
    exp_manager(trainer, cfg.exp_manager)

    model = MegatronGriffinModel(cfg.model, trainer).half()

    trainer.fit(model)


if __name__ == '__main__':
    main()

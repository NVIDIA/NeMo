from megatron.core.optimizer import MegatronOptimizer, get_megatron_optimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

from nemo.tron.config import ConfigContainer


def setup_optimizer(cfg: ConfigContainer, model, no_weight_decay_cond=None, scale_lr_cond=None, lr_mult=1.0):
    optimizer = get_megatron_optimizer(cfg.optimizer_config, model, no_weight_decay_cond, scale_lr_cond, lr_mult)
    scheduler = _get_scheduler(cfg, optimizer)

    return optimizer, scheduler


def _get_scheduler(cfg: ConfigContainer, optimizer: MegatronOptimizer):
    scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=cfg.megatron_lm_config.lr_warmup_init,
        max_lr=cfg.optimizer_config.lr,
        min_lr=cfg.optimizer_config.min_lr,
        lr_warmup_steps=cfg.scheduler_config.lr_warmup_steps,
        lr_decay_steps=cfg.scheduler_config.lr_decay_steps,
        lr_decay_style=cfg.megatron_lm_config.lr_decay_style,
        start_wd=cfg.megatron_lm_config.start_weight_decay,
        end_wd=cfg.megatron_lm_config.end_weight_decay,
        wd_incr_steps=cfg.scheduler_config.wd_incr_steps,
        wd_incr_style=cfg.megatron_lm_config.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=cfg.megatron_lm_config.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=cfg.megatron_lm_config.override_opt_param_scheduler,
        wsd_decay_steps=cfg.scheduler_config.wsd_decay_steps,
        lr_wsd_decay_style=cfg.megatron_lm_config.lr_wsd_decay_style,
    )

    return scheduler

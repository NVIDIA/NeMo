from megatron.core.num_microbatches_calculator import get_current_global_batch_size, update_num_microbatches
from megatron.core.optimizer import MegatronOptimizer, OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

from nemo.tron.config import FlatConfig


def setup_optimizer(cfg: FlatConfig, model, no_weight_decay_cond=None, scale_lr_cond=None, lr_mult=1.0):
    optim_cfg = cfg.to_module_cfg(OptimizerConfig)
    optimizer = get_megatron_optimizer(optim_cfg, model, no_weight_decay_cond, scale_lr_cond, lr_mult)
    scheduler = _get_scheduler(cfg, optimizer)

    return optimizer, scheduler


def _get_scheduler(cfg: FlatConfig, optimizer: MegatronOptimizer):
    # Iteration-based training.
    if cfg.train_iters:
        if cfg.lr_decay_iters is None:
            cfg.lr_decay_iters = cfg.train_iters
        lr_decay_steps = cfg.lr_decay_iters * cfg.global_batch_size
        wd_incr_steps = cfg.train_iters * cfg.global_batch_size
        wsd_decay_steps = None
        if cfg.lr_wsd_decay_iters is not None:
            wsd_decay_steps = cfg.lr_wsd_decay_iters * cfg.global_batch_size
        if cfg.lr_warmup_fraction is not None:
            lr_warmup_steps = cfg.lr_warmup_fraction * lr_decay_steps
        else:
            lr_warmup_steps = cfg.lr_warmup_iters * cfg.global_batch_size
    # Sample-based training.
    elif cfg.train_samples:
        # We need to set training iters for later use. Technically
        # we need to adjust the training samples too (due to last
        # batch being incomplete) but we leave it as is for now.
        _update_train_iters(cfg)
        if cfg.lr_decay_samples is None:
            cfg.lr_decay_samples = cfg.train_samples
        lr_decay_steps = cfg.lr_decay_samples
        wd_incr_steps = cfg.train_samples
        wsd_decay_steps = cfg.lr_wsd_decay_samples
        if cfg.lr_warmup_fraction is not None:
            lr_warmup_steps = cfg.lr_warmup_fraction * lr_decay_steps
        else:
            lr_warmup_steps = cfg.lr_warmup_samples
    else:
        raise Exception('either train-iters or train-samples should be provided.')
    # TODO (maanug): move above logic into config validation/post init

    scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=cfg.lr_warmup_init,
        max_lr=cfg.lr,
        min_lr=cfg.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=cfg.lr_decay_style,
        start_wd=cfg.start_weight_decay,
        end_wd=cfg.end_weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=cfg.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=cfg.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=cfg.override_opt_param_scheduler,
        wsd_decay_steps=wsd_decay_steps,
        lr_wsd_decay_style=cfg.lr_wsd_decay_style,
    )

    return scheduler


def _update_train_iters(cfg: FlatConfig):  # TODO (maanug): move into config validation/post init
    # For iteration-based training, we don't need to do anything
    if cfg.train_iters:
        return

    # Constant batch size with sample-based training.
    if cfg.rampup_batch_size is None:
        cfg.train_iters = cfg.train_samples // cfg.global_batch_size

    else:
        # Sample based training with rampup batch size.
        iterations = 0
        consumed_samples = 0
        # Rampup phase.
        while consumed_samples <= int(cfg.rampup_batch_size[2]) and consumed_samples <= cfg.train_samples:
            update_num_microbatches(consumed_samples, consistency_check=False)
            consumed_samples += get_current_global_batch_size()
            iterations += 1
        # Reset
        update_num_microbatches(0, consistency_check=False)
        # Constant phase
        # Note that we throw away any partial last batch.
        if cfg.train_samples > consumed_samples:
            iterations += (cfg.train_samples - consumed_samples) // cfg.global_batch_size
        cfg.train_iters = iterations

    print(f'setting training iterations to {cfg.train_iters}')

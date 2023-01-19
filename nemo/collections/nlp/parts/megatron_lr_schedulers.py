from nemo.core.optim.lr_scheduler import CosineAnnealing, AVAILABLE_SCHEDULERS

class CosineAnnealingExp(CosineAnnealing):
    """
    Setting max_steps_for_lr_sched for this scheduler in the config is experimental and "
    not recommended. The scheduler can use max_steps automatically from "
    trainer.max_steps.
    """
    def __init__(self, optimizer, *, max_steps, min_lr=0, last_epoch=-1, max_steps_for_lr_sched=None, **kwargs):
        super().__init__(optimizer=optimizer, max_steps=max_steps, last_epoch=last_epoch, min_lr=min_lr, **kwargs)
        if max_steps_for_lr_sched:
            self.max_steps = max_steps_for_lr_sched
            self.decay_steps = self.max_steps - (self.constant_steps + self.warmup_steps)

AVAILABLE_SCHEDULERS['CosineAnnealingExp'] = CosineAnnealingExp
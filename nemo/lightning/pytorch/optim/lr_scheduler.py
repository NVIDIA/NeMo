from typing import Optional

from nemo.core.optim.lr_scheduler import (
    InverseSquareRootAnnealing,
    MegatronAnnealing,
    NoamAnnealing,
    NoamHoldAnnealing,
    PolynomialDecayAnnealing,
    PolynomialHoldDecayAnnealing,
    SquareAnnealing,
    SquareRootAnnealing,
    T5InverseSquareRootAnnealing,
    WarmupAnnealing,
    WarmupHoldPolicy,
    WarmupPolicy,
)
from nemo.lightning.pytorch.optim.base import LRSchedulerModule


class WarmupPolicyScheduler(LRSchedulerModule):
    """Warmup Policy Learning Rate Scheduler."""

    def __init__(
        self,
        warmup_steps: int = 750,
        warmup_ratio: Optional[float] = None,
        max_steps: int = 10,
        min_lr: float = 0.0,
        interval: str = "epoch",
        frequency: int = 1,
        monitor: str = "val_loss",
    ):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.interval = interval
        self.frequency = frequency
        self.monitor = monitor

    def scheduler(self, model, optimizer):
        lr_scheduler = WarmupPolicy(
            optimizer,
            warmup_steps=self.warmup_steps,
            warmup_ratio=self.warmup_ratio,
            max_steps=self.max_steps,
            min_lr=self.min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": self.interval,
                "frequency": self.frequency,
            },
            "monitor": self.monitor,
        }


class WarmupHoldPolicyScheduler(LRSchedulerModule):
    """Warmup Hold Policy Learning Rate Scheduler."""

    def __init__(
        self,
        warmup_steps: int = 750,
        warmup_ratio: Optional[float] = None,
        hold_steps: Optional[int] = None,
        hold_ratio: Optional[float] = None,
        max_steps: int = 10,
        min_lr: float = 0.0,
        interval: str = "epoch",
        frequency: int = 1,
        monitor: str = "val_loss",
    ):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio
        self.hold_steps = hold_steps
        self.hold_ratio = hold_ratio
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.interval = interval
        self.frequency = frequency
        self.monitor = monitor

    def scheduler(self, model, optimizer):
        lr_scheduler = WarmupHoldPolicy(
            optimizer,
            warmup_steps=self.warmup_steps,
            warmup_ratio=self.warmup_ratio,
            hold_steps=self.hold_steps,
            hold_ratio=self.hold_ratio,
            max_steps=self.max_steps,
            min_lr=self.min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": self.interval,
                "frequency": self.frequency,
            },
            "monitor": self.monitor,
        }


class SquareAnnealingScheduler(LRSchedulerModule):
    """Square Annealing Learning Rate Scheduler."""

    def __init__(
        self,
        max_steps: int = 10,
        min_lr: float = 1e-5,
        interval: str = "epoch",
        frequency: int = 1,
        monitor: str = "val_loss",
    ):
        super().__init__()
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.interval = interval
        self.frequency = frequency
        self.monitor = monitor

    def scheduler(self, model, optimizer):
        lr_scheduler = SquareAnnealing(optimizer, max_steps=self.max_steps, min_lr=self.min_lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": self.interval,
                "frequency": self.frequency,
            },
            "monitor": self.monitor,
        }


class SquareRootAnnealingScheduler(LRSchedulerModule):
    """Square Root Annealing Learning Rate Scheduler."""

    def __init__(
        self,
        max_steps: int = 10,
        min_lr: float = 0.0,
        interval: str = "epoch",
        frequency: int = 1,
        monitor: str = "val_loss",
    ):
        super().__init__()
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.interval = interval
        self.frequency = frequency
        self.monitor = monitor

    def scheduler(self, model, optimizer):
        lr_scheduler = SquareRootAnnealing(optimizer, max_steps=self.max_steps, min_lr=self.min_lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": self.interval,
                "frequency": self.frequency,
            },
            "monitor": self.monitor,
        }


class NoamAnnealingScheduler(LRSchedulerModule):
    """Noam Annealing Learning Rate Scheduler."""

    def __init__(
        self,
        d_model: int,
        warmup_steps: int = 750,
        warmup_ratio: Optional[float] = None,
        max_steps: int = 10,
        min_lr: float = 0.0,
        interval: str = "epoch",
        frequency: int = 1,
        monitor: str = "val_loss",
    ):
        super().__init__()
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.interval = interval
        self.frequency = frequency
        self.monitor = monitor

    def scheduler(self, model, optimizer):
        lr_scheduler = NoamAnnealing(
            optimizer,
            d_model=self.d_model,
            warmup_steps=self.warmup_steps,
            warmup_ratio=self.warmup_ratio,
            max_steps=self.max_steps,
            min_lr=self.min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": self.interval,
                "frequency": self.frequency,
            },
            "monitor": self.monitor,
        }


class NoamHoldAnnealingScheduler(LRSchedulerModule):
    """Noam Hold Annealing Learning Rate Scheduler."""

    def __init__(
        self,
        max_steps: int = 10,
        decay_rate: float = 0.5,
        min_lr: float = 0.0,
        interval: str = "epoch",
        frequency: int = 1,
        monitor: str = "val_loss",
    ):
        super().__init__()
        self.max_steps = max_steps
        self.decay_rate = decay_rate
        self.min_lr = min_lr
        self.interval = interval
        self.frequency = frequency
        self.monitor = monitor

    def scheduler(self, model, optimizer):
        lr_scheduler = NoamHoldAnnealing(
            optimizer, max_steps=self.max_steps, decay_rate=self.decay_rate, min_lr=self.min_lr
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": self.interval,
                "frequency": self.frequency,
            },
            "monitor": self.monitor,
        }


class WarmupAnnealingScheduler(LRSchedulerModule):
    """Warmup Annealing Learning Rate Scheduler."""

    def __init__(
        self,
        max_steps: int = 10,
        min_lr: float = 0.0,
        interval: str = "epoch",
        frequency: int = 1,
        monitor: str = "val_loss",
    ):
        super().__init__()
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.interval = interval
        self.frequency = frequency
        self.monitor = monitor

    def scheduler(self, model, optimizer):
        lr_scheduler = WarmupAnnealing(optimizer, max_steps=self.max_steps, min_lr=self.min_lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": self.interval,
                "frequency": self.frequency,
            },
            "monitor": self.monitor,
        }


class InverseSquareRootAnnealingScheduler(LRSchedulerModule):
    """Inverse Square Root Annealing Learning Rate Scheduler."""

    def __init__(
        self,
        max_steps: int = 10,
        min_lr: float = 0.0,
        interval: str = "epoch",
        frequency: int = 1,
        monitor: str = "val_loss",
    ):
        super().__init__()
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.interval = interval
        self.frequency = frequency
        self.monitor = monitor

    def scheduler(self, model, optimizer):
        lr_scheduler = InverseSquareRootAnnealing(optimizer, max_steps=self.max_steps, min_lr=self.min_lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": self.interval,
                "frequency": self.frequency,
            },
            "monitor": self.monitor,
        }


class T5InverseSquareRootAnnealingScheduler(LRSchedulerModule):
    """T5 Inverse Square Root Annealing Learning Rate Scheduler."""

    def __init__(
        self,
        max_steps: int = 10,
        min_lr: float = 0.0,
        interval: str = "epoch",
        frequency: int = 1,
        monitor: str = "val_loss",
    ):
        super().__init__()
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.interval = interval
        self.frequency = frequency
        self.monitor = monitor

    def scheduler(self, model, optimizer):
        lr_scheduler = T5InverseSquareRootAnnealing(optimizer, max_steps=self.max_steps, min_lr=self.min_lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": self.interval,
                "frequency": self.frequency,
            },
            "monitor": self.monitor,
        }


class PolynomialDecayAnnealingScheduler(LRSchedulerModule):
    """Polynomial Decay Annealing Learning Rate Scheduler."""

    def __init__(
        self,
        max_steps: int = 10,
        min_lr: float = 0.0,
        power: float = 1.0,
        cycle: bool = False,
        interval: str = "epoch",
        frequency: int = 1,
        monitor: str = "val_loss",
    ):
        super().__init__()
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.power = power
        self.cycle = cycle
        self.interval = interval
        self.frequency = frequency
        self.monitor = monitor

    def scheduler(self, model, optimizer):
        lr_scheduler = PolynomialDecayAnnealing(
            optimizer, max_steps=self.max_steps, min_lr=self.min_lr, power=self.power, cycle=self.cycle
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": self.interval,
                "frequency": self.frequency,
            },
            "monitor": self.monitor,
        }


class PolynomialHoldDecayAnnealingScheduler(LRSchedulerModule):
    """Polynomial Hold Decay Annealing Learning Rate Scheduler."""

    def __init__(
        self,
        max_steps: int = 10,
        min_lr: float = 0.0,
        power: float = 1.0,
        cycle: bool = False,
        interval: str = "epoch",
        frequency: int = 1,
        monitor: str = "val_loss",
    ):
        super().__init__()
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.power = power
        self.cycle = cycle
        self.interval = interval
        self.frequency = frequency
        self.monitor = monitor

    def scheduler(self, model, optimizer):
        lr_scheduler = PolynomialHoldDecayAnnealing(
            optimizer, max_steps=self.max_steps, min_lr=self.min_lr, power=self.power, cycle=self.cycle
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": self.interval,
                "frequency": self.frequency,
            },
            "monitor": self.monitor,
        }


class CosineAnnealingScheduler(LRSchedulerModule):
    def __init__(
        self,
        max_steps=10,
        warmup_steps=750,
        constant_steps=80000,
        min_lr=int(6e-5),
        interval="epoch",
        frequency=1,
        monitor="val_loss",
    ):
        super().__init__()
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.constant_steps = constant_steps
        self.min_lr = min_lr
        self.interval = interval
        self.frequency = frequency
        self.monitor = monitor

    def scheduler(self, model, optimizer):
        from nemo.core.optim.lr_scheduler import CosineAnnealing

        lr_scheduler = CosineAnnealing(
            optimizer,
            max_steps=self.max_steps,
            warmup_steps=self.warmup_steps,
            constant_steps=self.constant_steps,
            min_lr=self.min_lr,
        )

        return {
            "optimizer": optimizer,
            "scheduler": lr_scheduler,
            "lr_scheduler": {
                # REQUIRED: The scheduler instance
                "scheduler": lr_scheduler,
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": self.interval,
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": self.frequency,
            },
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": self.monitor,
        }


class MegatronAnnealingScheduler(LRSchedulerModule):
    def __init__(
        self,
        init_lr=0.0,
        max_lr=0.00015,
        min_lr=1e-05,
        lr_warmup_steps=3200.0,
        lr_decay_steps=320000,
        lr_decay_style='cosine',
        start_wd=0.01,
        end_wd=0.01,
        wd_incr_steps=20,
        wd_incr_style='constant',
        interval="epoch",
        frequency=1,
        monitor="val_loss",
    ):
        super().__init__()
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_style = lr_decay_style
        self.start_wd = start_wd
        self.end_wd = end_wd
        self.wd_incr_steps = wd_incr_steps
        self.wd_incr_style = wd_incr_style
        self.interval = interval
        self.frequency = frequency
        self.monitor = monitor

    def scheduler(self, model, optimizer):

        lr_scheduler = MegatronAnnealing(
            optimizer=optimizer,
            init_lr=self.init_lr,
            max_lr=self.max_lr,
            min_lr=self.min_lr,
            lr_warmup_steps=self.lr_warmup_steps,
            lr_decay_steps=self.lr_decay_steps,
            lr_decay_style=self.lr_decay_style,
            start_wd=self.start_wd,
            end_wd=self.end_wd,
            wd_incr_steps=self.wd_incr_steps,
            wd_incr_style=self.wd_incr_style,
            use_checkpoint_opt_param_scheduler=False,
            override_opt_param_scheduler=False,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": self.interval,
                "frequency": self.frequency,
            },
            "monitor": self.monitor,
        }

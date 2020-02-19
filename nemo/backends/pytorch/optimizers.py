import math

import torch
from torch.optim import Optimizer


def _check_valid_opt_params(lr, eps, betas):
    if lr < 0:
        raise ValueError(f"Invalid learning rate: {lr}")
    if eps < 0:
        raise ValueError(f"Invalid epsilon value: {eps}")
    if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
        raise ValueError(f"Betas have to be between 0 and 1: {betas}")


def master_params(optimizer):
    """
    Generator expression that iterates over the params owned by ``optimizer``.
    Args:
        optimizer: An optimizer previously returned from ``amp.initialize``.
    """
    for group in optimizer.param_groups:
        for p in group['params']:
            yield p


class AdamW(Optimizer):
    """Implements AdamW algorithm.
    It has been proposed in "Decoupled Weight Decay Regularization"
    (https://arxiv.org/abs/1711.05101)

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and Beyond"
    """

    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False,
    ):
        _check_valid_opt_params(lr, eps, betas)
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad,)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
                amsgrad = group["amsgrad"]
                state = self.state[p]

                # State initialization
                if not state:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp moving avg of squared grad
                        state["max_exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # if group['weight_decay'] != 0:
                #     grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains max of all 2nd moment running avg till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                # p.data.addcdiv_(-step_size, exp_avg, denom)
                p.data.add_(
                    -step_size, torch.mul(p.data, group["weight_decay"]).addcdiv_(1, exp_avg, denom),
                )

        return loss


class Novograd(Optimizer):
    """Implements Novograd algorithm.
    It has been proposed  in "Stochastic Gradient Methods with Layer-wise
    Adaptive Moments for Training of Deep Networks"
    (https://arxiv.org/abs/1905.11286)

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and Beyond"
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.95, 0.98),
        eps=1e-8,
        weight_decay=0,
        grad_averaging=False,
        amsgrad=False,
        luc=False,
        luc_trust=1e-3,
        luc_eps=1e-8,
    ):
        _check_valid_opt_params(lr, eps, betas)
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, grad_averaging=grad_averaging, amsgrad=amsgrad,
        )
        self.luc = luc
        self.luc_trust = luc_trust
        self.luc_eps = luc_eps
        super(Novograd, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Novograd, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Sparse gradients are not supported.")
                amsgrad = group["amsgrad"]
                state = self.state[p]

                # State initialization
                if not state:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros([]).to(state["exp_avg"].device)
                    if amsgrad:
                        # Maintains max of all exp moving avg of squared grad
                        state["max_exp_avg_sq"] = torch.zeros([]).to(state["exp_avg"].device)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                norm = grad.norm().pow(2)

                if exp_avg_sq == 0:
                    exp_avg_sq.copy_(norm)
                else:
                    exp_avg_sq.mul_(beta2).add_(1 - beta2, norm)

                if amsgrad:
                    # Maintains max of all 2nd moment running avg till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                grad.div_(denom)
                if group["weight_decay"] != 0:
                    grad.add_(group["weight_decay"], p.data)
                if group["grad_averaging"]:
                    grad.mul_(1 - beta1)
                exp_avg.mul_(beta1).add_(grad)

                if self.luc:
                    # Clip update so that updates are less than eta*weights
                    data_norm = torch.norm(p.data)
                    grad_norm = torch.norm(exp_avg.data)
                    luc_factor = self.luc_trust * data_norm / (grad_norm + self.luc_eps)
                    luc_factor = min(luc_factor, group["lr"])
                    p.data.add_(-luc_factor, exp_avg)
                else:
                    p.data.add_(-group["lr"], exp_avg)

        return loss

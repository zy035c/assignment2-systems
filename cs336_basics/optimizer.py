from jaxtyping import Float, Int
import torch

from collections.abc import Callable, Iterable
from typing import Optional
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr * math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                grad = p.grad.data

                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad * grad)

                lr_t = lr * math.sqrt(1 - beta2**(t)) / (1 - beta1**(t))

                p.data -= lr_t * (m / (torch.sqrt(v) + eps))
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss


def lr_cosine_schedule(t, lr_max, lr_min, Tw, Tc):
    if t < Tw:
        return t / Tw * lr_max
    elif Tw <= t <= Tc:
        return lr_min + .5 * (lr_max - lr_min) * (1 + math.cos(math.pi * (t - Tw) / (Tc - Tw)))
    else:
        return lr_min

from torch.optim.lr_scheduler import _LRScheduler

class CosineWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, lr_max, lr_min, Tw, Tc, last_epoch=-1):
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.Tw = Tw
        self.Tc = Tc
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch
        lr = lr_cosine_schedule(t, self.lr_max, self.lr_min, self.Tw, self.Tc)
        return [lr for _ in self.optimizer.param_groups]


def gradient_clipping(
    inputs: Iterable[torch.nn.Parameter],
    M: float,
    eps=1e-6
):
    l2_norm = torch.norm(
        torch.stack([p.grad.detach().norm(2) for p in inputs if p.grad is not None]),
        2
    )
    for p in inputs:
        if p.grad is None: continue
        if l2_norm >= M:
            p.grad = p.grad * (M / (l2_norm + eps))
    pass

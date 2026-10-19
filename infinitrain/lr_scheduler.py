"""
学习率调度器 Python 实现。

本模块与 `demo/lr_scheduler.h` 的 C++ 实现**行为完全一致**（由 demo/align.py 逐步断言保证），
公式与 `torch.optim.lr_scheduler` 对齐。

对齐口径说明
------------
每个调度器以 **闭式公式（closed-form）** 计算绝对学习率，等价于 PyTorch 的
`_get_closed_form_lr()` 口径。约定 `step_count` 从 0 开始，**第一次调用 step() 后变为 1**，
对应 PyTorch 的 `last_epoch`。

已知与 PyTorch 的差异（不做伪装对齐，详见 align.py 中的用例）
--------------------------------------------------------------
1. `LinearLR` 采用**加性线性插值**（start + α·(end-start)），可从 0 开始 warmup；
   PyTorch 的 `LinearLR` 是乘性因子且要求 start_factor > 0，**无法表达从 0 开始的 warmup**。
   本实现的等价物是 `LambdaLR(lambda t: min(t / warmup_steps, 1.0))`，已用测试断言二者一致。

2. `ChainedScheduler` 中每个子调度器按闭式公式独立计算绝对 lr，
   因此**最终值由最后一个子调度器决定**；PyTorch 的 `ChainedScheduler` 中子调度器是
   相对于 optimizer 当前 lr 递推的（`return [group["lr"] for group in ...]`），语义为依次相乘。
"""

from typing import Callable, List, Sequence

__all__ = [
    "LRScheduler",
    "ConstantLR",
    "StepLR",
    "LinearLR",
    "LambdaLR",
    "SequentialLR",
    "ChainedScheduler",
]


class LRScheduler:
    """学习率调度器基类。

    Args:
        optimizer: 需带 `param_groups` 列表的优化器对象。
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.step_count = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.last_lr = self.base_lrs[0] if self.base_lrs else 0.0

    def step(self) -> None:
        """推进一个 step，并把新学习率写回所有 param_group。"""
        self.step_count += 1
        lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.last_lr = lr

    def get_lr(self) -> float:
        """返回当前 step 应使用的学习率（闭式计算，无副作用）。"""
        raise NotImplementedError

    def state_dict(self) -> dict:
        return {"step_count": self.step_count}

    def load_state_dict(self, state_dict: dict) -> None:
        self.step_count = state_dict["step_count"]

    def _group_count(self) -> int:
        return len(self.optimizer.param_groups)


class ConstantLR(LRScheduler):
    """常数学习率。"""

    def __init__(self, optimizer, lr: float):
        super().__init__(optimizer)
        self.lr = lr
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.last_lr = lr

    def get_lr(self) -> float:
        return self.lr


class StepLR(LRScheduler):
    """等间隔衰减：lr = initial_lr * gamma^(floor(step_count / step_size))。

    对齐 torch.optim.lr_scheduler.StepLR 的闭式口径。注意指数是
    `step_count // step_size` 而非 `(step_count - 1) // step_size`——
    后者会让每一次衰减都比 PyTorch 晚一步。
    """

    def __init__(self, optimizer, initial_lr: float, step_size: int, gamma: float):
        super().__init__(optimizer)
        if step_size <= 0:
            raise ValueError(f"step_size 必须为正整数，当前为 {step_size}")
        self.initial_lr = initial_lr
        self.step_size = step_size
        self.gamma = gamma
        for group in self.optimizer.param_groups:
            group["lr"] = initial_lr
        self.last_lr = initial_lr

    def get_lr(self) -> float:
        exponent = self.step_count // self.step_size
        return self.initial_lr * (self.gamma ** exponent)


class LinearLR(LRScheduler):
    """线性 warmup（加性插值）：在 warmup_steps 内从 start_lr 线性增长到 end_lr。

    与 PyTorch `LinearLR` 不同：本实现是加性的，允许 start_lr == 0。
    等价的 PyTorch 表达是 `LambdaLR(lambda t: start + min(t / warmup_steps, 1) * (end - start))`。
    """

    def __init__(self, optimizer, start_lr: float, end_lr: float, warmup_steps: int):
        super().__init__(optimizer)
        if warmup_steps <= 0:
            raise ValueError(f"warmup_steps 必须为正整数，当前为 {warmup_steps}")
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.warmup_steps = warmup_steps
        for group in self.optimizer.param_groups:
            group["lr"] = start_lr
        self.last_lr = start_lr

    def get_lr(self) -> float:
        if self.step_count >= self.warmup_steps:
            return self.end_lr
        alpha = self.step_count / self.warmup_steps
        return self.start_lr + alpha * (self.end_lr - self.start_lr)


class LambdaLR(LRScheduler):
    """Lambda 调度器：lr = initial_lr * lr_lambda(step_count)。

    对齐 torch.optim.lr_scheduler.LambdaLR 的闭式口径。
    """

    def __init__(self, optimizer, initial_lr: float, lr_lambda: Callable[[int], float]):
        super().__init__(optimizer)
        if not callable(lr_lambda):
            raise TypeError("lr_lambda 必须是可调用对象")
        self.initial_lr = initial_lr
        self.lr_lambda = lr_lambda
        for group in self.optimizer.param_groups:
            group["lr"] = initial_lr
        self.last_lr = initial_lr

    def get_lr(self) -> float:
        return self.initial_lr * self.lr_lambda(self.step_count)


class SequentialLR(LRScheduler):
    """按里程碑顺序切换子调度器。

    语义对齐 torch.optim.lr_scheduler.SequentialLR：
    每个子调度器**独立维护自己的 step_count，只在被选中激活时才推进**。
    因此切换后新的调度器从自己的第 1 步重新开始计数。

    Args:
        schedulers: 子调度器列表
        milestones: 切换点列表，长度必须为 len(schedulers) - 1。
            milestones[i] = k 表示第 k 步之后切换到 schedulers[i+1]。
    """

    def __init__(self, optimizer, schedulers: Sequence[LRScheduler], milestones: Sequence[int]):
        super().__init__(optimizer)
        if len(schedulers) != len(milestones) + 1:
            raise ValueError(
                "SequentialLR: schedulers 数量必须比 milestones 多 1，"
                f"当前 schedulers={len(schedulers)}, milestones={len(milestones)}"
            )
        self.schedulers = list(schedulers)
        self.milestones = list(milestones)
        self.idx = 0
        if self.schedulers:
            # 初始学习率取第一个调度器的当前值
            self.last_lr = self.schedulers[0].get_lr()
            for group in self.optimizer.param_groups:
                group["lr"] = self.last_lr

    def step(self) -> None:
        self.step_count += 1
        # 推进到当前应生效的调度器索引
        while self.idx < len(self.milestones) and self.step_count > self.milestones[self.idx]:
            self.idx += 1
        if self.idx < len(self.schedulers):
            # 不要在此同步子调度器的 step_count——那会导致 step() 内部重复计数，
            # 使曲线比 PyTorch 快一步。子调度器自行计数即可。
            self.schedulers[self.idx].step()
        self.last_lr = self.optimizer.param_groups[0]["lr"]

    def get_lr(self) -> float:
        if self.idx < len(self.schedulers):
            return self.schedulers[self.idx].get_lr()
        return self.last_lr

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["idx"] = self.idx
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        if "idx" in state_dict:
            self.idx = state_dict["idx"]


class ChainedScheduler(LRScheduler):
    """链式组合调度器：每步依次调用所有子调度器的 step()。

    ⚠️ 与 PyTorch `ChainedScheduler` 的语义差异（重要）：

    - 本实现中每个子调度器按闭式公式计算**绝对 lr** 并写回 optimizer，
      因此最终生效的是**最后一个**子调度器的结果。
    - PyTorch 的子调度器是相对于 optimizer 当前 lr 递推的
      （典型实现为 `return [group["lr"] for group in ...]`），语义为**依次相乘**。

    两者不可互换，混用会得到不同的学习率曲线。若需要 PyTorch 语义，
    请直接使用 `torch.optim.lr_scheduler.ChainedScheduler`。
    """

    def __init__(self, optimizer, schedulers: Sequence[LRScheduler]):
        super().__init__(optimizer)
        self.schedulers = list(schedulers)
        if self.schedulers:
            self.last_lr = self.schedulers[-1].get_lr()
            for group in self.optimizer.param_groups:
                group["lr"] = self.last_lr

    def step(self) -> None:
        self.step_count += 1
        for s in self.schedulers:
            s.step()
        if self.schedulers:
            self.last_lr = self.optimizer.param_groups[0]["lr"]

    def get_lr(self) -> float:
        if self.schedulers:
            return self.schedulers[-1].get_lr()
        return self.last_lr

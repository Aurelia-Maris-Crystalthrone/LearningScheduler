"""
学习率调度器 Python 实现（与 C++ 版本行为一致）
"""

class LRScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.step_count = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        self.step_count += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        raise NotImplementedError

    def state_dict(self):
        return {'step_count': self.step_count}

    def load_state_dict(self, state_dict):
        self.step_count = state_dict['step_count']


class ConstantLR(LRScheduler):
    def __init__(self, optimizer, lr):
        super().__init__(optimizer)
        self.lr = lr
        for group in optimizer.param_groups:
            group['lr'] = lr

    def get_lr(self):
        return self.lr


class StepLR(LRScheduler):
    def __init__(self, optimizer, initial_lr, step_size, gamma):
        super().__init__(optimizer)
        self.initial_lr = initial_lr
        self.step_size = step_size
        self.gamma = gamma
        for group in optimizer.param_groups:
            group['lr'] = initial_lr

    def get_lr(self):
        exponent = (self.step_count - 1) // self.step_size
        return self.initial_lr * (self.gamma ** exponent)


class LinearLR(LRScheduler):
    def __init__(self, optimizer, start_lr, end_lr, warmup_steps):
        super().__init__(optimizer)
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.warmup_steps = warmup_steps
        for group in optimizer.param_groups:
            group['lr'] = start_lr

    def get_lr(self):
        if self.step_count >= self.warmup_steps:
            return self.end_lr
        alpha = self.step_count / self.warmup_steps
        return self.start_lr + alpha * (self.end_lr - self.start_lr)


class LambdaLR(LRScheduler):
    def __init__(self, optimizer, initial_lr, lr_lambda):
        super().__init__(optimizer)
        self.initial_lr = initial_lr
        self.lr_lambda = lr_lambda
        for group in optimizer.param_groups:
            group['lr'] = initial_lr

    def get_lr(self):
        return self.initial_lr * self.lr_lambda(self.step_count)


class SequentialLR(LRScheduler):
    def __init__(self, optimizer, schedulers, milestones):
        super().__init__(optimizer)
        self.schedulers = schedulers
        self.milestones = milestones
        self.idx = 0
        # 初始化学习率为第一个调度器的值
        self.schedulers[0].step_count = 0  # 确保步数一致
        self.schedulers[0].get_lr()  # 触发内部状态（如果有）
        for group in optimizer.param_groups:
            group['lr'] = self.schedulers[0].get_lr()

    def step(self):
        self.step_count += 1
        # 检查是否需要切换
        while self.idx < len(self.milestones) and self.step_count > self.milestones[self.idx]:
            self.idx += 1
        if self.idx < len(self.schedulers):
            self.schedulers[self.idx].step_count = self.step_count  # 同步步数
            self.schedulers[self.idx].step()  # 会调用内部的 get_lr 并更新参数
        # 更新本调度器的学习率（实际已由子调度器更新，这里只需保存）
        self.last_lr = self.optimizer.param_groups[0]['lr']

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


class ChainedScheduler(LRScheduler):
    def __init__(self, optimizer, schedulers):
        super().__init__(optimizer)
        self.schedulers = schedulers
        # 初始化学习率为最后一个调度器的值
        for s in schedulers:
            s.step_count = 0
        self.last_lr = schedulers[-1].get_lr() if schedulers else 0
        for group in optimizer.param_groups:
            group['lr'] = self.last_lr

    def step(self):
        self.step_count += 1
        for s in self.schedulers:
            s.step_count = self.step_count
            s.step()
        self.last_lr = self.optimizer.param_groups[0]['lr']

    def get_lr(self):
        return self.last_lr
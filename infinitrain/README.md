# Python 学习率调度器实现

与 `demo/lr_scheduler.h` **行为完全一致**的 Python 实现，用于快速实验与数值对齐验证。

## 文件

| 文件 | 说明 |
| :--- | :--- |
| `lr_scheduler.py` | 调度器实现（核心） |
| `train.py` | 模拟训练脚本，支持单机与 DDP |
| `requirements.txt` | 依赖清单 |

## 安装

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 使用

```python
from lr_scheduler import LinearLR, StepLR, SequentialLR

# 任意带 param_groups 的优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

warmup = LinearLR(optimizer, 0.0, 0.001, warmup_steps=5)
decay  = StepLR(optimizer, 0.001, step_size=3, gamma=0.5)
sched  = SequentialLR(optimizer, [warmup, decay], milestones=[5])

for epoch in range(num_epochs):
    optimizer.step()
    sched.step()
```

## 与 PyTorch 的对齐口径

约定 `step_count` 从 0 开始，第一次 `step()` 后变为 1，对应 PyTorch 的 `last_epoch`。
核心公式：`StepLR` 为 `lr₀ · γ^(⌊step_count / step_size⌋)`，
即**在第 `step_size` 步完成首次衰减**。

### 两处已知差异

1. **`LinearLR` 是加性 warmup**：`start + min(step/warmup, 1)·(end-start)`，可从 0 起。
   PyTorch 的 `LinearLR` 是乘性因子且要求 `start_factor > 0`，无法表达从 0 开始的 warmup。
   等价表达为 `LambdaLR(lambda t: start/end + min(t/warmup, 1)·(1 - start/end))`。

2. **`ChainedScheduler` 语义不同**：本实现的最终值由最后一个子调度器决定；
   PyTorch 的子调度器相对 optimizer 当前 lr 依次递推（相乘）。

详见根目录 `README.md` 与 `demo/align.py` 中的说明与测试。

## 训练脚本

`train.py` 有无 torchrun 两种环境下都能运行——`WORLD_SIZE <= 1` 时自动跳过分布式
初始化并禁用 DDP，无 GPU 时回落到 CPU 与 gloo 后端。

```bash
# 单机单进程
python train.py --lr-scheduler sequential --total-steps 20

# 多卡
torchrun --nproc_per_node=8 train.py --lr-scheduler sequential --total-steps 20
```

可选调度器：`constant`、`steplr`、`linear`、`lambda`、`sequential`、`chained`。

> **注意**：脚本使用的模型是**随机初始化的极小配置**
> （GPT-2 为 `n_layer=2, n_embd=64, vocab_size=1000`），
> 目的是观察学习率曲线，**不含真实训练逻辑，也不是预训练模型**。

## 验证

```bash
python ../demo/align.py
pytest ../demo/align.py -v
```

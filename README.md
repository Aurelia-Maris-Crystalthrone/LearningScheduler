# LearningScheduler

> **学习率调度器模块** — 一套同时提供 C++ 与 Python 实现的学习率调度器，支持组合调度与状态恢复，并提供可执行的数值对齐验证。

**项目地址**：<https://github.com/Aurelia-Maris-Crystalthrone/LearningScheduler>

---

## 📖 项目简介

本仓库实现了一组常用学习率调度策略，用于验证同一套调度逻辑在 **C++ 训练框架**与 **Python/PyTorch 参考实现**之间能否给出完全一致的学习率曲线。

同一个库提供两份等价实现：

| 模块 | 语言 | 用途 |
| :--- | :--- | :--- |
| `demo/lr_scheduler.h` | C++17 | 头文件即用，面向 C++ 训练框架集成 |
| `infinitrain/lr_scheduler.py` | Python | 便于快速实验与和 PyTorch 对比 |

「两套实现行为一致」这一结论由 `demo/align.py` **自动编译并运行 C++ 代码后逐步断言**，而非人工声称。

---

## 🔗 项目来源

本项目的 C++ 调度器实现（`demo/`）是在 [TinyInfiniTensor](https://github.com/InfiniTensor/TinyInfiniTensor) 的调度器练习题目基础上完成的；Python 实现与训练脚本（`infinitrain/`）为本仓库补充。

> **⚠️ 需要澄清的一点**：本仓库**并非 GitHub 意义上的 fork**——GitHub API 对该仓库返回
> `fork: false`、`parent: null`，仓库中也不包含 TinyInfiniTensor 的原始代码或上游提交历史。
> 上文指的是「参考了其题目与课程要求」，而非代码层面的派生。

---

## 📂 仓库结构

```
LearningScheduler/
├── LICENSE                     # 根目录许可证（MIT）
├── README.md
├── .gitignore
├── demo/                       # C++ 实现与验证
│   ├── lr_scheduler.h          # 调度器头文件（核心）
│   ├── reference_cpp.cpp       # 参考序列生成器，输出学习率序列 JSON
│   ├── align.py                # ⭐ 三方数值对齐验证（C++ / Python / PyTorch）
│   ├── CMakeLists.txt          # CMake 构建配置
│   ├── LICENSE                 # demo 模块许可证（MIT, InfiniTensor）
│   ├── README.md               # demo 模块说明
│   └── examples/
│       └── main.cc             # 命令行示例
└── infinitrain/                # Python 实现与模拟训练脚本
    ├── lr_scheduler.py         # Python 调度器实现（核心）
    ├── train.py                # 模拟训练脚本（支持单机与 DDP）
    ├── requirements.txt        # Python 依赖
    └── README.md
```

---

## ✨ 调度器一览

| 类 | 公式 | 说明 |
| :--- | :--- | :--- |
| `ConstantLR` | `lr = const` | 常数学习率 |
| `StepLR` | `lr = lr₀ · γ^(⌊step/step_size⌋)` | 等间隔衰减 |
| `LinearLR` | `lr = start + min(step/warmup, 1)·(end-start)` | 线性 warmup（**加性**，可从 0 起） |
| `LambdaLR` | `lr = lr₀ · f(step)` | 自定义函数 |
| `SequentialLR` | 按里程碑切换子调度器 | 子调度器独立计数 |
| `ChainedScheduler` | 依次 `step()` 所有子调度器 | 见下方差异说明 |

约定：`step_count` 从 0 开始，**第一次调用 `step()` 后变为 1**，对应 PyTorch 的 `last_epoch`。

### 设计模式与接口

- **状态可恢复**：所有调度器实现 `State()` / `LoadState()`，支持 checkpoint/resume
- **零侵入**：仅通过优化器的 `SetLearningRate()` 公开接口交互
- **易扩展**：继承 `LRScheduler` 并实现 `ComputeLR()` 即可
- **错误处理统一**：`StateDict` 缺失键时保持当前值，不会一处静默、一处抛异常

---

## 🎯 数值对齐验证

这是本仓库的核心。运行：

```bash
python demo/align.py          # 打印对比表格
pytest demo/align.py -v       # 作为测试套件运行
```

`align.py` 会：

1. 自动编译并运行 `reference_cpp.cpp`，拿到 C++ 侧的学习率序列（**单一数据源**）；
2. 用相同参数驱动 `infinitrain/lr_scheduler.py`；
3. 用相同参数驱动 `torch.optim.lr_scheduler`；
4. 逐步断言三者一致。

实测输出（12 步，Python 3.12.7 + PyTorch 2.14.1 + MinGW g++）：

```
case                      C++ vs Python        Python vs PyTorch
------------------------------------------------------------------
constant_lr               ✅ 一致                ✅ 一致
step_lr                   ✅ 一致                ✅ 一致
step_lr_size1             ✅ 一致                ✅ 一致
linear_lr_warmup          ✅ 一致                ✅ 一致
linear_lr_nonzero_start   ✅ 一致                ✅ 一致
lambda_lr_stepwise        ✅ 一致                ✅ 一致
lambda_lr_reciprocal      ✅ 一致                ✅ 一致
sequential_warmup_step    ✅ 一致                ✅ 一致
sequential_three_stage    ✅ 一致                ✅ 一致
chained_warmup_step       ✅ 一致                不适用（语义差异）
```

若环境缺少 C++ 编译器或 PyTorch，相应用例会自动跳过并说明原因，不会静默伪装成通过。

### ⚠️ 与 PyTorch 的两处已知差异

这两处**不做伪装对齐**，此处明确记录：

**1. `LinearLR` 是加性线性 warmup，不是 PyTorch 的 `LinearLR`**

| | 本项目 `LinearLR` | PyTorch `LinearLR` |
| :--- | :--- | :--- |
| 语义 | 加性插值：`start + α·(end-start)` | 乘性因子：`lr × 递推因子` |
| 能否从 0 开始 | ✅ 可以 | ❌ 不可以，要求 `start_factor > 0` |

本实现的等价 PyTorch 表达是 `LambdaLR(lambda t: start/end + min(t/warmup, 1)·(1 - start/end))`，
`align.py` 中对该等价性有专门的测试用例。

**2. `ChainedScheduler` 语义不同**

- 本项目：每个子调度器按闭式公式计算**绝对 lr** 并写回 optimizer，最终值由**最后一个**决定。
- PyTorch：子调度器相对 optimizer 当前 lr **依次递推**（典型实现为 `return [group["lr"] for group in ...]`），效果是依次相乘。

两者不可互换。需要 PyTorch 语义时请直接使用 `torch.optim.lr_scheduler.ChainedScheduler`。

### 精度容差说明

- **Python vs PyTorch**：双精度计算，`rtol = 1e-9`
- **C++ vs Python**：C++ 侧内部为 `float`（有效位约 7 位十进制），`rtol = 1e-6`

后者是语言层面的存储精度差异，不是实现 bug。若需要更高精度，可把 `lr_scheduler.h` 中的
`float` 改为 `double`。

---

## 🛠️ 环境要求

### C++ 模块

| 依赖 | 要求 |
| :--- | :--- |
| C++ 编译器 | GCC ≥ 7 / Clang ≥ 5 / MSVC 2019+，需支持 **C++17** |
| CMake | ≥ 3.15（推荐 3.27+） |
| gflags | 仅构建 `examples/main.cc` 需要；`reference_cpp.cpp` 无此依赖 |

### Python 模块

| 依赖 | 要求 |
| :--- | :--- |
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.0（对齐基线为 2.14.1） |
| transformers | ≥ 4.30（仅 `train.py` 构造 tiny 模型时使用） |
| numpy | ≥ 1.24 |
| pytest | ≥ 7.0（运行 `align.py` 测试套件） |

---

## 🚀 快速开始

### 1. 数值对齐验证（推荐先做）

需要 C++ 编译器和 PyTorch：

```bash
python demo/align.py
```

### 2. C++ 模块

```bash
# Linux
sudo apt install -y cmake g++ libgflags-dev

cd demo
mkdir -p build && cd build
cmake ..
cmake --build . --config Release -j$(nproc)
```

运行示例：

```bash
./lr_scheduler_demo --lr_scheduler=sequential \
                    --scheduler_list=linear,steplr \
                    --scheduler_milestones=5 \
                    --warmup_steps=5 \
                    --step_size=3 \
                    --gamma=0.5 \
                    --total_steps=10
```

无 sudo 权限时可将 CMake / gflags 安装到 `$HOME/.local`，再指定 `CMAKE_PREFIX_PATH`：

```bash
cmake -DCMAKE_PREFIX_PATH=$HOME/.local ..
```

### 3. Python 模块

```bash
cd infinitrain
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

训练脚本在**有无 torchrun 两种环境下都能运行**：检测到 `WORLD_SIZE <= 1` 时会自动
跳过分布式初始化并禁用 DDP，无 GPU 时回落到 CPU 与 gloo 后端。

```bash
# 单机单进程
python train.py --lr-scheduler sequential --total-steps 20

# 多卡（由 torchrun 注入 LOCAL_RANK / WORLD_SIZE）
torchrun --nproc_per_node=8 train.py --lr-scheduler sequential --total-steps 20
```

---

## 📝 扩展自定义调度器

继承 `LRScheduler` 并实现 `ComputeLR()`（**无参数**，当前步数通过成员的 `step_count_` 获取）：

```cpp
class MyScheduler : public LRScheduler {
public:
    MyScheduler(Optimizer* optimizer, float base_lr, int decay_steps, float decay_rate)
        : LRScheduler(optimizer),
          base_lr_(base_lr), decay_steps_(decay_steps), decay_rate_(decay_rate) {
        last_lr_ = base_lr_;
        optimizer_->SetLearningRate(last_lr_);
    }

protected:
    float ComputeLR() override {
        int exponent = step_count_ / decay_steps_;
        return base_lr_ * std::pow(decay_rate_, exponent);
    }

private:
    float base_lr_;
    int decay_steps_;
    float decay_rate_;
};
```

> 注意签名是 `float ComputeLR() override`，**不带参数**。旧版本文档中写作
> `ComputeLR(int step)`，照抄会导致编译失败（`override` 不匹配）。

### 状态保存与恢复

```cpp
// 保存
StateDict state = scheduler->State();

// 恢复
scheduler->LoadState(state);
```

---

## 🧪 验证状态

本项目下述结论均在 Python 3.12.7 / PyTorch 2.14.1 / MinGW g++ 下实测通过：

- ✅ 10 个用例的 C++ 与 Python 实现逐步一致
- ✅ 其中 9 个可与 PyTorch 对齐的用例逐步一致
- ✅ StepLR 在第 `step_size` 步（而非 `step_size + 1` 步）完成首次衰减
- ✅ SequentialLR 切换后子调度器从自己的第 1 步重新计数，无重复自增
- ✅ 状态保存/加载后学习率曲线与未中断时一致（resume 正确性）
- ✅ 加性 LinearLR 与其等价 LambdaLR 表达式一致
- ✅ `examples/main.cc` 与 `reference_cpp.cpp` 在 `-Wall -Wextra -Werror` 下零警告编译

### 已知限制

- `train.py` 中的模型是**随机初始化的极小配置**（GPT-2 为 `n_layer=2, n_embd=64, vocab=1000`），
  用于验证调度器行为，**不是**真实的 LLaMA 3 / GPT-2 预训练模型，也不含任何真实训练逻辑。
- `ChainedScheduler` 与 PyTorch 语义不同，详见上文。
- C++ 侧使用 `float` 存储学习率，与 Python 侧的 `double` 存在约 1e-7 的相对误差。
- C++ 部分**未在 InfiniTrain 完整框架中集成验证**，仅在 demo 模块内编译与运行通过。

---

## 📄 许可证

- 根目录 [LICENSE](LICENSE)：MIT
- `demo/LICENSE`：MIT，Copyright (c) 2025 InfiniTensor

---

## 🙏 致谢

- 参考题目来源：[InfiniTensor/TinyInfiniTensor](https://github.com/InfiniTensor/TinyInfiniTensor)
- 训练框架参考：[InfiniTensor/InfiniTrain](https://github.com/InfiniTensor/InfiniTrain)

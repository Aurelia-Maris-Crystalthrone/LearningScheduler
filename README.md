# LearingScheduler

> **InfiniTrain 学习率调度器模块** — 一个独立、可扩展、支持状态恢复的学习率调度器，与 PyTorch `torch.optim.lr_scheduler` 数值对齐。

## 🔗 项目来源

本项目 fork 自 **[InfiniTensor/TinyInfiniTensor](https://github.com/InfiniTensor/TinyInfiniTensor)**。

TinyInfiniTensor 是 InfiniTensor 组织下的一个简化版 AI 编译器学习项目，旨在帮助初学者快速上手，保留了计算图和 kernel 层的核心概念，能够基于 C++ 搭建计算图进行推理计算，目前支持 CPU 平台。InfiniTensor 社区由启元实验室发起，聚焦人工智能编译器、大模型训练与推理优化、异构算力适配等核心领域。

本 fork 在原项目基础上，添加了完整的**学习率调度器模块**，作为 InfiniTrain 训练框架的组件实现与验证。InfiniTrain 是一个从零实现的 C++ 大规模模型训练框架，支持多维度分布式并行（DDP、TP、SP、PP 等），适配 GPT-2、LLaMA 3 等主流模型。

## 📖 项目简介

LearingScheduler 为 InfiniTrain 训练框架提供了一套完整的学习率调度器实现。该模块遵循单一职责、最小耦合和易于扩展的设计原则，支持 C++ 和 Python 两种实现，可满足从核心训练到快速验证的多种场景。

## 📂 仓库结构

```
LearingScheduler/
├── demo/                      # C++ 核心调度器实现
│   ├── examples/              # 使用示例
│   │   └── main.cc            # 示例主程序
│   ├── align.py               # PyTorch 数值对齐验证脚本
│   ├── lr_scheduler.h         # 调度器头文件
│   ├── CMakeLists.txt         # CMake 构建配置
│   ├── LICENSE                # 许可证
│   └── README.md              # demo 模块说明
├── infinitrain/               # Python 模拟训练框架（调度器测例专用）
│   ├── lr_scheduler.py        # Python 调度器实现
│   ├── train.py               # 训练脚本
│   ├── requirements.txt       # Python 依赖
│   ├── get-pip.py             # pip 安装脚本
│   └── README.md              # infinitrain 模块说明
├── .gitignore
├── venv_backup.tar.gz         # 虚拟环境备份
└── README.md
```

## ✨ 核心特性

- **PyTorch 数值对齐**：所有调度器的数值行为与 PyTorch 对应类完全一致
- **状态可恢复**：所有调度器实现 `State()` / `LoadState()` 接口，支持检查点保存与训练恢复
- **易于扩展**：继承 `LRScheduler` 并实现 `ComputeLR()` 即可添加新策略
- **组合支持**：通过 `SequentialLR` 和 `ChainedScheduler` 构建复杂的调度流水线
- **零侵入设计**：仅通过优化器的公开 `SetLearningRate()` API 进行交互

## 🛠️ 环境要求

### demo 模块（C++）

| 依赖项 | 版本要求 | 说明 |
|---|---|---|
| C++ 编译器 | GCC >= 7 或 MSVC 2019/2022 | 需支持 C++17 |
| CMake | >= 3.15 | 推荐 3.27+ |
| gflags | 最新版 | Google 命令行参数解析库 |

> **注意**：InfiniTrain 官方推荐 gcc/g++ 13+、CMake 3.13+。本 demo 模块的最低要求略低，但建议使用较新版本以获得最佳兼容性。

### infinitrain 模块（Python）

| 依赖项 | 版本要求 |
|---|---|
| Python | >= 3.10 |
| PyTorch | >= 1.10（支持 CUDA） |
| transformers | 最新版 |
| numpy | 最新版 |

## 🚀 快速开始

### 前置检查

```bash
# 检查编译器
g++ --version          # 应 >= 7

# 检查 CMake
cmake --version        # 应 >= 3.15

# 检查 Python
python3 --version      # 应 >= 3.10
```

### C++ 调度器模块

#### 1. 安装依赖

**Linux（有 sudo 权限）**：
```bash
sudo apt update
sudo apt install -y cmake g++ libgflags-dev
```

**Linux（受限环境 / 无 sudo 权限）**：

如果系统提示 `sudo: The "no new privileges" flag is set`，说明你处于 Docker/Podman/distrobox/devcontainer 等容器环境中，`sudo` 无法提权。此时可将 CMake 和 gflags 安装到用户目录：

**安装 CMake**：
```bash
cd ~
wget https://github.com/Kitware/CMake/releases/download/v3.27.8/cmake-3.27.8-linux-x86_64.tar.gz
tar -xzf cmake-3.27.8-linux-x86_64.tar.gz
export PATH="$HOME/cmake-3.27.8-linux-x86_64/bin:$PATH"

# 永久生效（追加到 ~/.bashrc）
echo 'export PATH="$HOME/cmake-3.27.8-linux-x86_64/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

cmake --version  # 验证安装
```

**安装 gflags**：
```bash
cd ~
git clone https://github.com/gflags/gflags.git
cd gflags
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$HOME/.local ..
make -j$(nproc)
make install

# 将 gflags 库路径加入环境变量
echo 'export CMAKE_PREFIX_PATH=$HOME/.local:$CMAKE_PREFIX_PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**Windows（使用 vcpkg）**：
```bash
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install
.\vcpkg install gflags:x64-windows
```

#### 2. 构建项目

```bash
cd demo
mkdir -p build && cd build

# 标准构建
cmake ..

# 如果 gflags 安装在 $HOME/.local（受限环境），需指定路径：
# cmake -DCMAKE_PREFIX_PATH=$HOME/.local ..

cmake --build . --config Release -j$(nproc)
```

> **常见问题**：如果之前构建失败过，CMake 会缓存错误配置。建议先清理缓存再重新构建：
> ```bash
> rm -rf build && mkdir build && cd build
> cmake -DCMAKE_PREFIX_PATH=$HOME/.local ..
> make -j$(nproc)
> ```

### Python 模拟训练模块

```bash
cd infinitrain

# 创建虚拟环境（推荐，避免 PEP 668 externally-managed-environment 错误）
python3 -m venv .venv
source .venv/bin/activate

# 升级 pip 并安装依赖
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> **常见问题**：如果在虚拟环境中仍然遇到 `externally-managed-environment` 错误，可能是因为 `pip` 仍指向系统版本。可用以下命令验证和解决：
> ```bash
> which pip    # 应输出 .venv/bin/pip，而非 /usr/bin/pip
> ./.venv/bin/python3 -m pip install -r requirements.txt
> ```
> 若虚拟环境创建失败（缺少 `python3-venv` 组件且无 sudo 权限），可尝试：
> ```bash
> python3 -m venv --without-pip .venv
> ```

运行训练脚本进行验证：

```bash
python train.py
```

## 📝 使用说明

### 扩展自定义调度器（C++）

继承 `LRScheduler` 基类并实现 `ComputeLR()` 方法即可定义新的学习率策略：

```cpp
class MyScheduler : public LRScheduler {
 public:
  float ComputeLR(int step) override {
    // 自定义学习率计算逻辑
    return base_lr_ * std::pow(decay_rate_, step / decay_steps_);
  }
};
```

### 状态保存与恢复

所有调度器均支持检查点功能：

```cpp
// 保存状态
auto state = scheduler->State();
SaveToCheckpoint(state, "checkpoint.bin");

// 恢复状态
auto state = LoadFromCheckpoint("checkpoint.bin");
scheduler->LoadState(state);
```

### PyTorch 数值对齐验证

`demo/align.py` 提供了与 PyTorch 调度器的数值对比脚本，可用于验证 C++ 实现的正确性。使用前请确保已安装 PyTorch：

```bash
pip install torch numpy
python demo/align.py
```

## 🧪 构建验证清单

完成构建后，可按以下步骤确认环境配置正确：

- [ ] `cmake --version` 输出 ≥ 3.15
- [ ] `g++ --version` 输出 ≥ 7
- [ ] gflags 已安装（`ls $HOME/.local/lib/libgflags*` 或 `dpkg -l libgflags-dev`）
- [ ] `cd demo/build && cmake -DCMAKE_PREFIX_PATH=$HOME/.local ..` 无报错，输出 `-- Configuring done` 和 `-- Generating done`
- [ ] `make -j$(nproc)` 成功生成可执行文件
- [ ] `python3 -c "import torch; print(torch.__version__)"` 正常输出

## 📄 许可证

本项目 demo 模块采用 MIT 许可证，详见 [demo/LICENSE](demo/LICENSE)。

## 🙏 致谢

- 原始项目：[InfiniTensor/TinyInfiniTensor](https://github.com/InfiniTensor/TinyInfiniTensor)
- 训练框架参考：[InfiniTensor/InfiniTrain](https://github.com/InfiniTensor/InfiniTrain)
- InfiniTensor 社区：由启元实验室与清华大学等机构支持

---

**项目地址**：https://github.com/Aurelia-Maris-Crystalthrone/LearingScheduler

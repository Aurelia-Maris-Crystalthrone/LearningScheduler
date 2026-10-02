# InfiniTrain Learning Rate Scheduler Module

## Overview
This module provides an independent, extensible, and state-recoverable Learning Rate Scheduler for the InfiniTrain training framework. It is numerically aligned with PyTorch's `torch.optim.lr_scheduler`. The design follows the principles of single responsibility, minimal coupling, and easy extensibility.

## Features
- **PyTorch Aligned**: Numerical behavior matches PyTorch's equivalent classes.
- **State Recoverable**: All schedulers implement `State()` / `LoadState()` for checkpoint/resume.
- **Easy to Extend**: Add new strategies by inheriting `LRScheduler` and implementing `ComputeLR()`.
- **Composition Support**: `SequentialLR` and `ChainedScheduler` allow complex scheduling pipelines.
- **Zero Intrusion**: Interacts with the optimizer only through its public `SetLearningRate()` API.

## Quick Start (Windows 11 x64 / Linux)

### 1. Install Dependencies
- **C++ Compiler**: MSVC (Visual Studio 2019/2022) on Windows, or GCC (>=7) on Linux.
- **CMake** (>=3.15): [Download](https://cmake.org/download/)
- **gflags Library**:
  - **Windows (vcpkg)**:
    ```powershell
    git clone https://github.com/Microsoft/vcpkg.git
    cd vcpkg
    .\bootstrap-vcpkg.bat
    .\vcpkg integrate install
    .\vcpkg install gflags:x64-windows
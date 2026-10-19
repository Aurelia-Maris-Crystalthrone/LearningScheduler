# C++ Learning Rate Scheduler Module

Header-only C++17 implementation of several learning rate schedulers, with an
executable three-way numerical alignment check against Python and PyTorch.

## Contents

| File | Purpose |
| :--- | :--- |
| `lr_scheduler.h` | The scheduler library (header-only) |
| `reference_cpp.cpp` | Emits learning-rate sequences as JSON, used by `align.py` |
| `align.py` | ⭐ Three-way alignment test: C++ vs Python vs PyTorch |
| `examples/main.cc` | CLI demo (requires gflags) |
| `CMakeLists.txt` | Build config for both targets |

## Features

- **State recoverable** — every scheduler implements `State()` / `LoadState()` for checkpoint/resume
- **Easy to extend** — inherit `LRScheduler`, implement `float ComputeLR()` (note: **no arguments**)
- **Composition** — `SequentialLR` (milestone-based) and `ChainedScheduler`
- **Zero intrusion** — touches the optimizer only via `SetLearningRate()`
- **Consistent error handling** — missing `StateDict` keys keep the current value everywhere

## Alignment with PyTorch

Claiming "numerically aligned with PyTorch" is only meaningful if it is verified.
`align.py` does exactly that: it compiles and runs `reference_cpp.cpp`, then drives
both the Python implementation and `torch.optim.lr_scheduler` with identical
parameters and asserts step-by-step equality.

```bash
python align.py            # print comparison table
pytest align.py -v         # run as a test suite
```

Expected result (12 steps, PyTorch 2.14.1):

```
case                      C++ vs Python        Python vs PyTorch
------------------------------------------------------------------
constant_lr               ✅ match                ✅ match
step_lr                   ✅ match                ✅ match
step_lr_size1             ✅ match                ✅ match
linear_lr_warmup          ✅ match                ✅ match
linear_lr_nonzero_start   ✅ match                ✅ match
lambda_lr_stepwise        ✅ match                ✅ match
lambda_lr_reciprocal      ✅ match                ✅ match
sequential_warmup_step    ✅ match                ✅ match
sequential_three_stage    ✅ match                ✅ match
chained_warmup_step       ✅ match                n/a (semantic difference)
```

### Two documented differences from PyTorch

These are recorded honestly rather than papered over — see the root README for detail.

1. **`LinearLR` is additive warmup**, not PyTorch's multiplicative `LinearLR`.
   It can start from exactly 0; PyTorch's requires `start_factor > 0`.
   Equivalent PyTorch form: `LambdaLR(lambda t: start/end + min(t/warmup, 1)*(1 - start/end))`.

2. **`ChainedScheduler` semantics differ.** Here each subscheduler computes an
   *absolute* lr, so the last one wins. PyTorch's subschedulers are *relative* to
   the optimizer's current lr, so they multiply cumulatively.

### Tolerances

- Python vs PyTorch (both double): `rtol = 1e-9`
- C++ vs Python (C++ stores `float`): `rtol = 1e-6`

## Build

```bash
# Linux
sudo apt install -y cmake g++ libgflags-dev

mkdir -p build && cd build
cmake ..
cmake --build . --config Release -j$(nproc)
```

Without sudo privileges, install CMake and gflags into `$HOME/.local` and point CMake at it:

```bash
cmake -DCMAKE_PREFIX_PATH=$HOME/.local ..
```

> `reference_cpp` does **not** depend on gflags; only `lr_scheduler_demo` does.

## Usage

```cpp
#include "lr_scheduler.h"

class MyOptimizer : public Optimizer {
public:
    void SetLearningRate(float lr) override { lr_ = lr; }
    float GetLearningRate() const override { return lr_; }
private:
    float lr_;
};

MyOptimizer opt(0.001f);
StepLR scheduler(&opt, /*initial_lr=*/0.001f, /*step_size=*/3, /*gamma=*/0.5f);

for (int step = 0; step < 100; ++step) {
    scheduler.Step();
    // opt.GetLearningRate() reflects the current schedule
}
```

Sequential composition — each subscheduler keeps its own step count and only
advances while it is the active one:

```cpp
auto* warmup = new LinearLR(&opt, 0.0f, 0.001f, 5);
auto* decay  = new StepLR(&opt, 0.001f, 3, 0.5f);
SequentialLR sched(&opt, {warmup, decay}, {5});   // switch after step 5
// sched does NOT own them; caller must free them, or use unique_ptr in the caller scope
```

## Conventions

`step_count` starts at 0 and becomes 1 after the first `Step()` call, matching
PyTorch's `last_epoch`. StepLR therefore decays at the `step_size`-th step:

```
lr = initial_lr * gamma^(floor(step_count / step_size))
```

An earlier revision used `(step_count - 1) / step_size`, which delayed every decay
by exactly one step relative to PyTorch. `align.py` carries a regression test for this.

## License

MIT — see `demo/LICENSE` (Copyright 2025 InfiniTensor).

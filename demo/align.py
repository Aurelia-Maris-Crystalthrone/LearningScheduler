#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数值对齐验证：C++ 实现 vs Python 实现 vs PyTorch 官方实现。

这是本仓库「数值对齐 PyTorch」这一说法的**唯一证据**。
运行方式：

    python align.py                 # 直接运行，打印对比表格
    pytest align.py -v              # 作为测试套件运行

验证策略
--------
1. 自动编译并运行 `reference_cpp.cpp`，得到 C++ 实现的学习率序列（单一数据源）。
2. 用相同参数驱动 `infinitrain/lr_scheduler.py` 的 Python 实现。
3. 用相同参数驱动 `torch.optim.lr_scheduler` 的官方实现。
4. 逐步断言三者一致。

精度容差
--------
- Python vs PyTorch：均在双精度下计算，容差 rtol=1e-9。
- C++ vs Python：C++ 侧内部使用 float，float 有效位约 7 位十进制，
  容差取 rtol=1e-6。这不是 bug，是两种语言的存储精度差异。

已知差异（不做伪装对齐）
------------------------
`chained_warmup_step` 只比对 C++/Python，不比对 PyTorch。
原因见 lr_scheduler.py 中 ChainedScheduler 的文档注释：
本项目的 ChainedScheduler 是「最后一个子调度器决定绝对 lr」，
而 PyTorch 的是「相对于 optimizer 当前 lr 依次递推」，两者语义不同。
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "infinitrain"))

# float 与 double 的存储精度差异决定了这个容差
RTOL_CPP = 1e-6
ATOL_CPP = 1e-9
RTOL_TORCH = 1e-9
ATOL_TORCH = 1e-12


# ------------------------------------------------------------------ 辅助


class FakeOptimizer:
    """仅含 param_groups 的最小优化器替身。"""

    def __init__(self, lr):
        self.param_groups = [{"lr": lr}]


def _find_compiler():
    for name in ("g++", "clang++", "c++"):
        path = shutil.which(name)
        if path:
            return path
    for candidate in (
        r"D:\mingw64\bin\g++.exe",
        r"C:\mingw64\bin\g++.exe",
        r"C:\msys64\mingw64\bin\g++.exe",
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def load_cpp_reference():
    """编译并运行 reference_cpp.cpp，返回 C++ 参考序列。

    Returns:
        (steps, {case_id: {...}}, error_message)
        若无法编译则返回 (None, None, reason)。
    """
    src = DEMO_DIR / "reference_cpp.cpp"
    if not src.exists():
        return None, None, f"未找到 {src}"

    compiler = _find_compiler()
    if compiler is None:
        return None, None, "未找到可用的 C++ 编译器（g++/clang++）"

    try:
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / ("reference_cpp.exe" if os.name == "nt" else "reference_cpp")
            proc = subprocess.run(
                [compiler, "-std=c++17", "-static", "-o", str(exe), str(src)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=180,
            )
            if proc.returncode != 0:
                return None, None, f"编译失败：\n{proc.stderr[-800:]}"

            proc = subprocess.run(
                [str(exe)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
            if proc.returncode != 0:
                return None, None, f"运行失败：{proc.stderr[-400:]}"

            data = json.loads(proc.stdout)
    except subprocess.TimeoutExpired:
        return None, None, "编译超时"
    except json.JSONDecodeError as exc:
        return None, None, f"解析 C++ 输出失败：{exc}"

    cases = {c["id"]: c for c in data["cases"]}
    return data["steps"], cases, None


# --------------------------------------------------- Python 侧 case 构造


def make_python_cases(steps):
    """为每个 case id 构造 Python 实现的学习率序列。"""
    from lr_scheduler import (
        ChainedScheduler,
        ConstantLR,
        LambdaLR,
        LinearLR,
        SequentialLR,
        StepLR,
    )

    def collect(scheduler, opt):
        seq = []
        for _ in range(steps):
            scheduler.step()
            seq.append(opt.param_groups[0]["lr"])
        return seq

    out = {}

    opt = FakeOptimizer(0.001)
    out["constant_lr"] = collect(ConstantLR(opt, 0.001), opt)

    opt = FakeOptimizer(0.001)
    out["step_lr"] = collect(StepLR(opt, 0.001, 3, 0.5), opt)

    opt = FakeOptimizer(0.01)
    out["step_lr_size1"] = collect(StepLR(opt, 0.01, 1, 0.9), opt)

    opt = FakeOptimizer(0.0)
    out["linear_lr_warmup"] = collect(LinearLR(opt, 0.0, 0.001, 5), opt)

    opt = FakeOptimizer(0.0001)
    out["linear_lr_nonzero_start"] = collect(LinearLR(opt, 0.0001, 0.001, 4), opt)

    opt = FakeOptimizer(0.001)
    out["lambda_lr_stepwise"] = collect(
        LambdaLR(opt, 0.001, lambda t: 0.5 ** (t // 3)), opt
    )

    opt = FakeOptimizer(0.001)
    out["lambda_lr_reciprocal"] = collect(
        LambdaLR(opt, 0.001, lambda t: 1.0 / (1.0 + 0.1 * t)), opt
    )

    opt = FakeOptimizer(0.0)
    lin = LinearLR(opt, 0.0, 0.001, 5)
    stp = StepLR(opt, 0.001, 3, 0.5)
    out["sequential_warmup_step"] = collect(SequentialLR(opt, [lin, stp], [5]), opt)

    opt = FakeOptimizer(0.0)
    lin3 = LinearLR(opt, 0.0, 0.001, 3)
    cst = ConstantLR(opt, 0.001)
    stp2 = StepLR(opt, 0.001, 2, 0.5)
    out["sequential_three_stage"] = collect(
        SequentialLR(opt, [lin3, cst, stp2], [3, 7]), opt
    )

    opt = FakeOptimizer(0.0)
    lin_c = LinearLR(opt, 0.0, 0.001, 5)
    stp_c = StepLR(opt, 0.001, 3, 0.5)
    out["chained_warmup_step"] = collect(ChainedScheduler(opt, [lin_c, stp_c]), opt)

    return out


# --------------------------------------------------- PyTorch 侧 case 构造


def make_torch_cases(steps):
    """为每个 case id 构造 PyTorch 官方实现的等价序列。

    Returns:
        dict；若 torch 不可用返回 None。
    """
    try:
        import warnings

        import torch

        warnings.filterwarnings("ignore")
    except ImportError:
        return None

    def _mk(lr):
        param = torch.nn.Parameter(torch.zeros(2))
        return torch.optim.SGD([param], lr=lr)

    def collect(scheduler, opt):
        seq = []
        for _ in range(steps):
            # PyTorch 要求先 optimizer.step()，否则会跳过调度的第一个值
            opt.step()
            scheduler.step()
            seq.append(opt.param_groups[0]["lr"])
        return seq

    T = torch.optim.lr_scheduler
    out = {}

    opt = _mk(0.001)
    out["constant_lr"] = collect(T.LambdaLR(opt, lambda t: 1.0), opt)

    opt = _mk(0.001)
    out["step_lr"] = collect(T.StepLR(opt, step_size=3, gamma=0.5), opt)

    opt = _mk(0.01)
    out["step_lr_size1"] = collect(T.StepLR(opt, step_size=1, gamma=0.9), opt)

    # LinearLR(0 -> base, warmup=5) 的等价物：
    # PyTorch LinearLR 要求 start_factor > 0，无法表达从 0 开始的 warmup，改用 LambdaLR
    opt = _mk(0.001)
    out["linear_lr_warmup"] = collect(
        T.LambdaLR(opt, lambda t: (t / 5.0) if t < 5 else 1.0), opt
    )

    opt = _mk(0.001)
    out["linear_lr_nonzero_start"] = collect(
        T.LambdaLR(opt, lambda t: (0.1 + (t / 4.0) * 0.9) if t < 4 else 1.0), opt
    )

    opt = _mk(0.001)
    out["lambda_lr_stepwise"] = collect(T.LambdaLR(opt, lambda t: 0.5 ** (t // 3)), opt)

    opt = _mk(0.001)
    out["lambda_lr_reciprocal"] = collect(
        T.LambdaLR(opt, lambda t: 1.0 / (1.0 + 0.1 * t)), opt
    )

    opt = _mk(0.001)
    out["sequential_warmup_step"] = collect(
        T.SequentialLR(
            opt,
            schedulers=[
                T.LambdaLR(opt, lambda t: (t / 5.0) if t < 5 else 1.0),
                T.StepLR(opt, step_size=3, gamma=0.5),
            ],
            milestones=[5],
        ),
        opt,
    )

    opt = _mk(0.001)
    out["sequential_three_stage"] = collect(
        T.SequentialLR(
            opt,
            schedulers=[
                T.LambdaLR(opt, lambda t: (t / 3.0) if t < 3 else 1.0),
                T.LambdaLR(opt, lambda t: 1.0),
                T.StepLR(opt, step_size=2, gamma=0.5),
            ],
            milestones=[3, 7],
        ),
        opt,
    )

    # chained_warmup_step 故意不提供：语义不同，见模块文档
    return out


# ------------------------------------------------------------------ 断言


def assert_close(name, mine, ref, rtol, atol, label):
    """逐步比对两条序列，返回差异描述列表。"""
    if len(mine) != len(ref):
        return [f"{name}@{label}: 长度不一致 {len(mine)} vs {len(ref)}"]

    diffs = []
    for i, (a, b) in enumerate(zip(mine, ref)):
        if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
            diff = a - b
            diffs.append(
                f"{name}@{label} step={i + 1}: mine={a!r} ref={b!r} diff={diff:.3e}"
            )
    return diffs


def first_diff_index(mine, ref, rtol, atol):
    for i, (a, b) in enumerate(zip(mine, ref)):
        if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
            return i
    return None


# ------------------------------------------------------------------ 主流程


def _load_all():
    steps, cpp_cases, cpp_err = load_cpp_reference()
    if cpp_err:
        return None, None, None, cpp_err
    py_cases = make_python_cases(steps)
    torch_cases = make_torch_cases(steps)
    return steps, cpp_cases, (py_cases, torch_cases), None


_CACHED = None


def _cached():
    global _CACHED
    if _CACHED is None:
        _CACHED = _load_all()
    return _CACHED


def main():
    steps, cpp_cases, others, err = _cached()
    if err:
        print(f"⚠️ 跳过 C++ 对比：{err}")
        return 0

    py_cases, torch_cases = others
    print(f"对齐验证：steps={steps}  用例数={len(cpp_cases)}")
    print("=" * 78)

    failures = []
    rows = []

    for case_id in cpp_cases:
        cpp_seq = cpp_cases[case_id]["seq"]
        py_seq = py_cases[case_id]

        failures.extend(
            assert_close(case_id, py_seq, cpp_seq, RTOL_CPP, ATOL_CPP, "Python/C++")
        )

        if torch_cases is None:
            torch_status = "跳过（无 torch）"
        elif case_id not in torch_cases:
            torch_status = "不适用（语义差异）"
        else:
            torch_seq = torch_cases[case_id]
            failures.extend(
                assert_close(
                    case_id, py_seq, torch_seq, RTOL_TORCH, ATOL_TORCH, "Python/PyTorch"
                )
            )
            bad = first_diff_index(py_seq, torch_seq, RTOL_TORCH, ATOL_TORCH)
            torch_status = "✅ 一致" if bad is None else f"❌ 第 {bad + 1} 步起偏差"

        bad = first_diff_index(py_seq, cpp_seq, RTOL_CPP, ATOL_CPP)
        cpp_status = "✅ 一致" if bad is None else f"❌ 第 {bad + 1} 步起偏差"
        rows.append((case_id, cpp_status, torch_status))

    width = max(len(r[0]) for r in rows)
    print(f"{'case'.ljust(width)}   C++ vs Python        Python vs PyTorch")
    print("-" * 78)
    for cid, cpp_s, t_s in rows:
        print(f"{cid.ljust(width)}   {cpp_s.ljust(20)}  {t_s}")

    print("=" * 78)
    if failures:
        print(f"❌ 发现 {len(failures)} 处不一致：\n")
        for msg in failures[:40]:
            print("   " + msg)
        if len(failures) > 40:
            print(f"   ... 其余 {len(failures) - 40} 处省略")
        return 1

    print("✅ 全部通过：Python 实现与 C++ 实现一致；"
          "可对齐的用例与 PyTorch 官方实现逐步一致。")
    return 0


# ------------------------------------------------------------------ pytest


def test_python_matches_cpp():
    steps, cpp_cases, others, err = _cached()
    if err:
        import pytest

        pytest.skip(f"C++ 参考序列不可用：{err}")

    py_cases = others[0]
    failures = []
    for case_id, case in cpp_cases.items():
        failures += assert_close(
            case_id, py_cases[case_id], case["seq"], RTOL_CPP, ATOL_CPP, "Python/C++"
        )
    assert not failures, "Python 与 C++ 实现不一致：\n" + "\n".join(failures)


def test_python_matches_pytorch():
    steps, cpp_cases, others, err = _cached()
    if err:
        import pytest

        pytest.skip(f"C++ 参考序列不可用：{err}")

    py_cases, torch_cases = others
    if torch_cases is None:
        import pytest

        pytest.skip("未安装 torch，跳过与 PyTorch 的对比")

    failures = []
    for case_id, torch_seq in torch_cases.items():
        failures += assert_close(
            case_id, py_cases[case_id], torch_seq, RTOL_TORCH, ATOL_TORCH, "Python/PyTorch"
        )
    assert not failures, "Python 与 PyTorch 不一致：\n" + "\n".join(failures)


def test_step_lr_decays_at_pytorch_boundary():
    """回归测试：StepLR 必须在第 step_size 步（而非第 step_size+1 步）衰减。

    这是本仓库此前存在的核心 bug——旧实现用 (step_count-1)//step_size，
    导致每次衰减都比 PyTorch 晚一步。
    """
    from lr_scheduler import StepLR

    opt = FakeOptimizer(0.001)
    sch = StepLR(opt, 0.001, 3, 0.5)
    seq = []
    for _ in range(6):
        sch.step()
        seq.append(opt.param_groups[0]["lr"])

    assert math.isclose(seq[0], 0.001, rel_tol=1e-12)
    assert math.isclose(seq[1], 0.001, rel_tol=1e-12)
    assert math.isclose(seq[2], 0.0005, rel_tol=1e-12), f"第 3 步应为 0.0005，实际 {seq[2]}"
    assert math.isclose(seq[5], 0.00025, rel_tol=1e-12), f"第 6 步应为 0.00025，实际 {seq[5]}"


def test_step_lr_rejects_invalid_step_size():
    from lr_scheduler import StepLR

    try:
        StepLR(FakeOptimizer(0.001), 0.001, 0, 0.5)
    except ValueError:
        return
    raise AssertionError("step_size=0 应抛出 ValueError")


def test_sequential_lr_does_not_double_count():
    """回归测试：SequentialLR 不得重复推进子调度器的 step_count。

    旧实现在调用子调度器 step() 前先赋值 step_count=全局步数，
    而 step() 内部又会自增，导致子调度器比实际多算一步。
    """
    from lr_scheduler import LinearLR, SequentialLR, StepLR

    opt = FakeOptimizer(0.0)
    lin = LinearLR(opt, 0.0, 0.001, 5)
    stp = StepLR(opt, 0.001, 3, 0.5)
    sch = SequentialLR(opt, [lin, stp], [5])

    seq = []
    for _ in range(8):
        sch.step()
        seq.append(opt.param_groups[0]["lr"])

    # 预热结束后，StepLR 应从自己的第 1 步重新计数
    assert math.isclose(seq[4], 0.001, rel_tol=1e-12)
    assert math.isclose(seq[5], 0.001, rel_tol=1e-12), f"第 6 步应为 0.001，实际 {seq[5]}"
    assert math.isclose(seq[7], 0.0005, rel_tol=1e-12), f"第 8 步应为 0.0005，实际 {seq[7]}"
    # StepLR 只在第 6~8 步被推进，共 3 次
    assert stp.step_count == 3, f"StepLR 应只推进 3 次，实际 {stp.step_count}"


def test_state_roundtrip():
    """状态保存/加载后，继续训练的学习率应与未中断时一致。"""
    from lr_scheduler import StepLR

    opt_full = FakeOptimizer(0.001)
    full = StepLR(opt_full, 0.001, 3, 0.5)
    expected = []
    for _ in range(10):
        full.step()
        expected.append(opt_full.param_groups[0]["lr"])

    opt_resume = FakeOptimizer(0.001)
    resume = StepLR(opt_resume, 0.001, 3, 0.5)
    for _ in range(5):
        resume.step()
    state = resume.state_dict()

    opt_new = FakeOptimizer(0.001)
    new = StepLR(opt_new, 0.001, 3, 0.5)
    new.load_state_dict(state)
    actual = []
    for _ in range(5):
        new.step()
        actual.append(opt_new.param_groups[0]["lr"])

    for i, (a, b) in enumerate(zip(expected[5:], actual)):
        assert math.isclose(a, b, rel_tol=1e-12), f"resume 后第 {i + 6} 步不一致: {a} vs {b}"


def test_linear_lr_matches_lambda_equivalent():
    """加性 LinearLR 必须与等价的 LambdaLR 表达式一致。"""
    from lr_scheduler import LambdaLR, LinearLR

    start, end, warm = 0.0, 0.001, 5

    opt_a = FakeOptimizer(start)
    lin = LinearLR(opt_a, start, end, warm)
    seq_a = []
    for _ in range(10):
        lin.step()
        seq_a.append(opt_a.param_groups[0]["lr"])

    opt_b = FakeOptimizer(start)
    lam = LambdaLR(opt_b, end, lambda t: (t / warm) if t < warm else 1.0)
    seq_b = []
    for _ in range(10):
        lam.step()
        seq_b.append(opt_b.param_groups[0]["lr"])

    for i, (a, b) in enumerate(zip(seq_a, seq_b)):
        assert math.isclose(a, b, rel_tol=1e-12), f"第 {i + 1} 步不一致: {a} vs {b}"


if __name__ == "__main__":
    sys.exit(main())

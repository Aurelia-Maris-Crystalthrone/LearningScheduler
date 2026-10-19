// 参考序列生成器：输出本仓库 C++ 实现的学习率序列，供 demo/align.py 做三方对比。
//
// 编译：
//   g++ -std=c++17 -static -o reference_cpp reference_cpp.cpp
//
// 输出：JSON，形如 {"steps": N, "cases": [{"id": ..., "params": {...}, "seq": [...]}, ...]}
//
// 本文件是 align.py 的**唯一数据源**：Python 侧与 PyTorch 侧的用例必须
// 与这里保持一致，由 align.py 按 id 分派构造，避免多处重复定义参数。

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "lr_scheduler.h"

// 测试用优化器：不产生任何输出，避免污染 stdout
class SilentOptimizer : public Optimizer {
public:
    explicit SilentOptimizer(double lr) : lr_(static_cast<float>(lr)) {}
    void SetLearningRate(float lr) override { lr_ = lr; }
    float GetLearningRate() const override { return lr_; }

private:
    float lr_;
};

// 跑 total_steps 步并收集每步之后的学习率
std::vector<float> Collect(LRScheduler* sched, int total_steps) {
    std::vector<float> seq;
    seq.reserve(total_steps);
    for (int i = 0; i < total_steps; ++i) {
        sched->Step();
        seq.push_back(sched->GetLR());
    }
    return seq;
}

std::string SeqToJson(const std::vector<float>& seq) {
    std::string out = "[";
    for (size_t i = 0; i < seq.size(); ++i) {
        if (i) out += ",";
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.10f", seq[i]);
        out += buf;
    }
    out += "]";
    return out;
}

void Emit(const std::string& id, const std::string& params, const std::vector<float>& seq) {
    std::printf("{\"id\":\"%s\",\"params\":%s,\"seq\":%s}", id.c_str(), params.c_str(),
                SeqToJson(seq).c_str());
}

int main() {
    const int steps = 12;
    const double base = 0.001;
    const double gamma = 0.5, warmup = 5.0;

    std::printf("{\"steps\":%d,\"cases\":[", steps);
    bool first = true;
    auto sep = [&first]() {
        if (!first) std::printf(",");
        first = false;
    };

    // ---- 1. ConstantLR ----
    {
        SilentOptimizer opt(base);
        ConstantLR s(&opt, static_cast<float>(base));
        auto seq = Collect(&s, steps);
        sep();
        Emit("constant_lr", "{\"base_lr\":0.001}", seq);
    }

    // ---- 2. StepLR(base, step_size=3, gamma=0.5) ----
    {
        SilentOptimizer opt(base);
        StepLR s(&opt, static_cast<float>(base), 3, static_cast<float>(gamma));
        auto seq = Collect(&s, steps);
        sep();
        Emit("step_lr", "{\"base_lr\":0.001,\"step_size\":3,\"gamma\":0.5}", seq);
    }

    // ---- 3. StepLR 小 step_size ----
    {
        SilentOptimizer opt(0.01);
        StepLR s(&opt, 0.01f, 1, 0.9f);
        auto seq = Collect(&s, steps);
        sep();
        Emit("step_lr_size1", "{\"base_lr\":0.01,\"step_size\":1,\"gamma\":0.9}", seq);
    }

    // ---- 4. LinearLR warmup from 0 ----
    {
        SilentOptimizer opt(0.0);
        LinearLR s(&opt, 0.0f, static_cast<float>(base), static_cast<int>(warmup));
        auto seq = Collect(&s, steps);
        sep();
        Emit("linear_lr_warmup", "{\"start_lr\":0.0,\"end_lr\":0.001,\"warmup_steps\":5}", seq);
    }

    // ---- 5. LinearLR 非零起点 ----
    {
        SilentOptimizer opt(0.0001);
        LinearLR s(&opt, 0.0001f, static_cast<float>(base), 4);
        auto seq = Collect(&s, steps);
        sep();
        Emit("linear_lr_nonzero_start",
             "{\"start_lr\":0.0001,\"end_lr\":0.001,\"warmup_steps\":4}", seq);
    }

    // ---- 6. LambdaLR 指数衰减 -> 等价于 StepLR ----
    {
        SilentOptimizer opt(base);
        LambdaLR s(&opt, static_cast<float>(base),
                   [](int t) { return std::pow(0.5f, static_cast<float>(t / 3)); });
        auto seq = Collect(&s, steps);
        sep();
        Emit("lambda_lr_stepwise", "{\"base_lr\":0.001,\"step_size\":3,\"gamma\":0.5}", seq);
    }

    // ---- 7. LambdaLR 倒数衰减 ----
    {
        SilentOptimizer opt(base);
        LambdaLR s(&opt, static_cast<float>(base),
                   [](int t) { return 1.0f / (1.0f + 0.1f * static_cast<float>(t)); });
        auto seq = Collect(&s, steps);
        sep();
        Emit("lambda_lr_reciprocal", "{\"base_lr\":0.001,\"scale\":0.1}", seq);
    }

    // ---- 8. SequentialLR: LinearLR warmup -> StepLR, milestone [5] ----
    {
        SilentOptimizer opt(0.0);
        auto* lin = new LinearLR(&opt, 0.0f, static_cast<float>(base), 5);
        auto* step = new StepLR(&opt, static_cast<float>(base), 3, static_cast<float>(gamma));
        SequentialLR s(&opt, {lin, step}, {5});
        auto seq = Collect(&s, steps);
        sep();
        Emit("sequential_warmup_step",
             "{\"base_lr\":0.001,\"warmup_steps\":5,\"step_size\":3,\"gamma\":0.5}", seq);
        delete lin;
        delete step;
    }

    // ---- 9. SequentialLR 三段：warmup -> constant -> step, milestones [3,7] ----
    {
        SilentOptimizer opt(0.0);
        auto* lin = new LinearLR(&opt, 0.0f, static_cast<float>(base), 3);
        auto* cst = new ConstantLR(&opt, static_cast<float>(base));
        auto* step = new StepLR(&opt, static_cast<float>(base), 2, static_cast<float>(gamma));
        SequentialLR s(&opt, {lin, cst, step}, {3, 7});
        auto seq = Collect(&s, steps);
        sep();
        Emit("sequential_three_stage",
             "{\"base_lr\":0.001,\"warmup_steps\":3,\"step_size\":2,\"gamma\":0.5,"
             "\"milestones\":[3,7]}",
             seq);
        delete lin;
        delete cst;
        delete step;
    }

    // ---- 10. ChainedScheduler（与 PyTorch 语义不同，仅保证 C++/Python 两侧一致）----
    {
        SilentOptimizer opt(0.0);
        auto* lin = new LinearLR(&opt, 0.0f, static_cast<float>(base), 5);
        auto* step = new StepLR(&opt, static_cast<float>(base), 3, static_cast<float>(gamma));
        ChainedScheduler s(&opt, {lin, step});
        auto seq = Collect(&s, steps);
        sep();
        Emit("chained_warmup_step",
             "{\"base_lr\":0.001,\"warmup_steps\":5,\"step_size\":3,\"gamma\":0.5}", seq);
        delete lin;
        delete step;
    }

    std::printf("]}\n");
    return 0;
}

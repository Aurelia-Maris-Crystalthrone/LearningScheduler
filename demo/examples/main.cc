#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <gflags/gflags.h>
#include "lr_scheduler.h"

// 命令行参数定义
DEFINE_string(lr_scheduler, "constant",
              "Type of LR scheduler: constant, steplr, linear, sequential, chained, lambda");
DEFINE_double(learning_rate, 0.001, "Initial learning rate");
DEFINE_int32(warmup_steps, 0, "Warmup steps for linear warmup");
DEFINE_int32(step_size, 30, "Step size for StepLR");
DEFINE_double(gamma, 0.1, "Gamma for StepLR");
DEFINE_string(scheduler_milestones, "",
              "Comma-separated milestones for SequentialLR (e.g., 1000,2000)");
DEFINE_string(scheduler_list, "",
              "Comma-separated scheduler types for Sequential/Chained (e.g., linear,steplr)");
DEFINE_int32(total_steps, 100, "Number of steps to simulate");

// 简单的 Optimizer 实现。
// 注意：不在此处打印日志——SetLearningRate 会被高频调用，且打印会污染 stdout，
// 使本程序的输出无法被其他工具解析。
class SimpleOptimizer : public Optimizer {
public:
    explicit SimpleOptimizer(float lr) : lr_(lr) {}
    void SetLearningRate(float lr) override { lr_ = lr; }
    float GetLearningRate() const override { return lr_; }

private:
    float lr_;
};

// 将逗号分隔的字符串切分为片段
std::vector<std::string> SplitComma(const std::string& str) {
    std::vector<std::string> result;
    size_t start = 0;
    while (start < str.length()) {
        size_t comma = str.find(',', start);
        if (comma == std::string::npos) {
            result.push_back(str.substr(start));
            break;
        }
        // 支持空片段（如 "a,,b"），保持元素数量与里程碑数量可对应
        result.push_back(str.substr(start, comma - start));
        start = comma + 1;
    }
    return result;
}

std::vector<int> ParseCommaSeparatedInts(const std::string& str) {
    std::vector<int> result;
    for (const auto& part : SplitComma(str)) {
        if (part.empty()) continue;
        result.push_back(std::stoi(part));
    }
    return result;
}

std::vector<std::string> ParseSchedulerTypes(const std::string& str) {
    std::vector<std::string> result;
    for (const auto& part : SplitComma(str)) {
        if (part.empty()) continue;
        result.push_back(part);
    }
    return result;
}

// 辅助函数：根据字符串创建调度器（工厂模式）
std::unique_ptr<LRScheduler> CreateScheduler(const std::string& type, Optimizer* opt,
                                             float base_lr, int warmup_steps,
                                             int step_size, float gamma) {
    if (type == "constant") {
        return std::make_unique<ConstantLR>(opt, base_lr);
    }
    if (type == "steplr") {
        return std::make_unique<StepLR>(opt, base_lr, step_size, gamma);
    }
    if (type == "linear") {
        // 线性从 0 增长到 base_lr
        return std::make_unique<LinearLR>(opt, 0.0f, base_lr, warmup_steps);
    }
    if (type == "lambda") {
        // 示例：每步衰减 0.99
        return std::make_unique<LambdaLR>(opt, base_lr, [](int step) {
            return static_cast<float>(std::pow(0.99, step));
        });
    }
    throw std::invalid_argument("Unknown scheduler type: " + type);
}

// 组合型调度器需要多个子调度器，统一在此构建。
// 使用 unique_ptr 持有子调度器：
// SequentialLR / ChainedScheduler 不拥有子调度器，若此处用裸指针 new
// 且后续不 delete，就会泄漏——这正是本示例之前的 bug。
struct CompositeScheduler {
    std::vector<std::unique_ptr<LRScheduler>> owners;
    std::vector<LRScheduler*> raw_ptrs;
};

CompositeScheduler BuildComposite(const std::vector<std::string>& types, Optimizer* opt,
                                  float base_lr, int warmup_steps, int step_size, float gamma) {
    CompositeScheduler comp;
    for (const auto& t : types) {
        comp.owners.push_back(CreateScheduler(t, opt, base_lr, warmup_steps, step_size, gamma));
        comp.raw_ptrs.push_back(comp.owners.back().get());
    }
    return comp;
}

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    SimpleOptimizer opt(static_cast<float>(FLAGS_learning_rate));
    std::cout << "Initial LR: " << opt.GetLearningRate() << std::endl;

    std::unique_ptr<LRScheduler> scheduler;
    // 组合调度器的子对象需要比 scheduler 存活得更久，因此在最外层声明
    CompositeScheduler composite;

    if (FLAGS_lr_scheduler == "sequential" || FLAGS_lr_scheduler == "chained") {
        std::vector<std::string> types = ParseSchedulerTypes(FLAGS_scheduler_list);
        std::vector<int> milestones = ParseCommaSeparatedInts(FLAGS_scheduler_milestones);

        composite = BuildComposite(types, &opt, static_cast<float>(FLAGS_learning_rate),
                                   FLAGS_warmup_steps, FLAGS_step_size,
                                   static_cast<float>(FLAGS_gamma));

        if (FLAGS_lr_scheduler == "sequential") {
            if (types.size() != milestones.size() + 1) {
                std::cerr << "SequentialLR requires len(scheduler_list) == "
                             "len(scheduler_milestones) + 1, got "
                          << types.size() << " vs " << milestones.size() << std::endl;
                return 1;
            }
            scheduler = std::make_unique<SequentialLR>(&opt, composite.raw_ptrs, milestones);
        } else {  // chained
            scheduler = std::make_unique<ChainedScheduler>(&opt, composite.raw_ptrs);
        }
    } else {
        scheduler = CreateScheduler(FLAGS_lr_scheduler, &opt,
                                    static_cast<float>(FLAGS_learning_rate),
                                    FLAGS_warmup_steps, FLAGS_step_size,
                                    static_cast<float>(FLAGS_gamma));
    }

    // 模拟训练循环
    for (int step = 1; step <= FLAGS_total_steps; ++step) {
        scheduler->Step();
        std::cout << "Step " << step << ": LR = " << scheduler->GetLR() << std::endl;
    }

    // 演示状态保存与加载
    std::cout << "\n--- Checkpoint example ---" << std::endl;
    StateDict state = scheduler->State();
    std::cout << "Saved state (first 5 entries):" << std::endl;
    int cnt = 0;
    for (const auto& kv : state) {
        if (cnt++ >= 5) break;
        std::cout << "  " << kv.first << " = " << kv.second << std::endl;
    }

    // 用相同配置重建一个新调度器并加载状态
    std::unique_ptr<LRScheduler> new_sched;
    CompositeScheduler composite2;

    if (FLAGS_lr_scheduler == "sequential" || FLAGS_lr_scheduler == "chained") {
        std::vector<std::string> types = ParseSchedulerTypes(FLAGS_scheduler_list);
        std::vector<int> milestones = ParseCommaSeparatedInts(FLAGS_scheduler_milestones);
        composite2 = BuildComposite(types, &opt, static_cast<float>(FLAGS_learning_rate),
                                    FLAGS_warmup_steps, FLAGS_step_size,
                                    static_cast<float>(FLAGS_gamma));
        if (FLAGS_lr_scheduler == "sequential") {
            new_sched = std::make_unique<SequentialLR>(&opt, composite2.raw_ptrs, milestones);
        } else {
            new_sched = std::make_unique<ChainedScheduler>(&opt, composite2.raw_ptrs);
        }
    } else {
        new_sched = CreateScheduler(FLAGS_lr_scheduler, &opt,
                                    static_cast<float>(FLAGS_learning_rate),
                                    FLAGS_warmup_steps, FLAGS_step_size,
                                    static_cast<float>(FLAGS_gamma));
    }

    new_sched->LoadState(state);
    std::cout << "After loading state, new scheduler LR = " << new_sched->GetLR() << std::endl;
    std::cout << "Optimizer LR after load = " << opt.GetLearningRate() << std::endl;

    // 所有资源由 unique_ptr 自动释放，无需手动 delete
    return 0;
}

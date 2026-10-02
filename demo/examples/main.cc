#include <iostream>
#include <string>
#include <vector>
#include <gflags/gflags.h>  
#include "lr_scheduler.h"

// 命令行参数定义
DEFINE_string(lr_scheduler, "constant", "Type of LR scheduler: constant, steplr, linear, sequential, chained, lambda");
DEFINE_double(learning_rate, 0.001, "Initial learning rate");
DEFINE_int32(warmup_steps, 0, "Warmup steps for linear warmup");
DEFINE_int32(step_size, 30, "Step size for StepLR");
DEFINE_double(gamma, 0.1, "Gamma for StepLR");
DEFINE_string(scheduler_milestones, "", "Comma-separated milestones for SequentialLR (e.g., 1000,2000)");
DEFINE_string(scheduler_list, "", "Comma-separated scheduler types for Sequential/Chained (e.g., linear,steplr)");
DEFINE_int32(total_steps, 100, "Number of steps to simulate");

// 简单的 Optimizer 实现
class SimpleOptimizer : public Optimizer {
public:
    SimpleOptimizer(float lr) : lr_(lr) {}
    virtual void SetLearningRate(float lr) override { 
        lr_ = lr;
        std::cout << "Optimizer LR set to " << lr_ << std::endl;
    }
    virtual float GetLearningRate() const override { return lr_; }
private:
    float lr_;
};

// 辅助函数：将逗号分隔的字符串解析为整数向量
std::vector<int> ParseCommaSeparatedInts(const std::string& str) {
    std::vector<int> result;
    size_t start = 0, end;
    while ((end = str.find(',', start)) != std::string::npos) {
        result.push_back(std::stoi(str.substr(start, end - start)));
        start = end + 1;
    }
    if (start < str.length()) {
        result.push_back(std::stoi(str.substr(start)));
    }
    return result;
}

// 辅助函数：根据字符串创建调度器（工厂模式）
LRScheduler* CreateScheduler(const std::string& type, Optimizer* opt, 
                              float base_lr, int warmup_steps, int step_size, float gamma) {
    if (type == "constant") {
        return new ConstantLR(opt, base_lr);
    } else if (type == "steplr") {
        return new StepLR(opt, base_lr, step_size, gamma);
    } else if (type == "linear") {
        // 假设线性从0增长到base_lr
        return new LinearLR(opt, 0.0f, base_lr, warmup_steps);
    } else if (type == "lambda") {
        // 示例：每步衰减0.99
        return new LambdaLR(opt, base_lr, [](int step) { return 0.99f; });
    } else {
        throw std::invalid_argument("Unknown scheduler type: " + type);
    }
}

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // 创建优化器
    SimpleOptimizer opt(FLAGS_learning_rate);
    std::cout << "Initial LR: " << opt.GetLearningRate() << std::endl;

    LRScheduler* scheduler = nullptr;

    // 根据命令行参数创建调度器
    if (FLAGS_lr_scheduler == "sequential" || FLAGS_lr_scheduler == "chained") {
        // 解析 scheduler_list 和 milestones
        std::vector<std::string> types;
        std::string list = FLAGS_scheduler_list;
        size_t start = 0, end;
        while ((end = list.find(',', start)) != std::string::npos) {
            types.push_back(list.substr(start, end - start));
            start = end + 1;
        }
        if (start < list.length()) types.push_back(list.substr(start));

        std::vector<int> milestones = ParseCommaSeparatedInts(FLAGS_scheduler_milestones);

        std::vector<LRScheduler*> sub_schedulers;
        for (const auto& t : types) {
            sub_schedulers.push_back(CreateScheduler(t, &opt, FLAGS_learning_rate, 
                                                      FLAGS_warmup_steps, FLAGS_step_size, FLAGS_gamma));
        }

        if (FLAGS_lr_scheduler == "sequential") {
            scheduler = new SequentialLR(&opt, sub_schedulers, milestones);
        } else { // chained
            scheduler = new ChainedScheduler(&opt, sub_schedulers);
        }
    } else {
        // 单个调度器
        scheduler = CreateScheduler(FLAGS_lr_scheduler, &opt, FLAGS_learning_rate,
                                     FLAGS_warmup_steps, FLAGS_step_size, FLAGS_gamma);
    }

    // 模拟训练循环
    for (int step = 1; step <= FLAGS_total_steps; ++step) {
        scheduler->Step();
        std::cout << "Step " << step << ": LR = " << opt.GetLearningRate() << std::endl;
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

    // 创建一个新调度器并加载状态（假设相同配置）
    LRScheduler* new_sched = nullptr;
    if (FLAGS_lr_scheduler == "sequential") {
        // 重新创建子调度器
        std::vector<std::string> types;
        std::string list = FLAGS_scheduler_list;
        size_t start = 0, end;
        while ((end = list.find(',', start)) != std::string::npos) {
            types.push_back(list.substr(start, end - start));
            start = end + 1;
        }
        if (start < list.length()) types.push_back(list.substr(start));
        std::vector<int> milestones = ParseCommaSeparatedInts(FLAGS_scheduler_milestones);
        std::vector<LRScheduler*> sub_schedulers;
        for (const auto& t : types) {
            sub_schedulers.push_back(CreateScheduler(t, &opt, FLAGS_learning_rate,
                                                      FLAGS_warmup_steps, FLAGS_step_size, FLAGS_gamma));
        }
        new_sched = new SequentialLR(&opt, sub_schedulers, milestones);
    } else if (FLAGS_lr_scheduler == "chained") {
        // 类似...
        std::vector<std::string> types;
        std::string list = FLAGS_scheduler_list;
        size_t start = 0, end;
        while ((end = list.find(',', start)) != std::string::npos) {
            types.push_back(list.substr(start, end - start));
            start = end + 1;
        }
        if (start < list.length()) types.push_back(list.substr(start));
        std::vector<LRScheduler*> sub_schedulers;
        for (const auto& t : types) {
            sub_schedulers.push_back(CreateScheduler(t, &opt, FLAGS_learning_rate,
                                                      FLAGS_warmup_steps, FLAGS_step_size, FLAGS_gamma));
        }
        new_sched = new ChainedScheduler(&opt, sub_schedulers);
    } else {
        new_sched = CreateScheduler(FLAGS_lr_scheduler, &opt, FLAGS_learning_rate,
                                     FLAGS_warmup_steps, FLAGS_step_size, FLAGS_gamma);
    }

    new_sched->LoadState(state);
    std::cout << "After loading state, new scheduler LR = " << new_sched->GetLR() << std::endl;

    delete scheduler;
    delete new_sched;
    return 0;
}
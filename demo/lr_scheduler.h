#ifndef INFINITRAIN_LR_SCHEDULER_H_
#define INFINITRAIN_LR_SCHEDULER_H_

#include <map>
#include <string>
#include <vector>
#include <functional>
#include <cmath>
#include <stdexcept>

// 前向声明 Optimizer 类（需由用户实现）
class Optimizer {
public:
    virtual void SetLearningRate(float lr) = 0;
    virtual float GetLearningRate() const = 0;
    virtual ~Optimizer() = default;
};

// 状态字典类型：使用浮点数保存所有状态（整数会隐式转换）
using StateDict = std::map<std::string, float>;

/**
 * @brief 学习率调度器抽象基类
 * 
 * 所有具体调度器需继承此类并实现 ComputeLR() 方法。
 * 调度器持有 Optimizer 指针，在 Step() 时计算新学习率并设置到优化器。
 * 支持状态保存与加载，便于 checkpoint/resume。
 */
class LRScheduler {
public:
    /**
     * @param optimizer 关联的优化器（非 owning 指针）
     */
    explicit LRScheduler(Optimizer* optimizer)
        : optimizer_(optimizer), last_lr_(0.0f), step_count_(0) {}

    virtual ~LRScheduler() = default;

    /**
     * @brief 推进调度器一步，更新学习率
     * 
     * 调用 ComputeLR() 计算新学习率，并通过 optimizer_->SetLearningRate() 设置。
     * step_count_ 在调用 ComputeLR() 前递增。
     */
    virtual void Step() {
        step_count_++;
        last_lr_ = ComputeLR();
        optimizer_->SetLearningRate(last_lr_);
    }

    /**
     * @brief 获取当前学习率（最近一次设置的值）
     */
    virtual float GetLR() const { return last_lr_; }

    /**
     * @brief 导出调度器内部状态
     * @return 状态字典（键值对）
     */
    virtual StateDict State() const {
        return {
            {"last_lr", last_lr_},
            {"step_count", static_cast<float>(step_count_)}
        };
    }

    /**
     * @brief 从状态字典加载调度器状态
     * @param state 之前保存的状态字典
     */
    virtual void LoadState(const StateDict& state) {
        auto it = state.find("last_lr");
        if (it != state.end()) last_lr_ = it->second;
        it = state.find("step_count");
        if (it != state.end()) step_count_ = static_cast<int>(it->second);
    }

protected:
    /**
     * @brief 由派生类实现具体学习率计算公式
     * @return 当前步应使用的学习率
     */
    virtual float ComputeLR() = 0;

    Optimizer* optimizer_;      // 关联的优化器（派生类可能需要访问）
    float last_lr_;             // 最近一次设置的学习率
    int step_count_;            // 当前步数（从1开始）
};

// ==================== 基础调度策略 ====================

/**
 * @brief 常数学习率调度器
 */
class ConstantLR : public LRScheduler {
public:
    /**
     * @param optimizer 优化器指针
     * @param lr 常数学习率
     */
    ConstantLR(Optimizer* optimizer, float lr)
        : LRScheduler(optimizer), constant_lr_(lr) {
        last_lr_ = constant_lr_;
        optimizer_->SetLearningRate(last_lr_);
    }

    virtual StateDict State() const override {
        auto state = LRScheduler::State();
        state["constant_lr"] = constant_lr_;
        return state;
    }

    virtual void LoadState(const StateDict& state) override {
        LRScheduler::LoadState(state);
        auto it = state.find("constant_lr");
        if (it != state.end()) constant_lr_ = it->second;
    }

protected:
    virtual float ComputeLR() override {
        return constant_lr_;
    }

private:
    float constant_lr_;
};

/**
 * @brief 步进衰减学习率调度器
 * 
 * 每 step_size 步，学习率乘以 gamma。
 * lr = initial_lr * gamma^(floor((step-1)/step_size))
 */
class StepLR : public LRScheduler {
public:
    /**
     * @param optimizer 优化器指针
     * @param initial_lr 初始学习率
     * @param step_size 衰减步长
     * @param gamma 衰减因子
     */
    StepLR(Optimizer* optimizer, float initial_lr, int step_size, float gamma)
        : LRScheduler(optimizer), initial_lr_(initial_lr),
          step_size_(step_size), gamma_(gamma) {
        last_lr_ = initial_lr_;
        optimizer_->SetLearningRate(last_lr_);
    }

    virtual StateDict State() const override {
        auto state = LRScheduler::State();
        state["initial_lr"] = initial_lr_;
        state["step_size"] = static_cast<float>(step_size_);
        state["gamma"] = gamma_;
        return state;
    }

    virtual void LoadState(const StateDict& state) override {
        LRScheduler::LoadState(state);
        initial_lr_ = state.at("initial_lr");
        step_size_ = static_cast<int>(state.at("step_size"));
        gamma_ = state.at("gamma");
    }

protected:
    virtual float ComputeLR() override {
        // 使用 step_count_ 从1开始，公式为 gamma^((step-1)//step_size)
        int exponent = (step_count_ - 1) / step_size_;
        return initial_lr_ * std::pow(gamma_, exponent);
    }

private:
    float initial_lr_;
    int step_size_;
    float gamma_;
};

/**
 * @brief 线性 warmup 调度器
 * 
 * 在 warmup_steps 内从 start_lr 线性增长到 end_lr，之后保持 end_lr。
 */
class LinearLR : public LRScheduler {
public:
    /**
     * @param optimizer 优化器指针
     * @param start_lr 起始学习率（通常为0或很小）
     * @param end_lr 目标学习率
     * @param warmup_steps warmup 步数
     */
    LinearLR(Optimizer* optimizer, float start_lr, float end_lr, int warmup_steps)
        : LRScheduler(optimizer), start_lr_(start_lr), end_lr_(end_lr),
          warmup_steps_(warmup_steps) {
        last_lr_ = start_lr_;
        optimizer_->SetLearningRate(last_lr_);
    }

    virtual StateDict State() const override {
        auto state = LRScheduler::State();
        state["start_lr"] = start_lr_;
        state["end_lr"] = end_lr_;
        state["warmup_steps"] = static_cast<float>(warmup_steps_);
        return state;
    }

    virtual void LoadState(const StateDict& state) override {
        LRScheduler::LoadState(state);
        start_lr_ = state.at("start_lr");
        end_lr_ = state.at("end_lr");
        warmup_steps_ = static_cast<int>(state.at("warmup_steps"));
    }

protected:
    virtual float ComputeLR() override {
        if (step_count_ >= warmup_steps_) {
            return end_lr_;
        }
        float alpha = static_cast<float>(step_count_) / warmup_steps_;
        return start_lr_ + alpha * (end_lr_ - start_lr_);
    }

private:
    float start_lr_;
    float end_lr_;
    int warmup_steps_;
};

/**
 * @brief Lambda 调度器：通过用户提供的函数计算学习率乘因子
 * 
 * lr = initial_lr * lambda(step)
 */
class LambdaLR : public LRScheduler {
public:
    /**
     * @param optimizer 优化器指针
     * @param initial_lr 初始学习率
     * @param lambda_func 函数对象，接受 int step，返回 float 乘因子
     */
    LambdaLR(Optimizer* optimizer, float initial_lr, std::function<float(int)> lambda_func)
        : LRScheduler(optimizer), initial_lr_(initial_lr), lambda_func_(lambda_func) {
        last_lr_ = initial_lr_;
        optimizer_->SetLearningRate(last_lr_);
    }

    virtual StateDict State() const override {
        auto state = LRScheduler::State();
        state["initial_lr"] = initial_lr_;
        // lambda 函数无法序列化，恢复时需要由外部重新设置
        return state;
    }

    virtual void LoadState(const StateDict& state) override {
        LRScheduler::LoadState(state);
        initial_lr_ = state.at("initial_lr");
        // 注意：lambda_func_ 需在外部重建
    }

protected:
    virtual float ComputeLR() override {
        return initial_lr_ * lambda_func_(step_count_);
    }

private:
    float initial_lr_;
    std::function<float(int)> lambda_func_;
};

// ==================== 组合调度策略 ====================

/**
 * @brief 顺序组合调度器：按里程碑分段切换调度器
 * 
 * 提供多个调度器和对应的里程碑步数，在达到里程碑时切换到下一个调度器。
 * 注意：所有子调度器必须关联同一个优化器（但实现中不强制检查）。
 */
class SequentialLR : public LRScheduler {
public:
    /**
     * @param optimizer 优化器指针
     * @param schedulers 子调度器列表，按顺序使用
     * @param milestones 里程碑列表，大小应为 schedulers.size() - 1
     *                   表示切换点的步数（例如 milestones[0]=5 表示第6步切换到第二个调度器）
     */
    SequentialLR(Optimizer* optimizer,
                 const std::vector<LRScheduler*>& schedulers,
                 const std::vector<int>& milestones)
        : LRScheduler(optimizer), schedulers_(schedulers), milestones_(milestones), idx_(0) {
        if (schedulers_.size() != milestones_.size() + 1) {
            throw std::invalid_argument("SequentialLR: schedulers size must be milestones size + 1");
        }
        if (!schedulers_.empty()) {
            last_lr_ = schedulers_[0]->GetLR();
            optimizer_->SetLearningRate(last_lr_);
        }
    }

    virtual void Step() override {
        step_count_++;
        // 检查是否需要切换到下一个调度器
        while (idx_ < milestones_.size() && step_count_ > milestones_[idx_]) {
            idx_++;
        }
        if (idx_ < schedulers_.size()) {
            schedulers_[idx_]->Step();  // 当前调度器推进
            last_lr_ = schedulers_[idx_]->GetLR();
            optimizer_->SetLearningRate(last_lr_);
        }
    }

    virtual float GetLR() const override {
        return last_lr_;
    }

    virtual StateDict State() const override {
        StateDict state = LRScheduler::State();
        state["idx"] = static_cast<float>(idx_);
        // 保存每个子调度器的状态，用前缀区分
        for (size_t i = 0; i < schedulers_.size(); ++i) {
            auto substate = schedulers_[i]->State();
            for (const auto& kv : substate) {
                state["sub_" + std::to_string(i) + "_" + kv.first] = kv.second;
            }
        }
        return state;
    }

    virtual void LoadState(const StateDict& state) override {
        LRScheduler::LoadState(state);
        idx_ = static_cast<int>(state.at("idx"));
        // 恢复每个子调度器的状态
        for (size_t i = 0; i < schedulers_.size(); ++i) {
            StateDict substate;
            std::string prefix = "sub_" + std::to_string(i) + "_";
            for (const auto& kv : state) {
                if (kv.first.compare(0, prefix.size(), prefix) == 0) {
                    std::string key = kv.first.substr(prefix.size());
                    substate[key] = kv.second;
                }
            }
            schedulers_[i]->LoadState(substate);
        }
        if (idx_ < schedulers_.size()) {
            last_lr_ = schedulers_[idx_]->GetLR();
        }
    }

protected:
    virtual float ComputeLR() override {
        // 返回当前活跃调度器的学习率（主要为了满足基类纯虚函数要求）
        if (idx_ < schedulers_.size()) {
            return schedulers_[idx_]->GetLR();
        }
        return last_lr_;
    }

private:
    std::vector<LRScheduler*> schedulers_;
    std::vector<int> milestones_;
    int idx_;  // 当前使用的调度器索引
};

/**
 * @brief 链式组合调度器：每一步依次调用所有子调度器的 Step()
 * 
 * 每个子调度器依次作用，后一个调度器会覆盖前一个设置的学习率。
 * 最终学习率由最后一个调度器决定。
 */
class ChainedScheduler : public LRScheduler {
public:
    /**
     * @param optimizer 优化器指针
     * @param schedulers 子调度器列表，按调用顺序
     */
    ChainedScheduler(Optimizer* optimizer, const std::vector<LRScheduler*>& schedulers)
        : LRScheduler(optimizer), schedulers_(schedulers) {
        if (!schedulers_.empty()) {
            // 取最后一个调度器的当前学习率作为初始值
            last_lr_ = schedulers_.back()->GetLR();
            optimizer_->SetLearningRate(last_lr_);
        }
    }

    virtual void Step() override {
        step_count_++;
        if (schedulers_.empty()) return;
        for (auto* sched : schedulers_) {
            sched->Step();
        }
        last_lr_ = schedulers_.back()->GetLR();
        optimizer_->SetLearningRate(last_lr_);
    }

    virtual float GetLR() const override {
        return last_lr_;
    }

    virtual StateDict State() const override {
        StateDict state = LRScheduler::State();
        for (size_t i = 0; i < schedulers_.size(); ++i) {
            auto substate = schedulers_[i]->State();
            for (const auto& kv : substate) {
                state["sub_" + std::to_string(i) + "_" + kv.first] = kv.second;
            }
        }
        return state;
    }

    virtual void LoadState(const StateDict& state) override {
        LRScheduler::LoadState(state);
        for (size_t i = 0; i < schedulers_.size(); ++i) {
            StateDict substate;
            std::string prefix = "sub_" + std::to_string(i) + "_";
            for (const auto& kv : state) {
                if (kv.first.compare(0, prefix.size(), prefix) == 0) {
                    std::string key = kv.first.substr(prefix.size());
                    substate[key] = kv.second;
                }
            }
            schedulers_[i]->LoadState(substate);
        }
        if (!schedulers_.empty()) {
            last_lr_ = schedulers_.back()->GetLR();
        }
    }

protected:
    virtual float ComputeLR() override {
        // 返回最后一个子调度器的学习率（主要为了满足基类纯虚函数要求）
        if (!schedulers_.empty()) {
            return schedulers_.back()->GetLR();
        }
        return last_lr_;
    }

private:
    std::vector<LRScheduler*> schedulers_;
};

#endif  // INFINITRAIN_LR_SCHEDULER_H_
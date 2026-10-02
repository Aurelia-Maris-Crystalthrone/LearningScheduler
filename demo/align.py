import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    StepLR, LinearLR, SequentialLR, ChainedScheduler, ConstantLR, LambdaLR
)
import argparse
import random
import numpy as np
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_model_and_data(model_name):
    """根据飞书文档中的服务器路径加载模型和数据集"""
    if model_name == 'gpt2':
        # 模型路径
        model_path = '/data/shared/InfiniTrain-dev/data/llmc/gpt2/gpt2_124M.bin'
        # 数据集路径
        data_path = '/data/shared/InfiniTrain-dev/data/llmc/gpt2/tinyshakespeare/tiny_shakespeare_train.bin'
        config = GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=1000)
        model = GPT2LMHeadModel(config)
    elif model_name == 'llama':
        model_path = '/data/shared/InfiniTrain-dev/data/llmc/llama3/llama3.2_1B_fp32.bin'
        data_path = '/data/shared/InfiniTrain-dev/data/llmc/llama3/tinyshakespeare/tiny_shakespeare_train.bin'
        config = LlamaConfig(
            num_hidden_layers=2,
            num_attention_heads=4,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=1000
        )
        model = LlamaForCausalLM(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # 注意：实际加载预训练权重的代码需要根据文件格式实现
    # 这里仅做示例，真正的加载逻辑需要学员补充
    print(f"Model path: {model_path}")
    print(f"Data path: {data_path}")
    return model

def get_scheduler(name, optimizer, base_lr, warmup_steps, step_size, gamma,
                  milestones=None, scheduler_list=None):
    # (与之前相同的调度器创建逻辑)
    # ... (略，与之前提供的 align.py 一致)
    pass

def train_step(model, optimizer, scheduler, input_ids):
    model.train()
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    return loss.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2',
                        choices=['gpt2', 'llama'])
    parser.add_argument('--scheduler', type=str, default='steplr',
                        choices=['constant', 'steplr', 'linear',
                                 'sequential', 'chained'])
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--base_lr', type=float, default=0.001)
    parser.add_argument('--warmup_steps', type=int, default=5)
    parser.add_argument('--step_size', type=int, default=3)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--milestones', type=str, default='5')
    parser.add_argument('--scheduler_list', type=str,
                        default='linear,steplr')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model_and_data(args.model).to(device)
    optimizer = AdamW(model.parameters(), lr=args.base_lr)
    scheduler = get_scheduler(
        args.scheduler, optimizer, args.base_lr, args.warmup_steps,
        args.step_size, args.gamma, args.milestones, args.scheduler_list
    )

    # 生成随机输入（实际应用中应使用真实数据）
    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)

    print(f"Model: {args.model}, Scheduler: {args.scheduler}, Steps: {args.steps}")
    print("Step\tLoss\tLR")
    for step in range(1, args.steps + 1):
        loss = train_step(model, optimizer, scheduler, input_ids)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"{step}\t{loss:.6f}\t{current_lr:.6f}")

if __name__ == '__main__':
    main()
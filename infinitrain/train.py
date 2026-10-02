#!/usr/bin/env python3
"""
InfiniTrain 模拟训练脚本（支持 DDP）
用法：torchrun --nproc_per_node=8 train.py [参数]
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
import random
import numpy as np

# 导入自定义调度器
from lr_scheduler import (
    ConstantLR, StepLR, LinearLR, LambdaLR,
    SequentialLR, ChainedScheduler
)

def setup_distributed(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_model(model_name):
    if model_name == 'gpt2':
        from transformers import GPT2Config, GPT2LMHeadModel
        config = GPT2Config(
            n_layer=2,
            n_head=2,
            n_embd=64,
            vocab_size=1000
        )
        model = GPT2LMHeadModel(config)
    elif model_name == 'llama':
        from transformers import LlamaConfig, LlamaForCausalLM
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
    return model

def get_scheduler(name, optimizer, args):
    if name == 'constant':
        return ConstantLR(optimizer, args.lr)
    elif name == 'steplr':
        return StepLR(optimizer, args.lr, args.step_size, args.gamma)
    elif name == 'linear':
        return LinearLR(optimizer, 0.0, args.lr, args.warmup_steps)
    elif name == 'chained':
        types = args.scheduler_list.split(',')
        schedulers = []
        for t in types:
            if t == 'linear':
                schedulers.append(LinearLR(optimizer, 0.0, args.lr, args.warmup_steps))
            elif t == 'steplr':
                schedulers.append(StepLR(optimizer, args.lr, args.step_size, args.gamma))
            else:
                raise ValueError(f"Unsupported scheduler type in chain: {t}")
        return ChainedScheduler(optimizer, schedulers)
    elif name == 'sequential':
        types = args.scheduler_list.split(',')
        milestones = [int(m) for m in args.scheduler_milestones.split(',')]
        schedulers = []
        for t in types:
            if t == 'linear':
                schedulers.append(LinearLR(optimizer, 0.0, args.lr, args.warmup_steps))
            elif t == 'steplr':
                schedulers.append(StepLR(optimizer, args.lr, args.step_size, args.gamma))
            else:
                raise ValueError(f"Unsupported scheduler type in sequential: {t}")
        return SequentialLR(optimizer, schedulers, milestones)
    else:
        raise ValueError(f"Unknown scheduler: {name}")

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
    parser.add_argument('--model', type=str, default='gpt2', choices=['gpt2', 'llama'])
    parser.add_argument('--lr-scheduler', type=str, default='steplr',
                        choices=['constant', 'steplr', 'linear', 'chained', 'sequential'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--warmup-steps', type=int, default=5)
    parser.add_argument('--step-size', type=int, default=3)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--scheduler-list', type=str, default='linear,steplr')
    parser.add_argument('--scheduler-milestones', type=str, default='5')
    parser.add_argument('--total-steps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-interval', type=int, default=1)
    args = parser.parse_args()

    # 分布式初始化
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ['WORLD_SIZE'])
    setup_distributed(local_rank, world_size)

    set_seed(args.seed + local_rank)  # 不同进程不同种子保证输入不同？但为了对齐，可以固定种子
    # 为了 PyTorch 对齐，最好所有进程使用相同数据，因此种子应相同
    set_seed(args.seed)  # 所有进程相同种子

    device = torch.device(f'cuda:{local_rank}')
    model = get_model(args.model).to(device)
    model = DDP(model, device_ids=[local_rank])

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_scheduler(args.lr_scheduler, optimizer, args)

    # 生成随机输入（所有进程相同，保证 loss 一致）
    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)

    if local_rank == 0:
        print(f"Model: {args.model}, Scheduler: {args.lr_scheduler}, Steps: {args.total_steps}")
        print("Step\tLoss\tLR")

    for step in range(1, args.total_steps + 1):
        loss = train_step(model, optimizer, scheduler, input_ids)
        current_lr = optimizer.param_groups[0]['lr']
        if local_rank == 0 and (step % args.log_interval == 0 or step == 1):
            print(f"{step}\t{loss:.6f}\t{current_lr:.6f}")

    cleanup()

if __name__ == '__main__':
    main()
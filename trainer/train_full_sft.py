"""
script for supervised finetuning
"""

import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.distributed as dist
from contextlib import nullcontext

import argparse

from model import MiniGPTConfig
from .trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniGPT Full SFT")
    
    # Base Training Arguments
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument("--save_weight", type=str, default="full_sft", help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮次（通常2-5）")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (SFT通常使用小batch)")
    parser.add_argument("--learning_rate", type=float, default=5e-7, help=("初始学习率")) # commonly lower lr than pretrain
    
    # Hardware Configuration
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="Dataloader线程数量")
    
    # Training Policy 
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累计步数")   # 1 for SFT
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    
    # Model Architecture 
    parser.add_argument("--hidden_size", default=512, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument("--max_seq_len", default=512, type=int, help="训练的最大截断长度")
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1], help="是否开启MoE架构（0:False, 1: True）")
    
    # Data & Model 
    parser.add_argument("--data_path", type=str, default="../dataset/sft_mini_512.jsonl", help="SFT训练数据路径")
    parser.add_argument("--from_weight", type=str, default="pretrain", help="基础模型类型权重") # default SFT on pretrain model
    parser.add_argument("--from_resume", default=1, type=int, choices=[0, 1], help="是否开启自动续训（0:False, 1:True）")
    
    # Experiments
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb监控实验")
    parser.add_argument("--wandb_project", type=str, default="MiniGPT-Full-SFT", help="wandb项目名")
    
    args = parser.parse_args()
   
   
    # 1.  config env and seed
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # 2. dir, params and ckpt
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniGPTConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir="../checkpoints") if args.from_resume==1 else None

    # 3. mix dtype
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=dtype)
    
    # 4. wandb config
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniGPT-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

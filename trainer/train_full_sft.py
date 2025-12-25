"""
script for supervised finetuning
"""

import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time

import torch
import torch.distributed as dist
from torch import optim
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler, DataLoader

from model import MiniGPTConfig
from dataset import SFTDataset
from .trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler


def train_epoch(epoch ,loader, iters, start_step=0, wandb=None):
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    
    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step+1):
        X = X.to(args.device)   # (B, L)
        Y = Y.to(args.device)   # (B, L)
        loss_mask = loss_mask.to(args.device)   # (B, L)

        lr = get_lr(current_step=epoch*iters + step, 
                    total_steps=args.epochs*iters,
                    lr=args.learning_rate)       # adjust lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        with autocast_ctx:
            res = model(X)  # forward

            # input: [B, L, V] -> [B*L, V] 
            # target: [B, L] -> [B*L]
            loss = loss_fn(res.logits.view(-1, res.logits.size(-1)), Y.view(-1)).view(Y.size())   # (B, L)
            loss = (loss*loss_mask).sum() / loss_mask.sum()

            loss = loss / args.accumulation_steps
        
        
        scaler.scale(loss).backward()

        if (step+1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        # log
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:')
            
            if wandb:
                wandb.log({"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min})

        # ckpt
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            
            if isinstance(model, nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()  # DDP module
            else:
                state_dict = model.state_dict()
            
            state_dict = {k: v.half() for k, v in state_dict.items()} # save as half precision
            
            torch.save(state_dict, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler)
            
            model.train()


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
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    
    # Model Architecture 
    parser.add_argument("--hidden_size", default=512, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument("--max_seq_len", default=512, type=int, help="训练的最大截断长度")
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1], help="是否开启MoE架构（0:False, 1: True）")
    
    # Data & Model 
    parser.add_argument("--data_path", type=str, default="dataset/sft_mini_512.jsonl", help="SFT训练数据路径")
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

    # 5. model, data and optimizer
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = SFTDataset(args.data_path, tokenizer, args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr = args.learning_rate)
    
    # 6. load from ckpt
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data['step', 0]
    
    # 7. DDP
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
        
    # 8. begain training
    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        if epoch == start_epoch and start_step > 0:
            # exits ckpt
            batch_sampler = SkipBatchSampler(sampler=train_sampler or range(len(train_ds)),
                                             batch_size=args.batch_size,
                                             skip_batches=start_step+1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: Exclude {start_step} stepes before and training from  step {start_step + 1}')
            
            train_epoch(epoch, loader, len(loader)+start_step+1, start_step, wandb)
        else:
            loader = DataLoader(train_ds, batch_size=args.batch_size, 
                                shuffle=(train_sampler is None), sampler=train_sampler,
                                num_workers=args.num_workers, pin_memory=True)
            
            train_epoch(epoch, loader, len(loader), 0, wandb)
            
        
            
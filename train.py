# train.py
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from tqdm import tqdm
import os
import argparse

from dataset import ParamDataset
from models import RPG
from visualize import setup_global_fonts, plot_heatmap, plot_gaussian, plot_histogram, plot_violin
from scheduler import get_lr_scheduler

def train(cfg):
    # 设置设备 (Setup Device)
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建保存目录 (Create Save Dir)
    os.makedirs(cfg.train.save_dir, exist_ok=True)
    
    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(cfg.train.save_dir, 'logs'))
    
    # 数据 (Data)
    full_dataset = ParamDataset(cfg)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(cfg.train.seed)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.train.batch_size, 
        shuffle=True, 
        num_workers=cfg.train.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers
    )
    
    print(f"Dataset Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # 模型 (Model)
    model = RPG(cfg).to(device)
    model.train()
    
    # 优化器 (Optimizer)
    optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    
    # 创建学习率调度器
    if cfg.train.lr_scheduler == 'cosine_warmup':
        kwargs = {
            'scheduler_type': cfg.train.lr_scheduler,
            'warmup_epochs': 0.1 * cfg.train.num_epochs,
            'max_epochs': cfg.train.num_epochs,
            'warmup_start_lr': cfg.train.min_lr,
            'eta_min': cfg.train.min_lr
        }
    elif cfg.train.lr_scheduler == 'cosine':
        kwargs = {
            'scheduler_type': cfg.train.lr_scheduler,
            'T_max': cfg.train.num_epochs,
            'eta_min': cfg.train.min_lr
        }
    elif cfg.train.lr_scheduler == 'const':
        kwargs = {
            'scheduler_type': cfg.train.lr_scheduler
        }
    scheduler = get_lr_scheduler(optimizer, **kwargs)
    
    # 训练循环 (Training Loop)
    for epoch in range(1, cfg.train.num_epochs + 1):
        # Training Phase
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.train.num_epochs} [Train]")
        total_loss = 0
        
        for batch_idx, (condition, w1, w2, seed_idx) in enumerate(pbar):
            condition = condition.to(device)
            w1 = w1.to(device)
            w2 = w2.to(device)
            seed_idx = seed_idx.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播 (Forward pass) (内部计算扩散损失)
            loss, pre, tgt= model(condition, w1, w2, seed_idx)
            if epoch % 10 == 0 and batch_idx == len(train_loader) - 1:
                plot_heatmap(pre, tgt, filename=os.path.join(cfg.train.save_dir, "results", f"[T]heatmap_epoch_{epoch}.png"))
                plot_histogram(pre, tgt, filename=os.path.join(cfg.train.save_dir, "results", f"[T]histogram_epoch_{epoch}.png"))
                plot_violin(pre, tgt, filename=os.path.join(cfg.train.save_dir, "results", f"[T]violin_epoch_{epoch}.png"))
                plot_gaussian(pre - tgt, filename=os.path.join(cfg.train.save_dir, "results", f"[T]gaussian_epoch_{epoch}.png"))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation Phase
        if epoch % cfg.train.eval_freq == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_idx, (condition, w1, w2, seed_idx) in enumerate(val_loader):
                    condition = condition.to(device)
                    w1 = w1.to(device)
                    w2 = w2.to(device)
                    seed_idx = seed_idx.to(device)
                    loss, pre, tgt= model(condition, w1, w2, seed_idx)
                    if batch_idx == len(val_loader) - 1:
                        plot_heatmap(pre, tgt, filename=os.path.join(cfg.train.save_dir, "results", f"[V]heatmap_epoch_{epoch}.png"))
                        plot_histogram(pre, tgt, filename=os.path.join(cfg.train.save_dir, "results", f"[V]histogram_epoch_{epoch}.png"))
                        plot_violin(pre, tgt, filename=os.path.join(cfg.train.save_dir, "results", f"[V]violin_epoch_{epoch}.png"))
                        plot_gaussian(pre - tgt, filename=os.path.join(cfg.train.save_dir, "results", f"[V]gaussian_epoch_{epoch}.png"))
                    val_loss += loss.item()
        
            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
            writer.add_scalar('epoch/val_loss', avg_val_loss, epoch)
            print(f"Epoch {epoch} Val Loss: {avg_val_loss:.4e}")
            
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Logging
        writer.add_scalar('epoch/train_loss', avg_train_loss, epoch)
        writer.add_scalar('epoch/lr', current_lr, epoch)
        
        # 打印学习率
        print(f"Epoch {epoch} Train Loss: {avg_train_loss:.4e}, LR: {current_lr:.4e}")
        
        # 保存检查点 (Save Checkpoint)
        if epoch % 50 == 0:
            torch.save(model.state_dict(), os.path.join(cfg.train.save_dir, f"rpg_epoch_{epoch}.pth"))
            
    writer.close()
    
    # Test Phase
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for condition, w1, w2, seed_idx in test_loader:
            condition = condition.to(device)
            w1 = w1.to(device)
            w2 = w2.to(device)
            seed_idx = seed_idx.to(device)
            loss, pre, tgt = model(condition, w1, w2, seed_idx)
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0
    print(f"Test Loss: {avg_test_loss:.4e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RPG Model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    setup_global_fonts()
    
    cfg = OmegaConf.load(args.config)
    train(cfg)

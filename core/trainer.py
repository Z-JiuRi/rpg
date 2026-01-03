import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from omegaconf import OmegaConf

from models.rpg import RecurrentParameterGenerator
from utils.dataset import LoRADataset
from utils.tools import EMA, set_seed


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        set_seed(cfg.train.seed)
        os.makedirs(cfg.train.log_dir, exist_ok=True)
        
        # 1. Dataset
        self.full_dataset = LoRADataset(cfg.data.data_dir, cfg.data.token_size)
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(
            self.full_dataset, 
            [0.9, 0.1],
            generator=torch.Generator().manual_seed(cfg.train.seed)
        )
        # 更新 config 中的动态参数
        cfg.data.num_seeds = self.full_dataset.num_seeds
        # 简单估算最大 token 长度 (例如132)，或取 dataset 中最长的
        cfg.data.max_token_idx = 200 # Safe margin
        
        self.train_dataloader = DataLoader(
            self.train_dataset, 
            batch_size=cfg.data.batch_size, 
            shuffle=True, 
            num_workers=cfg.data.num_workers
        )
        self.test_dataloader = DataLoader(
            self.test_dataset, 
            batch_size=cfg.data.batch_size, 
            shuffle=False, 
            num_workers=cfg.data.num_workers
        )
        
        # 2. Model
        # Construct model kwargs
        model_kwargs = {
            "pos_embedding": {
                "hidden_dim": cfg.model.rpg.hidden_dim,
                "pe_fusion_type": cfg.model.rpg.pos_fusion,
                "token_pe_type": cfg.model.rpg.pos_type,
                "max_layers": 10,
                "max_tokens": cfg.data.max_token_idx
            },
            "proto": {
                "proto_drop": cfg.model.rpg.null_seed_prob,
                "cond_dim": cfg.model.rpg.cond_dim,
                "hidden_dim": cfg.model.rpg.hidden_dim,
                "num_seeds": cfg.data.num_seeds,
                "learnable_null": True,
                "prototype_dim": cfg.model.rpg.hidden_dim,
                # RNN args
                "d_model": cfg.model.rpg.hidden_dim,
                **OmegaConf.to_container(cfg.model.rnn, resolve=True)
            },
            "denoiser": {
                "type": cfg.model.type,
                # UNet args
                "dim": cfg.model.unet.dim,
                "layer_channels": [cfg.model.unet.dim * m for m in cfg.model.unet.dim_mults],
                "init_ch": cfg.model.unet.input_channels,
                "final_ch": cfg.model.unet.input_channels,
                "cond_dim": cfg.model.rpg.cond_dim,
                "proto_dim": cfg.model.rpg.hidden_dim,
                # DiT args could be added here if needed
            },
            "ddpm": {
                "timesteps": cfg.model.diffusion.timesteps,
                "beta_scheduler": cfg.model.diffusion.beta_schedule,
                "prediction_type": cfg.model.diffusion.prediction_type,
                "loss_type": cfg.model.diffusion.loss_type,
                "beta_start": 0.0001,
                "beta_end": 0.02
            }
        }

        self.model = RecurrentParameterGenerator(**model_kwargs).cuda()
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.train.lr)
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.use_amp)
        
        if cfg.train.use_ema:
            self.ema = EMA(self.model, cfg.train.ema_decay)
            self.ema.register()

    def run(self):
        print("Starting training...")
        best_loss = float('inf')
        
        for epoch in range(self.cfg.train.num_epochs):
            self.model.train()
            loss_acc = 0
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
            
            for batch in pbar:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.cuda()
                
                self.optimizer.zero_grad()
                
                with torch.cuda.amp.autocast(enabled=self.cfg.train.use_amp):
                    # RPG model takes unpacked batch args or we need to unpack them
                    # Check RPG forward: x_0, condition, layer_indices, token_indices, permutation_state_indices
                    loss_dict = self.model(
                        x_0=batch['data'], # Assuming 'data' key from dataset
                        condition=batch['condition'],
                        layer_indices=batch['layer_ids'],
                        token_indices=batch['token_ids'],
                        permutation_state_indices=batch.get('seed_ids', None) # Assuming 'seed_ids' or similar
                    )
                    loss = loss_dict['loss']
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                if self.cfg.train.use_ema:
                    self.ema.update()
                    
                loss_acc += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            # Validation
            val_loss = self.validate()
            print(f"Epoch {epoch} Val Loss: {val_loss:.6f}")

            if (epoch + 1) % self.cfg.train.save_interval == 0:
                self.save(epoch)
            
            if val_loss < best_loss:
                best_loss = val_loss
                self.save('best')

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        if self.cfg.train.use_ema:
            self.ema.apply_shadow()
            
        loss_acc = 0
        for batch in self.test_dataloader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            
            with torch.cuda.amp.autocast(enabled=self.cfg.train.use_amp):
                loss_dict = self.model(
                    x_0=batch['data'],
                    condition=batch['condition'],
                    layer_indices=batch['layer_ids'],
                    token_indices=batch['token_ids'],
                    permutation_state_indices=batch.get('seed_ids', None)
                )
                loss_acc += loss_dict['loss'].item()
        
        if self.cfg.train.use_ema:
            self.ema.restore()
            
        return loss_acc / len(self.test_dataloader)

    def save(self, epoch_or_tag):
        path = os.path.join(self.cfg.train.log_dir, f"ckpt_{epoch_or_tag}.pt")
        if self.cfg.train.use_ema:
            self.ema.apply_shadow()
            torch.save(self.model.state_dict(), path)
            self.ema.restore()
        else:
            torch.save(self.model.state_dict(), path)
        print(f"Saved to {path}")
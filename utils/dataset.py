import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
import numpy as np

class LoRADataset(Dataset):
    def __init__(self, root_dir, token_size=64):
        super().__init__()
        self.root_dir = root_dir
        self.token_size = token_size  # 要求1: 64
        
        self.cond_root = os.path.join(root_dir, 'conditions')
        self.param_root = os.path.join(root_dir, 'params')
        
        # 1. 扫描所有的 Seed 文件夹
        # 假设结构: params/<seed_name>/...
        self.seed_names = sorted([d for d in os.listdir(self.param_root) if os.path.isdir(os.path.join(self.param_root, d))])
        self.seed_to_idx = {name: i for i, name in enumerate(self.seed_names)}
        self.num_seeds = len(self.seed_names)
        
        # 2. 收集所有数据样本
        self.samples = []
        for seed_name in self.seed_names:
            seed_idx = self.seed_to_idx[seed_name]
            param_seed_dir = os.path.join(self.param_root, seed_name)
            cond_seed_dir = os.path.join(self.cond_root, seed_name)
            
            # 找到该seed下的所有 dataid (通过扫描 _a1.pth 结尾的文件)
            files = glob.glob(os.path.join(param_seed_dir, "*_a1.pth"))
            for f in files:
                # 提取 dataid
                filename = os.path.basename(f)
                data_id = filename.replace("_a1.pth", "")
                
                # 检查所有文件是否存在
                p_a1 = os.path.join(param_seed_dir, f"{data_id}_a1.pth")
                p_b1 = os.path.join(param_seed_dir, f"{data_id}_b1.pth")
                p_a2 = os.path.join(param_seed_dir, f"{data_id}_a2.pth")
                p_b2 = os.path.join(param_seed_dir, f"{data_id}_b2.pth")
                p_cond = os.path.join(cond_seed_dir, f"{data_id}.pth")
                
                if all(os.path.exists(p) for p in [p_a1, p_b1, p_a2, p_b2, p_cond]):
                    self.samples.append({
                        'seed_name': seed_name,
                        'seed_idx': seed_idx,
                        'data_id': data_id,
                        'paths': [p_a1, p_b1, p_a2, p_b2],
                        'cond_path': p_cond
                    })

        print(f"Found {len(self.samples)} samples across {self.num_seeds} seeds.")

    def __len__(self):
        return len(self.samples)

    def _process_matrix(self, path, layer_idx):
        """读取矩阵，flatten，并reshape为 (N, 64)"""
        mat = torch.load(path, map_location='cpu')
        
        # Flatten parameters
        flat = mat.flatten() # [M]
        
        # Padding or check divisibility?
        # 题目要求: token_size取64，不用padding。这意味着参数总数必须能被64整除。
        # 检查: 
        # a1 (2,64)=128 -> 2 tokens
        # b1 (2048,2)=4096 -> 64 tokens
        # a2 (2,2048)=4096 -> 64 tokens
        # b2 (64,2)=128 -> 2 tokens
        # Total 132 tokens. Perfect.
        
        if flat.numel() % self.token_size != 0:
            raise ValueError(f"Matrix {path} size {flat.numel()} not divisible by {self.token_size}")
            
        num_tokens = flat.numel() // self.token_size
        reshaped = flat.view(num_tokens, self.token_size)
        
        # 生成对应的 Layer ID
        layer_ids = torch.full((num_tokens,), layer_idx, dtype=torch.long)
        
        return reshaped, layer_ids

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # 1. Load Condition
        condition = torch.load(item['cond_path'], map_location='cpu') # [512]
        
        # 2. Load and Process Matrices
        # Order: a1, b1, a2, b2 -> Layer IDs: 0, 1, 2, 3
        all_tokens = []
        all_layer_ids = []
        
        for i, path in enumerate(item['paths']):
            tokens, lids = self._process_matrix(path, i)
            all_tokens.append(tokens)
            all_layer_ids.append(lids)
            
        # Concatenate sequence
        data_seq = torch.cat(all_tokens, dim=0) # [L_total, 64]
        layer_ids = torch.cat(all_layer_ids, dim=0) # [L_total]
        
        # Generate Token IDs (0 to L-1)
        seq_len = data_seq.shape[0]
        token_ids = torch.arange(seq_len, dtype=torch.long)
        
        # 转置 data_seq 以适配 UNet (Channel, Length) -> [64, L_total]
        # 注意：Config里的 token_size 是 64，对应 UNet 的 input_channels
        data_seq = data_seq.transpose(0, 1) # [64, L_total]
        
        return {
            "x": data_seq.float(),               # [64, 132]
            "condition": condition.float(),      # [512]
            "layer_id": layer_ids,               # [132]
            "token_id": token_ids,               # [132]
            "seed_idx": torch.tensor(item['seed_idx'], dtype=torch.long),
            "seq_len": seq_len # FIXME: 如果不同样本长度不同，需要CollateFn处理，这里假设结构固定长度一致
        }


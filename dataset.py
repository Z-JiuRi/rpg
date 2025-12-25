# dataset.py
import os
import torch
from torch.utils.data import Dataset
import glob

class ParamDataset(Dataset):
    def __init__(self, cfg):
        self.root = cfg.data.root_dir
        self.cond_dir = os.path.join(self.root, "conditions")
        self.param_dir = os.path.join(self.root, "params")
        
        # 扫描所有 params 下的 seed 文件夹
        self.seeds = sorted(os.listdir(self.param_dir))
        # 构建 seed 到 index 的映射 (用于 Permutation State Embedding)
        self.seed_to_idx = {seed_name: i for i, seed_name in enumerate(self.seeds)}
        
        self.samples = []
        
        # 遍历收集所有样本
        for seed_name in self.seeds:
            curr_param_path = os.path.join(self.param_dir, seed_name)
            curr_cond_path = os.path.join(self.cond_dir, seed_name)
            
            # 找到 linear1 的文件，并推断对应的 linear2
            # 假设文件名格式: <dataid>_linear1_weight.pth
            linear1_files = glob.glob(os.path.join(curr_param_path, "*_linear1_weight.pth"))
            
            for f1 in linear1_files:
                filename = os.path.basename(f1)
                dataid = filename.replace("_linear1_weight.pth", "")
                
                f2 = os.path.join(curr_param_path, f"{dataid}_linear2_weight.pth")
                cond_file = os.path.join(curr_cond_path, f"{dataid}.pth")
                
                if os.path.exists(f2) and os.path.exists(cond_file):
                    self.samples.append({
                        "seed_name": seed_name,
                        "dataid": dataid,
                        "linear1_path": f1,
                        "linear2_path": f2,
                        "cond_path": cond_file,
                        "seed_idx": self.seed_to_idx[seed_name]
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # 加载 Condition (512,)
        condition = torch.load(item['cond_path'], map_location='cpu', weights_only=True)
        
        # 加载 Params
        # 注意：这里加载的是 tensor
        w1 = torch.load(item['linear1_path'], map_location='cpu', weights_only=True) # (64, 2048)
        w2 = torch.load(item['linear2_path'], map_location='cpu', weights_only=True) # (2048, 64)
        
        seed_idx = torch.tensor(item['seed_idx'], dtype=torch.long)
        
        return condition, w1, w2, seed_idx
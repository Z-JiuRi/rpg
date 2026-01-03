import torch
import os
from omegaconf import OmegaConf
from models.rpg import RecurrentParameterGenerator

class Inferencer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda')
        
        # Construct model kwargs (same as Trainer)
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
                "d_model": cfg.model.rpg.hidden_dim,
                **OmegaConf.to_container(cfg.model.rnn, resolve=True)
            },
            "denoiser": {
                "type": cfg.model.type,
                "dim": cfg.model.unet.dim,
                "layer_channels": [cfg.model.unet.dim * m for m in cfg.model.unet.dim_mults],
                "init_ch": cfg.model.unet.input_channels,
                "final_ch": cfg.model.unet.input_channels,
                "cond_dim": cfg.model.rpg.cond_dim,
                "proto_dim": cfg.model.rpg.hidden_dim,
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
        
        self.model = RecurrentParameterGenerator(**model_kwargs).to(self.device)
        
        print(f"Loading from {cfg.inference.ckpt_path}")
        state = torch.load(cfg.inference.ckpt_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def run(self, cond_file, output_dir):
        """
        根据 Cond 生成参数并保存
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Load Condition
        cond = torch.load(cond_file, map_location=self.device)
        if cond.dim() == 1:
            cond = cond.unsqueeze(0) # [1, CondDim]
        
        # 2. Construct Target Structure
        # Default structure (Legacy)
        default_structure = [
            {"count": 2, "id": 0, "name": "a1", "shape": [2, 64]},
            {"count": 64, "id": 1, "name": "b1", "shape": [2048, 2]},
            {"count": 64, "id": 2, "name": "a2", "shape": [2, 2048]},
            {"count": 2, "id": 3, "name": "b2", "shape": [64, 2]}
        ]
        
        structure_cfg = self.cfg.inference.get('structure', default_structure)
        # Convert OmegaConf list to python list if needed
        if hasattr(structure_cfg, 'to_container'): # or just iterate
             pass
        
        structure = []
        for item in structure_cfg:
            structure.append(item)

        total_tokens = sum(x['count'] for x in structure)
        
        # 构造 Layer IDs 和 Token IDs
        layer_ids_list = []
        for item in structure:
            layer_ids_list.append(torch.full((item['count'],), item['id'], dtype=torch.long))
        layer_ids = torch.cat(layer_ids_list).unsqueeze(0).to(self.device) # [1, TotalTokens]
        
        token_ids = torch.arange(total_tokens, dtype=torch.long).unsqueeze(0).to(self.device) # [1, TotalTokens]
        
        # 3. Generate
        with torch.no_grad():
            output = self.model.sample(
                condition=cond,
                layer_indices=layer_ids,
                token_indices=token_ids,
                use_ddim=False
            )
            
        # 4. Reconstruction
        # output: [1, C, L] -> transpose to [L, C]
        output = output.squeeze(0).transpose(0, 1) # [L, C]
        
        results = {}
        ptr = 0
        
        for item in structure:
            count = item['count']
            name = item.get('name', f'layer_{item["id"]}')
            shape = item.get('shape', None)
            
            chunk = output[ptr:ptr+count] # [count, C]
            
            if shape:
                # flatten and view
                # 注意：这里假设 flatten 顺序一致。
                # 原始代码: output[ptr:ptr+2].flatten().view(2, 64)
                # output slice is [2, 64]. flatten is 128. view(2, 64) is same.
                # b1: output slice [64, 64]. flatten 4096. view(2048, 2).
                try:
                    chunk = chunk.flatten().view(*shape)
                except Exception as e:
                    print(f"Warning: Failed to reshape {name} to {shape}: {e}. Keeping original shape.")
            
            results[name] = chunk
            ptr += count
        
        # Save
        save_path = os.path.join(output_dir, "generated_params.pth")
        torch.save(results, save_path)
        print(f"Saved to {save_path}")
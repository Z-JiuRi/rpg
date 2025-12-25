# models.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_2d_sincos_pos_embed

class ParamTokenizer(nn.Module):
    def __init__(self, token_size, layer_shapes):
        """
        Args:
            token_size: 每个 token 包含多少个参数 (K)
            layer_shapes: list of tuples, e.g. [(64, 2048), (2048, 64)]
        """
        super().__init__()
        self.token_size = token_size
        self.layer_shapes = layer_shapes
        
    def forward(self, w1, w2):
        """
        Input: Batch of raw weights
        Output: 
            tokens: (B, Seq_Len, token_size)
            pos_info: (Seq_Len, 2) -> (layer_idx, token_idx)
            stats: list of (mean, std) for denormalization
        """
        batch_size = w1.shape[0]
        device = w1.device
        
        all_tokens = []
        layer_indices = []
        token_indices_in_layer = []
        stats = []

        weights_list = [w1, w2]
        
        for layer_idx, w in enumerate(weights_list):
            # 1. Flatten
            w_flat = w.view(batch_size, -1) # (B, N_params)
            
            # 2. Layer-wise Normalization 
            # 计算这一层所有参数的均值和方差
            mean = w_flat.mean(dim=1, keepdim=True)
            std = w_flat.std(dim=1, keepdim=True) + 1e-6
            w_norm = (w_flat - mean) / std
            
            stats.append((mean, std))
            
            # 3. Padding if necessary
            num_params = w_norm.shape[1]
            pad_len = (self.token_size - (num_params % self.token_size)) % self.token_size
            if pad_len > 0:
                # Pad with zeros, these won't contribute to loss ideally
                w_norm = F.pad(w_norm, (0, pad_len))
            
            # 4. Tokenization 
            # Chunk into (B, num_chunks, token_size)
            tokens = w_norm.view(batch_size, -1, self.token_size)
            all_tokens.append(tokens)
            
            # 记录位置信息
            num_chunks = tokens.shape[1]
            layer_indices.extend([layer_idx] * num_chunks)
            token_indices_in_layer.extend(list(range(num_chunks)))

        # Concatenate all layers
        full_sequence = torch.cat(all_tokens, dim=1) # (B, Total_Seq, K)
        
        pos_ids = {
            'layer': torch.tensor(layer_indices, device=device),
            'token': torch.tensor(token_indices_in_layer, device=device)
        }
        
        return full_sequence, pos_ids, stats

    def inverse(self, tokens, stats, original_shapes):
        """用于推理：将 Token 还原为权重"""
        # 此处省略具体实现，逻辑是 Tokenizer 的逆过程：
        # 1. Flatten tokens -> 2. Remove padding -> 3. Denormalize ( * std + mean) -> 4. Reshape
        pass

class RecurrentModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.hidden_dim = cfg.model.hidden_dim
        self.token_size = cfg.model.token_size
        
        # Project Token K -> Hidden
        self.token_proj = nn.Linear(self.token_size, self.hidden_dim)
        
        # Condition Projection (512 -> Hidden)
        self.cond_proj = nn.Linear(cfg.data.condition_dim, self.hidden_dim)
        
        # Permutation State Embedding 
        self.perm_emb = nn.Embedding(cfg.train.max_seeds, self.hidden_dim)
        
        # RNN Backbone (User requested RNN, paper uses Mamba/Transformer)
        # Using GRU for stability
        self.rnn = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=cfg.model.rnn_layers,
            batch_first=True,
            bidirectional=False 
        )
        
        # Output Prototype Projection [cite: 138]
        self.proto_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, x_tokens, condition, perm_idx, pos_embed):
        """
        x_tokens: (B, L, token_size)
        condition: (B, cond_dim)
        perm_idx: (B,)
        pos_embed: (L, hidden_dim) - Fixed 2D embedding
        """
        B, L, _ = x_tokens.shape
        
        # 1. Embed Inputs
        h_tokens = self.token_proj(x_tokens) # (B, L, H)
        h_cond = self.cond_proj(condition).unsqueeze(1) # (B, 1, H)
        h_perm = self.perm_emb(perm_idx).unsqueeze(1)   # (B, 1, H)
        
        # 2. Add Position Embedding [cite: 128]
        # pos_embed 不需要加到 condition 和 perm 上，只加到 parameter tokens 上
        h_tokens = h_tokens + pos_embed.unsqueeze(0)
        
        # 3. Construct Sequence: [Condition, Permutation, Token_1, Token_2, ...]
        # 这样 RNN 可以先看到任务条件和置换状态，再处理参数序列
        rnn_input = torch.cat([h_cond, h_perm, h_tokens], dim=1) # (B, L+2, H)
        
        # 4. Recurrent Processing
        rnn_out, _ = self.rnn(rnn_input)
        
        # 5. Extract Prototypes (Remove the first 2 context tokens)
        prototypes = rnn_out[:, 2:, :] # (B, L, H)
        
        return self.proto_proj(prototypes) # Corresponds to Eq (4)

class Diffusion1D(nn.Module):
    """
    Simple 1D ResNet for Diffusion, conditioned on Prototypes
    """
    def __init__(self, cfg):
        super().__init__()
        dim = cfg.model.hidden_dim
        self.time_emb = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        
        # Main processing blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, dim, 3, padding=1),
                nn.GroupNorm(8, dim),
                nn.SiLU(),
                nn.Conv1d(dim, dim, 3, padding=1),
            ) for _ in range(cfg.model.diff_layers)
        ])
        
        self.token_proj = nn.Linear(cfg.model.token_size, dim)
        self.final_proj = nn.Conv1d(dim, cfg.model.token_size, 1)

    def forward(self, x_noisy, t, prototypes):
        """
        x_noisy: (B, L, token_size) - Noisy parameter tokens
        t: (B,) - Time steps
        prototypes: (B, L, hidden_dim) - From RNN
        """
        B, L, K = x_noisy.shape
        H = prototypes.shape[2]
        
        # Time Embedding
        # Sinusoidal time embedding (omitted for brevity, assume t is already embedded or use simple projection)
        # Here we assume t passed in is (B, H) for simplicity, or we compute it.
        # Let's implement basic sinusoidal:
        half_dim = H // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        # Pad if H is odd
        if H % 2 == 1: emb = F.pad(emb, (0, 1))
        t_emb = self.time_emb(emb).unsqueeze(-1) # (B, H, 1)
        
        # Input Projection (B, L, K) -> (B, H, L) for Conv1d
        h = self.token_proj(x_noisy).transpose(1, 2) 
        proto = prototypes.transpose(1, 2) # (B, H, L)
        
        # Add Prototype Condition [cite: 656] (Directly add to feature map)
        h = h + proto
        
        for block in self.blocks:
            h_in = h
            # Add time embedding
            h = h + t_emb
            h = block(h)
            h = h + h_in # Residual
            
        out = self.final_proj(h).transpose(1, 2) # (B, L, K)
        return out


class RPG(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = ParamTokenizer(cfg.model.token_size, [tuple(cfg.data.linear1_shape), tuple(cfg.data.linear2_shape)])
        self.rnn_model = RecurrentModel(cfg)
        self.diffusion = Diffusion1D(cfg)
        
        # Register fixed 2D position embedding buffer
        # This is computed dynamically in forward first time or pre-computed if length is fixed
        self.pos_embed = None

    def forward(self, condition, w1, w2, seed_idx):
        """
        Training Forward Pass
        """
        # 1. Tokenize Parameters
        # gt_tokens: (B, L, K)
        gt_tokens, pos_ids, stats = self.tokenizer(w1, w2)
        device = gt_tokens.device
        
        # 2. Get 2D Position Embedding (Fixed)
        if self.pos_embed is None:
            self.pos_embed = get_2d_sincos_pos_embed(
                self.cfg.model.hidden_dim, 
                pos_ids['layer'].cpu(), 
                pos_ids['token'].cpu()
            ).to(device)
            
        # 3. Recurrent Model -> Prototypes
        # [cite: 136] P = f(H, S)
        prototypes = self.rnn_model(gt_tokens, condition, seed_idx, self.pos_embed)
        
        # 4. Diffusion Process
        # Sample noise
        noise = torch.randn_like(gt_tokens)
        B = gt_tokens.shape[0]
        t = torch.randint(0, self.cfg.model.timesteps, (B,), device=device).long()
        
        # Add noise (Standard Gaussian Diffusion Scheduler - Simplified)
        # In production, use a proper scheduler (e.g., DDPMScheduler from diffusers)
        # Here assumes linear beta schedule manually
        beta = torch.linspace(self.cfg.model.beta_start, self.cfg.model.beta_end, self.cfg.model.timesteps).to(device)
        alpha = 1. - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        
        a_bar_t = alpha_bar[t].view(B, 1, 1)
        noisy_tokens = torch.sqrt(a_bar_t) * gt_tokens + torch.sqrt(1 - a_bar_t) * noise
        
        # 5. Predict Noise [cite: 143]
        noise_pred = self.diffusion(noisy_tokens, t, prototypes)
        
        # 6. Loss
        loss = F.mse_loss(noise_pred, noise)
        return loss, noise_pred, noise
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tools import z_score


# ======================================================================
class SinusoidalTimeEmbedding(nn.Module):
    """时间步的正弦位置编码"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class TimeMLPEmbedding(nn.Module):
    """
    时间嵌入MLP
    """
    def __init__(self, dim: int, time_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            SinusoidalTimeEmbedding(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        return self.mlp(time)

# ======================================================================
class ConditionEmbedding(nn.Module):
    """
    条件嵌入，将条件投影到隐藏维度
    """
    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return z_score(self.cond_proj(cond)) # (batch_size, hidden_dim)

# ======================================================================
class SinusoidalPositionalEmbedding(nn.Module):
    """正弦位置编码"""
    
    def __init__(self, output_dim: int, max_tokens: int = 5000):
        super().__init__()
        self.output_dim = output_dim
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_tokens, output_dim)
        position = torch.arange(0, max_tokens, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, output_dim, 2).float() * (-math.log(10000.0) / output_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)  # (max_tokens, output_dim)
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (batch_size, seq_len) 或 (seq_len,)
        
        Returns:
            position_embedding: (batch_size, seq_len, output_dim) 或 (seq_len, output_dim)
        """
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)
        
        batch_size, seq_len = positions.shape
        
        # 获取位置编码
        pe = self.pe[positions.view(-1)]  # (batch_size*seq_len, output_dim)
        pe = pe.view(batch_size, seq_len, self.output_dim)
        
        return pe


class LearnablePositionalEmbedding(nn.Module):
    """可学习位置编码"""
    
    def __init__(self, output_dim: int, max_tokens: int):
        super().__init__()
        self.embedding = nn.Embedding(max_tokens, output_dim)
        
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (batch_size, seq_len) 或 (seq_len,)
        
        Returns:
            position_embedding: (batch_size, seq_len, output_dim) 或 (seq_len, output_dim)
        """
        return self.embedding(positions)


class LayerTokenPositionEmbedding(nn.Module):
    def __init__(self, **kwargs):
        """
        两层位置编码：层号 + 层内token位置
        
        Args:
            hidden_dim: 隐藏维度
            pe_fusion_type: 位置编码融合方式 (add/concat)
            token_pe_type: token位置编码类型 (sinusoidal/learnable)
            max_layers: 最大层数
            max_tokens: 每层最大token数
        """
        super().__init__()
        self.hidden_dim = kwargs["hidden_dim"]
        self.pe_fusion_type = kwargs["pe_fusion_type"]
        self.token_pe_type = kwargs["token_pe_type"]
        self.max_layers = kwargs["max_layers"]
        self.max_tokens = kwargs["max_tokens"]
        
        # 层编码
        self.layer_embedding = nn.Embedding(self.max_layers, self.hidden_dim)
        
        # token位置编码
        if self.token_pe_type == "sinusoidal":
            self.token_pos_embedding = SinusoidalPositionalEmbedding(self.hidden_dim, self.max_tokens)
        elif self.token_pe_type == "learnable":
            self.token_pos_embedding = LearnablePositionalEmbedding(self.hidden_dim, self.max_tokens)
        else:
            raise ValueError(f"未知的token_pe_type: {self.token_pe_type}")
        
        # 如果是concat方式，需要线性投影
        if self.pe_fusion_type == "concat":
            self.proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        elif self.pe_fusion_type == "add":
            self.proj = nn.Identity()
        else:
            raise ValueError(f"未知的pe_fusion_type: {self.pe_fusion_type}")
    
    def forward(self, layer_indices: torch.Tensor, token_in_layer_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            layer_indices: (batch_size, seq_len) 或 (seq_len,)
            token_in_layer_indices: (batch_size, seq_len) 或 (seq_len,)
        
        Returns:
            position_embedding: (batch_size, seq_len, hidden_dim)
        """
        # 确保是2D张量
        if layer_indices.dim() == 1:
            layer_indices = layer_indices.unsqueeze(0)
            token_in_layer_indices = token_in_layer_indices.unsqueeze(0)
                
        # 层编码
        layer_emb = self.layer_embedding(layer_indices)  # (batch_size, seq_len, hidden_dim)
        
        # token位置编码
        token_emb = self.token_pos_embedding(token_in_layer_indices)  # (batch_size, seq_len, hidden_dim)
        
        # 融合
        if self.pe_fusion_type == "add":
            pos_emb = layer_emb + token_emb
        elif self.pe_fusion_type == "concat":
            combined = torch.cat([layer_emb, token_emb], dim=-1)  # (batch_size, seq_len, hidden_dim*2)
            pos_emb = self.proj(combined)  # (batch_size, seq_len, hidden_dim)
        
        return pos_emb




# models/dit.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from einops import rearrange

from models.embedding import TimeMLPEmbedding 

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    AdaLN 调制操作：对归一化后的特征进行平移(shift)和缩放(scale)
    
    Args:
        x: 输入特征 (B, L, D)
        shift: 平移参数 (B, D)
        scale: 缩放参数 (B, D)
        
    Returns:
        out: 调制后的特征 (B, L, D)
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    DiT Transformer Block (基于 DiT 论文架构)
    
    包含以下关键特性：
    1. AdaLN-Zero: 使用条件向量预测 6 个调制参数 (shift, scale, gate)，并采用零初始化策略加速收敛。
    2. Flash Attention: 使用 PyTorch 的 scaled_dot_product_attention 进行高效计算。
    
    Args:
        hidden_dim: Transformer隐藏层维度
        num_heads: 注意力头数
        mlp_ratio: MLP扩展比例 (通常为 4.0)
        cond_dim: 条件向量维度 (时间步 + 外部条件)
        dropout: dropout概率
    
    Returns:
        out: (batch_size, seq_len, hidden_dim)
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        cond_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.scale = head_dim ** -0.5
        self.dropout = dropout

        # 1. Attention 部分的层归一化 (使用 AdaLN，这里仅实例化 LayerNorm 基础结构，不带可学习参数)
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        
        # 2. QKV 投影层
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=True)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

        # 3. MLP 部分的层归一化
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        
        # 4. 前馈网络 (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, hidden_dim),
        )

        # 5. AdaLN-Zero 调制网络
        # 预测 6 个参数: 
        # shift_msa, scale_msa, gate_msa (用于 Attention 块)
        # shift_mlp, scale_mlp, gate_mlp (用于 MLP 块)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_dim, bias=True)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """
        初始化权重，特别是实现 AdaLN-Zero 策略
        """
        # AdaLN-Zero 策略核心：将调制层的最后一层全零初始化
        # 这样初始状态下 gate=0，shift=0，scale=0
        # 使得整个 Block 在训练初期表现为恒等映射 (Identity mapping)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        # 其他层使用 Xavier 初始化
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.xavier_uniform_(self.mlp[2].weight)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入序列特征 (batch_size, seq_len, hidden_dim)
            cond: 条件特征 (batch_size, cond_dim)
            
        Returns:
            x: 输出序列特征 (batch_size, seq_len, hidden_dim)
        """
        B, L, D = x.shape
        
        # 1. 计算 AdaLN 参数
        # cond: (B, cond_dim) -> (B, 6 * hidden_dim)
        # 将结果切分为 6 份，每份维度为 (B, hidden_dim)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(cond).chunk(6, dim=1)

        # -----------------------------------------------------------
        # Block 1: Self-Attention (With Flash Attention)
        # -----------------------------------------------------------
        
        # 归一化 + 调制
        # x: (B, L, D) -> norm -> modulate -> (B, L, D)
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)

        # 计算 Q, K, V
        # qkv: (B, L, 3 * D)
        qkv = self.qkv(x_norm)
        
        # 重排维度以适应 Attention 计算
        # q, k, v: (B, num_heads, seq_len, head_dim)
        q, k, v = rearrange(qkv, 'b l (three h d) -> three b h l d', three=3, h=self.num_heads)
        
        # Flash Attention
        dropout_p = self.dropout if self.training else 0.0
        # x_attn: (B, num_heads, seq_len, head_dim)
        x_attn = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        
        # 输出投影
        # (B, num_heads, seq_len, head_dim) -> (B, seq_len, hidden_dim)
        x_attn = rearrange(x_attn, 'b h l d -> b l (h d)')
        x_attn = self.proj(x_attn)

        # 应用门控 (Gate) 并执行残差连接
        # gate_msa: (B, D) -> (B, 1, D)
        x = x + gate_msa.unsqueeze(1) * x_attn

        # -----------------------------------------------------------
        # Block 2: MLP (Feed Forward)
        # -----------------------------------------------------------
        
        # 归一化 + 调制
        # x_norm: (B, L, D)
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        
        # MLP 计算
        # x_mlp: (B, L, D)
        x_mlp = self.mlp(x_norm)
        
        # 应用门控 (Gate) 并执行残差连接
        x = x + gate_mlp.unsqueeze(1) * x_mlp

        return x

class DiT(nn.Module):
    """
    DiT 主模型架构
    
    Args:
        input_dim: 输入通道数
        hidden_dim: Transformer隐藏层维度
        depth: Transformer Block 层数
        num_heads: 注意力头数
        dim_feedforward: MLP隐藏层维度
        cond_dim: 全局条件向量维度
        proto_dim: 原型向量维度
        dropout: dropout概率
        use_pos_emb: 是否使用位置编码
        max_seq_len: 最大序列长度 (用于初始化位置编码表)
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.input_dim = kwargs["input_dim"]
        self.hidden_dim = kwargs["hidden_dim"]
        self.cond_dim = kwargs["cond_dim"]

        # 时间嵌入网络
        self.time_dim = self.hidden_dim * 4
        self.time_embed = TimeMLPEmbedding(self.hidden_dim, self.time_dim)
        
        # 输入投影层: (Channels) -> (Hidden Dim)
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        
        # 投影层: (Proto Dim) -> (Input Dim)
        if kwargs["proto_dim"] != self.input_dim:
            self.proto_proj = nn.Linear(kwargs["proto_dim"], self.input_dim)
        else:
            self.proto_proj = nn.Identity()

        # 位置编码 (可学习参数)
        if kwargs["use_pos_emb"]:
            # pos_embed: (1, max_seq_len, hidden_dim)
            self.pos_embed = nn.Parameter(torch.zeros(1, kwargs["max_seq_len"], self.hidden_dim))
            # 使用截断正态分布初始化
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 条件融合层: 融合时间嵌入和外部条件
        self.cond_fusion = nn.Sequential(
            nn.Linear(self.time_dim + self.cond_dim, self.cond_dim),
            nn.SiLU(),
            nn.Linear(self.cond_dim, self.cond_dim)
        )
        
        # 堆叠 DiT Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=self.hidden_dim,
                num_heads=kwargs["num_heads"],
                dim_feedforward=kwargs["dim_feedforward"],
                cond_dim=self.cond_dim,
                dropout=kwargs["dropout"]
            )
            for _ in range(kwargs["depth"])
        ])
        
        # 最终输出层结构 (参考 DiT 论文)
        # 包括: LayerNorm -> AdaLN Modulation -> Linear Projection
        self.final_norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(self.cond_dim, 2 * self.hidden_dim, bias=True) # 预测 shift 和 scale
        )
        
        # 对最终的 AdaLN 调制层也进行零初始化
        nn.init.constant_(self.final_adaLN[-1].weight, 0)
        nn.init.constant_(self.final_adaLN[-1].bias, 0)

        # 输出投影层: (Hidden Dim) -> (Input Dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.input_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        proto: torch.Tensor,
        cond: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 噪声输入 (batch_size, channels, seq_length)
            time: 时间步索引 (batch_size,)
            proto: 原型输入 (batch_size, channels, seq_length)
            cond: 全局条件向量 (batch_size, cond_dim)
            
        Returns:
            out: 去噪结果 (batch_size, channels, seq_length)
        """
        
        # 1. 维度调整与原型处理
        # proto: (B, C, L) -> (B, L, C)
        proto = rearrange(proto, 'b c l -> b l c')
        
        # 投影原型到输入维度 (如果需要)
        # proto: (B, L, C) -> (B, L, input_dim)
        if isinstance(self.proto_proj, nn.Linear):
            proto = self.proto_proj(proto)

        # 2. 输入特征投影
        # x: (B, C, L) -> (B, L, C)
        x = rearrange(x, 'b c l -> b l c')
        
        # 3. 注入原型 (残差连接)
        # 在投影前注入，假设原型已经在输入空间
        x = x + proto
        
        # x: (B, L, C) -> (B, L, hidden_dim)
        x = self.input_proj(x)

        # 4. 注入位置编码
        if self.use_pos_emb:
            seq_len = x.shape[1]
            # 检查序列长度是否越界
            if seq_len > self.pos_embed.shape[1]:
                 raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.pos_embed.shape[1]}")
            # 切片对应长度的位置编码并相加
            # pos_embed: (1, L, D) -> 广播到 (B, L, D)
            x = x + self.pos_embed[:, :seq_len, :]

        # 5. 时间和条件嵌入融合
        # time: (B,) -> time_emb: (B, time_dim)
        time_emb = self.time_embed(time)
        
        # 拼接时间嵌入和条件向量
        # fused_cond: (B, time_dim + cond_dim)
        fused_cond = torch.cat([time_emb, cond], dim=1)
        
        # 融合后的条件向量 c: (B, cond_dim)
        c = self.cond_fusion(fused_cond)

        # 6. 通过 Transformer Blocks
        for block in self.blocks:
            # x: (B, L, hidden_dim)
            x = block(x, c)

        # 7. 最终输出层
        # 预测最终的平移和缩放参数
        # shift, scale: (B, hidden_dim)
        shift, scale = self.final_adaLN(c).chunk(2, dim=1)
        
        # 归一化并调制
        # x: (B, L, hidden_dim)
        x = modulate(self.final_norm(x), shift, scale)
        
        # 投影回输入维度
        # x: (B, L, input_dim)
        x = self.output_proj(x)
        
        # 8. 恢复原始形状
        # x: (B, L, C) -> (B, C, L)
        x = rearrange(x, 'b l c -> b c l')
        
        return x
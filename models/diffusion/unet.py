# models/unet.py
"""
UNet模型
用于扩散模型的去噪网络
集成AdaGN和Prototype条件
"""

import math
from functools import partial
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.embedding import TimeMLPEmbedding


class AdaGN(nn.Module):
    """
    Adaptive Group Normalization
    将条件注入到GroupNorm中
    在diffusion模型中，用于条件化生成
    """
    def __init__(self, num_groups: int, num_channels: int, cond_dim: int):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        
        # 条件投影层
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, cond_dim // 2),  # 512 -> 256
            nn.SiLU(),
            nn.Linear(cond_dim // 2, num_channels * 2)  # 256 -> 128/256
        )
        
        # GroupNorm
        self.norm = nn.GroupNorm(num_groups, num_channels)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channels, length)  # GroupNorm 的标准格式
            cond: (batch_size, cond_dim)
        
        Returns:
            normalized_x: (batch_size, channels, length)
        """
        # 计算GroupNorm
        x_norm = self.norm(x)
        
        # 从条件计算scale和shift
        cond_params = self.cond_proj(cond)  # (batch_size, num_channels*2)
        scale, shift = cond_params.chunk(2, dim=1)  # 各(batch_size, num_channels)
        scale = scale.unsqueeze(-1)  # (batch_size, num_channels, 1)
        shift = shift.unsqueeze(-1)  # (batch_size, num_channels, 1)
        
        # 应用自适应归一化
        return x_norm * (1 + scale) + shift


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class Downsample(nn.Module):
    """下采样模块"""
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.conv = nn.Conv1d(dim_in, dim_out, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """上采样模块"""
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim_in, dim_out, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Block(nn.Module):
    """
    基础卷积块，使用AdaGN进行条件归一化
    """
    def __init__(
        self, 
        dim_in: int, 
        dim_out: int, 
        cond_dim: int,
        groups: int = 8,
        dropout: float = 0.0
    ):
        super().__init__()
        self.proj = nn.Conv1d(dim_in, dim_out, kernel_size=3, padding=1)
        self.norm = AdaGN(num_groups=min(groups, dim_out), num_channels=dim_out, cond_dim=cond_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(
        self, 
        x: torch.Tensor, 
        cond: torch.Tensor
    ) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x, cond)
        x = self.act(x)
        x = self.dropout(x)
        return x

class ResnetBlock(nn.Module):
    """
    ResNet块，集成AdaGN和时间嵌入
    """
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        time_emb_dim: int,
        cond_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out)
        )
        
        self.block1 = Block(dim_in, dim_out, cond_dim=cond_dim, dropout=dropout)
        self.block2 = Block(dim_out, dim_out, cond_dim=cond_dim, dropout=dropout)
        
        self.res_conv = nn.Conv1d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        cond: torch.Tensor
    ) -> torch.Tensor:
        t = self.time_mlp(time_emb)
        t = rearrange(t, 'b c -> b c 1')
        
        h = self.block1(x, cond)
        h = h + t
        h = self.block2(h, cond)
        
        return h + self.res_conv(x)

class Attention(nn.Module):
    """自注意力模块"""
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            nn.GroupNorm(min(8, dim), dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h=self.heads), qkv)
        
        q = q * self.scale
        sim = torch.einsum('b h c i, b h c j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        
        out = torch.einsum('b h i j, b h c j -> b h c i', attn, v)
        out = rearrange(out, 'b h c n -> b (h c) n')
        
        return self.to_out(out)

class LinearAttention(nn.Module):
    """线性注意力模块"""
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            nn.GroupNorm(min(8, dim), dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h=self.heads), qkv)
        
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n')
        
        return self.to_out(out)

class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(min(8, dim), dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.fn(x)

class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.fn(x, *args, **kwargs) + x

class UNet(nn.Module):
    """
    条件UNet模型
    输入包含 x_t, time, proto, cond
    """
    def __init__(
        self,
        dim: int = 64,
        layer_channels: List[int] = None,
        init_ch: int = 1,
        final_ch: int = 1,
        cond_dim: int = 192,
        proto_dim: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = None,
        dropout: float = 0.0,
        use_linear_attention: bool = True,
        **kwargs
    ):
        super().__init__()
        
        layer_channels = default(layer_channels, [64, 128, 256])
        attention_resolutions = default(attention_resolutions, [16])
        
        self.dim = dim
        self.init_ch = init_ch
        
        # --- 修改点 1: 输入通道仅为 init_ch ---
        input_ch = init_ch
        
        # --- 修改点 2: 原型投影层 ---
        # 如果原型维度和输入维度不一致，添加一个 1x1 卷积进行投影
        if proto_dim != init_ch:
            self.proto_proj = nn.Conv1d(proto_dim, init_ch, 1)
        else:
            self.proto_proj = nn.Identity()
        
        # 初始卷积
        self.init_conv = nn.Conv1d(input_ch, layer_channels[0], kernel_size=3, padding=1)
        
        # 时间嵌入
        time_dim = dim * 4
        self.time_mlp = TimeMLPEmbedding(dim, time_dim)
        
        # 注意力类型
        AttentionClass = LinearAttention if use_linear_attention else Attention
        
        # 构建层
        in_out = list(zip(layer_channels[:-1], layer_channels[1:]))
        num_resolutions = len(in_out)
        
        # 下采样
        self.downs = nn.ModuleList([])
        current_resolution = 64
        
        for ind, (ch_in, ch_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            use_attention = current_resolution in attention_resolutions
            
            blocks = []
            for _ in range(num_res_blocks):
                blocks.append(
                    ResnetBlock(
                        ch_in, ch_in,
                        time_emb_dim=time_dim,
                        cond_dim=cond_dim,
                        dropout=dropout
                    )
                )
            
            if use_attention:
                blocks.append(Residual(PreNorm(ch_in, AttentionClass(ch_in))))
            else:
                blocks.append(nn.Identity())
            
            if not is_last:
                blocks.append(Downsample(ch_in, ch_out))
                current_resolution //= 2
            else:
                blocks.append(nn.Conv1d(ch_in, ch_out, 3, padding=1))
            
            self.downs.append(nn.ModuleList(blocks))
        
        # 中间块
        mid_dim = layer_channels[-1]
        self.mid_block1 = ResnetBlock(
            mid_dim, mid_dim,
            time_emb_dim=time_dim,
            cond_dim=cond_dim,
            dropout=dropout
        )
        self.mid_attn = Residual(PreNorm(mid_dim, AttentionClass(mid_dim)))
        self.mid_block2 = ResnetBlock(
            mid_dim, mid_dim,
            time_emb_dim=time_dim,
            cond_dim=cond_dim,
            dropout=dropout
        )
        
        # 上采样
        self.ups = nn.ModuleList([])
        for ind, (ch_in, ch_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            use_attention = current_resolution in attention_resolutions
            
            blocks = []
            for i in range(num_res_blocks):
                res_in_ch = ch_out + ch_in if i == 0 else ch_out
                blocks.append(
                    ResnetBlock(
                        res_in_ch, ch_out,
                        time_emb_dim=time_dim,
                        cond_dim=cond_dim,
                        dropout=dropout
                    )
                )
            
            if use_attention:
                blocks.append(Residual(PreNorm(ch_out, AttentionClass(ch_out))))
            else:
                blocks.append(nn.Identity())
            
            if not is_last:
                blocks.append(Upsample(ch_out, ch_in))
                current_resolution *= 2
            else:
                blocks.append(nn.Conv1d(ch_out, ch_in, 3, padding=1))
            
            self.ups.append(nn.ModuleList(blocks))
        
        # 最终块
        self.final_res_block = ResnetBlock(
            layer_channels[0] * 2, layer_channels[0],
            time_emb_dim=time_dim,
            cond_dim=cond_dim,
            dropout=dropout
        )
        self.final_conv = nn.Conv1d(layer_channels[0], final_ch, 1)
    
    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        proto: torch.Tensor,
        cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: 噪声输入 [B, C, L]
            time: 时间步 [B]
            proto: 原型 [B, proto_C, L]
            cond: 全局条件 [B, cond_dim]
        """
        # --- 修改点 3: 投影并相加 ---
        # 确保 prototype 通道数与 x 一致
        proto = self.proto_proj(proto)
        # 将原型加到输入上，而不是拼接
        x = x + proto
        
        # 初始卷积
        x = self.init_conv(x)
        r = x.clone()
        
        # 计算时间嵌入
        t = self.time_mlp(time)
        
        # 下采样
        h = []
        for blocks in self.downs:
            for block in blocks[:-2]:
                if isinstance(block, ResnetBlock):
                    x = block(x, t, cond)
                else:
                    x = block(x)
            x = blocks[-2](x)
            h.append(x)
            x = blocks[-1](x)
        
        # 中间层
        x = self.mid_block1(x, t, cond)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, cond)
        
        # 上采样
        for blocks in self.ups:
            for i, block in enumerate(blocks[:-2]):
                if i == 0:
                    x = torch.cat([x, h.pop()], dim=1)
                
                if isinstance(block, ResnetBlock):
                    x = block(x, t, cond)
                else:
                    x = block(x)
            x = blocks[-2](x)
            x = blocks[-1](x)
        
        # 最终层
        x = torch.cat([x, r], dim=1)
        x = self.final_res_block(x, t, cond)
        x = self.final_conv(x)
        
        return x
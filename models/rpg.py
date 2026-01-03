import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Any

from models.prototype.proto import PrototypeGenerator
from models.diffusion.ddpm import GaussianDiffusion
from models.diffusion.denoiser import get_denoiser
from models.embedding import LayerTokenPositionEmbedding

class RecurrentParameterGenerator(nn.Module):
    def __init__(self, **kwargs):
        """
        端到端的 RecurrentParameterGenerator 模型
        包含模块:
        1. PositionEmbedding: 生成层/Token位置编码
        2. PrototypeGenerator: 生成每个Token对应的Prototype
        3. Denoiser: 扩散模型的去噪网络，接收 (x_t, t, proto, cond)
        4. DDPM: 控制扩散过程的模块
        
        Args:
            __init__:
                - pos_embedding: 位置编码配置
                    - hidden_dim: 位置编码维度
                    - pe_fusion_type: 位置编码融合类型
                    - token_pe_type: Token位置编码类型
                    - max_layers: 最大层索引
                    - max_tokens: 最大Token索引
                - proto: 原型生成器配置
                    - hidden_dim: 原型隐藏维度
                - denoiser: 去噪网络配置
                - ddpm: DDPM配置
            forward:
                - num_seeds: 种子数量
                - layer_indices: 层索引 [Batch, SeqLen]
                - token_indices: Token索引 [Batch, SeqLen]
                - permutation_state_indices: 置换状态索引 [Batch]
        
        Returns:
            - loss_dict: 包含 loss, mse, cos 等指标
        """
        super().__init__()
        # 1. 位置编码模块
        self.pos_embedding = LayerTokenPositionEmbedding(**kwargs["pos_embedding"])
        
        # 2. 原型生成模块
        self.prototype_generator = PrototypeGenerator(**kwargs["proto"])
        
        # 3. 去噪网络模块
        self.denoiser = get_denoiser(**kwargs["denoiser"])
        
        # 4. DDPM 模块
        self.diffusion = GaussianDiffusion(self.denoiser, **kwargs["ddpm"])

    def get_prototypes(
        self,
        condition: torch.Tensor,
        layer_indices: torch.Tensor,
        token_indices: torch.Tensor,
        permutation_state_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        辅助函数：生成Prototypes
        """
        # 生成位置编码
        pos_emb = self.pos_embedding(layer_indices, token_indices)
        
        # 生成Prototypes
        # pos_emb: (batch, seq_len, hidden_dim)
        # prototypes: (batch, prototype_dim, seq_len) - 内部已经做了transpose
        prototypes = self.prototype_generator(
            condition=condition,
            permutation_state_indices=permutation_state_indices,
            position_emb=pos_emb
        )
        return prototypes

    def forward(
        self, 
        x_0: torch.Tensor, 
        condition: torch.Tensor,
        layer_indices: torch.Tensor,
        token_indices: torch.Tensor,
        permutation_state_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        训练前向传播
        
        Args:
            x_0: 目标数据 [Batch, Channels, SeqLen]
            condition: 条件向量 [Batch, CondDim]
            layer_indices: 层索引 [Batch, SeqLen]
            token_indices: Token索引 [Batch, SeqLen]
            permutation_state_indices: 置换状态索引 [Batch]
            
        Returns:
            loss_dict: 包含 loss, mse, cos 等
        """
        # 1. 获取 Prototypes
        # [Batch, ProtoDim, SeqLen]
        prototypes = self.get_prototypes(condition, layer_indices, token_indices, permutation_state_indices)
        
        # 2. 计算 DDPM Loss
        # 这里会将 prototypes 作为条件传入 UNet
        loss_dict = self.diffusion(
            x_0=x_0,
            proto=prototypes,
            cond=condition,
            noise=None # 自动生成噪声
        )
        
        return loss_dict

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        layer_indices: torch.Tensor,
        token_indices: torch.Tensor,
        use_ddim: bool = False,
        ddim_steps: int = 50,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        推理/采样函数
        
        Args:
            condition: 条件向量
            layer_indices: 目标生成的层索引
            token_indices: 目标生成的Token索引
            use_ddim: 是否使用DDIM加速
            ddim_steps: DDIM步数
            eta: DDIM超参数
            
        Returns:
            generated_data: [Batch, Channels, SeqLen]
        """
        # 1. 获取 Prototypes (推理模式下 permutation_state_indices 为 None)
        prototypes = self.get_prototypes(
            condition=condition,
            layer_indices=layer_indices,
            token_indices=token_indices,
            permutation_state_indices=None
        )
        
        # 2. DDPM 采样
        samples = self.diffusion.sample(
            proto=prototypes,
            cond=condition,
            use_ddim=use_ddim,
            ddim_steps=ddim_steps,
            eta=eta
        )
        
        return samples
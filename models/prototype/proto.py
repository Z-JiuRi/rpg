# models/rpg.py
from torch import _conj
from ninja import expand
import torch
import torch.nn as nn
from typing import Optional

from models.prototype.rnn import get_rnn_model
from models.embedding import ConditionEmbedding

class PrototypeGenerator(nn.Module):
    def __init__(self, **kwargs):
        """
        原型生成器
        
        mamba:
            d_model (int): 输入特征维度
            d_state (int, optional): 状态空间维度（状态矩阵的大小），默认为16
            d_conv (int, optional): 卷积核大小，用于离散化状态空间模型，默认为4
            expand (int, optional): 扩展因子，内部特征维度为d_model*expand，默认为2
            dropout (float, optional): dropout率，应用于输出，默认为0.0
        transformer:
            d_model (int): 输入特征维度
            num_layers (int, optional): 层数，默认为4
            nhead (int, optional): 头数，默认为4
            dim_feedforward (int, optional): 前馈网络维度，默认为2048
            dropout (float, optional): dropout率，应用于输出，默认为0.1
        gru:
            d_model (int): 输入特征维度
            num_layers (int, optional): 层数，默认为1
            dropout (float, optional): dropout率，应用于输出，默认为0.0
            bidirectional (bool, optional): 是否为双向GRU，默认为False
        lstm:
            d_model (int): 输入特征维度
            num_layers (int, optional): 层数，默认为1
            dropout (float, optional): dropout率，应用于输出，默认为0.0
            bidirectional (bool, optional): 是否为双向LSTM，默认为False
        """
        super().__init__()
        self.proto_drop = kwargs["proto_drop"]
        
        # 条件嵌入
        self.condition_embedding = ConditionEmbedding(kwargs["cond_dim"], kwargs["hidden_dim"])
        
        # 置换状态嵌入
        self.permutation_embedding = nn.Embedding(kwargs["num_seeds"], kwargs["hidden_dim"])
        
        # 可学习的空嵌入
        if kwargs["learnable_null"]:
            self.null_embedding = nn.Parameter(torch.randn(1, kwargs["hidden_dim"]))    
        else:
            self.register_buffer('null_embedding', torch.zeros(1, kwargs["hidden_dim"]))
        
        # rnn输入序列前缀生成
        self.prefix_generator = nn.Sequential(
            nn.Linear(kwargs["hidden_dim"], kwargs["hidden_dim"]),
            nn.GELU(),
            nn.Linear(kwargs["hidden_dim"], kwargs["hidden_dim"])
        )
        
        # RNN
        self.rnn = get_rnn_model(**kwargs)
        
        # 原型投影层
        self.prototype_proj = nn.Linear(kwargs["hidden_dim"], kwargs["prototype_dim"])
    
    def forward(self, condition: torch.Tensor, 
                permutation_state_indices: Optional[torch.Tensor] = None,
                position_emb: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """
        Returns:
            prototypes: (batch_size, prototype_dim, seq_len) -> 注意转置了
        """
        batch_size = condition.shape[0]
        
        # 编码条件
        cond_emb = self.condition_embedding(condition)
        drop_mask = torch.rand(batch_size, device=condition.device) < self.proto_drop
        cond_emb = torch.where(drop_mask.view(-1, 1), self.null_embedding.expand(batch_size, -1), cond_emb)
        
        # 编码置换状态
        if permutation_state_indices is not None:
            perm_emb = self.permutation_embedding(permutation_state_indices)
            drop_mask = torch.rand(batch_size, device=condition.device) < self.proto_drop
            perm_emb = torch.where(drop_mask.view(-1, 1), self.null_embedding.expand(batch_size, -1), perm_emb)
        else:
            perm_emb = self.null_embedding.expand(batch_size, -1)
        
        # 生成RNN输入前缀
        rnn_input0 = self.prefix_generator(cond_emb).unsqueeze(1)
        rnn_input1 = self.prefix_generator(perm_emb).unsqueeze(1)
        
        # 拼接: [P0, P1, Pos_Embs...]
        combined_input = torch.cat([rnn_input0, rnn_input1, position_emb], dim=1)
        
        # RNN生成
        rnn_output = self.rnn(combined_input)
        
        # 投影
        prototypes = self.prototype_proj(rnn_output)
        
        # 切片 (去除前两个) 并转置 (B, L, C) -> (B, C, L)
        # 以适配 UNet 的 (Batch, Channel, Length)
        return prototypes[:, 2:, :].transpose(1, 2)
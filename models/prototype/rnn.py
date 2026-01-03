import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba as MambaSSM

# 导出所有模型类
__all__ = ["get_rnn"]


class GRU(nn.Module):
    """
    Args:
        d_model (int): 输入特征维度，也是隐藏层维度
        num_layers (int, optional): GRU层数，默认为2。多层GRU会堆叠在一起
        dropout (float, optional): 层间dropout率，仅在num_layers>1时生效，默认为0.0
        bidirectional (bool, optional): 是否使用双向GRU，默认为False
    
    forward:
        Args:

    """
    def __init__(self, d_model, num_layers=2, dropout=0.0, bidirectional=False):
        super().__init__()
        self.core = nn.GRU(
            input_size=d_model,        # 输入特征的维度
            hidden_size=d_model,       # 隐藏状态的维度
            num_layers=num_layers,     # GRU层的堆叠数量
            dropout=dropout if num_layers > 1 else 0.0,  # 层间dropout，单层时不使用
            batch_first=True,          # 输入输出格式为(batch, seq_len, feature)
            bidirectional=bidirectional,  # 是否为双向GRU
        )
        # 如果是双向GRU，需要将前向和后向的输出拼接后投影回原维度
        self.out_proj = nn.Linear(d_model * 2, d_model) if bidirectional else nn.Identity()

    def forward(self, x):
        """
        Args:
            x (Tensor): 输入张量，形状为(batch_size, seq_len, d_model)
            
        Returns:
            Tensor: 输出张量，形状为(batch_size, seq_len, d_model)
        """
        out, _ = self.core(x)  # 忽略隐藏状态输出
        return self.out_proj(out)


class LSTM(nn.Module):
    """
    Args:
        d_model (int): 输入特征维度，也是隐藏层维度
        num_layers (int, optional): LSTM层数，默认为2
        dropout (float, optional): 层间dropout率，仅在num_layers>1时生效，默认为0.0
        bidirectional (bool, optional): 是否使用双向LSTM，默认为False
    """
    def __init__(self, d_model, num_layers=2, dropout=0.0, bidirectional=False):
        super().__init__()
        self.core = nn.LSTM(
            input_size=d_model,        # 输入特征的维度
            hidden_size=d_model,       # 隐藏状态的维度
            num_layers=num_layers,     # LSTM层的堆叠数量
            dropout=dropout if num_layers > 1 else 0.0,  # 层间dropout
            batch_first=True,          # 输入输出格式为(batch, seq_len, feature)
            bidirectional=bidirectional,  # 是否为双向LSTM
        )
        # 双向LSTM需要处理输出维度翻倍的情况
        self.out_proj = nn.Linear(d_model * 2, d_model) if bidirectional else nn.Identity()

    def forward(self, x):
        """
        Args:
            x (Tensor): 输入张量，形状为(batch_size, seq_len, d_model)
            
        Returns:
            Tensor: 输出张量，形状为(batch_size, seq_len, d_model)
        """
        out, _ = self.core(x)  # 忽略隐藏状态和细胞状态输出
        return self.out_proj(out)


class Transformer(nn.Module):
    """
    Args:
        d_model (int): 输入特征维度
        num_layers (int, optional): Transformer编码器层数，默认为4
        nhead (int, optional): 注意力头的数量，默认为8
        dim_feedforward (int, optional): 前馈神经网络的隐藏层维度，默认为2048
        dropout (float, optional): dropout率，应用于注意力和前馈网络，默认为0.0
        activation (str, optional): 激活函数类型，如'relu'、'gelu'等，默认为"gelu"
        norm_first (bool, optional): 是否采用Pre-LN结构（LayerNorm在子层之前），默认为True
        final_norm (bool, optional): 是否在编码器最后添加LayerNorm，默认为True
    """
    def __init__(
        self,
        d_model,
        num_layers=4,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.0,
        activation="gelu",
        norm_first=True,
        final_norm=True,
    ):
        super().__init__()
        # 创建单个Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,           # 输入维度
            nhead=nhead,               # 多头注意力的头数（d_model需能被nhead整除）
            dim_feedforward=dim_feedforward,  # 前馈网络中间层维度
            dropout=dropout,           # dropout概率
            activation=activation,     # 激活函数
            batch_first=True,          # 输入输出格式为(batch, seq_len, feature)
            norm_first=norm_first,     # Pre-LN或Post-LN结构
        )
        # 最终的LayerNorm层（可选）
        norm = nn.LayerNorm(d_model) if final_norm else None
        # 堆叠多个编码器层形成完整编码器
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=norm)

    def forward(self, x):
        """
        Args:
            x (Tensor): 输入张量，形状为(batch_size, seq_len, d_model)
            
        Returns:
            Tensor: 输出张量，形状为(batch_size, seq_len, d_model)
        """
        return self.encoder(x)


class Mamba(nn.Module):
    """
    Args:
        d_model (int): 输入特征维度
        d_state (int, optional): 状态空间维度（状态矩阵的大小），默认为16
        d_conv (int, optional): 卷积核大小，用于离散化状态空间模型，默认为4
        expand (int, optional): 扩展因子，内部特征维度为d_model*expand，默认为2
        dropout (float, optional): dropout率，应用于输出，默认为0.0
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.0):
        super().__init__()
        if MambaSSM is None:
            raise ImportError("mamba_ssm is not installed. Please install it to use Mamba.")
            
        # Mamba核心模块
        self.core = MambaSSM(
            d_model=d_model,           # 输入输出维度
            d_state=int(d_state),      # 状态维度（状态空间模型的隐藏状态大小）
            d_conv=int(d_conv),        # 卷积核大小（用于离散化的卷积维度）
            expand=int(expand),        # 扩展因子（内部表示的放大倍数）
        )
        # 输出dropout（可选）
        self.dropout = nn.Dropout(float(dropout)) if dropout and float(dropout) > 0 else nn.Identity()

    def forward(self, x):
        """
        Args:
            x (Tensor): 输入张量，形状为(batch_size, seq_len, d_model)
            
        Returns:
            Tensor: 输出张量，形状为(batch_size, seq_len, d_model)
        """
        return self.dropout(self.core(x))


def get_rnn_model(**kwargs):
    """
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
    rnn_type = kwargs['rnn_type']
    if rnn_type == 'gru':
        return GRU(
            d_model=kwargs['hidden_dim'],
            num_layers=kwargs['num_layers'],
            dropout=kwargs['dropout'],
            bidirectional=kwargs['bidirectional']
        )
    elif rnn_type == 'lstm':
        return LSTM(
            d_model=kwargs['hidden_dim'],
            num_layers=kwargs['num_layers'],
            dropout=kwargs['dropout'],
            bidirectional=kwargs['bidirectional']
        )
    elif rnn_type == 'transformer':
        return Transformer(
            d_model=kwargs['hidden_dim'],
            num_layers=kwargs['num_layers'],
            nhead=kwargs['nhead'],
            dim_feedforward=kwargs['dim_feedforward'],
            dropout=kwargs['dropout'],
        )
    elif rnn_type == 'mamba':
        return Mamba(
            d_model=kwargs['hidden_dim'],
            d_state=kwargs['d_state'],
            d_conv=kwargs['d_conv'],
            expand=kwargs['expand'],
            dropout=kwargs['dropout']
        )
    else:
        raise ValueError(f"Unknown RNN type: {rnn_type}")
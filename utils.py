# utils.py
import torch
import numpy as np
import math

def get_2d_sincos_pos_embed(embed_dim, layer_ids, token_ids):
    """
    实现论文公式 (3) 的 2D 位置编码。
    e[i] = [e_i^layer, e_i^token]
    这里简化为将 layer_pos 和 token_pos 的编码相加或拼接。
    为了简单且有效，我们使用标准的 Sinusoidal 编码并将 Layer 和 Token 的编码相加。
    
    Args:
        embed_dim: 输出维度
        layer_ids: (Seq_Len,) 张量，表示每个 token 属于哪一层
        token_ids: (Seq_Len,) 张量，表示每个 token 在层内的索引
    Returns:
        pos_embed: (Seq_Len, embed_dim)
    """
    def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
        """
        embed_dim: output dimension for each position
        pos: list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float32)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

        emb_sin = np.sin(out) # (M, D/2)
        emb_cos = np.cos(out) # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb

    # 生成 Layer 的位置编码
    layer_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, layer_ids.numpy())
    # 生成 Token 在层内的位置编码
    token_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, token_ids.numpy())
    
    # 论文中隐含的是融合这两个信息，相加是 Transformer 中的标准做法
    pos_embed = layer_embed + token_embed
    return torch.from_numpy(pos_embed).float()
import torch
import torch.nn as nn
import torch.nn.functional as F
#### 4.27修改： AttentionPhi 模块，
# 采用 Pre-LN 架构彻底解决深层梯度消失问题，
# 并引入 MLP 预测头提升性能。
class AttentionPhi(nn.Module):
    """Transformer-based Attention Generator (SpatialAttentionGenerator).
    Maps DINO features (B, 768, 28, 28) to CT Attention Map (B, 1, 28, 28).
    """
    def __init__(self, in_channels=768, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024):
        super().__init__()
        
        # 1. 线性降维 (Projection)
        self.input_proj = nn.Linear(in_channels, d_model)
        
        # 2. 可学习的位置编码 (Positional Encoding)
        self.pos_embed = nn.Parameter(torch.zeros(1, 784, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # 3. 自注意力特征提取 (Transformer Encoder)
        # 【修改点 1】: 设置 norm_first=True 开启 Pre-LN 架构，彻底解决深层梯度消失问题
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation='gelu',
            dropout=0.1,
            norm_first=True  # <-- 核心修改
        )
        
        # 【修改点 2】: Pre-LN 架构要求在整个 Encoder 结束、进入预测头之前进行一次最终的 LayerNorm
        final_norm = nn.LayerNorm(d_model)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers, 
            norm=final_norm  # <-- 核心修改：接管你原来外部的 self.norm
        )
        
        # 4. 预测头 (Regression Head)
        # 【修改点 3】: 从 Linear(256, 1) 升级为 MLP 漏斗结构，提供高维到1维的非线性缓冲
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),  # 256 -> 128
            nn.GELU(),                         # 引入非线性流形折叠
            nn.LayerNorm(d_model // 2),        # (可选) 稳定最后一层的输出方差
            nn.Linear(d_model // 2, 1)         # 128 -> 1
        )

    def forward(self, x):
        # Input x: (B, 768, 28, 28)
        B, C, H, W = x.shape
        
        # 1. 序列化 (Flatten & Permute): (B, 768, 784) -> (B, 784, 768)
        x = x.flatten(2).transpose(1, 2)
        
        # 2. 线性降维 (Projection): (B, 784, 256)
        x = self.input_proj(x)
        
        # 3. 位置编码 (Positional Encoding)
        x = x + self.pos_embed
        
        # 【注意】：这里删除了你原本的 x = self.norm(x)
        # 因为 Pre-LN 的机制是：进入 self.transformer 后，每一层内部会首先做 LN
        
        # 4. 自注意力特征提取 (Transformer Encoder)
        # 输出的 x 已经被 final_norm 归一化过
        x = self.transformer(x)
        
        # 5. 预测头 (Regression Head): (B, 784, 1)
        x = self.regression_head(x)
        
        # 6. 空间重构与激活 (Reshape & Activation)
        # (B, 784, 1) -> (B, 1, 28, 28)
        x = x.transpose(1, 2).reshape(B, 1, H, W)
        
        # Sigmoid 激活到 [0, 1]
        output = torch.sigmoid(x)
        
        return output
class AttentionCompoundLoss(nn.Module):
    """Compound Loss: L1 + alpha * Soft Dice."""
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.epsilon = 1e-5

    def soft_dice_loss(self, preds, targets):
        batch_size = preds.size(0)
        p = preds.view(batch_size, -1)
        t = targets.view(batch_size, -1)
        
        intersection = torch.sum(p * t, dim=1)
        cardinality = torch.sum(p**2 + t**2, dim=1)
        
        dice_score = (2. * intersection + self.epsilon) / (cardinality + self.epsilon)
        return 1.0 - dice_score.mean()

    def forward(self, preds, targets):
        l1 = self.l1_loss(preds, targets)
        dice = self.soft_dice_loss(preds, targets)
        total = l1 + self.alpha * dice
        return total, l1, dice

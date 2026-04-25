import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPhi(nn.Module):
    """Transformer-based Attention Generator (SpatialAttentionGenerator).
    Maps DINO features (B, 768, 28, 28) to CT Attention Map (B, 1, 28, 28).
    """
    def __init__(self, in_channels=768, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024):
        super().__init__()
        
        # 2. 线性降维 (Projection)
        self.input_proj = nn.Linear(in_channels, d_model)
        
        # 3. 可学习的位置编码 (Positional Encoding) for 28x28 grid
        self.pos_embed = nn.Parameter(torch.zeros(1, 784, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # 4. 自注意力特征提取 (Transformer Encoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation='gelu',
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 5. 预测头 (Regression Head)
        self.regression_head = nn.Linear(d_model, 1)
        
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # Input x: (B, 768, 28, 28)
        B, C, H, W = x.shape
        
        # 1. 序列化 (Flatten & Permute): (B, 768, 784) -> (B, 784, 768)
        x = x.flatten(2).transpose(1, 2)
        
        # 2. 线性降维 (Projection): (B, 784, 256)
        x = self.input_proj(x)
        
        # 3. 位置编码 (Positional Encoding)
        x = x + self.pos_embed
        x = self.norm(x)
        
        # 4. 自注意力特征提取 (Transformer Encoder)
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

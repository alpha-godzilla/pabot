import torch
import torch.nn as nn


class AttentionPhi(nn.Module):
    """Small residual mapper from MRI DINO attention map to CT-style attention map."""

    def __init__(self, in_channels=1, hidden_channels=32):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.body = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.tail = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, attn_map):
        feat = self.head(attn_map)
        residual = self.tail(self.body(feat))
        return attn_map + residual

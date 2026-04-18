import torch
import torch.nn as nn


class LightweightSpatialAttention(nn.Module):
    """Token attention on small attention maps (e.g., 28x28) with low overhead."""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(1, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w

        qkv = self.qkv(self.norm(x))
        q, k, v = torch.chunk(qkv, chunks=3, dim=1)

        q = q.view(b, self.num_heads, self.head_dim, n).transpose(-2, -1)
        k = k.view(b, self.num_heads, self.head_dim, n).transpose(-2, -1)
        v = v.view(b, self.num_heads, self.head_dim, n).transpose(-2, -1)

        attn = torch.softmax((q * self.scale) @ k.transpose(-2, -1), dim=-1)
        out = attn @ v
        out = out.transpose(-2, -1).contiguous().view(b, c, h, w)
        return self.proj(out)


class AttentionPhi(nn.Module):
    """Residual CNN + lightweight attention + post gate.

    Learns delta first, then applies suppression gate on Y = A + delta:
        output = (1 - G) * Y
    where G in [0, 1] is predicted from [A, delta, |delta|].
    """

    def __init__(self, in_channels=1, hidden_channels=32, attn_heads=4):
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
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.attn = LightweightSpatialAttention(hidden_channels, num_heads=attn_heads)
        self.tail = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

        gate_mid = max(8, hidden_channels // 2)
        self.post_gate = nn.Sequential(
            nn.Conv2d(in_channels * 3, gate_mid, kernel_size=3, padding=1),
            nn.InstanceNorm2d(gate_mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_mid, gate_mid, kernel_size=3, padding=1),
            nn.InstanceNorm2d(gate_mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_mid, in_channels, kernel_size=1, padding=0),
        )

        # Start with near-identity behavior: G ~= 0, so output ~= Y.
        nn.init.zeros_(self.post_gate[-1].weight)
        nn.init.constant_(self.post_gate[-1].bias, -4.0)

    def forward(self, attn_map):
        feat = self.head(attn_map)
        feat = self.body(feat)
        feat = feat + self.attn(feat)
        delta = self.tail(feat)
        y = attn_map + delta

        gate_in = torch.cat([attn_map, delta, delta.abs()], dim=1)
        g = torch.sigmoid(self.post_gate(gate_in))
        return (1.0 - g) * y

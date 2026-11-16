import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(DownBlock, self).__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = ConvBlock(in_channels, out_channels, dropout)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels, dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if x1.shape[2:] != x2.shape[2:]:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='trilinear', align_corners=False)
        return self.conv(torch.cat([x2, x1], dim=1))


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        x = x + self.attn(x)
        x = self.norm2(x)
        x = x + self.mlp(x)
        return x


class AdaptiveTeU(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=32,
                 transformer_embed_dim=128, transformer_num_heads=4, transformer_depth=2,
                 dropout=0.1):
        super(AdaptiveTeU, self).__init__()

        self.inc = ConvBlock(in_channels, base_filters, dropout)
        self.down1 = DownBlock(base_filters, base_filters * 2, dropout)
        self.down2 = DownBlock(base_filters * 2, base_filters * 4, dropout)
        self.down3 = DownBlock(base_filters * 4, base_filters * 8, dropout)

        self.bottleneck_conv = ConvBlock(base_filters * 8, base_filters * 16, dropout)

        self.adaptive_pool = nn.AdaptiveAvgPool3d((8, 8, 8))

        self.transformer_proj = nn.Conv3d(base_filters * 16, transformer_embed_dim, kernel_size=1)
        self.transformers = nn.ModuleList([
            TransformerBlock(transformer_embed_dim, transformer_num_heads, dropout=dropout)
            for _ in range(transformer_depth)
        ])
        self.transformer_proj_back = nn.Conv3d(transformer_embed_dim, base_filters * 16, kernel_size=1)

        self.up1 = UpBlock(base_filters * 16, base_filters * 8, dropout)
        self.up2 = UpBlock(base_filters * 8, base_filters * 4, dropout)
        self.up3 = UpBlock(base_filters * 4, base_filters * 2, dropout)
        self.up4 = UpBlock(base_filters * 2, base_filters, dropout)

        self.outc = nn.Sequential(
            nn.Conv3d(base_filters, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        input_size = x.shape[2:]

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x5 = self.bottleneck_conv(x4)

        x5_pooled = self.adaptive_pool(x5)

        B, C, D, H, W = x5_pooled.shape
        x5_pooled = self.transformer_proj(x5_pooled)
        x5_pooled = x5_pooled.flatten(2).transpose(1, 2)

        for transformer in self.transformers:
            x5_pooled = transformer(x5_pooled)

        x5_pooled = x5_pooled.transpose(1, 2).reshape(B, -1, D, H, W)
        x5_pooled = self.transformer_proj_back(x5_pooled)

        x5_pooled = F.interpolate(x5_pooled, size=x5.shape[2:], mode='trilinear', align_corners=False)

        x = self.up1(x5_pooled, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        out = self.outc(x)

        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='trilinear', align_corners=False)

        return out


TeU = AdaptiveTeU

__all__ = ['AdaptiveTeU', 'TeU']
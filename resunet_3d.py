"""
3D ResUNet backbone for fault segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.15):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        if in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        identity = self.proj(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class DownRes3D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool3d(2),
            ResidualBlock3D(in_channels, out_channels, dropout)
        )

    def forward(self, x):
        return self.block(x)


class UpRes3D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout=0.15):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.reduce = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.skip_proj = nn.Conv3d(skip_channels, out_channels, kernel_size=1)
        self.block = ResidualBlock3D(out_channels * 2, out_channels, dropout)

    def forward(self, x, skip):
        x = self.reduce(self.up(x))
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        skip = self.skip_proj(skip)
        return self.block(torch.cat([skip, x], dim=1))


class ResUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=16, dropout=0.15):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters

        f = base_filters
        self.inc = ResidualBlock3D(in_channels, f, dropout)
        self.down1 = DownRes3D(f, f * 2, dropout)
        self.down2 = DownRes3D(f * 2, f * 4, dropout)
        self.down3 = DownRes3D(f * 4, f * 8, dropout)
        self.bottleneck = DownRes3D(f * 8, f * 8, dropout)

        self.up1 = UpRes3D(f * 8, f * 8, f * 4, dropout)
        self.up2 = UpRes3D(f * 4, f * 4, f * 2, dropout)
        self.up3 = UpRes3D(f * 2, f * 2, f, dropout)
        self.up4 = UpRes3D(f, f, f, dropout)
        self.outc = nn.Sequential(
            nn.Conv3d(f, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        input_size = x.shape[2:]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode="trilinear", align_corners=False)
        return out

    def get_features(self, x):
        features = {}
        x1 = self.inc(x)
        features["enc1"] = x1
        x2 = self.down1(x1)
        features["enc2"] = x2
        x3 = self.down2(x2)
        features["enc3"] = x3
        x4 = self.down3(x3)
        features["enc4"] = x4
        x5 = self.bottleneck(x4)
        features["bottleneck"] = x5
        x = self.up1(x5, x4)
        features["dec1"] = x
        x = self.up2(x, x3)
        features["dec2"] = x
        x = self.up3(x, x2)
        features["dec3"] = x
        x = self.up4(x, x1)
        features["dec4"] = x
        features["output"] = self.outc(x)
        return features


class AdaptiveResUNet3D(ResUNet3D):
    pass


__all__ = ["ResUNet3D", "AdaptiveResUNet3D"]

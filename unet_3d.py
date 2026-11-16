import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super(DoubleConv3D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super(Down3D, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels, dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=True, dropout=0.2):
        super(Up3D, self).__init__()
        self.out_channels = out_channels
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv_reduce = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv_reduce = None
        self.double_conv = DoubleConv3D(out_channels * 2, out_channels, dropout)
        self.skip_proj = None

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.conv_reduce is not None:
            x1 = self.conv_reduce(x1)
        x1 = self._match_size(x1, x2)
        if x2.shape[1] != self.out_channels:
            if self.skip_proj is None or self.skip_proj.in_channels != x2.shape[1]:
                self.skip_proj = nn.Conv3d(x2.shape[1], self.out_channels, kernel_size=1).to(x2.device)
            x2 = self.skip_proj(x2)
        x = torch.cat([x2, x1], dim=1)
        return self.double_conv(x)

    def _match_size(self, x, target):
        if x.shape[2:] != target.shape[2:]:
            x = F.interpolate(x, size=target.shape[2:], mode='trilinear', align_corners=False)
        return x


class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv(x))


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=32, trilinear=True, dropout=0.2):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters
        self.trilinear = trilinear
        factor = 2 if trilinear else 1
        self.inc = DoubleConv3D(in_channels, base_filters, dropout)
        self.down1 = Down3D(base_filters, base_filters * 2, dropout)
        self.down2 = Down3D(base_filters * 2, base_filters * 4, dropout)
        self.down3 = Down3D(base_filters * 4, base_filters * 8, dropout)
        self.down4 = Down3D(base_filters * 8, base_filters * 16 // factor, dropout)
        self.up1 = Up3D(base_filters * 16, base_filters * 8 // factor, trilinear, dropout)
        self.up2 = Up3D(base_filters * 8, base_filters * 4 // factor, trilinear, dropout)
        self.up3 = Up3D(base_filters * 4, base_filters * 2 // factor, trilinear, dropout)
        self.up4 = Up3D(base_filters * 2, base_filters, trilinear, dropout)
        self.outc = OutConv3D(base_filters, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return out

    def get_features(self, x):
        features = {}
        x1 = self.inc(x)
        features['level1'] = x1
        x2 = self.down1(x1)
        features['level2'] = x2
        x3 = self.down2(x2)
        features['level3'] = x3
        x4 = self.down3(x3)
        features['level4'] = x4
        x5 = self.down4(x4)
        features['bottleneck'] = x5
        return features


class AdaptiveUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=32, trilinear=True, dropout=0.2):
        super(AdaptiveUNet3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters
        self.trilinear = trilinear
        factor = 2 if trilinear else 1
        self.inc = DoubleConv3D(in_channels, base_filters, dropout)
        self.down1 = Down3D(base_filters, base_filters * 2, dropout)
        self.down2 = Down3D(base_filters * 2, base_filters * 4, dropout)
        self.down3 = Down3D(base_filters * 4, base_filters * 8, dropout)
        self.bottleneck_conv = DoubleConv3D(base_filters * 8, base_filters * 16 // factor, dropout)
        self.adaptive_pool = nn.AdaptiveAvgPool3d((8, 8, 8))
        self.up1 = Up3D(base_filters * 16 // factor, base_filters * 8 // factor, trilinear, dropout)
        self.up2 = Up3D(base_filters * 8 // factor, base_filters * 4 // factor, trilinear, dropout)
        self.up3 = Up3D(base_filters * 4 // factor, base_filters * 2 // factor, trilinear, dropout)
        self.up4 = Up3D(base_filters * 2 // factor, base_filters, trilinear, dropout)
        self.outc = OutConv3D(base_filters, out_channels)

    def forward(self, x):
        input_size = x.shape[2:]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck_conv(x4)
        x5_pooled = self.adaptive_pool(x5)
        x = self.up1(x5_pooled, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='trilinear', align_corners=False)
        return out

    def get_features(self, x):
        features = {}
        x1 = self.inc(x)
        features['enc1'] = x1
        x2 = self.down1(x1)
        features['enc2'] = x2
        x3 = self.down2(x2)
        features['enc3'] = x3
        x4 = self.down3(x3)
        features['enc4'] = x4
        x5 = self.bottleneck_conv(x4)
        x5_pooled = self.adaptive_pool(x5)
        features['bottleneck'] = x5_pooled
        x = self.up1(x5_pooled, x4)
        features['dec1'] = x
        x = self.up2(x, x3)
        features['dec2'] = x
        x = self.up3(x, x2)
        features['dec3'] = x
        x = self.up4(x, x1)
        features['dec4'] = x
        out = self.outc(x)
        features['output'] = out
        return features


__all__ = ['UNet3D', 'AdaptiveUNet3D']
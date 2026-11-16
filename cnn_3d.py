"""
3D CNN models for fault segmentation.
Includes both fixed-size and adaptive versions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3DBlock(nn.Module):
    """Basic 3D convolutional block: Conv3D + BatchNorm + ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(Conv3DBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class CNN3D(nn.Module):
    """
    Original 3D CNN with fixed architecture.
    Input size must be divisible by 8.
    """

    def __init__(self, in_channels=1, base_filters=32, num_classes=1):
        """
        Args:
            in_channels: Number of input channels (1 for seismic)
            base_filters: Base number of filters
            num_classes: Number of output classes (1 for binary segmentation)
        """
        super(CNN3D, self).__init__()

        # Encoder path
        self.enc1 = self._make_layer(in_channels, base_filters)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = self._make_layer(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = self._make_layer(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = self._make_layer(base_filters * 4, base_filters * 8)

        # Decoder
        self.up3 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = self._make_layer(base_filters * 8, base_filters * 4)

        self.up2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = self._make_layer(base_filters * 4, base_filters * 2)

        self.up1 = nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = self._make_layer(base_filters * 2, base_filters)

        # Final classification
        self.final_conv = nn.Conv3d(base_filters, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, in_channels, out_channels):
        """Create a convolutional layer block"""
        return nn.Sequential(
            Conv3DBlock(in_channels, out_channels),
            Conv3DBlock(out_channels, out_channels)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder with skip connections
        d3 = self.up3(b)
        d3 = self._match_size(d3, e3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._match_size(d2, e2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._match_size(d1, e1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Final output
        out = self.final_conv(d1)
        out = self.sigmoid(out)

        return out

    def _match_size(self, x, target):
        """Match the size of x to target using interpolation if needed"""
        if x.shape[2:] != target.shape[2:]:
            x = F.interpolate(x, size=target.shape[2:], mode='trilinear', align_corners=False)
        return x


class AdaptiveCNN3D(nn.Module):
    """
    Enhanced 3D CNN that adapts to any input size.
    Features adaptive pooling to handle variable dimensions.
    """

    def __init__(self, in_channels=1, base_filters=32, num_classes=1):
        """
        Args:
            in_channels: Number of input channels (1 for seismic)
            base_filters: Base number of filters
            num_classes: Number of output classes (1 for binary segmentation)
        """
        super(AdaptiveCNN3D, self).__init__()

        # Encoder path
        self.enc1 = self._make_layer(in_channels, base_filters)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = self._make_layer(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = self._make_layer(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = self._make_layer(base_filters * 4, base_filters * 8)

        # Adaptive pooling to handle variable sizes
        self.adaptive_pool = nn.AdaptiveAvgPool3d((8, 8, 8))

        # Decoder
        self.up3 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = self._make_layer(base_filters * 8, base_filters * 4)

        self.up2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = self._make_layer(base_filters * 4, base_filters * 2)

        self.up1 = nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = self._make_layer(base_filters * 2, base_filters)

        # Final convolution
        self.final_conv = nn.Conv3d(base_filters, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, in_channels, out_channels):
        """Create a convolutional layer block"""
        return nn.Sequential(
            Conv3DBlock(in_channels, out_channels),
            Conv3DBlock(out_channels, out_channels)
        )

    def forward(self, x):
        # Store input size for later resizing
        input_size = x.shape[2:]

        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck with adaptive pooling
        b = self.bottleneck(p3)
        b = self.adaptive_pool(b)

        # Decoder
        d3 = self.up3(b)
        d3 = self._match_size(d3, e3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._match_size(d2, e2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._match_size(d1, e1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Final output
        out = self.final_conv(d1)

        # Resize to original input size if needed
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='trilinear', align_corners=False)

        out = self.sigmoid(out)

        return out

    def _match_size(self, x, target):
        """Match the size of x to target using interpolation"""
        if x.shape[2:] != target.shape[2:]:
            x = F.interpolate(x, size=target.shape[2:], mode='trilinear', align_corners=False)
        return x


# Quick test
if __name__ == "__main__":
    print("Testing AdaptiveCNN3D with different input sizes:")
    model_adaptive = AdaptiveCNN3D(in_channels=1, base_filters=16)

    test_sizes = [
        (2, 1, 64, 64, 64),
        (2, 1, 128, 128, 128),
        (2, 1, 256, 256, 256),
        (2, 1, 50, 50, 50),
    ]

    for size in test_sizes:
        x = torch.randn(*size)
        y = model_adaptive(x)
        print(f"✅ Input {size} -> Output {tuple(y.shape)}")

    print("\n" + "=" * 50)
    print("Testing original CNN3D (fixed size):")
    model_fixed = CNN3D(in_channels=1, base_filters=16)

    # Test with compatible size
    x = torch.randn(2, 1, 64, 64, 64)
    y = model_fixed(x)
    print(f"✅ Input (2, 1, 64, 64, 64) -> Output {tuple(y.shape)}")

    # Test parameter counts
    print(f"\nParameter counts:")
    print(f"  CNN3D: {sum(p.numel() for p in model_fixed.parameters()):,}")
    print(f"  AdaptiveCNN3D: {sum(p.numel() for p in model_adaptive.parameters()):,}")
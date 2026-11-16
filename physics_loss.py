import torch
import torch.nn as nn
import torch.nn.functional as F


class Laplacian3D(nn.Module):
    def __init__(self):
        super(Laplacian3D, self).__init__()
        kernel = torch.zeros(3, 3, 3)
        kernel[1, 1, 0] = 1.0
        kernel[1, 1, 2] = 1.0
        kernel[1, 0, 1] = 1.0
        kernel[1, 2, 1] = 1.0
        kernel[0, 1, 1] = 1.0
        kernel[2, 1, 1] = 1.0
        kernel[1, 1, 1] = -6.0
        self.register_buffer('laplacian_kernel', kernel.unsqueeze(0).unsqueeze(0).float())

    def _get_kernel(self, device):
        if self.laplacian_kernel.device != device:
            return self.laplacian_kernel.to(device)
        return self.laplacian_kernel

    def forward(self, x):
        kernel = self._get_kernel(x.device)
        return F.conv3d(x, kernel, padding=1)


class AllenCahnPhysicsLoss(nn.Module):
    """Standard Allen-Cahn physics loss (fallback when innovation 2 disabled)"""

    def __init__(self, epsilon=0.05, boundary_weight=0.0):
        super(AllenCahnPhysicsLoss, self).__init__()
        self.epsilon = epsilon
        self.laplacian = Laplacian3D()

    def forward(self, pred, target=None, seismic=None):
        pred = torch.clamp(pred, min=0.0, max=1.0)
        laplacian = self.laplacian(pred)
        double_well = pred - pred ** 3
        residual = self.epsilon ** 2 * laplacian - double_well
        return (residual ** 2).mean(), {'pde_loss': (residual ** 2).mean().item()}


class MultiScaleAllenCahnLoss(nn.Module):
    """Innovation 2: Multi-Scale Physical Constraint (Multi-Scale Allen-Cahn)"""

    def __init__(self, epsilon_base=0.05, scales=[0.5, 1.0, 2.0], scale_weights=[0.3, 1.0, 0.3]):
        super(MultiScaleAllenCahnLoss, self).__init__()
        self.epsilon_base = epsilon_base
        self.scales = scales
        self.scale_weights = scale_weights
        self.laplacian = Laplacian3D()

    def gaussian_smooth(self, x, sigma):
        if sigma <= 0:
            return x
        kernel_size = int(2 * round(3 * sigma) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_1d = torch.arange(kernel_size, dtype=torch.float32, device=x.device)
        kernel_1d = kernel_1d - (kernel_size - 1) / 2
        kernel_1d = torch.exp(-kernel_1d ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_3d = kernel_1d[:, None, None] * kernel_1d[None, :, None] * kernel_1d[None, None, :]
        kernel_3d = kernel_3d.reshape(1, 1, kernel_size, kernel_size, kernel_size)
        return F.conv3d(x, kernel_3d, padding=kernel_size // 2)

    def compute_pde_residual(self, u, epsilon):
        laplacian_u = self.laplacian(u)
        double_well = u - u ** 3
        residual = epsilon ** 2 * laplacian_u - double_well
        return torch.clamp(residual, min=-2, max=2)

    def forward(self, pred, seismic=None, target=None):
        pred = torch.clamp(pred, min=0.0, max=1.0)
        total_loss = 0.0
        for scale, weight in zip(self.scales, self.scale_weights):
            epsilon = self.epsilon_base * scale
            if scale != 1.0:
                pred_scaled = self.gaussian_smooth(pred, sigma=abs(scale - 1.0) * 0.3)
            else:
                pred_scaled = pred
            residual = self.compute_pde_residual(pred_scaled, epsilon)
            total_loss = total_loss + weight * (residual ** 2).mean()
        return total_loss, {'multiscale_physics': total_loss.item()}
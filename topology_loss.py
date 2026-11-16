import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import gudhi as gd
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False


class PersistentHomologyTopologyLoss(nn.Module):
    """Standard persistent homology topology loss (fallback)"""

    def __init__(self, target_betti_numbers=[1, 0, 0], threshold=0.5, weights=[1.0, 1.0, 1.0]):
        super(PersistentHomologyTopologyLoss, self).__init__()
        self.target_betti_numbers = target_betti_numbers
        self.threshold = threshold
        self.weights = weights
        self.gudhi_available = GUDHI_AVAILABLE

    def compute_betti_numbers_simple(self, binary_volume):
        from scipy import ndimage
        labeled, num_components = ndimage.label(binary_volume)
        beta_0 = num_components
        cubes = np.sum(binary_volume)
        faces = 0
        faces += np.sum(binary_volume[1:, :, :] & binary_volume[:-1, :, :])
        faces += np.sum(binary_volume[:, 1:, :] & binary_volume[:, :-1, :])
        faces += np.sum(binary_volume[:, :, 1:] & binary_volume[:, :, :-1])
        edges = 0
        edges += np.sum(binary_volume[1:, 1:, :] & binary_volume[:-1, :-1, :])
        edges += np.sum(binary_volume[1:, :, 1:] & binary_volume[:-1, :, :-1])
        edges += np.sum(binary_volume[:, 1:, 1:] & binary_volume[:, :-1, :-1])
        vertices = np.sum(binary_volume[::2, ::2, ::2])
        euler = vertices - edges + faces - cubes
        beta_1 = max(0, beta_0 - euler)
        beta_2 = max(0, euler - beta_0 + beta_1)
        return [beta_0, beta_1, beta_2]

    def compute_topology_loss(self, pred_binary):
        B = pred_binary.shape[0]
        total_loss = 0.0
        for b in range(B):
            vol = pred_binary[b, 0].detach().cpu().numpy().astype(np.uint8)
            betti = self.compute_betti_numbers_simple(vol)
            loss_b = 0.0
            for i in range(3):
                diff = abs(betti[i] - self.target_betti_numbers[i])
                loss_b += self.weights[i] * diff ** 2
            total_loss += loss_b
        return total_loss / B

    def differentiable_approximation(self, pred):
        binary_like = torch.sigmoid(10 * (pred - self.threshold))
        pool = nn.AvgPool3d(kernel_size=3, stride=1, padding=1)
        local_density = pool(binary_like)
        isolation_penalty = binary_like * (1 - local_density)
        background = 1 - binary_like
        background_density = pool(background)
        hole_penalty = background * background_density
        return (isolation_penalty.mean() + hole_penalty.mean()) * 0.5

    def forward(self, pred, target=None):
        approx_loss = self.differentiable_approximation(pred)
        exact_loss = approx_loss.item()
        return approx_loss, {'topology_approx': approx_loss.item(), 'topology_exact': exact_loss}


class RobustTopologyLoss(nn.Module):
    """Innovation 3: Robust Topology Loss (soft threshold + multi-scale + confidence)"""

    def __init__(self, target_betti_numbers=[1, 0, 0], use_multiscale=True,
                 use_soft_threshold=True, noise_robust_sigma=0.1):
        super(RobustTopologyLoss, self).__init__()
        self.target_betti_numbers = target_betti_numbers
        self.use_multiscale = use_multiscale
        self.use_soft_threshold = use_soft_threshold
        self.noise_robust_sigma = noise_robust_sigma

    def gaussian_smooth(self, x, sigma=None):
        if sigma is None:
            sigma = self.noise_robust_sigma
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
        kernel_3d = kernel_3d.to(dtype=x.dtype)
        return F.conv3d(x, kernel_3d, padding=kernel_size // 2)

    def soft_threshold(self, x, temperature=0.5):
        if not self.use_soft_threshold:
            return (x > 0.5).float()
        return torch.sigmoid(torch.clamp((x - 0.5) / temperature, min=-5, max=5))

    def compute_local_connectivity(self, binary, kernel_size=3):
        kernel = torch.ones(
            1, 1, kernel_size, kernel_size, kernel_size,
            device=binary.device,
            dtype=binary.dtype
        )
        kernel = kernel / (kernel_size ** 3)
        return F.conv3d(binary, kernel, padding=kernel_size // 2)

    def compute_persistence_approximation(self, pred):
        binary_like = self.soft_threshold(pred)
        binary_smooth = self.gaussian_smooth(binary_like, sigma=0.3)
        local_conn = self.compute_local_connectivity(binary_smooth)
        isolation_penalty = binary_smooth * (1 - local_conn)
        background = 1 - binary_smooth
        background_conn = self.compute_local_connectivity(background)
        hole_penalty = background * background_conn

        if self.use_multiscale:
            binary_large = self.gaussian_smooth(binary_like, sigma=0.8)
            local_conn_large = self.compute_local_connectivity(binary_large, kernel_size=5)
            isolation_penalty_large = binary_large * (1 - local_conn_large)
            isolation_penalty = 0.7 * isolation_penalty + 0.3 * isolation_penalty_large

        confidence = 1 - torch.abs(pred - 0.5) * 2
        isolation_penalty = isolation_penalty * confidence
        hole_penalty = hole_penalty * confidence

        return (isolation_penalty.mean() + hole_penalty.mean()) * 0.05

    def forward(self, pred, target=None):
        approx_loss = self.compute_persistence_approximation(pred)
        return approx_loss, {'topology_approx': approx_loss.item()}

    

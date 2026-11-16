import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveNoiseConstraint(nn.Module):

    def __init__(self, snr_ref=30.0, min_weight=0.5, max_weight=2.0):
        super(AdaptiveNoiseConstraint, self).__init__()
        self.snr_ref = snr_ref
        self.min_weight = min_weight
        self.max_weight = max_weight

    def estimate_local_snr(self, seismic, window_size=5):
        kernel = torch.ones(
            1, 1, window_size, window_size, window_size,
            device=seismic.device,
            dtype=seismic.dtype
        )
        kernel = kernel / (window_size ** 3)
        local_mean = F.conv3d(seismic, kernel, padding=window_size // 2)
        local_var = F.conv3d(seismic ** 2, kernel, padding=window_size // 2) - local_mean ** 2
        local_var = torch.clamp(local_var, min=1e-4)
        snr_local = 10 * torch.log10(local_mean ** 2 / (local_var + 1e-4))
        normalized_snr = torch.clamp((snr_local - self.snr_ref) / 10.0, min=-5, max=5)
        weight_map = self.min_weight + (self.max_weight - self.min_weight) * torch.sigmoid(-normalized_snr)
        return snr_local, weight_map

    def forward(self, seismic, physics_loss):
        with torch.no_grad():
            _, weight_map = self.estimate_local_snr(seismic)
        return (weight_map * physics_loss).mean(), weight_map


class AdaptiveMultiScalePhysicsLoss(nn.Module):

    def __init__(self, epsilon=0.05, snr_ref=30.0, adaptive_weight_range=(0.5, 2.0),
                 scales=[0.5, 1.0, 2.0], scale_weights=[0.3, 1.0, 0.3]):
        super(AdaptiveMultiScalePhysicsLoss, self).__init__()
        self.adaptive_constraint = AdaptiveNoiseConstraint(snr_ref, adaptive_weight_range[0], adaptive_weight_range[1])
        from models.losses.physics_loss import MultiScaleAllenCahnLoss
        self.multiscale_loss = MultiScaleAllenCahnLoss(epsilon, scales, scale_weights)

    def forward(self, pred, seismic=None, target=None):
        multiscale_loss, _ = self.multiscale_loss(pred)
        if seismic is not None:
            weighted_loss, weight_map = self.adaptive_constraint(seismic, multiscale_loss)
            return weighted_loss, {'physics_loss': weighted_loss.item(), 'weight_mean': weight_map.mean().item()}
        else:
            return multiscale_loss, {'physics_loss': multiscale_loss.item()}

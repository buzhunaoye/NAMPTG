import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):

    def __init__(self, reduction='mean'):
        super(BCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)
        bce_loss = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))

        if self.reduction == 'mean':
            return bce_loss.mean()
        elif self.reduction == 'sum':
            return bce_loss.sum()
        else:
            return bce_loss


class BalancedBCELoss(nn.Module):

    def __init__(self, reduction='mean'):
        super(BalancedBCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)

        count_pos = target.sum()
        count_neg = (1 - target).sum()
        total = count_pos + count_neg

        beta = count_neg / (total + 1e-7)
        pos_weight = beta / (1 - beta + 1e-7)

        bce_loss = -(target * torch.log(pred) * pos_weight +
                     (1 - target) * torch.log(1 - pred))

        if self.reduction == 'mean':
            return bce_loss.mean()
        elif self.reduction == 'sum':
            return bce_loss.sum()
        else:
            return bce_loss


class FocalLoss(nn.Module):

    def __init__(self, gamma=2.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)

        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma

        bce = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        focal_loss = focal_weight * bce

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice_coef = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_coef

        return dice_loss.mean()


class TverskyLoss(nn.Module):
    """Tversky loss. A larger alpha penalizes false positives more strongly."""

    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-7):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        tp = (pred_flat * target_flat).sum(dim=1)
        fp = (pred_flat * (1 - target_flat)).sum(dim=1)
        fn = ((1 - pred_flat) * target_flat).sum(dim=1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1 - tversky).mean()


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss for imbalanced segmentation."""

    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-7):
        super(FocalTverskyLoss, self).__init__()
        self.tversky_loss = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
        self.gamma = gamma

    def forward(self, pred, target):
        loss = self.tversky_loss(pred, target)
        return torch.pow(loss, self.gamma)


class BoundaryDiceLoss(nn.Module):
    """Dice loss on soft 3D boundaries to improve thin fault localization."""

    def __init__(self, width=1, smooth=1e-7):
        super(BoundaryDiceLoss, self).__init__()
        self.width = max(1, int(width))
        self.smooth = smooth

    def _boundary(self, x):
        kernel_size = 2 * self.width + 1
        dilation = F.max_pool3d(x, kernel_size=kernel_size, stride=1, padding=self.width)
        erosion = -F.max_pool3d(-x, kernel_size=kernel_size, stride=1, padding=self.width)
        return torch.clamp(dilation - erosion, min=0.0, max=1.0)

    def forward(self, pred, target):
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)
        pred_boundary = self._boundary(pred)
        target_boundary = self._boundary(target)

        pred_flat = pred_boundary.view(pred.size(0), -1)
        target_flat = target_boundary.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return (1 - dice).mean()


class HardVoxelBCELoss(nn.Module):
    """BCE on the hardest voxels in each batch for imbalance-aware fine tuning."""

    def __init__(self, fraction=0.10):
        super(HardVoxelBCELoss, self).__init__()
        self.fraction = min(1.0, max(1e-4, float(fraction)))

    def forward(self, pred, target):
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)
        voxel_loss = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        flat = voxel_loss.view(voxel_loss.size(0), -1)
        k = max(1, int(flat.size(1) * self.fraction))
        hard_loss = torch.topk(flat, k=k, dim=1, largest=True, sorted=False).values
        return hard_loss.mean()


class CompositeDataLoss(nn.Module):

    def __init__(self, bce_weight=1.0, focal_weight=1.0, dice_weight=1.0,
                 focal_gamma=2.5, use_balanced_bce=True,
                 extra_loss_type='none', extra_loss_weight=0.0,
                 tversky_alpha=0.7, tversky_beta=0.3,
                 focal_tversky_gamma=0.75,
                 boundary_loss_weight=0.0, boundary_width=1,
                 hard_mining_weight=0.0, hard_mining_fraction=0.10):
        super(CompositeDataLoss, self).__init__()

        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.extra_loss_type = extra_loss_type
        self.extra_loss_weight = extra_loss_weight
        self.boundary_loss_weight = boundary_loss_weight
        self.hard_mining_weight = hard_mining_weight

        if use_balanced_bce:
            self.bce_loss = BalancedBCELoss()
        else:
            self.bce_loss = BCELoss()

        self.focal_loss = FocalLoss(gamma=focal_gamma)
        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryDiceLoss(width=boundary_width)
        self.hard_mining_loss = HardVoxelBCELoss(fraction=hard_mining_fraction)
        if extra_loss_type == 'tversky':
            self.extra_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        elif extra_loss_type == 'focal_tversky':
            self.extra_loss = FocalTverskyLoss(
                alpha=tversky_alpha,
                beta=tversky_beta,
                gamma=focal_tversky_gamma
            )
        else:
            self.extra_loss = None

    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)

        total = (self.bce_weight * bce +
                 self.focal_weight * focal +
                 self.dice_weight * dice)

        log_dict = {'bce': bce.item(), 'focal': focal.item(), 'dice': dice.item()}

        if self.extra_loss is not None and self.extra_loss_weight > 0:
            extra = self.extra_loss(pred, target)
            total = total + self.extra_loss_weight * extra
            log_dict[self.extra_loss_type] = extra.item()

        if self.boundary_loss_weight > 0:
            boundary = self.boundary_loss(pred, target)
            total = total + self.boundary_loss_weight * boundary
            log_dict['boundary_dice'] = boundary.item()

        if self.hard_mining_weight > 0:
            hard = self.hard_mining_loss(pred, target)
            total = total + self.hard_mining_weight * hard
            log_dict['hard_voxel_bce'] = hard.item()

        return total, log_dict

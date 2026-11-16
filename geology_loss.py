
import torch
import torch.nn as nn
import torch.nn.functional as F


class TotalVariationLoss(nn.Module):


    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, pred):

        diff_x = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1])
        diff_y = torch.abs(pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :])
        diff_z = torch.abs(pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :])

        tv = diff_x.mean() + diff_y.mean() + diff_z.mean()

        return tv


class AngleSmoothnessLoss(nn.Module):

    def __init__(self, epsilon=1e-7):

        super(AngleSmoothnessLoss, self).__init__()
        self.epsilon = epsilon

    def compute_gradients(self, x):

        x_padded = F.pad(x, (1, 1, 1, 1, 1, 1), mode='replicate')

        # Central differences
        grad_x = (x_padded[:, :, 1:-1, 1:-1, 2:] - x_padded[:, :, 1:-1, 1:-1, :-2]) / 2
        grad_y = (x_padded[:, :, 1:-1, 2:, 1:-1] - x_padded[:, :, 1:-1, :-2, 1:-1]) / 2
        grad_z = (x_padded[:, :, 2:, 1:-1, 1:-1] - x_padded[:, :, :-2, 1:-1, 1:-1]) / 2

        return grad_x, grad_y, grad_z

    def compute_normals(self, grad_x, grad_y, grad_z):

        gradients = torch.stack([grad_x, grad_y, grad_z], dim=2)  # [B, 1, 3, D, H, W]
        gradients = gradients.squeeze(1)  # [B, 3, D, H, W]


        norm = torch.norm(gradients, dim=1, keepdim=True) + self.epsilon


        normals = gradients / norm

        return normals

    def forward(self, pred):

        grad_x, grad_y, grad_z = self.compute_gradients(pred)


        normals = self.compute_normals(grad_x, grad_y, grad_z)  # [B, 3, D, H, W]


        loss = 0.0
        count = 0

        # x-direction differences
        n1 = normals[:, :, :, :, :-1]
        n2 = normals[:, :, :, :, 1:]
        dot_product = (n1 * n2).sum(dim=1)
        loss += (1 - dot_product).mean()
        count += 1

        n1 = normals[:, :, :, :-1, :]
        n2 = normals[:, :, :, 1:, :]
        dot_product = (n1 * n2).sum(dim=1)
        loss += (1 - dot_product).mean()
        count += 1


        n1 = normals[:, :, :-1, :, :]
        n2 = normals[:, :, 1:, :, :]
        dot_product = (n1 * n2).sum(dim=1)
        loss += (1 - dot_product).mean()
        count += 1

        return loss / count


class ConnectivityLoss(nn.Module):

    def __init__(self, window_size=5, threshold=0.3):

        super(ConnectivityLoss, self).__init__()

        self.window_size = window_size
        self.threshold = threshold
        self.pool = nn.AvgPool3d(kernel_size=window_size, stride=1, padding=window_size // 2)

    def forward(self, pred):

        local_avg = self.pool(pred)


        penalty = F.relu(self.threshold - local_avg) * pred

        return penalty.mean()


class BranchPenaltyLoss(nn.Module):

    def __init__(self, branch_threshold=4, threshold=0.5):

        super(BranchPenaltyLoss, self).__init__()

        self.branch_threshold = branch_threshold
        self.threshold = threshold

        kernel = torch.zeros(3, 3, 3)
        kernel[1, 1, 0] = 1.0  # z-1
        kernel[1, 1, 2] = 1.0  # z+1
        kernel[1, 0, 1] = 1.0  # y-1
        kernel[1, 2, 1] = 1.0  # y+1
        kernel[0, 1, 1] = 1.0  # x-1
        kernel[2, 1, 1] = 1.0  # x+1

        self.register_buffer('neighbor_kernel', kernel.unsqueeze(0).unsqueeze(0).float())

    def forward(self, pred):

        with torch.no_grad():
            binary = (pred > self.threshold).float()

        kernel = self.neighbor_kernel.to(device=binary.device, dtype=binary.dtype)
        neighbor_count = F.conv3d(binary, kernel, padding=1)

        excess = F.relu(neighbor_count - self.branch_threshold)

        penalty = excess * pred

        return penalty.mean()


class CompositeGeologyLoss(nn.Module):

    def __init__(self, tv_weight=1.0, angle_weight=0.5, conn_weight=2.0, branch_weight=1.0,
                 window_size=5, conn_threshold=0.3, branch_threshold=4):

        super(CompositeGeologyLoss, self).__init__()

        self.tv_weight = tv_weight
        self.angle_weight = angle_weight
        self.conn_weight = conn_weight
        self.branch_weight = branch_weight

        self.tv_loss = TotalVariationLoss()
        self.angle_loss = AngleSmoothnessLoss()
        self.connectivity_loss = ConnectivityLoss(
            window_size=window_size,
            threshold=conn_threshold
        )
        self.branch_loss = BranchPenaltyLoss(branch_threshold=branch_threshold)

    def forward(self, pred):

        tv = self.tv_loss(pred)
        angle = self.angle_loss(pred)
        conn = self.connectivity_loss(pred)
        branch = self.branch_loss(pred)

        total = (self.tv_weight * tv +
                 self.angle_weight * angle +
                 self.conn_weight * conn +
                 self.branch_weight * branch)

        return total, {
            'tv': tv.item(),
            'angle': angle.item(),
            'connectivity': conn.item(),
            'branch': branch.item()
        }

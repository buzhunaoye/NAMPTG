import torch.nn as nn
from models.losses.data_loss import CompositeDataLoss
from models.losses.geology_loss import CompositeGeologyLoss
from models.losses.adaptive_physics_loss import AdaptiveMultiScalePhysicsLoss
from models.losses.topology_loss import RobustTopologyLoss
from models.losses.physics_loss import AllenCahnPhysicsLoss


class PTGLoss(nn.Module):
    """PTG Loss with Innovations 1, 2, 3"""

    def __init__(self, config):
        super(PTGLoss, self).__init__()
        data_weights = getattr(config, 'DATA_LOSS_WEIGHTS', [0.3, 0.5, 1.0])
        focal_gamma = getattr(config, 'FOCAL_GAMMA', 2.0)

        self.data_loss = CompositeDataLoss(
            bce_weight=data_weights[0],
            focal_weight=data_weights[1],
            dice_weight=data_weights[2],
            focal_gamma=focal_gamma,
            extra_loss_type=getattr(config, 'DATA_EXTRA_LOSS_TYPE', 'none'),
            extra_loss_weight=getattr(config, 'DATA_EXTRA_LOSS_WEIGHT', 0.0),
            tversky_alpha=getattr(config, 'TVERSKY_ALPHA', 0.7),
            tversky_beta=getattr(config, 'TVERSKY_BETA', 0.3),
            focal_tversky_gamma=getattr(config, 'FOCAL_TVERSKY_GAMMA', 0.75),
            boundary_loss_weight=getattr(config, 'BOUNDARY_LOSS_WEIGHT', 0.0),
            boundary_width=getattr(config, 'BOUNDARY_WIDTH', 1),
            hard_mining_weight=getattr(config, 'HARD_MINING_WEIGHT', 0.0),
            hard_mining_fraction=getattr(config, 'HARD_MINING_FRACTION', 0.10)
        )

        physics_type = getattr(config, 'PHYSICS_LOSS_TYPE', 'adaptive_multiscale')
        if physics_type == 'adaptive_multiscale':
            self.physics_loss = AdaptiveMultiScalePhysicsLoss(
                epsilon=getattr(config, 'ALLEN_CAHN_EPSILON', 0.05),
                snr_ref=getattr(config, 'SNR_REF', 30.0),
                adaptive_weight_range=getattr(config, 'ADAPTIVE_WEIGHT_RANGE', (0.5, 2.0)),
                scales=getattr(config, 'MULTISCALE_SCALES', [0.5, 1.0, 2.0]),
                scale_weights=getattr(config, 'MULTISCALE_WEIGHTS', [0.3, 1.0, 0.3])
            )
        else:
            self.physics_loss = AllenCahnPhysicsLoss(epsilon=getattr(config, 'ALLEN_CAHN_EPSILON', 0.05))

        topology_type = getattr(config, 'TOPOLOGY_LOSS_TYPE', 'robust')
        if topology_type == 'robust':
            self.topology_loss = RobustTopologyLoss(
                target_betti_numbers=getattr(config, 'TARGET_BETTI_NUMBERS', [1, 0, 0]),
                use_multiscale=getattr(config, 'TOPO_USE_MULTISCALE', True),
                use_soft_threshold=getattr(config, 'TOPO_USE_SOFT_THRESHOLD', True),
                noise_robust_sigma=getattr(config, 'TOPO_NOISE_ROUBUST_SIGMA', 0.1)
            )
        else:
            from models.losses.topology_loss import PersistentHomologyTopologyLoss
            self.topology_loss = PersistentHomologyTopologyLoss()

        self.geology_loss = CompositeGeologyLoss()

        self.lambda_data = 1.0
        self.lambda_phys = getattr(config, 'PHYSICS_LOSS_WEIGHT', 0.30)
        self.lambda_topo = getattr(config, 'TOPOLOGY_LOSS_WEIGHT', 0.20)
        self.lambda_geo = getattr(config, 'GEOLOGY_LOSS_WEIGHT', 0.20)

    def forward(self, pred, target, seismic=None):
        data_loss, data_dict = self.data_loss(pred, target)
        phys_loss, phys_dict = self.physics_loss(pred, seismic=seismic)
        topo_loss, topo_dict = self.topology_loss(pred, target)
        geo_loss, geo_dict = self.geology_loss(pred)

        total_loss = (self.lambda_data * data_loss +
                      self.lambda_phys * phys_loss +
                      self.lambda_topo * topo_loss +
                      self.lambda_geo * geo_loss)

        log_dict = {
            'total': total_loss.item(),
            'data': data_loss.item(),
            'physics': phys_loss.item(),
            'topology': topo_loss.item(),
            'geology': geo_loss.item(),
            **{f'data_{k}': v for k, v in data_dict.items()},
            **{f'phys_{k}': v for k, v in phys_dict.items()},
            **{f'topo_{k}': v for k, v in topo_dict.items()},
            **{f'geo_{k}': v for k, v in geo_dict.items()}
        }
        return total_loss, log_dict

import os
import sys

os.environ['TORCH_DYNAMO_DISABLE'] = '1'
os.environ['PYTORCH_JIT'] = '0'

current_file = os.path.abspath(__file__)
project_root = os.path.dirname(current_file)
sys.path.insert(0, project_root)

import argparse
import json
import random

import numpy as np
import torch

from config.config import PTGConfig
from data.dataset import create_data_loaders
from training.trainer import Trainer


DEFAULT_DATA_DIR = os.path.abspath(os.path.join(project_root, '..', '毕设数据', 'archive'))
DEFAULT_CHECKPOINT_DIR = os.path.join(project_root, 'checkpoints')
DEFAULT_LOG_DIR = os.path.join(project_root, 'logs')


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def build_model(args):
    if args.model == 'TeU':
        base_filters = args.base_filters or 32
        if args.adaptive:
            from models.backbones.teu import AdaptiveTeU
            return AdaptiveTeU(
                in_channels=1,
                out_channels=1,
                base_filters=base_filters,
                transformer_embed_dim=args.embed_dim,
                transformer_num_heads=args.num_heads,
                transformer_depth=args.transformer_depth
            )
        from models.backbones.teu import TeU
        return TeU(
            in_channels=1,
            out_channels=1,
            base_filters=base_filters,
            transformer_embed_dim=args.embed_dim,
            transformer_num_heads=args.num_heads,
            transformer_depth=args.transformer_depth
        )

    if args.model == 'ViT3D':
        if args.adaptive:
            from models.backbones.vit_3d import AdaptiveViT3D
            return AdaptiveViT3D(
                in_channels=1,
                out_channels=1,
                embed_dim=args.embed_dim,
                depth=args.transformer_depth,
                num_heads=args.num_heads,
                grid_size=args.vit_grid_size
            )
        from models.backbones.vit_3d import ViT3D
        return ViT3D(
            volume_size=(128, 128, 128),
            patch_size=(16, 16, 16),
            in_channels=1,
            out_channels=1,
            embed_dim=args.embed_dim,
            depth=args.transformer_depth,
            num_heads=args.num_heads
        )

    if args.model == 'UNet3D':
        base_filters = args.base_filters or 16
        if args.adaptive:
            from models.backbones.unet_3d import AdaptiveUNet3D
            return AdaptiveUNet3D(in_channels=1, out_channels=1, base_filters=base_filters)
        from models.backbones.unet_3d import UNet3D
        return UNet3D(in_channels=1, out_channels=1, base_filters=base_filters)

    if args.model == 'ResUNet3D':
        base_filters = args.base_filters or 16
        if args.adaptive:
            from models.backbones.resunet_3d import AdaptiveResUNet3D
            return AdaptiveResUNet3D(in_channels=1, out_channels=1, base_filters=base_filters)
        from models.backbones.resunet_3d import ResUNet3D
        return ResUNet3D(in_channels=1, out_channels=1, base_filters=base_filters)

    if args.model == 'CNN3D':
        base_filters = args.base_filters or 32
        if args.adaptive:
            from models.backbones.cnn_3d import AdaptiveCNN3D
            return AdaptiveCNN3D(in_channels=1, base_filters=base_filters, num_classes=1)
        from models.backbones.cnn_3d import CNN3D
        return CNN3D(in_channels=1, base_filters=base_filters, num_classes=1)

    raise ValueError(f"Unknown model type: {args.model}")


def make_arg_parser():
    parser = argparse.ArgumentParser(description='Train PTG-based fault segmentation model')

    parser.add_argument('--model', type=str, default='TeU',
                        choices=['CNN3D', 'UNet3D', 'ResUNet3D', 'ViT3D', 'TeU'],
                        help='Model architecture to train')
    parser.add_argument('--adaptive', action='store_true', default=True,
                        help='Use adaptive version of the model')
    parser.add_argument('--non_adaptive', action='store_false', dest='adaptive',
                        help='Use the non-adaptive model variant')

    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                        help='Root directory containing train/ and validation/')
    parser.add_argument('--checkpoint_dir', type=str, default=DEFAULT_CHECKPOINT_DIR,
                        help='Directory for saved checkpoints')
    parser.add_argument('--log_dir', type=str, default=DEFAULT_LOG_DIR,
                        help='Directory for logs and JSON summaries')

    parser.add_argument('--apply_denoising', action='store_true', default=False,
                        help='Apply denoising preprocessing')
    parser.add_argument('--no_augment', action='store_true', default=False,
                        help='Disable data augmentation')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader worker processes')
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='Limit training samples for smoke tests')
    parser.add_argument('--max_val_samples', type=int, default=None,
                        help='Limit validation samples for smoke tests')

    parser.add_argument('--run_name', type=str, default='kaggle_teu_innovations',
                        help='Name for this training run')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use, such as cuda, cuda:0, or cpu')
    parser.add_argument('--require_cuda', action='store_true',
                        help='Exit immediately if CUDA is unavailable')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use deterministic cuDNN settings; slower but more reproducible')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from a saved checkpoint path')
    parser.add_argument('--resume_model_only', action='store_true',
                        help='Load only model weights from --resume and start a fresh optimizer/history')
    parser.add_argument('--resume_non_strict', action='store_true',
                        help='Allow missing/unexpected model keys when loading --resume')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save numbered checkpoints every N epochs; 0 disables them')
    parser.add_argument('--early_stopping_patience', type=int, default=None,
                        help='Override early stopping patience')
    parser.add_argument('--ema', action='store_true',
                        help='Validate and checkpoint with exponential moving average model weights')
    parser.add_argument('--ema_decay', type=float, default=0.995,
                        help='EMA decay used when --ema is enabled')

    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument('--amp', action='store_true', dest='amp', default=None,
                           help='Enable CUDA automatic mixed precision')
    amp_group.add_argument('--no_amp', action='store_false', dest='amp',
                           help='Disable automatic mixed precision')

    parser.add_argument('--base_filters', type=int, default=None,
                        help='Base convolution filters for CNN3D/UNet3D/TeU')
    parser.add_argument('--embed_dim', type=int, default=128,
                        help='Transformer embedding dimension for TeU/ViT3D')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Transformer attention heads for TeU/ViT3D')
    parser.add_argument('--transformer_depth', type=int, default=2,
                        help='Transformer depth for TeU/ViT3D')
    parser.add_argument('--vit_grid_size', type=int, default=4,
                        help='Adaptive ViT token grid size; 4 means 4x4x4 tokens, 8 keeps more spatial detail')

    parser.add_argument('--lambda_phys', type=float, default=0.30,
                        help='Weight for physics-informed loss')
    parser.add_argument('--lambda_topo', type=float, default=0.20,
                        help='Weight for topology-informed loss')
    parser.add_argument('--lambda_geo', type=float, default=0.20,
                        help='Weight for geology-informed loss')
    parser.add_argument('--disable_seismic_physics', action='store_true',
                        help='Do not pass seismic amplitudes into the adaptive physics term')

    parser.add_argument('--data_bce_weight', type=float, default=None,
                        help='Override BCE term weight inside data loss')
    parser.add_argument('--data_focal_weight', type=float, default=None,
                        help='Override focal term weight inside data loss')
    parser.add_argument('--data_dice_weight', type=float, default=None,
                        help='Override Dice term weight inside data loss')

    parser.add_argument('--data_extra_loss_type', type=str, default='none',
                        choices=['none', 'tversky', 'focal_tversky'],
                        help='Optional extra data loss for F1-oriented experiments')
    parser.add_argument('--data_extra_loss_weight', type=float, default=0.0,
                        help='Weight of the optional extra data loss')
    parser.add_argument('--tversky_alpha', type=float, default=0.7,
                        help='False-positive penalty in Tversky loss')
    parser.add_argument('--tversky_beta', type=float, default=0.3,
                        help='False-negative penalty in Tversky loss')
    parser.add_argument('--focal_tversky_gamma', type=float, default=0.75,
                        help='Gamma in focal Tversky loss')
    parser.add_argument('--boundary_loss_weight', type=float, default=0.0,
                        help='Weight for 3D boundary Dice loss')
    parser.add_argument('--boundary_width', type=int, default=1,
                        help='Boundary width in voxels for boundary Dice loss')
    parser.add_argument('--hard_mining_weight', type=float, default=0.0,
                        help='Weight for hard-voxel BCE on the most difficult voxels')
    parser.add_argument('--hard_mining_fraction', type=float, default=0.10,
                        help='Fraction of voxels used in hard-voxel BCE')

    parser.add_argument('--train_crop_size', type=int, default=None,
                        help='Optional cubic crop size for fault-centered crop training')
    parser.add_argument('--samples_per_volume', type=int, default=1,
                        help='Training crop samples drawn per volume each epoch')
    parser.add_argument('--fault_crop_prob', type=float, default=0.0,
                        help='Probability of centering a training crop on a fault voxel')
    parser.add_argument('--min_positive_voxels', type=int, default=1,
                        help='Minimum fault voxels requested in a positive crop')

    parser.add_argument('--eval_threshold_min', type=float, default=None,
                        help='Minimum threshold for validation F1 search')
    parser.add_argument('--eval_threshold_max', type=float, default=None,
                        help='Maximum threshold for validation F1 search')
    parser.add_argument('--eval_threshold_step', type=float, default=None,
                        help='Step size for validation F1 threshold search')
    parser.add_argument('--eval_postprocess_modes', nargs='+', default=None,
                        help='Validation postprocess modes, e.g. none open1 close1_open1')

    return parser


def save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    parser = make_arg_parser()
    args = parser.parse_args()

    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this run, but no CUDA device is available.")

    set_random_seed(args.seed, deterministic=args.deterministic)

    config = PTGConfig()
    if args.device:
        config.DEVICE = args.device
    config.BATCH_SIZE = args.batch_size
    config.NUM_EPOCHS = args.epochs
    config.LEARNING_RATE = args.lr
    config.PHYSICS_LOSS_WEIGHT = args.lambda_phys
    config.TOPOLOGY_LOSS_WEIGHT = args.lambda_topo
    config.GEOLOGY_LOSS_WEIGHT = args.lambda_geo
    config.DATA_EXTRA_LOSS_TYPE = args.data_extra_loss_type
    config.DATA_EXTRA_LOSS_WEIGHT = args.data_extra_loss_weight
    config.TVERSKY_ALPHA = args.tversky_alpha
    config.TVERSKY_BETA = args.tversky_beta
    config.FOCAL_TVERSKY_GAMMA = args.focal_tversky_gamma
    config.BOUNDARY_LOSS_WEIGHT = args.boundary_loss_weight
    config.BOUNDARY_WIDTH = args.boundary_width
    config.HARD_MINING_WEIGHT = args.hard_mining_weight
    config.HARD_MINING_FRACTION = args.hard_mining_fraction
    if args.data_bce_weight is not None or args.data_focal_weight is not None or args.data_dice_weight is not None:
        weights = list(getattr(config, 'DATA_LOSS_WEIGHTS', [1.0, 1.0, 1.5]))
        if args.data_bce_weight is not None:
            weights[0] = args.data_bce_weight
        if args.data_focal_weight is not None:
            weights[1] = args.data_focal_weight
        if args.data_dice_weight is not None:
            weights[2] = args.data_dice_weight
        config.DATA_LOSS_WEIGHTS = weights
    config.TRAIN_CROP_SIZE = args.train_crop_size
    config.SAMPLES_PER_VOLUME = args.samples_per_volume
    config.FAULT_CROP_PROB = args.fault_crop_prob
    config.MIN_POSITIVE_VOXELS = args.min_positive_voxels
    config.NUM_WORKERS = args.num_workers
    config.CHECKPOINT_DIR = args.checkpoint_dir
    config.LOG_DIR = args.log_dir
    config.CHECKPOINT_SAVE_INTERVAL = args.save_every
    config.PASS_SEISMIC_TO_PHYSICS = not args.disable_seismic_physics
    config.USE_EMA = args.ema
    config.EMA_DECAY = args.ema_decay
    if args.eval_postprocess_modes is not None:
        config.EVAL_POSTPROCESS_MODES = args.eval_postprocess_modes
    if args.early_stopping_patience is not None:
        config.EARLY_STOPPING_PATIENCE = args.early_stopping_patience

    requested_device = torch.device(config.DEVICE)
    if args.amp is None:
        config.USE_AMP = requested_device.type == 'cuda' and torch.cuda.is_available()
    else:
        config.USE_AMP = bool(args.amp)

    if args.eval_threshold_min is not None or args.eval_threshold_max is not None or args.eval_threshold_step is not None:
        t_min = 0.05 if args.eval_threshold_min is None else args.eval_threshold_min
        t_max = 0.95 if args.eval_threshold_max is None else args.eval_threshold_max
        t_step = 0.01 if args.eval_threshold_step is None else args.eval_threshold_step
        config.EVAL_THRESHOLDS = [round(float(t), 4) for t in np.arange(t_min, t_max + 0.5 * t_step, t_step)]

    print("\n" + "=" * 70)
    print("PTG-BASED FAULT SEGMENTATION TRAINING")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Adaptive version: {args.adaptive}")
    print(f"  Device: {config.DEVICE}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"  AMP: {config.USE_AMP}")
    print(f"  Random seed: {args.seed}")
    print(f"  Run name: {args.run_name}")
    print(f"\nData Configuration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Apply denoising: {args.apply_denoising}")
    print(f"  Data augmentation: {not args.no_augment}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Num workers: {config.NUM_WORKERS}")
    print(f"  Max train/val samples: {args.max_train_samples}/{args.max_val_samples}")
    print(f"  Train crop size: {config.TRAIN_CROP_SIZE}")
    print(f"  Samples per volume: {config.SAMPLES_PER_VOLUME}")
    print(f"  Fault crop probability: {config.FAULT_CROP_PROB}")
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {config.NUM_EPOCHS}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Weight decay: {config.WEIGHT_DECAY}")
    print(f"  Gradient clipping: {config.GRADIENT_CLIP}")
    print(f"  Checkpoint save interval: {config.CHECKPOINT_SAVE_INTERVAL}")
    print(f"\nPTG Loss Weights:")
    print(f"  Physics loss: {config.PHYSICS_LOSS_WEIGHT}")
    print(f"  Topology loss: {config.TOPOLOGY_LOSS_WEIGHT}")
    print(f"  Geology loss: {config.GEOLOGY_LOSS_WEIGHT}")
    print(f"  Pass seismic to physics loss: {config.PASS_SEISMIC_TO_PHYSICS}")
    print(f"\nF1-Boost Settings:")
    print(f"  Extra data loss: {config.DATA_EXTRA_LOSS_TYPE}")
    print(f"  Extra loss weight: {config.DATA_EXTRA_LOSS_WEIGHT}")
    print(f"  Tversky alpha/beta: {config.TVERSKY_ALPHA}/{config.TVERSKY_BETA}")
    print(f"  Boundary loss weight: {config.BOUNDARY_LOSS_WEIGHT}")
    print(f"  Hard mining weight/fraction: {config.HARD_MINING_WEIGHT}/{config.HARD_MINING_FRACTION}")
    print(f"  Data loss weights: {config.DATA_LOSS_WEIGHTS}")
    print(f"  Threshold candidates: {len(getattr(config, 'EVAL_THRESHOLDS', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))}")
    print(f"  Eval postprocess modes: {getattr(config, 'EVAL_POSTPROCESS_MODES', ['none'])}")
    print(f"  EMA: {config.USE_EMA} (decay={config.EMA_DECAY})")

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "validation")

    if not os.path.exists(train_dir):
        raise SystemExit(f"Training directory not found: {train_dir}")
    if not os.path.exists(val_dir):
        raise SystemExit(f"Validation directory not found: {val_dir}")

    print("\nCreating data loaders...")
    train_loader, val_loader = create_data_loaders(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=config.BATCH_SIZE,
        apply_denoising=args.apply_denoising,
        augment_train=not args.no_augment,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY and requested_device.type == 'cuda',
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        train_crop_size=config.TRAIN_CROP_SIZE,
        samples_per_volume=config.SAMPLES_PER_VOLUME,
        fault_crop_prob=config.FAULT_CROP_PROB,
        min_positive_voxels=config.MIN_POSITIVE_VOXELS
    )
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")

    print(f"\nCreating {args.model} model...")
    model = build_model(args)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model created successfully")
    print(f"  Total parameters: {total_params:,}")

    print("\nInitializing trainer...")
    trainer = Trainer(model, config, run_name=args.run_name)
    print(f"  Trainer initialized")
    print(f"  Checkpoint directory: {trainer.checkpoint_dir}")
    print(f"  Log directory: {trainer.log_dir}")

    experiment_config = vars(args).copy()
    experiment_config['resolved_device'] = str(trainer.device)
    experiment_config['cuda_available'] = torch.cuda.is_available()
    experiment_config['amp_enabled'] = trainer.use_amp
    experiment_config['total_parameters'] = total_params
    save_json(os.path.join(trainer.log_dir, 'experiment_config.json'), experiment_config)

    if args.resume:
        trainer.load_checkpoint(
            args.resume,
            model_only=args.resume_model_only,
            strict=not args.resume_non_strict
        )

    print("\nStarting training...")
    print("=" * 70)

    try:
        trainer.fit(train_loader, val_loader)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        raise SystemExit(130)
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        raise

    print("\nFinal evaluation on validation set...")
    val_loss, val_metrics, best_threshold, best_postprocess = trainer.validate(val_loader)
    print(f"\nFinal Validation Results:")
    print(f"  Loss: {val_loss:.6f}")
    print(f"  Best threshold: {best_threshold:.3f}")
    print(f"  Best postprocess: {best_postprocess}")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall: {val_metrics['recall']:.4f}")
    print(f"  F1 Score: {val_metrics['f1']:.4f}")

    final_summary = {
        'run_name': args.run_name,
        'final_val_loss': float(val_loss),
        'final_metrics': {k: float(v) for k, v in val_metrics.items()},
        'final_best_threshold': float(best_threshold),
        'final_best_postprocess': best_postprocess,
        'best_val_f1': float(trainer.best_val_f1),
        'best_threshold': float(trainer.best_threshold),
        'best_postprocess': trainer.best_postprocess,
        'checkpoint_dir': trainer.checkpoint_dir,
        'log_dir': trainer.log_dir
    }
    save_json(os.path.join(trainer.log_dir, 'final_metrics.json'), final_summary)

    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print(f"Checkpoints saved in: {trainer.checkpoint_dir}")
    print(f"Logs saved in: {trainer.log_dir}")
    print(f"Best validation F1: {trainer.best_val_f1:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()

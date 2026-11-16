# NAMPTG

**Local-Statistics-Guided Multiscale Regularization for 3D Seismic Fault Segmentation**

NAMPTG is a PyTorch framework for three-dimensional seismic fault segmentation. It combines a composite data-fitting objective with multiscale Allen-Cahn regularization, a differentiable topology-inspired surrogate, and geological shape constraints.

This repository provides selected core training and evaluation code associated with the manuscript. The implementation supports CNN, U-Net, ResUNet, and TeU backbones.

## Repository Contents

```text
NAMPTG/
├── README.md
├── requirements_training_gpu.txt
├── .gitignore
├── run_training_TeU.py
├── evaluate_checkpoint_f1_thresholds.py
├── config/
├── models/
│   ├── backbones/
│   └── losses/
├── data/
│   ├── dataset.py
│   └── preprocessing/
└── training/
```

- **Backbones:** CNN, U-Net, ResUNet, and a convolution-Transformer hybrid model, TeU.
- **Data-fitting losses:** balanced binary cross-entropy, focal loss, Dice loss, focal Tversky loss, boundary loss, and hard-voxel mining.
- **Physical regularization:** multiscale Allen-Cahn residuals and weights derived from local seismic statistics.
- **Topology-inspired regularization:** differentiable local isolation and hole penalties.
- **Geological constraints:** total variation, orientation smoothness, connectivity, and branching penalties.
- **Training utilities:** fault-centered patch sampling, exponential moving average (EMA), checkpoint saving, and validation threshold search.

The auxiliary ViT implementation is retained for compatibility with the backbone package imports. Raw datasets, trained checkpoints, and the field-data candidate-ranking workflow are not included in the selected repository contents.

## Installation

Install a PyTorch build compatible with your Python version and CPU/CUDA environment:

[PyTorch installation instructions](https://pytorch.org/get-started/locally/)

Then install the remaining dependencies using the existing dependency file:

```bash
python -m pip install -r requirements_training_gpu.txt
```

The core training dependencies are:

```text
torch
numpy
scipy
scikit-learn
PyWavelets
tqdm
```

The supplied dependency file also includes packages used by visualization and supporting utilities. GPU execution is recommended for three-dimensional training.

## Data Preparation

The training code expects the following directory structure:

```text
archive/
├── train/
│   ├── 0.npz
│   ├── 1.npz
│   └── ...
└── validation/
    ├── 0.npz
    ├── 1.npz
    └── ...
```

Each `.npz` file must contain two aligned three-dimensional arrays:

| Key | Description |
| --- | --- |
| `seis` | Seismic amplitude volume |
| `fault` | Corresponding binary fault mask |

Both arrays must have the same shape and axis order. Fault voxels are represented by 1 and background voxels by 0.

The manuscript describes 200 training volumes and 20 validation volumes, each containing 128 x 128 x 128 voxels. Training patches are sampled from these volumes.

Synthetic data and their associated labels are available through the [FaultSeg3D repository](https://github.com/xinwucwp/faultSeg). The upstream data must be converted into the `.npz` format expected by this implementation; automatic downloading and conversion are not provided.

Preprocessing depends on both the stored input arrays and the selected loader options. Preserve the original data preparation procedure when comparing experimental results.

## Training

Run commands from the repository root. The following example specifies a TeU training configuration:

```bash
python run_training_TeU.py \
  --model TeU \
  --adaptive \
  --data_dir /path/to/archive \
  --run_name namptg_teu \
  --device cuda \
  --require_cuda \
  --epochs 80 \
  --batch_size 1 \
  --num_workers 0 \
  --base_filters 32 \
  --embed_dim 128 \
  --num_heads 4 \
  --transformer_depth 2 \
  --lr 6e-6 \
  --lambda_phys 0.03 \
  --lambda_topo 0.03 \
  --lambda_geo 0.03 \
  --data_bce_weight 0.30 \
  --data_focal_weight 1.00 \
  --data_dice_weight 2.50 \
  --data_extra_loss_type focal_tversky \
  --data_extra_loss_weight 0.70 \
  --tversky_alpha 0.70 \
  --tversky_beta 0.30 \
  --focal_tversky_gamma 0.75 \
  --boundary_loss_weight 0.25 \
  --hard_mining_weight 0.15 \
  --hard_mining_fraction 0.08 \
  --train_crop_size 96 \
  --samples_per_volume 4 \
  --fault_crop_prob 0.90 \
  --eval_threshold_min 0.40 \
  --eval_threshold_max 0.95 \
  --eval_threshold_step 0.05 \
  --eval_postprocess_modes none \
  --ema \
  --ema_decay 0.997 \
  --no_amp \
  --no_augment
```

This example disables online augmentation and optional denoising. It is an example for executing the released code, rather than a guarantee of reproducing the manuscript results.

The line-continuation syntax above is for Bash. In Windows PowerShell, place the command on one line or use PowerShell line continuations.

Supported backbone settings are:

| Backbone | `--model` | Base filters |
| --- | --- | ---: |
| CNN | `CNN3D` | 32 |
| U-Net | `UNet3D` | 16 |
| ResUNet | `ResUNet3D` | 16 |
| TeU | `TeU` | 32 |

To initialize a run from existing model weights, add:

```text
--resume /path/to/checkpoint.pth --resume_model_only
```

The checkpoint architecture must match the selected model. Initialization, preprocessing, early stopping, and validation settings must be matched when comparing runs.

For all available options:

```bash
python run_training_TeU.py --help
```

## Evaluation

The standalone evaluator supports the TeU architecture and a legacy TeU variant. To evaluate a TeU checkpoint:

```bash
python evaluate_checkpoint_f1_thresholds.py \
  --model TeU \
  --checkpoint checkpoints/namptg_teu/best_model.pth \
  --data_dir /path/to/archive \
  --device cuda \
  --threshold_min 0.40 \
  --threshold_max 0.95 \
  --threshold_step 0.05
```

The evaluator accumulates voxel-level confusion counts across the validation data and reports accuracy, precision, recall, and F1 score. Binary morphology postprocessing is not applied by this standalone evaluator.

A threshold selected using validation data should be reported as validation-based selection rather than an independent test evaluation.

## Configuration and Outputs

Model and loss defaults are defined in `config/config.py`. Command-line arguments override the settings exposed by the training entry point.

In the supplied source, the physical reference parameter `SNR_REF` is set to 30.0 dB. The adaptive physical module applies the mean local weight to a scalar multiscale residual loss. These implementation details should be checked against the intended mathematical formulation before interpreting experimental results.

Training outputs are stored under:

```text
checkpoints/<run_name>/
logs/<run_name>/
```

They include model checkpoints, training histories, experiment configurations, and validation metrics.

## Data Attribution

Please acknowledge the original data providers and follow their usage terms.

- **Synthetic seismic data:** [FaultSeg3D](https://github.com/xinwucwp/faultSeg).
- **Netherlands F3 survey:** [dGB Earth Sciences / TerraNubis](https://terranubis.com/datainfo/F3-Demo-2023).

For the FaultSeg3D dataset, please cite:

> Wu, X., Liang, L., Shi, Y., and Fomel, S., 2019. FaultSeg3D: Using synthetic data sets to train an end-to-end convolutional neural network for 3D seismic fault segmentation. *GEOPHYSICS*, 84(3), IM35-IM45.

## Associated Manuscript

*Local-Statistics-Guided Multiscale Regularization for 3D Seismic Fault Segmentation.*

Publication details and the article DOI will be added when available.

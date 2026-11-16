import torch

class PTGConfig:
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Training
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 100
    BATCH_SIZE = 1
    GRADIENT_CLIP = 0.5
    USE_SCHEDULER = False
    USE_AMP = False
    USE_EMA = False
    EMA_DECAY = 0.995
    NUM_WORKERS = 0
    PIN_MEMORY = True
    CHECKPOINT_SAVE_INTERVAL = 10
    PASS_SEISMIC_TO_PHYSICS = True

    # Early stopping
    EARLY_STOPPING_PATIENCE = 20
    EARLY_STOPPING_MIN_DELTA = 0.001

    # Loss type selection
    PHYSICS_LOSS_TYPE = 'adaptive_multiscale'  # 'adaptive_multiscale' or 'standard'
    TOPOLOGY_LOSS_TYPE = 'robust'  # 'robust' or 'standard'

    # Data loss
    DATA_LOSS_WEIGHTS = [1.0, 1.0, 1.5]
    FOCAL_GAMMA = 2.0
    DATA_EXTRA_LOSS_TYPE = 'none'  # 'none', 'tversky', or 'focal_tversky'
    DATA_EXTRA_LOSS_WEIGHT = 0.0
    TVERSKY_ALPHA = 0.7  # false-positive penalty
    TVERSKY_BETA = 0.3   # false-negative penalty
    FOCAL_TVERSKY_GAMMA = 0.75
    BOUNDARY_LOSS_WEIGHT = 0.0
    BOUNDARY_WIDTH = 1
    HARD_MINING_WEIGHT = 0.0
    HARD_MINING_FRACTION = 0.10

    # Optional fault-centered crop training for F1-oriented retraining
    TRAIN_CROP_SIZE = None
    SAMPLES_PER_VOLUME = 1
    FAULT_CROP_PROB = 0.0
    MIN_POSITIVE_VOXELS = 1

    # Validation-time F1 search. Postprocess modes are lightweight 3D morphology
    # options applied after thresholding; "none" keeps the raw binary prediction.
    EVAL_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    EVAL_POSTPROCESS_MODES = ['none']

    # PTG loss weights
    PHYSICS_LOSS_WEIGHT = 0.30
    TOPOLOGY_LOSS_WEIGHT = 0.20
    GEOLOGY_LOSS_WEIGHT = 0.20

    # Allen-Cahn parameters
    ALLEN_CAHN_EPSILON = 0.05

    # Innovation 1: Adaptive Noise Constraint
    SNR_REF = 30.0
    ADAPTIVE_WEIGHT_RANGE = (0.5, 2.0)

    # Innovation 2: Multi-Scale Physics Constraint
    MULTISCALE_SCALES = [0.5, 1.0, 2.0]
    MULTISCALE_WEIGHTS = [0.3, 1.0, 0.3]

    # Innovation 3: Robust Topology Loss
    TOPO_USE_MULTISCALE = True
    TOPO_USE_SOFT_THRESHOLD = True
    TOPO_NOISE_ROUBUST_SIGMA = 0.1

    # Target Betti numbers
    TARGET_BETTI_NUMBERS = [1, 0, 0]

    # Geology loss weights
    TV_LOSS_WEIGHT = 1.0
    ANGLE_LOSS_WEIGHT = 0.5
    CONNECTIVITY_LOSS_WEIGHT = 2.0
    BRANCH_LOSS_WEIGHT = 1.0
    CONNECTIVITY_WINDOW_SIZE = 5
    CONNECTIVITY_THRESHOLD = 0.3
    BRANCH_THRESHOLD = 4

    # Paths
    CHECKPOINT_DIR = './checkpoints'
    LOG_DIR = './logs'

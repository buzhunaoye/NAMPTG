import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
import random


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _parse_crop_size(crop_size):
    if crop_size is None:
        return None
    if isinstance(crop_size, int):
        return (crop_size, crop_size, crop_size)
    if isinstance(crop_size, (list, tuple)):
        if len(crop_size) == 1:
            size = int(crop_size[0])
            return (size, size, size)
        if len(crop_size) == 3:
            return tuple(int(v) for v in crop_size)
    raise ValueError(f"Invalid crop_size: {crop_size}")


class FaultDataset(Dataset):
    def __init__(self, data_dir, apply_denoising=False, normalize_method='zscore',
                 augment=False, clip_percentile=99.5, max_samples=None,
                 crop_size=None, samples_per_volume=1, fault_crop_prob=0.0,
                 min_positive_voxels=1):
        self.data_dir = data_dir
        self.apply_denoising = apply_denoising
        self.normalize_method = normalize_method
        self.augment = augment
        self.clip_percentile = clip_percentile
        self.crop_size = _parse_crop_size(crop_size)
        self.samples_per_volume = max(1, int(samples_per_volume))
        self.fault_crop_prob = float(fault_crop_prob)
        self.min_positive_voxels = int(min_positive_voxels)
        self.file_list = glob.glob(os.path.join(data_dir, "*.npz"))
        self.file_list.sort()
        if max_samples is not None:
            self.file_list = self.file_list[:max_samples]
        print(f"Found {len(self.file_list)} .npz files in {data_dir}")
        if self.apply_denoising or self.augment:
            from data.preprocessing.denoising import combined_denoising
            from data.preprocessing.normalization import normalize_volume
            from data.preprocessing.augmentation import AugmentationPipeline
            self.denoising_func = combined_denoising
            self.normalize_func = normalize_volume
            if self.augment:
                self.augmentor = AugmentationPipeline(rotation=True, flip=True, shift=False, noise_snr=None)
            else:
                self.augmentor = None
        else:
            self.denoising_func = None
            self.normalize_func = None
            self.augmentor = None

    def __len__(self):
        if self.crop_size is None:
            return len(self.file_list)
        return len(self.file_list) * self.samples_per_volume

    def _preprocess_seismic(self, seismic):
        if self.apply_denoising and self.denoising_func is not None:
            seismic = self.denoising_func(seismic)
        if self.clip_percentile is not None:
            p_low = (100 - self.clip_percentile) / 2
            p_high = 100 - p_low
            low = np.percentile(seismic, p_low)
            high = np.percentile(seismic, p_high)
            seismic = np.clip(seismic, low, high)
        if self.normalize_func is not None:
            seismic, _ = self.normalize_func(seismic, method=self.normalize_method)
        return seismic

    def _preprocess_label(self, label):
        label = (label > 0.5).astype(np.float32)
        return label

    def _random_start(self, shape):
        starts = []
        for dim, crop in zip(shape, self.crop_size):
            if dim <= crop:
                starts.append(0)
            else:
                starts.append(random.randint(0, dim - crop))
        return starts

    def _centered_start(self, center, shape):
        starts = []
        for c, dim, crop in zip(center, shape, self.crop_size):
            if dim <= crop:
                starts.append(0)
                continue
            jitter = random.randint(-max(1, crop // 4), max(1, crop // 4))
            start = int(c) - crop // 2 + jitter
            start = max(0, min(start, dim - crop))
            starts.append(start)
        return starts

    def _crop_pair_once(self, seismic, fault, prefer_fault):
        shape = seismic.shape
        use_fault_center = prefer_fault and fault.sum() >= self.min_positive_voxels
        if use_fault_center:
            coords = np.argwhere(fault > 0.5)
            center = coords[random.randrange(len(coords))]
            starts = self._centered_start(center, shape)
        else:
            starts = self._random_start(shape)

        slices = tuple(slice(s, min(s + c, dim)) for s, c, dim in zip(starts, self.crop_size, shape))
        seismic_crop = seismic[slices]
        fault_crop = fault[slices]
        return seismic_crop, fault_crop

    def _crop_pair(self, seismic, fault):
        if self.crop_size is None:
            return seismic, fault

        prefer_fault = random.random() < self.fault_crop_prob
        seismic_crop, fault_crop = self._crop_pair_once(seismic, fault, prefer_fault)

        retries = 8
        while fault_crop.sum() < self.min_positive_voxels and retries > 0:
            seismic_crop, fault_crop = self._crop_pair_once(seismic, fault, True)
            retries -= 1

        return seismic_crop, fault_crop

    def __getitem__(self, idx):
        if self.crop_size is None:
            file_idx = idx
        else:
            file_idx = idx % len(self.file_list)
        file_path = self.file_list[file_idx]
        data = np.load(file_path)
        seismic = data['seis'].astype(np.float32)
        fault = data['fault'].astype(np.float32)
        seismic = self._preprocess_seismic(seismic)
        fault = self._preprocess_label(fault)
        if self.augment and self.augmentor is not None:
            seismic, fault = self.augmentor(seismic, fault)
        seismic, fault = self._crop_pair(seismic, fault)
        seismic = seismic[np.newaxis, ...]
        fault = fault[np.newaxis, ...]
        seismic = torch.from_numpy(seismic).float()
        fault = torch.from_numpy(fault).float()
        return seismic, fault


def create_data_loaders(train_dir, val_dir, batch_size=1, apply_denoising=False,
                        augment_train=True, num_workers=0, pin_memory=None,
                        seed=None, max_train_samples=None, max_val_samples=None,
                        train_crop_size=None, samples_per_volume=1,
                        fault_crop_prob=0.0, min_positive_voxels=1):
    train_dataset = FaultDataset(
        train_dir,
        apply_denoising=apply_denoising,
        augment=augment_train,
        max_samples=max_train_samples,
        crop_size=train_crop_size,
        samples_per_volume=samples_per_volume,
        fault_crop_prob=fault_crop_prob,
        min_positive_voxels=min_positive_voxels
    )
    val_dataset = FaultDataset(
        val_dir,
        apply_denoising=False,
        augment=False,
        max_samples=max_val_samples
    )
    print(f"\nDataset Statistics:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    loader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'worker_init_fn': seed_worker if seed is not None else None,
        'generator': generator
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs
    )
    return train_loader, val_loader


class F3RealDataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.dat_files = glob.glob(os.path.join(data_dir, "*.dat"))
        self.dat_files.sort()
        print(f"Found {len(self.dat_files)} .dat files in {data_dir}")
        if len(self.dat_files) == 0:
            print(f"Warning: No .dat files found in {data_dir}")

    def load_seismic(self, file_idx=0):
        if not self.dat_files:
            print("  No .dat files found, returning dummy data for testing")
            return np.random.randn(128, 128, 128).astype(np.float32)
        file_path = self.dat_files[file_idx]
        print(f"  Loading seismic data from: {os.path.basename(file_path)}")
        try:
            import segyio
            with segyio.open(file_path, ignore_geometry=True) as f:
                data = segyio.tools.cube(f)
                print(f"  Loaded as SEGY with shape: {data.shape}")
                return data.astype(np.float32)
        except ImportError:
            print("  segyio not installed, trying binary format...")
        except Exception as e:
            print(f"  Not a valid SEGY file: {e}")
        try:
            data = np.fromfile(file_path, dtype=np.float32)
            possible_shapes = [(128, 128, 128), (256, 256, 256), (512, 512, 512), (64, 64, 64)]
            total_elements = len(data)
            loaded = False
            for shape in possible_shapes:
                expected = shape[0] * shape[1] * shape[2]
                if total_elements >= expected:
                    data = data[:expected].reshape(shape)
                    print(f"  Loaded as binary with shape: {shape}")
                    loaded = True
                    break
            if not loaded:
                cube_size = int(round(total_elements ** (1 / 3)))
                data = data[:cube_size ** 3].reshape(cube_size, cube_size, cube_size)
                print(f"  Loaded as binary with inferred shape: {data.shape}")
            return data.astype(np.float32)
        except Exception as e:
            print(f"  Error loading as binary: {e}")
            print("  Returning dummy data for testing")
            return np.random.randn(128, 128, 128).astype(np.float32)

    def load_sample_data(self):
        print("  Loading sample synthetic data for testing")
        x = np.linspace(-10, 10, 128)
        y = np.linspace(-10, 10, 128)
        z = np.linspace(-10, 10, 128)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        data = np.sin(0.5 * X) * np.cos(0.5 * Y) + 0.3 * Z
        data += 0.1 * np.random.randn(128, 128, 128)
        return data.astype(np.float32)

    def preprocess_for_model(self, seismic):
        seismic = seismic.astype(np.float32)
        mean = np.mean(seismic)
        std = np.std(seismic)
        if std > 0:
            seismic = (seismic - mean) / (std + 1e-7)
        else:
            seismic = seismic - mean
        seismic = seismic[np.newaxis, np.newaxis, ...]
        return torch.from_numpy(seismic).float()

    def get_file_list(self):
        return self.dat_files

    def get_file_name(self, idx):
        if 0 <= idx < len(self.dat_files):
            return os.path.basename(self.dat_files[idx])
        return None

    def __len__(self):
        return len(self.dat_files)

    def __getitem__(self, idx):
        seismic = self.load_seismic(idx)
        tensor = self.preprocess_for_model(seismic)
        return tensor, seismic.shape


if __name__ == "__main__":
    print("Testing dataset classes...")
    test_loader = F3RealDataLoader(".")
    seismic = test_loader.load_sample_data()
    tensor = test_loader.preprocess_for_model(seismic)
    print(f"Sample data shape: {seismic.shape}")
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor range: [{tensor.min():.4f}, {tensor.max():.4f}]")

import numpy as np
import random
from scipy.signal import butter, filtfilt


def random_rotation_3d(seismic, label):

    k = random.choice([1, 2, 3])
    if k > 0:
        seismic = np.rot90(seismic, k, axes=(1, 2))
        label = np.rot90(label, k, axes=(1, 2))
    return seismic, label


def fixed_rotation_90(seismic, label):
    seismic = np.rot90(seismic, 1, axes=(1, 2))
    label = np.rot90(label, 1, axes=(1, 2))
    return seismic, label


def fixed_rotation_180(seismic, label):
    seismic = np.rot90(seismic, 2, axes=(1, 2))
    label = np.rot90(label, 2, axes=(1, 2))
    return seismic, label


def fixed_rotation_270(seismic, label):
    seismic = np.rot90(seismic, 3, axes=(1, 2))
    label = np.rot90(label, 3, axes=(1, 2))
    return seismic, label


def random_flip_3d(seismic, label, axis_prob=0.5):
    if random.random() < axis_prob:
        seismic = np.flip(seismic, axis=0)
        label = np.flip(label, axis=0)
    if random.random() < axis_prob:
        seismic = np.flip(seismic, axis=1)
        label = np.flip(label, axis=1)
    if random.random() < axis_prob:
        seismic = np.flip(seismic, axis=2)
        label = np.flip(label, axis=2)
    seismic = seismic.copy()
    label = label.copy()
    return seismic, label


def fixed_flip_horizontal(seismic, label):
    seismic = np.flip(seismic, axis=1)
    label = np.flip(label, axis=1)
    return seismic, label


def fixed_flip_vertical(seismic, label):
    seismic = np.flip(seismic, axis=0)
    label = np.flip(label, axis=0)
    return seismic, label


def random_shift_3d(seismic, label, max_shift=5):
    from scipy.ndimage import shift
    shift_d = random.randint(-max_shift, max_shift)
    shift_h = random.randint(-max_shift, max_shift)
    shift_w = random.randint(-max_shift, max_shift)
    seismic = shift(seismic, (shift_d, shift_h, shift_w), mode='nearest')
    label = shift(label, (shift_d, shift_h, shift_w), mode='nearest')
    return seismic, label


def add_gaussian_noise(seismic, snr_db=30):
    signal_power = np.mean(seismic ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power) * np.random.randn(*seismic.shape)
    return seismic + noise


def add_bandlimited_noise(seismic, lowcut=2, highcut=80, fs=250, scale=0.1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(5, [low, high], btype='band')
    noise = np.random.normal(0, 1, seismic.shape)
    for i in range(seismic.shape[0]):
        for j in range(seismic.shape[1]):
            if seismic.ndim == 3:
                noise[i, j, :] = filtfilt(b, a, noise[i, j, :])
            else:
                noise[i, :] = filtfilt(b, a, noise[i, :])
    noise = noise / np.std(noise) * scale * np.std(seismic)
    return seismic + noise


def add_trace_wise_noise(seismic, num_noisy_traces=10, noisy_value=1.0):
    noisy_data = seismic.copy()
    for i in range(noisy_data.shape[0]):
        for j in range(noisy_data.shape[1]):
            trace_indices = np.random.choice(noisy_data.shape[2], num_noisy_traces, replace=False)
            noisy_data[i, j, trace_indices] = noisy_value
    return noisy_data


def enhance_contrast(data, percentile_low=2, percentile_high=98):
    p_low = np.percentile(data, percentile_low)
    p_high = np.percentile(data, percentile_high)
    data_clipped = np.clip(data, p_low, p_high)
    return (data_clipped - p_low) / (p_high - p_low + 1e-7)


class AugmentationPipeline:
    def __init__(self, rotation=True, flip=True, shift=True, noise_snr=None, bandlimited_noise=None):
        self.rotation = rotation
        self.flip = flip
        self.shift = shift
        self.noise_snr = noise_snr
        self.bandlimited_noise = bandlimited_noise

    def __call__(self, seismic, label):
        if self.rotation:
            seismic, label = random_rotation_3d(seismic, label)
        if self.flip:
            seismic, label = random_flip_3d(seismic, label)
        if self.shift:
            seismic, label = random_shift_3d(seismic, label)
        if self.noise_snr is not None:
            seismic = add_gaussian_noise(seismic, self.noise_snr)
        if self.bandlimited_noise is not None:
            seismic = add_bandlimited_noise(seismic, scale=self.bandlimited_noise)
        return seismic, label
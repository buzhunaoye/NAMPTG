import numpy as np
from scipy.ndimage import zoom


def resize_to_uniform(data, target_size=(128, 128, 128), method='crop_pad'):
    current_size = data.shape
    if current_size == target_size:
        return data
    if method == 'crop_pad':
        return _crop_pad_to_size(data, target_size)
    elif method == 'interpolate':
        return _interpolate_to_size(data, target_size)
    else:
        raise ValueError(f"Unknown resize method: {method}")


def _crop_pad_to_size(data, target_size):
    result = np.zeros(target_size, dtype=data.dtype)
    slices_in = []
    slices_out = []
    for dim, (current, target) in enumerate(zip(data.shape, target_size)):
        if current > target:
            start = (current - target) // 2
            end = start + target
            slices_in.append(slice(start, end))
            slices_out.append(slice(None))
        else:
            start = (target - current) // 2
            end = start + current
            slices_in.append(slice(None))
            slices_out.append(slice(start, end))
    result[tuple(slices_out)] = data[tuple(slices_in)]
    return result


def _interpolate_to_size(data, target_size):
    zoom_factors = [t / c for c, t in zip(data.shape, target_size)]
    return zoom(data, zoom_factors, order=1)


def crop_center(data, crop_size):
    start = [(s - c) // 2 for s, c in zip(data.shape, crop_size)]
    end = [s + c for s, c in zip(start, crop_size)]
    slices = tuple(slice(s, e) for s, e in zip(start, end))
    return data[slices]


def pad_to_size(data, target_size, mode='constant'):
    pad_width = []
    for current, target in zip(data.shape, target_size):
        diff = target - current
        if diff > 0:
            pad_before = diff // 2
            pad_after = diff - pad_before
            pad_width.append((pad_before, pad_after))
        else:
            pad_width.append((0, 0))
    return np.pad(data, pad_width, mode=mode)
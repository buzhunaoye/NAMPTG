import numpy as np


def zscore_normalize(data, eps=1e-7):
    print("Applying z-score normalization...")
    mean = np.mean(data)
    std = np.std(data)
    normalized = (data - mean) / (std + eps)
    params = {'mean': mean, 'std': std, 'eps': eps}
    print(f"  Mean: {mean:.6f} -> 0")
    print(f"  Std: {std:.6f} -> 1")
    return normalized, params


def minmax_normalize(data, min_val=None, max_val=None, target_range=(-1, 1)):
    print("Applying min-max normalization...")
    if min_val is None:
        min_val = np.min(data)
    if max_val is None:
        max_val = np.max(data)
    target_min, target_max = target_range
    normalized = (data - min_val) / (max_val - min_val + 1e-7)
    normalized = normalized * (target_max - target_min) + target_min
    params = {
        'min_val': min_val,
        'max_val': max_val,
        'target_min': target_min,
        'target_max': target_max
    }
    print(f"  Range: [{min_val:.6f}, {max_val:.6f}] -> [{target_min}, {target_max}]")
    return normalized, params


def inverse_zscore(normalized_data, params):
    return normalized_data * params['std'] + params['mean']


def inverse_minmax(normalized_data, params):
    normalized_01 = (normalized_data - params['target_min']) / (params['target_max'] - params['target_min'])
    return normalized_01 * (params['max_val'] - params['min_val']) + params['min_val']


def normalize_volume(seismic, method='zscore', clip_percentile=None):
    if clip_percentile is not None:
        p_low = (100 - clip_percentile) / 2
        p_high = 100 - p_low
        low = np.percentile(seismic, p_low)
        high = np.percentile(seismic, p_high)
        seismic = np.clip(seismic, low, high)
    if method == 'zscore':
        return zscore_normalize(seismic)
    elif method == 'minmax':
        return minmax_normalize(seismic)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
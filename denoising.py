import numpy as np
import pywt
from scipy.ndimage import gaussian_filter


def _safe_normalize(data):
    data = data.astype(np.float32)
    if data.size > 0:
        clip_max = np.percentile(data, 99.5)
        clip_min = np.percentile(data, 0.5)
        data = np.clip(data, clip_min, clip_max)
    return data


def wavelet_denoising(data, wavelet='db4', mode='soft', threshold_factor=0.3):
    print(f"  Applying wavelet denoising with {wavelet} wavelet...")
    data = _safe_normalize(data)
    denoised = np.zeros_like(data, dtype=np.float32)
    for d in range(data.shape[0]):
        slice_2d = data[d, :, :].astype(np.float64)
        try:
            coeffs = pywt.wavedec2(slice_2d, wavelet, level=3)
            detail_coeffs = coeffs[-1]
            detail_flat = np.concatenate([c.flatten() for c in detail_coeffs])
            sigma = np.median(np.abs(detail_flat)) / 0.6745
            threshold = threshold_factor * sigma
            coeffs_thresh = list(coeffs)
            for i in range(1, len(coeffs_thresh)):
                coeffs_thresh[i] = tuple(
                    pywt.threshold(c, threshold, mode=mode) for c in coeffs_thresh[i]
                )
            reconstructed = pywt.waverec2(coeffs_thresh, wavelet)
            h, w = slice_2d.shape
            denoised[d, :, :] = reconstructed[:h, :w].astype(np.float32)
        except Exception as e:
            print(f"    Wavelet denoising failed for slice {d}: {e}")
            denoised[d, :, :] = slice_2d.astype(np.float32)
    denoised = _safe_normalize(denoised)
    return denoised


def structure_oriented_filtering(data, sigma=1.0, alpha=0.3):
    print(f"  Applying structure-oriented filtering (sigma={sigma}, alpha={alpha})...")
    data = _safe_normalize(data)
    try:
        from scipy.ndimage import sobel, gaussian_filter
        grad_x = sobel(data, axis=2, mode='reflect')
        grad_y = sobel(data, axis=1, mode='reflect')
        grad_z = sobel(data, axis=0, mode='reflect')
        grad_x_smooth = gaussian_filter(grad_x, sigma=sigma)
        grad_y_smooth = gaussian_filter(grad_y, sigma=sigma)
        grad_z_smooth = gaussian_filter(grad_z, sigma=sigma)
        grad_norm = np.sqrt(grad_x_smooth ** 2 + grad_y_smooth ** 2 + grad_z_smooth ** 2)
        coherence = grad_norm / (np.max(grad_norm) + 1e-7)
        coherence = np.clip(coherence, 0, 1)
        filtered = gaussian_filter(data, sigma=sigma)
        result = filtered * coherence + data * (1 - coherence)
        result = result * alpha + data * (1 - alpha)
        result = _safe_normalize(result)
        return result
    except Exception as e:
        print(f"    Structure-oriented filtering failed: {e}")
        return data.copy()


def combined_denoising(data, wavelet='db4', sigma=1.0, alpha=0.3):
    print("=" * 50)
    print("Applying combined denoising:")
    print("=" * 50)
    data = _safe_normalize(data)
    denoised_wavelet = wavelet_denoising(data, wavelet=wavelet, threshold_factor=0.3)
    denoised_final = structure_oriented_filtering(denoised_wavelet, sigma=sigma, alpha=alpha)
    if np.isnan(denoised_final).any() or np.isinf(denoised_final).any():
        print("  Warning: NaN/Inf detected in denoised output, returning original data")
        return data.copy()
    print("Combined denoising completed")
    return denoised_final


def adaptive_median_filter(data, window_size=3):
    from scipy.ndimage import median_filter
    print(f"  Applying adaptive median filter (window={window_size})...")
    data = _safe_normalize(data)
    filtered = median_filter(data, size=window_size, mode='reflect')
    local_mean = gaussian_filter(data, sigma=1.0)
    local_var = gaussian_filter((data - local_mean) ** 2, sigma=1.0)
    weight = 1.0 - np.exp(-local_var / (np.mean(local_var) + 1e-7))
    weight = np.clip(weight, 0, 1)
    result = filtered * (1 - weight) + data * weight
    result = _safe_normalize(result)
    return result
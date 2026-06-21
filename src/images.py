import numpy as np
import cv2
from PIL import Image
from skimage.restoration import denoise_wavelet
from scipy.fft import fft

from coupling import (THRESHOLD, MINIMUM_POINTS,
                      compute_marginals, compute_gaps,
                      verdict, which_pairs_broke, surviving_pairs)
from binary_search import binary_search


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def _load_image(path: str) -> np.ndarray:
    """
    Open image, convert to float32 grayscale.
    Returns 2D array shape (H, W).
    """
    pil_img = Image.open(path)
    arr = np.array(pil_img).astype(np.float32)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return arr


# ---------------------------------------------------------------------------
# Extractions
# ---------------------------------------------------------------------------

def _extract_noise(img: np.ndarray, n_samples: int = 1000) -> np.ndarray:
    """
    Energy read: noise residual via wavelet denoising.
    Returns shape (n_samples, 1).
    """
    denoised = denoise_wavelet(img, rescale_sigma=True)
    noise_residual = img - denoised
    flat = noise_residual.flatten()
    samples = flat[:n_samples] if len(flat) >= n_samples else flat
    return samples.reshape(-1, 1)


def _extract_geometry(img: np.ndarray, n_samples: int = 500) -> np.ndarray:
    """
    Space read: deviation of detected edges from ideal straight lines.
    Returns shape (n_samples, 1).
    """
    uint8 = np.clip(img, 0, 255).astype(np.uint8)
    edges = cv2.Canny(uint8, 50, 150)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                             threshold=50, minLineLength=10, maxLineGap=5)

    deviations = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.hypot(x2 - x1, y2 - y1)
            if length == 0:
                continue
            # Direction vector of ideal line
            dx, dy = (x2 - x1) / length, (y2 - y1) / length
            # Sample points along the line
            n_pts = max(int(length), 2)
            xs = np.linspace(x1, x2, n_pts).astype(int)
            ys = np.linspace(y1, y2, n_pts).astype(int)
            # Clip to image bounds
            xs = np.clip(xs, 0, img.shape[1] - 1)
            ys = np.clip(ys, 0, img.shape[0] - 1)
            # Perpendicular deviation: distance from ideal line
            for i, (x, y) in enumerate(zip(xs, ys)):
                ideal_x = x1 + i * dx
                ideal_y = y1 + i * dy
                dev = np.sqrt((x - ideal_x) ** 2 + (y - ideal_y) ** 2)
                deviations.append(dev)

    if len(deviations) == 0:
        deviations = [0.0] * n_samples

    arr = np.array(deviations, dtype=np.float32)
    samples = arr[:n_samples] if len(arr) >= n_samples else np.pad(
        arr, (0, n_samples - len(arr)), constant_values=0.0)
    return samples.reshape(-1, 1)


def _extract_clock(img: np.ndarray,
                   noise_residual: np.ndarray,
                   n_samples: int = 500) -> np.ndarray:
    """
    Time read: row-level noise correlation from CMOS rolling shutter timing.
    FFT along axis=0 (column-wise across rows) gives frequency content
    of the row-by-row readout timing pattern.
    Returns shape (n_samples, 1).
    """
    freq = fft(noise_residual, axis=0)
    magnitude = np.abs(freq).flatten()
    # Skip DC component at index 0, take from middle frequency range
    mid_start = len(magnitude) // 4
    mid_end = mid_start + n_samples
    if mid_end <= len(magnitude):
        samples = magnitude[mid_start:mid_end]
    else:
        samples = magnitude[1:n_samples + 1]
    if len(samples) < n_samples:
        samples = np.pad(samples, (0, n_samples - len(samples)),
                         constant_values=0.0)
    return samples.astype(np.float32).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Region-level extraction for binary search
# ---------------------------------------------------------------------------

def _make_extract_fn(img: np.ndarray):
    """
    Returns a callable that extracts reads from a spatial region.
    Region format: (row_start, row_end, col_start, col_end)
    Returns None if region produces fewer than MINIMUM_POINTS.
    """
    def extract_fn(region):
        r0, r1, c0, c1 = region
        patch = img[r0:r1, c0:c1]
        n_pixels = (r1 - r0) * (c1 - c0)
        if n_pixels < MINIMUM_POINTS:
            return None
        noise = _extract_noise(patch, n_samples=min(1000, n_pixels))
        geom  = _extract_geometry(patch, n_samples=min(500, n_pixels))
        # recompute noise_residual for clock
        denoised = denoise_wavelet(patch, rescale_sigma=True)
        noise_residual = patch - denoised
        clock = _extract_clock(patch, noise_residual,
                               n_samples=min(500, n_pixels))
        if any(len(r) < 2 for r in [noise, geom, clock]):
            return None
        return noise, geom, clock
    return extract_fn


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyse_image(path: str) -> dict:
    """
    Run physical causality verification on an image file.

    Returns
    -------
    dict with keys:
        noise_geometry_gap, noise_clock_gap,
        geometry_clock_gap, joint_all_gap,
        verdict
        — and if manipulated —
        location, which_pair_broke, surviving_pairs, magnitude
    """
    img = _load_image(path)

    # Full-image extractions
    denoised = denoise_wavelet(img, rescale_sigma=True)
    noise_residual = img - denoised

    noise_read    = _extract_noise(img)
    geometry_read = _extract_geometry(img)
    clock_read    = _extract_clock(img, noise_residual)

    marginals = compute_marginals(noise_read, geometry_read, clock_read)
    gaps      = compute_gaps(noise_read, geometry_read, clock_read, marginals)
    v         = verdict(gaps)

    result = {**gaps, "verdict": v}

    if v == "manipulated":
        h, w = img.shape
        initial_region = (0, h, 0, w)
        extract_fn = _make_extract_fn(img)
        location_data = binary_search(extract_fn, initial_region)
        result["location"]         = location_data["location"]
        result["which_pair_broke"] = which_pairs_broke(gaps)
        result["surviving_pairs"]  = surviving_pairs(gaps)
        result["magnitude"]        = location_data["magnitude"]

    return result

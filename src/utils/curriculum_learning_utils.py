from typing import Any
from numpy import dtype, ndarray
from scipy.ndimage import gaussian_filter
import numpy as np
from scipy.ndimage import convolve1d


def apply_gaussian_smoothening(
    y: ndarray[Any, dtype[Any]], sigma: float
) -> ndarray[Any, dtype[Any]]:
    gaussian_filtered = gaussian_filter(y, sigma=sigma, axes=1)
    return gaussian_filtered


def apply_asymmetric_gaussian_smoothening(y: ndarray, sigma: float) -> ndarray:
    # Define a left-sided Gaussian kernel
    sigma = max(sigma, 0.01)  # Prevent division by zero
    kernel_size = int(3 * sigma)  # Approximate effective support
    x = np.arange(
        -kernel_size, 1
    )  # Only take the left side (including the center)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)  # Gaussian formula
    kernel /= kernel.sum()  # Normalize

    # Apply the filter along the time axis
    filtered = convolve1d(y, kernel, axis=-1, mode="nearest")

    return filtered

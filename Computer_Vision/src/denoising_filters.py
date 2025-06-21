import numpy as np
from scipy.ndimage import gaussian_filter1d

def detect_outliers(positions, threshold=2):
    """
    Detect outliers in ball positions based on standard deviation.
    :param positions: array-like of ball positions
    :param threshold: number of standard deviations to consider as outlier
    :return: indices of non-outlier positions
    """
    x_positions = positions[:, 0]
    y_positions = positions[:, 1]

    x_mean = np.mean(x_positions)
    y_mean = np.mean(y_positions)
    x_std = np.std(x_positions)
    y_std = np.std(y_positions)

    non_outlier_indices = np.where(
        (np.abs(x_positions - x_mean) <= threshold * x_std) &
        (np.abs(y_positions - y_mean) <= threshold * y_std)
    )[0]

    return non_outlier_indices


def apply_gaussian_filter(positions, sigma=1):
    """
    Apply Gaussian filter to smooth the ball positions.
    :param positions: array-like of ball positions
    :param sigma: standard deviation for Gaussian kernel
    :return: smoothed positions
    """
    smoothed_x = gaussian_filter1d(positions[:, 0], sigma=sigma)
    smoothed_y = gaussian_filter1d(positions[:, 1], sigma=sigma)
    return np.vstack((smoothed_x, smoothed_y)).T
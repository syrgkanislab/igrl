import numpy as np
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
import warnings
import multiprocessing


def get_reward_from_rgb(rgb, dgp):
    if dgp in ['dgp1', 'dgp2', 'dgp3']:
        return np.sum(rgb, axis=1)
    else:
        return rgb[:, 0]


def get_y_max(x, dgp):
    '''
    x is assumed to be a list of numpy arrays
    '''
    x = np.stack(x)
    pixels = x.reshape((x.shape[0], 3, -1))
    intensity = np.linalg.norm(pixels, axis=1)
    rgb = pixels[
        np.arange(x.shape[0]), :, np.argmax(intensity, axis=-1)
    ]
    y = get_reward_from_rgb(rgb, dgp)
    return y, rgb


def _get_single_rgb(xi):
    warnings.simplefilter('ignore')
    pixels = xi.reshape(3, -1).T
    n_colors = 2
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(pixels)
    dominant_colors = kmeans.cluster_centers_
    if np.sum(dominant_colors[1]) > np.sum(dominant_colors[0]):
        idx = 1
    else:
        idx = 0
    return dominant_colors[idx]


def get_y(x, dgp):
    '''
    x is assumed to be a list of numpy arrays
    '''
    # Reserve 5 cores for the system:
    n_jobs = max(1, multiprocessing.cpu_count() - 5)
    rgb = Parallel(n_jobs=n_jobs, verbose=3)(
        delayed(_get_single_rgb)(x[i]) for i in range(len(x))
    )
    rgb = np.array(rgb)
    y = get_reward_from_rgb(rgb, dgp)
    return y, rgb


def get_y_percentile(x, dgp, percentile=70):
    '''
    x is assumed to be a list of numpy arrays
    Returns the reward and rgb color of the pixel at the given percentile
    of intensity (default 70), excluding background (black) pixels.
    '''
    x = np.stack(x)
    pixels = x.reshape((x.shape[0], 3, -1))  # (batch, 3, num_pixels)
    intensity = np.linalg.norm(pixels, axis=1)  # (batch, num_pixels)
    # Mask for non-background pixels (at least one channel nonzero)
    mask = np.any(pixels != 0, axis=1)  # (batch, num_pixels)
    # Set background intensities to NaN
    intensity_bg_nan = np.where(mask, intensity, np.nan)
    # Compute percentile for each image, ignoring NaNs
    percentile_values = np.nanpercentile(
        intensity_bg_nan, percentile, axis=1
    )  # (batch,)
    # Compute absolute difference from percentile value, set background to inf
    abs_diff = np.abs(intensity - percentile_values[:, None])
    abs_diff[~mask] = np.inf
    # Find index of pixel closest to percentile value (among non-background)
    idx = np.argmin(abs_diff, axis=1)
    rgb = pixels[np.arange(x.shape[0]), :, idx]
    y = get_reward_from_rgb(rgb, dgp)
    return y, rgb

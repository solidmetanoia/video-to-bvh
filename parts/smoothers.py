from scipy.signal import butter, filtfilt
from scipy.signal import savgol_filter
import numpy as np

def moving_average_smoothing(keypoints, window_size=5):
    """
    Apply a moving average filter to smooth the keypoints.
    :param keypoints: Input keypoints of shape (num_frames, num_joints, 3).
    :param window_size: Size of the sliding window.
    :return: Smoothed keypoints.
    """
    num_frames, num_joints, _ = keypoints.shape
    smoothed_keypoints = np.zeros_like(keypoints)

    # Pad the keypoints to handle the edges
    pad_width = ((window_size // 2, window_size // 2), (0, 0), (0, 0))
    padded_keypoints = np.pad(keypoints, pad_width, mode='edge')

    # Apply moving average
    for i in range(num_frames):
        smoothed_keypoints[i] = np.mean(padded_keypoints[i:i + window_size], axis=0)

    return smoothed_keypoints

def savgol_smoothing(keypoints, window_size=5, polyorder=2):
    """
    Apply a Savitzky-Golay filter to smooth the keypoints.
    :param keypoints: Input keypoints of shape (num_frames, num_joints, 3).
    :param window_size: Size of the sliding window (must be odd).
    :param polyorder: Order of the polynomial to fit.
    :return: Smoothed keypoints.
    """
    num_frames, num_joints, _ = keypoints.shape
    smoothed_keypoints = np.zeros_like(keypoints)

    # Apply Savitzky-Golay filter to each joint and dimension
    for joint in range(num_joints):
        for dim in range(3):
            smoothed_keypoints[:, joint, dim] = savgol_filter(
                keypoints[:, joint, dim], window_size, polyorder
            )

    return smoothed_keypoints

def low_pass_filter(keypoints, cutoff=0.1, fs=30, order=5):
    """
    Apply a low-pass filter to smooth the keypoints.
    :param keypoints: Input keypoints of shape (num_frames, num_joints, 3).
    :param cutoff: Cutoff frequency (normalized to 0.5 * fs).
    :param fs: Sampling frequency (frames per second).
    :param order: Order of the filter.
    :return: Smoothed keypoints.
    """
    num_frames, num_joints, _ = keypoints.shape
    smoothed_keypoints = np.zeros_like(keypoints)

    # Design the filter
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # Apply the filter to each joint and dimension
    for joint in range(num_joints):
        for dim in range(3):
            smoothed_keypoints[:, joint, dim] = filtfilt(b, a, keypoints[:, joint, dim])

    return smoothed_keypoints

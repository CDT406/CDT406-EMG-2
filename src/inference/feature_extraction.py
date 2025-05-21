import numpy as np
from scipy.signal import butter, filtfilt



def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')

    return filtfilt(b, a, signal)


def compute_mav(window):
    return np.mean(np.abs(window))


def compute_wl(window):
    return np.sum(np.abs(np.diff(window)))


def compute_wamp(window, threshold=0.02):
    return np.sum(np.abs(np.diff(window)) > threshold)


def compute_mavs(window):
    half = len(window) // 2
    return np.abs(compute_mav(window[:half]) - compute_mav(window[half:]))


def extract_features(window, features = ['mav', 'wl', 'wamp', 'mavs'], wamp_threshold=0.02):
    extracted_features = []
    for feature in features:
        feature_lower_case = feature.lower()
        if feature_lower_case == 'mav':
            extracted_features.append(compute_mav(window))
        elif feature_lower_case == 'wl':
            extracted_features.append(compute_wl(window))
        elif feature_lower_case == 'wamp':
            extracted_features.append(compute_wamp(window, threshold=wamp_threshold))
        elif feature_lower_case == 'mavs':
            extracted_features.append(compute_mavs(window))
    return extracted_features

import numpy as np

def compute_mav(window):
    return np.mean(np.abs(window))

def compute_wl(window):
    return np.sum(np.abs(np.diff(window)))

def compute_wamp(window, threshold=0.02):
    return np.sum(np.abs(np.diff(window)) > threshold)

def compute_mavs(window):
    half = len(window) // 2
    return np.abs(compute_mav(window[:half]) - compute_mav(window[half:]))

def extract_features(window, wamp_threshold=0.02):
    return [
        compute_mav(window),
        compute_wl(window),
        compute_wamp(window, threshold=wamp_threshold),
        compute_mavs(window)
    ]

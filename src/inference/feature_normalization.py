import numpy as np

# These values are derived from the training data statistics
FEATURE_STATS = {
    'mav': {'mean': 0.5726, 'std': 0.7123},  # Estimated std based on range
    'wl': {'mean': 7.3511, 'std': 9.4432},   # Estimated std based on range
    'wamp': {'mean': 88.8112, 'std': 31.0},  # Estimated std based on range
    'mavs': {'mean': 0.2507, 'std': 0.8001}  # Estimated std based on range
}

def normalize_features(features):
    """
    Normalize features using training data statistics
    Args:
        features: numpy array of shape (4,) containing [mav, wl, wamp, mavs]
    Returns:
        normalized features array of same shape
    """
    feature_names = ['mav', 'wl', 'wamp', 'mavs']
    normalized = np.zeros_like(features)
    
    for i, name in enumerate(feature_names):
        stats = FEATURE_STATS[name]
        normalized[i] = (features[i] - stats['mean']) / (stats['std'] + 1e-8)
    
    return normalized 
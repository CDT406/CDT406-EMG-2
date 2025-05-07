import numpy as np
from collections import defaultdict

def create_sequences(samples, seq_len=1):
    if len(samples) < seq_len:
        return None, None
    features, labels = zip(*samples)
    X, y = [], []
    for i in range(len(features) - seq_len):
        X.append(features[i:i + seq_len])
        y.append(labels[i + seq_len - 1])
    return np.array(X), np.array(y)

def split_by_person(records, sequence_length, val_ids={1, 2}):
    """
    Splits data by person: persons in val_ids are used for validation, others for training.
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for rec in records:
        grouped[rec["person_id"]][rec["cycle_id"]].append(
            (np.array(rec["features"], dtype=np.float32), rec["label"])
        )

    X_train, y_train, X_val, y_val = [], [], [], []
    for person_id, cycles in grouped.items():
        target_X, target_y = (X_val, y_val) if person_id in val_ids else (X_train, y_train)
        for samples in cycles.values():
            X, y = create_sequences(samples, sequence_length)
            if X is not None:
                target_X.append(X)
                target_y.append(y)

    return X_train, y_train, X_val, y_val

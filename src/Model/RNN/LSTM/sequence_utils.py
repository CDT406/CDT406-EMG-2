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

def split_by_person(records, sequence_length):
    grouped = defaultdict(lambda: defaultdict(list))
    for rec in records:
        grouped[rec["person_id"]][rec["cycle_id"]].append((np.array(rec["features"], dtype=np.float32), rec["label"]))

    X_train, y_train, X_val, y_val = [], [], [], []
    for person_id, cycles in grouped.items():
        if person_id <= 7:
            for samples in cycles.values():
                Xt, yt = create_sequences(samples, sequence_length)
                if Xt is not None:
                    X_train.append(Xt)
                    y_train.append(yt)
        elif person_id >= 8:
            for samples in cycles.values():
                Xv, yv = create_sequences(samples, sequence_length)
                if Xv is not None:
                    X_val.append(Xv)
                    y_val.append(yv)

    return X_train, y_train, X_val, y_val

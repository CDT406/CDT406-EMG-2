from collections import Counter

def get_majority_label(labels):
    """
    4-Class: Keeps original label values from the signal.
    0 = Rest
    1 = Grip
    2 = Hold
    3 = Release
    Returns the most frequent label in the window.
    """
    return Counter(labels).most_common(1)[0][0]

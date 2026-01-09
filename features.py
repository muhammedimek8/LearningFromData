import numpy as np

def custom_features(texts):
    features = np.zeros((len(texts), 3))

    for i, text in enumerate(texts):
        features[i, 0] = len(text)                  # review length
        features[i, 1] = text.count("!")            # exclamation marks
        features[i, 2] = sum(1 for c in text if c.isupper()) / max(len(text), 1)

    return features

import numpy as np
from src.data.features import frame_to_vector
from src.data.labels import events_to_label

def build_sequence(frames, K: int):#, feature_cfg: dict, label_cfg: dict):
    X, Y = [], []
    for frame in frames:
        x_t = frame_to_vector(frame, K)#, feature_cfg)
        #print(x_t.shape)
        y_t = events_to_label(frame)#, label_cfg)
        X.append(x_t)
        Y.append(y_t)
    
    X = np.stack(X).astype(np.float32)     # (T, C) or (T, K, C)
    Y = np.asarray(Y, dtype=np.float32)    # (T,)
    return X, Y


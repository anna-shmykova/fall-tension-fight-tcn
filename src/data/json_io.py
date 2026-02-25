import json

def read_json_frames(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["frames"]
    
def json_to_sequence(path, K):

    features = []
    labels = []

    for frame in frames:
        x_t = frame_to_vector(frame, K)
        y_t = events_to_label(frame)

        features.append(x_t)
        labels.append(y_t)

    features = np.stack(features).astype(np.float32)  # (N, C)
    labels   = np.array(labels, dtype=np.float32)       # (N,)
    #labels_ramped = add_ramp_before_after_events(labels)
    return features, labels


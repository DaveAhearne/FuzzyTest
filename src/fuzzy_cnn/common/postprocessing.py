import json
import numpy as np
from fuzzy_cnn.common.config import LABELS_PATH

with open(LABELS_PATH, 'r') as f:
    LABEL_DATA = json.load(f)

def postprocess(logits, top_k=5):
    probabilities = softmax(logits)
    labelled_all = label_data(probabilities)
    return get_top_k_probs(labelled_all, top_k)

def label_data(probs):
    labelled_data = []
    for i, p in enumerate(probs):
        labelled_data.append({
            "label": LABEL_DATA[i],
            "prob": float(p)
        })
        
    return labelled_data 

def softmax(logits):
    # Shifting the whole thing down by the max, just prevents overflow
    # if we ever had something like a logit of 1000
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()

def get_top_k_probs(probs, top_k=5):
    return sorted(probs, key=lambda x: x['prob'], reverse=True)[:top_k]
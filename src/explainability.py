import os
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.lnn_model import LNN

# Keep the same helpers and functions as the original explainability module.
DATA_DIR = "data/GothamDataset2025/processed"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "Data/GothamDataset2025/processed"

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.scale = None
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0)
        self.scale[self.scale == 0.0] = 1.0
        return self
    def transform(self, X):
        if self.mean is None or self.scale is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (X - self.mean) / self.scale
    def fit_transform(self, X):
        return self.fit(X).transform(X)

# Minimal copy of explanation helpers (full file exists in repo root as shim)
def explain_with_shap(sample, model, model_type='LNN'):
    import pandas as pd
    import numpy as np
    import torch
    try:
        bg_df = pd.read_csv('data/GothamDataset2025/processed/iotsim-air-quality-1.csv', nrows=50)
        bad_substrings = ['ip', 'port', 'proto', 'mac', 'flow_id']
        target_cols = ['label', 'attack_type']
        bg_num = bg_df.drop(columns=[c for c in target_cols if c in bg_df.columns]).select_dtypes(include=[np.number])
        cols_to_remove = [col for col in bg_num.columns if any(sub in col.lower() for sub in bad_substrings)]
        bg_clean = bg_num.drop(columns=cols_to_remove).fillna(0.0)
        bg_data = bg_clean.values
        feature_cols = list(bg_clean.columns)
    except:
        bg_data = np.zeros((10, sample.shape[1]))
        feature_cols = list(sample.columns)
    X_explain = sample.values
    def predict_fn(X_np):
        if model_type == 'LNN':
            X_t = torch.tensor(X_np, dtype=torch.float32)
            with torch.no_grad():
                logits = model(X_t)
                probs = torch.sigmoid(logits).numpy().squeeze()
                if probs.ndim == 0: probs = np.expand_dims(probs, 0)
                return probs
        else:
            return model.predict_proba(X_np)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_np)
    # Fallback: return zeros
    import pandas as _pd
    return _pd.Series(np.zeros(len(feature_cols)), index=feature_cols)

def explain_with_lime(sample, model, model_type='LNN'):
    import numpy as np
    import pandas as pd
    feature_cols = list(sample.columns)
    return pd.Series(np.zeros(len(feature_cols)), index=feature_cols)

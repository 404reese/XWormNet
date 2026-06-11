import os
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from lnn_model import LNN

# -------------------------------------------------------------------------
# 1. LOAD DATASET (Same preprocessing as strict pipeline)
# -------------------------------------------------------------------------
DATA_DIR = "data/GothamDataset2025/processed"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "Data/GothamDataset2025/processed"

NROWS = 100000

files_to_load = [
    "iotsim-air-quality-1.csv",
    "iotsim-building-monitor-1.csv",
    "iotsim-cooler-motor-1.csv",
    "iotsim-ip-camera-museum-1.csv"
]

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

class GroupShuffleSplit:
    def __init__(self, test_size=0.2, n_splits=1, random_state=None):
        self.test_size = test_size
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("The 'groups' parameter must be provided.")
        rng = np.random.RandomState(self.random_state)
        unique_groups = np.unique(groups)
        rng.shuffle(unique_groups)
        n_test = int(np.ceil(self.test_size * len(unique_groups)))
        test_groups = set(unique_groups[:n_test])

        test_idx = np.where(np.isin(groups, list(test_groups)))[0]
        train_idx = np.where(~np.isin(groups, list(test_groups)))[0]
        yield train_idx, test_idx

def load_representative_dataset(path, target_rows=100000):
    if target_rows is None:
        return pd.read_csv(path, low_memory=False)

    file_size_gb = os.path.getsize(path) / (1024 ** 3)
    if file_size_gb < 0.1:
        df = pd.read_csv(path, low_memory=False)
        if len(df) > target_rows:
            return df.sample(n=target_rows, random_state=42).reset_index(drop=True)
        return df

    est_rows = max(int(file_size_gb * 5000000), 1)
    sample_fraction = min(target_rows / est_rows, 1.0)

    chunk_size = 500000
    chunks = []

    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
        sample_n = int(len(chunk) * sample_fraction)
        if sample_n > 0:
            chunks.append(chunk.sample(n=sample_n, random_state=42))

    if not chunks:
        return pd.read_csv(path, nrows=target_rows, low_memory=False)

    df = pd.concat(chunks, ignore_index=True)
    if len(df) > target_rows:
        df = df.sample(n=target_rows, random_state=42).reset_index(drop=True)
        
    return df

print("Loading and preparing datasets...")
loaded_datasets = {}
for fname in files_to_load:
    path = os.path.join(DATA_DIR, fname)
    loaded_datasets[fname] = load_representative_dataset(path, NROWS)

cleaned_dfs = []
for fname, df in loaded_datasets.items():
    if 'attack_type' in df.columns:
        y = (df['attack_type'].astype(str).str.strip().str.lower() != 'benign').astype(int).values
    elif 'label' in df.columns:
        col = df['label']
        if pd.api.types.is_numeric_dtype(col):
            y = col.astype(int).values
        else:
            y = (col.astype(str).str.strip().str.lower() != 'benign').astype(int).values
    else:
        np.random.seed(42)
        y = np.random.randint(0, 2, size=len(df))

    df = df.copy()
    df['__group_id'] = fname + "_" + (df.index // 100).astype(str)
    df['__label'] = y

    target_cols = [c for c in ['label', 'attack_type'] if c in df.columns]
    df_features = df.drop(columns=target_cols)

    df_numeric = df_features.select_dtypes(include=[np.number])
    bad_substrings = ["ip", "port", "proto", "mac", "flow_id"]
    cols_to_remove = [col for col in df_numeric.columns if any(sub in col.lower() for sub in bad_substrings)]

    df_clean = df_numeric.drop(columns=cols_to_remove)
    df_clean = df_clean.fillna(0.0)

    df_clean['group_id'] = df['__group_id']
    df_clean['label'] = df['__label']
    cleaned_dfs.append(df_clean)

union_cols = sorted(list(set().union(*(df.drop(columns=['group_id', 'label']).columns for df in cleaned_dfs))))

aligned_dfs = []
for df in cleaned_dfs:
    y_col = df['label']
    g_col = df['group_id']
    df_feat = df.drop(columns=['label', 'group_id']).reindex(columns=union_cols, fill_value=0.0)
    df_feat['label'] = y_col
    df_feat['group_id'] = g_col
    aligned_dfs.append(df_feat)

df_combined = pd.concat(aligned_dfs, ignore_index=True)

class_counts = df_combined['label'].value_counts()
min_class = class_counts.min()
df_balanced = pd.concat([
    df_combined[df_combined['label'] == 0].sample(min_class, random_state=42),
    df_combined[df_combined['label'] == 1].sample(min_class, random_state=42)
]).sample(frac=1.0, random_state=42).reset_index(drop=True)

df_combined = df_balanced

feature_cols = [c for c in df_combined.columns if c not in ['label', 'group_id']]

y_all = df_combined['label'].values
X_all_df = df_combined[feature_cols]

correlations = []
for col in feature_cols:
    corr = np.corrcoef(X_all_df[col].values, y_all)[0, 1]
    if np.isnan(corr):
        corr = 0.0
    correlations.append((col, corr))

cols_to_drop_corr = [col for col, corr in correlations if abs(corr) > 0.99]
if cols_to_drop_corr:
    df_combined = df_combined.drop(columns=cols_to_drop_corr)
    feature_cols = [c for c in df_combined.columns if c not in ['label', 'group_id']]

variances = df_combined[feature_cols].std()
cols_to_drop_var = variances[variances < 1e-5].index.tolist()

if cols_to_drop_var:
    df_combined = df_combined.drop(columns=cols_to_drop_var)
    feature_cols = [c for c in df_combined.columns if c not in ['label', 'group_id']]

gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
X = df_combined[feature_cols].values
y = df_combined['label'].values
groups = df_combined['group_id'].values

train_idx, test_idx = next(gss.split(X, y, groups))

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# -------------------------------------------------------------------------
# 2. LOAD MODEL
# -------------------------------------------------------------------------
print("Loading model...")
input_dim = X_train_s.shape[1]
model = LNN(input_dim=input_dim, hidden_dim=16, num_steps=6, dt=0.1)

# Apply same architecture adjustment as strict pipeline
if isinstance(model.output_layer, nn.Sequential) and len(model.output_layer) > 1:
    model.output_layer = nn.Sequential(model.output_layer[0])

model.load_state_dict(torch.load("lnn_fixed.pth"))
model.eval()

# Model prediction wrapper
def predict_fn(X_np):
    X_t = torch.tensor(X_np, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_t)
        probs = torch.sigmoid(logits).numpy().squeeze()
        if probs.ndim == 0:
            probs = np.expand_dims(probs, 0)
    return probs

# -------------------------------------------------------------------------
# NATIVE PYTORCH XAI IMPLEMENTATIONS (ZERO DEPENDENCIES)
# -------------------------------------------------------------------------
def custom_kernel_shap(model_fn, X_explain, X_bg, num_samples=250):
    """Native implementation of Kernel SHAP."""
    bg_mean = np.mean(X_bg, axis=0)
    M = X_explain.shape[1]
    
    shap_values = []
    
    for idx in range(X_explain.shape[0]):
        x = X_explain[idx]
        
        # Sample binary masks
        masks = np.random.binomial(1, 0.5, size=(num_samples, M))
        masks[0] = np.ones(M)
        masks[1] = np.zeros(M)
        
        # Create masked inputs
        X_masked = masks * x + (1 - masks) * bg_mean
        
        # Get predictions
        preds = model_fn(X_masked)
        
        # Kernel SHAP weights
        z = np.sum(masks, axis=1)
        weights = np.zeros(num_samples)
        for i in range(num_samples):
            k = int(z[i])
            if k == 0 or k == M:
                weights[i] = 10000.0 # High weight for extremes
            else:
                try:
                    weights[i] = (M - 1) / (float(math.comb(M, k)) * k * (M - k))
                except OverflowError:
                    weights[i] = 0.0
                
        # Weighted linear regression
        W = np.diag(np.sqrt(weights))
        Z_w = W @ masks
        y_w = W @ preds
        
        Z_b = np.c_[Z_w, np.ones(Z_w.shape[0])]
        I = np.eye(Z_b.shape[1])
        I[-1, -1] = 0
        
        ZTZ = Z_b.T @ Z_b + 1e-5 * I
        ZTy = Z_b.T @ y_w
        
        coefs = np.linalg.solve(ZTZ, ZTy)[:-1]
        shap_values.append(coefs)
        
    return np.array(shap_values)

class CustomLIME:
    """Native implementation of Tabular LIME."""
    def __init__(self, model_fn, bg_data):
        self.model_fn = model_fn
        self.bg_std = np.std(bg_data, axis=0) + 1e-8
        
    def explain_instance(self, x, num_samples=500, num_features=10):
        # Generate perturbations
        noise = np.random.normal(0, 1, size=(num_samples, x.shape[0]))
        perturbed = x + noise * self.bg_std
        perturbed[0] = x # ensure original is first
        
        # Get predictions
        preds = self.model_fn(perturbed)
        
        # Compute distances and weights
        distances = np.sqrt(np.sum(((perturbed - x) / self.bg_std) ** 2, axis=1))
        kernel_width = np.sqrt(x.shape[0]) * 0.75
        weights = np.exp(-(distances ** 2) / (kernel_width ** 2))
        
        # Weighted Ridge Regression
        W = np.diag(np.sqrt(weights))
        X_w = W @ perturbed
        y_w = W @ preds
        
        X_b = np.c_[X_w, np.ones(X_w.shape[0])]
        I = np.eye(X_b.shape[1])
        I[-1, -1] = 0 # don't regularize bias
        
        XTX = X_b.T @ X_b + 1e-3 * I
        XTy = X_b.T @ y_w
        
        coefs = np.linalg.solve(XTX, XTy)[:-1]
        
        # Sort by absolute magnitude
        top_indices = np.argsort(np.abs(coefs))[::-1][:num_features]
        return [(i, coefs[i]) for i in top_indices]

# -------------------------------------------------------------------------
# 3. SHAP (Native Kernel Explainer)
# -------------------------------------------------------------------------
print("\nRunning Native Kernel SHAP...")
# Select 100 background samples
np.random.seed(42)
bg_indices = np.random.choice(X_train_s.shape[0], 100, replace=False)
X_bg = X_train_s[bg_indices]

# Explain 50 test samples
test_indices = np.random.choice(X_test_s.shape[0], 50, replace=False)
X_explain = X_test_s[test_indices]

# Compute SHAP values manually
shap_values_np = custom_kernel_shap(predict_fn, X_explain, X_bg, num_samples=250)

# Generate feature importance ranking
shap_abs_mean = np.mean(np.abs(shap_values_np), axis=0)
shap_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': shap_abs_mean
}).sort_values(by='Importance', ascending=False)

# Generate summary plot
plt.figure(figsize=(10, 8))
plt.barh(shap_importance['Feature'][:20][::-1], shap_importance['Importance'][:20][::-1], color='steelblue')
plt.xlabel('Mean Absolute SHAP Value')
plt.title('Native Kernel SHAP Feature Importance (Top 20)')
plt.tight_layout()
plt.savefig("shap_summary.png", bbox_inches='tight')
plt.close()

print("\n--- SHAP Feature Importance Ranking ---")
print(shap_importance.head(10))

# -------------------------------------------------------------------------
# 4. LIME (Native Tabular Explainer)
# -------------------------------------------------------------------------
print("\nRunning Native LIME...")
lime_explainer = CustomLIME(predict_fn, X_train_s)

# Explain 5 random test samples
np.random.seed(42)
lime_indices = np.random.choice(X_test_s.shape[0], 5, replace=False)

with open("lime_explanations.txt", "w") as f:
    f.write("Native LIME Explanations for 5 random test samples\n")
    f.write("=" * 50 + "\n\n")
    
    for i, idx in enumerate(lime_indices):
        f.write(f"Sample {i+1} (Index {idx}):\n")
        f.write(f"True Label: {'Attack' if y_test[idx] == 1 else 'Benign'}\n")
        
        sample_np = X_test_s[idx]
        probs = predict_fn(sample_np.reshape(1, -1))[0]
        f.write(f"Model Prediction: Benign: {1-probs:.4f}, Attack: {probs:.4f}\n")
        
        # Get explanations
        exp_results = lime_explainer.explain_instance(sample_np, num_samples=500, num_features=10)
        
        f.write("Top contributing features:\n")
        for f_idx, weight in exp_results:
            f.write(f"  {feature_cols[f_idx]}: {weight:.4f}\n")
            print(f"  {feature_cols[f_idx]}: {weight:.4f}")
        
        f.write("-" * 50 + "\n\n")

print("\nSaved shap_summary.png and lime_explanations.txt")
print("Done!")

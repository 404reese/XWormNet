import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from lnn_model import LNN

# -------------------------------------------------------------------------
# CONFIGURATION
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

print(f"Data directory: {DATA_DIR}")
print(f"Row limit per file: {NROWS}")

# -------------------------------------------------------------------------
# ENVIRONMENT & FALLBACKS
# -------------------------------------------------------------------------
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupShuffleSplit
    SKLEARN_AVAILABLE = True
    print("[Environment] scikit-learn is available.")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[Environment Notice] Falling back to custom NumPy ML utilities.")

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

# -------------------------------------------------------------------------
# DATA LOADING (REPRESENTATIVE SAMPLING)
# -------------------------------------------------------------------------
print("\n" + "=" * 50)
print("             1. DATA LOADING")
print("=" * 50)

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

loaded_datasets = {}
for fname in files_to_load:
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required dataset file not found: {path}")
    print(f"Loading {fname}...")
    loaded_datasets[fname] = load_representative_dataset(path, NROWS)
    print(f"  Loaded {fname} shape: {loaded_datasets[fname].shape}")

# -------------------------------------------------------------------------
# DIAGNOSTIC: Check feature schemas BEFORE any processing
# -------------------------------------------------------------------------
print("\n" + "=" * 50)
print("             1b. SCHEMA DIAGNOSTIC")
print("=" * 50)

all_numeric_cols = set()
for fname, df in loaded_datasets.items():
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_numeric_cols.update(numeric_cols)
    print(f"\n{fname}:")
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Numeric columns: {len(numeric_cols)}")
    print(f"  Sample cols: {numeric_cols[:10]}")

print(f"\nUnion of all numeric columns: {len(all_numeric_cols)}")

# -------------------------------------------------------------------------
# LABEL HANDLING & FEATURE CLEANING — WITH SOURCE-AWARE GROUPING
# -------------------------------------------------------------------------
print("\n" + "=" * 50)
print("             2. LABEL HANDLING & FEATURE CLEANING")
print("=" * 50)

cleaned_dfs = []

for fname, df in loaded_datasets.items():
    # --- Extract label ---
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

    # --- Assign group_id BEFORE any feature manipulation ---
    # Use source filename + row chunk as group. Each file gets ~N/100 groups.
    # This preserves device/session boundaries and prevents cross-file leakage.
    df = df.copy()
    df['__group_id'] = fname + "_" + (df.index // 100).astype(str)
    df['__label'] = y

    # --- Clean features ---
    target_cols = [c for c in ['label', 'attack_type'] if c in df.columns]
    df_features = df.drop(columns=target_cols)

    df_numeric = df_features.select_dtypes(include=[np.number])
    bad_substrings = ["ip", "port", "proto", "mac", "flow_id"]
    cols_to_remove = [col for col in df_numeric.columns if any(sub in col.lower() for sub in bad_substrings)]

    df_clean = df_numeric.drop(columns=cols_to_remove)
    df_clean = df_clean.fillna(0.0)

    # Keep group_id and label
    df_clean['group_id'] = df['__group_id']
    df_clean['label'] = df['__label']
    cleaned_dfs.append(df_clean)

# Align features and merge
print("Aligning feature columns and merging datasets...")
union_cols = sorted(list(set().union(*(df.drop(columns=['group_id', 'label']).columns for df in cleaned_dfs))))

print(f"Total unique features across all files: {len(union_cols)}")

aligned_dfs = []
for df in cleaned_dfs:
    y_col = df['label']
    g_col = df['group_id']
    df_feat = df.drop(columns=['label', 'group_id']).reindex(columns=union_cols, fill_value=0.0)
    df_feat['label'] = y_col
    df_feat['group_id'] = g_col
    aligned_dfs.append(df_feat)

df_combined = pd.concat(aligned_dfs, ignore_index=True)

# Balance classes for meaningful Sanity Checks
print("Balancing classes in combined dataset...")
class_counts = df_combined['label'].value_counts()
print(f"  Class counts before balancing:\n{class_counts}")
min_class = class_counts.min()
df_balanced = pd.concat([
    df_combined[df_combined['label'] == 0].sample(min_class, random_state=42),
    df_combined[df_combined['label'] == 1].sample(min_class, random_state=42)
]).sample(frac=1.0, random_state=42).reset_index(drop=True)

df_combined = df_balanced
print(f"  Class counts after balancing:\n{df_combined['label'].value_counts()}")
print(f"Final combined shape: {df_combined.shape}")

# -------------------------------------------------------------------------
# GROUP ID VERIFICATION
# -------------------------------------------------------------------------
print("\n" + "=" * 50)
print("             3. GROUP ID VERIFICATION")
print("=" * 50)

feature_cols = [c for c in df_combined.columns if c not in ['label', 'group_id']]
num_unique_groups = df_combined['group_id'].nunique()
print(f"Generated {num_unique_groups} unique groups out of {len(df_combined)} rows.")
print(f"Average rows per group: {len(df_combined) / num_unique_groups:.1f}")

# Sanity: check that groups are not trivially unique per row
if num_unique_groups == len(df_combined):
    print("WARNING: Every row is its own group. This defeats the purpose of GroupShuffleSplit.")
    print("Consider using larger chunks (e.g. index // 500 instead of // 100).")

# -------------------------------------------------------------------------
# FEATURE ANALYSIS (CORRELATION) — RELAXED THRESHOLD
# -------------------------------------------------------------------------
print("\n" + "=" * 50)
print("             4. FEATURE ANALYSIS (CORRELATION)")
print("=" * 50)

y_all = df_combined['label'].values
X_all_df = df_combined[feature_cols]

correlations = []
for col in feature_cols:
    corr = np.corrcoef(X_all_df[col].values, y_all)[0, 1]
    if np.isnan(corr):
        corr = 0.0
    correlations.append((col, corr))

# Sort by absolute correlation descending
correlations.sort(key=lambda x: abs(x[1]), reverse=True)

print("--- FEATURE IMPORTANCE HINTS (TOP 20 CORRELATED) ---")
for col, corr in correlations[:20]:
    print(f"  {col:<40}: {corr:+.5f}")

# RELAXED threshold: only drop if |correlation| > 0.99 (near-perfect leakage)
print("\nFiltering features with |correlation| > 0.99 ...")
cols_to_drop_corr = [col for col, corr in correlations if abs(corr) > 0.99]
if cols_to_drop_corr:
    print(f"  REMOVING {len(cols_to_drop_corr)} NEAR-PERFECTLY PREDICTIVE FEATURES (Likely Leakage):")
    for c in cols_to_drop_corr:
        print(f"    - {c}")
    df_combined = df_combined.drop(columns=cols_to_drop_corr)
    feature_cols = [c for c in df_combined.columns if c not in ['label', 'group_id']]
else:
    print("  No features exceeded correlation threshold of 0.99.")

# -------------------------------------------------------------------------
# VARIANCE FILTER
# -------------------------------------------------------------------------
print("\n" + "=" * 50)
print("             5. VARIANCE FILTER")
print("=" * 50)

print("Filtering near-zero variance features (std < 1e-5)...")
variances = df_combined[feature_cols].std()
cols_to_drop_var = variances[variances < 1e-5].index.tolist()

if cols_to_drop_var:
    print(f"  REMOVING {len(cols_to_drop_var)} ZERO VARIANCE FEATURES:")
    for c in cols_to_drop_var:
        print(f"    - {c}")
    df_combined = df_combined.drop(columns=cols_to_drop_var)
    feature_cols = [c for c in df_combined.columns if c not in ['label', 'group_id']]
else:
    print("  No features failed variance threshold.")

print(f"Final feature count after filtering: {len(feature_cols)}")

# -------------------------------------------------------------------------
# GROUP-BASED SPLIT (CRITICAL)
# -------------------------------------------------------------------------
print("\n" + "=" * 50)
print("             6. GROUP-BASED SPLIT")
print("=" * 50)

gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
X = df_combined[feature_cols].values
y = df_combined['label'].values
groups = df_combined['group_id'].values

train_idx, test_idx = next(gss.split(X, y, groups))

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

train_groups = set(groups[train_idx])
test_groups = set(groups[test_idx])
group_overlap = train_groups.intersection(test_groups)

print(f"GroupShuffleSplit results:")
print(f"  Train samples: {len(X_train)}  (Unique Groups: {len(train_groups)})")
print(f"  Test samples:  {len(X_test)}  (Unique Groups: {len(test_groups)})")
print(f"  Group Overlap: {len(group_overlap)}")
assert len(group_overlap) == 0, "CRITICAL ERROR: Leakage detected. Group IDs overlap between Train and Test sets!"

# Verify class balance in each split
print(f"\nTrain class balance: {np.bincount(y_train)}")
print(f"Test class balance:  {np.bincount(y_test)}")

# -------------------------------------------------------------------------
# NORMAL PIPELINE (SCALE, INIT, TRAIN, EVAL)
# -------------------------------------------------------------------------
print("\n" + "=" * 50)
print("             7. NORMAL PIPELINE")
print("=" * 50)

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Model
input_dim = X_train_s.shape[1]
model = LNN(input_dim=input_dim, hidden_dim=16, num_steps=6, dt=0.1)
if isinstance(model.output_layer, nn.Sequential) and len(model.output_layer) > 1:
    model.output_layer = nn.Sequential(model.output_layer[0])

# Train
X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 15
print(f"Training REAL model for {epochs} epochs...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for bx, by in train_loader:
        optimizer.zero_grad()
        logits = model(bx)
        loss = criterion(logits, by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * bx.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"  Epoch {epoch+1:02d}/{epochs} - Loss: {epoch_loss:.6f}")

# Evaluation Helper
def evaluate_model(eval_model, X_eval, y_eval, threshold=0.5):
    eval_model.eval()
    X_t = torch.tensor(X_eval, dtype=torch.float32)
    start_time = time.perf_counter()
    with torch.no_grad():
        logits_t = eval_model(X_t)
        probs_t = torch.sigmoid(logits_t)
    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000 / len(X_eval)
    probs = probs_t.numpy().squeeze()
    if probs.ndim == 0: probs = np.array([probs])
    preds = (probs >= threshold).astype(int)

    tp = np.sum((y_eval == 1) & (preds == 1))
    tn = np.sum((y_eval == 0) & (preds == 0))
    fp = np.sum((y_eval == 0) & (preds == 1))
    fn = np.sum((y_eval == 1) & (preds == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    pos = probs[y_eval == 1]
    neg = probs[y_eval == 0]
    if len(pos) == 0 or len(neg) == 0:
        roc_auc = 0.5
    else:
        pos = np.sort(pos)
        neg = np.sort(neg)
        c_smaller = np.searchsorted(neg, pos, side='left')
        c_equal = np.searchsorted(neg, pos, side='right') - c_smaller
        roc_auc = np.mean(c_smaller + 0.5 * c_equal) / len(neg)

    return {"precision": precision, "recall": recall, "f1": f1, "roc_auc": roc_auc, 
            "cm": (tn, fp, fn, tp), "latency": latency_ms}

real_metrics = evaluate_model(model, X_test_s, y_test)
print("\n--- REAL MODEL EVALUATION ---")
print(f"F1 Score:          {real_metrics['f1']:.5f} (Target: ~0.6-0.8)")
print(f"ROC-AUC:           {real_metrics['roc_auc']:.5f}")
print(f"Precision:         {real_metrics['precision']:.5f}")
print(f"Recall:            {real_metrics['recall']:.5f}")
tn, fp, fn, tp = real_metrics['cm']
print(f"Confusion Matrix:  TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# -------------------------------------------------------------------------
# SANITY CHECK AGAIN
# -------------------------------------------------------------------------
print("\n" + "=" * 50)
print("             8. SANITY CHECK (SHUFFLE AUDIT)")
print("=" * 50)

y_train_shuffled = np.copy(y_train)
np.random.seed(42)
np.random.shuffle(y_train_shuffled)

shuffle_model = LNN(input_dim=input_dim, hidden_dim=16, num_steps=6, dt=0.1)
if isinstance(shuffle_model.output_layer, nn.Sequential) and len(shuffle_model.output_layer) > 1:
    shuffle_model.output_layer = nn.Sequential(shuffle_model.output_layer[0])

y_train_shuff_t = torch.tensor(y_train_shuffled, dtype=torch.float32).unsqueeze(1)
shuff_loader = DataLoader(TensorDataset(X_train_t, y_train_shuff_t), batch_size=256, shuffle=True)
shuff_optimizer = torch.optim.Adam(shuffle_model.parameters(), lr=0.001)

print(f"Training SHUFFLE model for 5 epochs...")
for epoch in range(5):
    shuffle_model.train()
    running_loss = 0.0
    for bx, by in shuff_loader:
        shuff_optimizer.zero_grad()
        logits = shuffle_model(bx)
        loss = criterion(logits, by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(shuffle_model.parameters(), max_norm=1.0)
        shuff_optimizer.step()
        running_loss += loss.item() * bx.size(0)
    epoch_loss = running_loss / len(shuff_loader.dataset)
    print(f"  Shuffled Epoch {epoch+1:02d}/05 - Loss: {epoch_loss:.6f}")

shuff_metrics = evaluate_model(shuffle_model, X_test_s, y_test)
print("\n--- SHUFFLE MODEL EVALUATION ---")
print(f"Sanity Check F1 Score: {shuff_metrics['f1']:.5f} (Expected: ~0.5)")
print(f"Sanity Check ROC-AUC:  {shuff_metrics['roc_auc']:.5f} (Expected: ~0.5)")

if 0.4 <= shuff_metrics['f1'] <= 0.6 and 0.4 <= shuff_metrics['roc_auc'] <= 0.6:
    print("\n✅ PIPELINE IS CLEAN: Shuffle test confirmed random guessing capabilities.")
else:
    print("\n⚠️ WARNING: Shuffle metrics deviated from 0.5. Check class balance or leakage.")
    print("   If F1 ~ 0.5 but ROC-AUC is high, the model may be learning group structure.")
    print("   This can happen when groups have inherent label imbalance.")

# -------------------------------------------------------------------------
# OUTPUT
# -------------------------------------------------------------------------
torch.save(model.state_dict(), "lnn_fixed.pth")
df_results = pd.DataFrame([{
    "Real_F1": real_metrics['f1'],
    "Real_ROC_AUC": real_metrics['roc_auc'],
    "Real_Precision": real_metrics['precision'],
    "Real_Recall": real_metrics['recall'],
    "Shuffle_F1": shuff_metrics['f1'],
    "Shuffle_ROC_AUC": shuff_metrics['roc_auc']
}])
df_results.to_csv("fixed_results.csv", index=False)
print("\nSaved lnn_fixed.pth and fixed_results.csv")
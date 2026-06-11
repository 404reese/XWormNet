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

# Row limit per file to optimize memory and runtime on CPU
# Set to None to run on full datasets (Warning: ~15M rows total, very slow on CPU)
NROWS = 100000

train_files = [
    "iotsim-air-quality-1.csv",
    "iotsim-building-monitor-1.csv",
    "iotsim-cooler-motor-1.csv"
]

test_files = [
    "iotsim-ip-camera-museum-1.csv"
]

print(f"Data directory: {DATA_DIR}")
print(f"Row limit (NROWS): {NROWS}")

# -------------------------------------------------------------------------
# SCIKIT-LEARN DETECTION & STANDARDS SCALER FALLBACK
# -------------------------------------------------------------------------
try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
    print("[Environment] scikit-learn is available and will be used.")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[Environment Notice] Falling back to custom NumPy ML utilities.")
    
    class StandardScaler:
        """
        Standardize features by removing the mean and scaling to unit variance.
        Equivalent to sklearn.preprocessing.StandardScaler.
        """
        def __init__(self):
            self.mean = None
            self.scale = None
            
        def fit(self, X):
            self.mean = np.mean(X, axis=0)
            self.scale = np.std(X, axis=0)
            self.scale[self.scale == 0.0] = 1.0  # Avoid division by zero
            return self
            
        def transform(self, X):
            if self.mean is None or self.scale is None:
                raise ValueError("Scaler has not been fitted yet.")
            return (X - self.mean) / self.scale
            
        def fit_transform(self, X):
            return self.fit(X).transform(X)

# -------------------------------------------------------------------------
# DATA LOADING (REPRESENTATIVE SAMPLING)
# -------------------------------------------------------------------------
def load_representative_dataset(path, target_rows=100000):
    """
    Loads a representative sample of a large CSV dataset using chunks to avoid 
    out-of-memory errors and ensure that both benign and attack categories 
    (which are distributed sequentially in the files) are loaded.
    """
    if target_rows is None:
        return pd.read_csv(path, low_memory=False)
        
    file_size_gb = os.path.getsize(path) / (1024 ** 3)
    # If the file is small (under 100 MB), load it fully and then sample
    if file_size_gb < 0.1:
        df = pd.read_csv(path, low_memory=False)
        if len(df) > target_rows:
            return df.sample(n=target_rows, random_state=42).reset_index(drop=True)
        return df
        
    # Estimate total rows (approx 5 million rows per GB)
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

print("\n" + "=" * 50)
print("             1. DATA LOADING")
print("=" * 50)

loaded_datasets = {}
all_files = train_files + test_files

for fname in all_files:
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required dataset file not found: {path}")
    print(f"Loading {fname}...")
    # Store each dataset separately
    loaded_datasets[fname] = load_representative_dataset(path, NROWS)
    print(f"  Loaded {fname} with shape: {loaded_datasets[fname].shape}")

# -------------------------------------------------------------------------
# LABEL HANDLING & FEATURE CLEANING
# -------------------------------------------------------------------------
print("\n" + "=" * 50)
print("             2. LABEL HANDLING & FEATURE CLEANING")
print("=" * 50)

cleaned_dfs = {}
labels = {}

for fname in all_files:
    df = loaded_datasets[fname]
    
    # Label Handling: Extract binary label
    if 'attack_type' in df.columns:
        y = (df['attack_type'].astype(str).str.strip().str.lower() != 'benign').astype(int).values
    elif 'label' in df.columns:
        col = df['label']
        if pd.api.types.is_numeric_dtype(col):
            y = col.astype(int).values
        else:
            y = (col.astype(str).str.strip().str.lower() != 'benign').astype(int).values
    else:
        # Fallback to random binary labels (only for testing)
        print(f"  Warning: No label column in {fname}. Generating random binary labels.")
        np.random.seed(42)
        y = np.random.randint(0, 2, size=len(df))
        
    labels[fname] = y
    
    # Feature Cleaning: Drop non-numeric and shortcut leakage columns
    # Drop label columns if they exist
    target_cols = [c for c in ['label', 'attack_type'] if c in df.columns]
    df_features = df.drop(columns=target_cols)
    
    # Drop non-numeric columns
    df_numeric = df_features.select_dtypes(include=[np.number])
    
    # Drop columns containing "ip", "port", "proto", "mac", "flow_id"
    bad_substrings = ["ip", "port", "proto", "mac", "flow_id"]
    cols_to_remove = [col for col in df_numeric.columns if any(sub in col.lower() for sub in bad_substrings)]
    
    df_clean = df_numeric.drop(columns=cols_to_remove)
    
    # Fill NaN with 0
    df_clean = df_clean.fillna(0.0)
    
    cleaned_dfs[fname] = df_clean
    print(f"  Processed {fname} -> Cleaned features shape: {df_clean.shape}")

# -------------------------------------------------------------------------
# FEATURE ALIGNMENT (UNION OF COLUMNS)
# -------------------------------------------------------------------------
print("\n" + "=" * 50)
print("             3. FEATURE ALIGNMENT")
print("=" * 50)

# Determine the union of all cleaned numeric feature columns across datasets
union_cols = sorted(list(set().union(*(df.columns for df in cleaned_dfs.values()))))
print(f"Aligning features using column union (total features: {len(union_cols)})")

for fname in all_files:
    cleaned_dfs[fname] = cleaned_dfs[fname].reindex(columns=union_cols, fill_value=0.0)
    print(f"  Aligned {fname} shape: {cleaned_dfs[fname].shape}")

# -------------------------------------------------------------------------
# SCENARIO-BASED SPLIT & MERGING
# -------------------------------------------------------------------------
print("\n" + "=" * 50)
print("             4. SCENARIO-BASED SPLITTING & MERGING")
print("=" * 50)
print("TRAIN datasets:")
for f in train_files:
    print(f"  - {f}")
print("TEST dataset:")
for f in test_files:
    print(f"  - {f}")

# Concatenate TRAIN datasets together
train_features_list = []
train_labels_list = []
for fname in train_files:
    df_feat = cleaned_dfs[fname]
    y = labels[fname]
    train_features_list.append(df_feat)
    train_labels_list.append(pd.Series(y, index=df_feat.index))

df_train_features = pd.concat(train_features_list, ignore_index=True)
y_train_series = pd.concat(train_labels_list, ignore_index=True)

df_train = df_train_features.copy()
df_train['label'] = y_train_series

# Keep TEST dataset completely separate
df_test_features = cleaned_dfs[test_files[0]].copy()
y_test_series = pd.Series(labels[test_files[0]], index=df_test_features.index)

df_test = df_test_features.copy()
df_test['label'] = y_test_series

print(f"Concatenated Train set shape: {df_train.shape}")
print(f"Test set shape: {df_test.shape}")

# -------------------------------------------------------------------------
# DUPLICATE REMOVAL & OVERLAP CHECK
# -------------------------------------------------------------------------
print("\n" + "=" * 50)
print("             5. DUPLICATE & OVERLAP REMOVAL")
print("=" * 50)

# 1. Remove duplicates in TRAIN set
train_len_before = len(df_train)
df_train_clean = df_train.drop_duplicates().copy()
train_dups_removed = train_len_before - len(df_train_clean)

# 2. Remove duplicates in TEST set
test_len_before = len(df_test)
df_test_clean = df_test.drop_duplicates().copy()
test_dups_removed = test_len_before - len(df_test_clean)

# 3. Remove overlap between TRAIN and TEST based on features
feature_cols = [c for c in df_train_clean.columns if c != 'label']
train_tuples = set(map(tuple, df_train_clean[feature_cols].values))
test_values = df_test_clean[feature_cols].values

is_not_overlapping = np.array([tuple(x) not in train_tuples for x in test_values])
overlap_count = np.sum(~is_not_overlapping)

df_test_no_overlap = df_test_clean[is_not_overlapping].copy()

print(f"Train set internal duplicates removed: {train_dups_removed}")
print(f"Test set internal duplicates removed:  {test_dups_removed}")
print(f"Overlap records removed from Test set: {overlap_count}")
print(f"Final Train set shape:                 {df_train_clean.shape}")
print(f"Final Test set shape (clean):          {df_test_no_overlap.shape}")

# -------------------------------------------------------------------------
# CONVERT TO NUMPY & SCALE
# -------------------------------------------------------------------------
print("\n" + "=" * 50)
print("             6. CONVERT TO NUMPY & SCALE")
print("=" * 50)

X_train = df_train_clean[feature_cols].values
y_train = df_train_clean['label'].values

X_test = df_test_no_overlap[feature_cols].values
y_test = df_test_no_overlap['label'].values

# Standard scaling to prevent continuous ODE numerical instability in LNN
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")

# -------------------------------------------------------------------------
# MODEL (LNN)
# -------------------------------------------------------------------------
print("\n" + "=" * 50)
print("             7. MODEL INITIALIZATION (LNN)")
print("=" * 50)

input_size = X_train.shape[1]
print(f"Loading LNN model from lnn_model.py")
model = LNN(input_dim=input_size, hidden_dim=16, num_steps=6, dt=0.1)
print(f"Input size = {input_size}")

# Strip Sigmoid activation from model.output_layer to output logits for BCEWithLogitsLoss
if isinstance(model.output_layer, nn.Sequential) and len(model.output_layer) > 1:
    print("Modifying final layer: replacing Sigmoid with raw logits (for BCEWithLogitsLoss)")
    model.output_layer = nn.Sequential(model.output_layer[0])

# -------------------------------------------------------------------------
# TRAINING
# -------------------------------------------------------------------------
print("\n" + "=" * 50)
print("             8. TRAINING")
print("=" * 50)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 15  # Epochs: 15-20
print(f"Training LNN model for {epochs} epochs...")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        
        # Clip gradients to ensure stable continuous dynamics
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        running_loss += loss.item() * batch_x.size(0)
        
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1:02d}/{epochs} - Loss: {epoch_loss:.6f}")

# -------------------------------------------------------------------------
# EVALUATION HELPER
# -------------------------------------------------------------------------
def evaluate_model(eval_model, X, y, threshold=0.5):
    eval_model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    
    start_time = time.perf_counter()
    with torch.no_grad():
        logits_t = eval_model(X_t)
        probs_t = torch.sigmoid(logits_t)
    end_time = time.perf_counter()
    
    # Latency calculation
    total_time_ms = (end_time - start_time) * 1000
    latency_per_sample = total_time_ms / len(X)
    
    probs = probs_t.numpy().squeeze()
    if probs.ndim == 0:
        probs = np.array([probs])
    preds = (probs >= threshold).astype(int)
    
    # Calculate confusion matrix components
    tp = np.sum((y == 1) & (preds == 1))
    tn = np.sum((y == 0) & (preds == 0))
    fp = np.sum((y == 0) & (preds == 1))
    fn = np.sum((y == 1) & (preds == 0))
    
    # Metric calculations
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Area under ROC (fast O(N log N) rank-based calculation)
    pos = probs[y == 1]
    neg = probs[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        roc_auc = 0.5
    else:
        pos = np.sort(pos)
        neg = np.sort(neg)
        count_smaller = np.searchsorted(neg, pos, side='left')
        count_equal = np.searchsorted(neg, pos, side='right') - count_smaller
        roc_auc = np.mean(count_smaller + 0.5 * count_equal) / len(neg)
        
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": (tn, fp, fn, tp),
        "latency_ms_per_sample": latency_per_sample
    }

# -------------------------------------------------------------------------
# EVALUATION
# -------------------------------------------------------------------------
print("\n" + "=" * 50)
print("             9. EVALUATION")
print("=" * 50)
print("Evaluating trained model on the clean scenario-based Test Set...")
test_metrics = evaluate_model(model, X_test, y_test)

print(f"Precision:         {test_metrics['precision']:.5f}")
print(f"Recall:            {test_metrics['recall']:.5f}")
print(f"F1 Score:          {test_metrics['f1']:.5f}")
print(f"ROC-AUC:           {test_metrics['roc_auc']:.5f}")
tn, fp, fn, tp = test_metrics['confusion_matrix']
print(f"Confusion Matrix:  TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Inference Latency: {test_metrics['latency_ms_per_sample']:.6f} ms/sample")

# -------------------------------------------------------------------------
# SANITY CHECK (MANDATORY SHUFFLE TEST)
# -------------------------------------------------------------------------
print("\n" + "=" * 50)
print("             10. SANITY CHECK (SHUFFLE LABEL AUDIT)")
print("=" * 50)
print("Shuffling training labels randomly to destroy feature correlations...")
y_train_shuffled = np.copy(y_train)
np.random.seed(42)
np.random.shuffle(y_train_shuffled)

print("Instantiating and retraining clean model instance for 5 epochs...")
shuffled_model = LNN(input_dim=input_size, hidden_dim=16, num_steps=6, dt=0.1)
if isinstance(shuffled_model.output_layer, nn.Sequential) and len(shuffled_model.output_layer) > 1:
    shuffled_model.output_layer = nn.Sequential(shuffled_model.output_layer[0])

y_train_shuffled_t = torch.tensor(y_train_shuffled, dtype=torch.float32).unsqueeze(1)
shuffled_dataset = TensorDataset(X_train_t, y_train_shuffled_t)
shuffled_loader = DataLoader(shuffled_dataset, batch_size=256, shuffle=True)

shuffled_optimizer = torch.optim.Adam(shuffled_model.parameters(), lr=0.001)
shuffled_model.train()
for epoch in range(5):
    running_loss = 0.0
    for batch_x, batch_y in shuffled_loader:
        shuffled_optimizer.zero_grad()
        logits = shuffled_model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(shuffled_model.parameters(), max_norm=1.0)
        shuffled_optimizer.step()
        running_loss += loss.item() * batch_x.size(0)
    epoch_loss = running_loss / len(shuffled_loader.dataset)
    print(f"Shuffled Epoch {epoch+1:02d}/05 - Loss: {epoch_loss:.6f}")

print("Evaluating shuffled model on the test set...")
shuffle_metrics = evaluate_model(shuffled_model, X_test, y_test)
print(f"Sanity Check F1 Score: {shuffle_metrics['f1']:.5f} (Expected F1 ~ 0.5)")
print(f"Sanity Check ROC-AUC:  {shuffle_metrics['roc_auc']:.5f}")

if shuffle_metrics['f1'] > 0.65 or shuffle_metrics['roc_auc'] > 0.65:
    print("\n[Sanity Check] WARNING: High performance on shuffled labels! Potential leakage remains.")
else:
    print("\n[Sanity Check] SUCCESS: Model performs close to random guessing. Valid feature associations confirmed.")

# -------------------------------------------------------------------------
# SAVING OUTPUTS
# -------------------------------------------------------------------------
print("\n" + "=" * 50)
print("             11. SAVING OUTPUTS")
print("=" * 50)

# Save Model State Dict
model_save_path = "models/lnn_multi.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model weights saved to: {model_save_path}")

# Save metrics CSV
results_save_path = "outputs/csv/multi_results.csv"
tn, fp, fn, tp = test_metrics['confusion_matrix']
metrics_df = pd.DataFrame([{
    "Precision": test_metrics["precision"],
    "Recall": test_metrics["recall"],
    "F1_Score": test_metrics["f1"],
    "ROC_AUC": test_metrics["roc_auc"],
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "TP": tp,
    "Inference_Latency_ms_per_sample": test_metrics["latency_ms_per_sample"],
    "Shuffle_Test_F1": shuffle_metrics["f1"],
    "Shuffle_Test_ROC_AUC": shuffle_metrics["roc_auc"]
}])
metrics_df.to_csv(results_save_path, index=False)
print(f"Validation metrics saved to: {results_save_path}")
print("=" * 50)

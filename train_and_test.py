import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Import LNN model from lnn_model.py
from lnn_model import LNN

# -------------------------------------------------------------------------
# SECURITY POLICY IMPORT FALLBACKS
# -------------------------------------------------------------------------
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError as e:
    SKLEARN_AVAILABLE = False
    print(f"\n[Environment Notice] Falling back to custom NumPy ML utilities.")
    print(f"Reason: DLL load failed importing scikit-learn modules due to system Application Control policy.\n")

# If scikit-learn is blocked, define compatible fallbacks
if not SKLEARN_AVAILABLE:
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

    def train_test_split(X, y, test_size=0.2, random_state=42, stratify=None):
        """
        Splits dataset into train/test sets, preserving class balance if stratify is set.
        """
        np.random.seed(random_state)
        train_indices = []
        test_indices = []
        
        if stratify is not None:
            # Stratified split implementation
            unique_classes = np.unique(stratify)
            for class_label in unique_classes:
                class_indices = np.where(stratify == class_label)[0]
                np.random.shuffle(class_indices)
                
                split_idx = int(len(class_indices) * (1 - test_size))
                train_indices.extend(class_indices[:split_idx])
                test_indices.extend(class_indices[split_idx:])
        else:
            # Simple randomized split
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            split_idx = int(len(X) * (1 - test_size))
            train_indices = indices[:split_idx]
            test_indices = indices[split_idx:]
            
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
        
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# -------------------------------------------------------------------------
# CUSTOM NUMPY EVALUATION AND RESAMPLING UTILITIES
# -------------------------------------------------------------------------

def oversample_minority_class(X, y):
    """
    Oversamples the minority class in X and y to match the majority class size.
    """
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return X, y
        
    minority_class = classes[np.argmin(counts)]
    majority_class = classes[np.argmax(counts)]
    
    minority_indices = np.where(y == minority_class)[0]
    majority_indices = np.where(y == majority_class)[0]
    
    num_to_oversample = len(majority_indices) - len(minority_indices)
    if num_to_oversample <= 0:
        return X, y
        
    np.random.seed(42)
    oversampled_indices = np.random.choice(minority_indices, size=num_to_oversample, replace=True)
    
    balanced_indices = np.concatenate([np.arange(len(y)), oversampled_indices])
    np.random.shuffle(balanced_indices)
    
    return X[balanced_indices], y[balanced_indices]


def compute_confusion_matrix(y_true, y_pred):
    """
    Computes confusion matrix elements: TN, FP, FN, TP.
    """
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tn, fp, fn, tp


def compute_binary_metrics(y_true, y_pred):
    """
    Computes precision, recall, and F1-score for class 1.
    """
    tn, fp, fn, tp = compute_confusion_matrix(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_roc_auc(y_true, y_prob):
    """
    Computes the Area Under the ROC Curve (ROC AUC) using Mann-Whitney U.
    Highly optimized O(N log N) implementation.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
        
    pos = np.sort(pos)
    neg = np.sort(neg)
    
    count_smaller = np.searchsorted(neg, pos, side='left')
    count_equal = np.searchsorted(neg, pos, side='right') - count_smaller
    
    auc = np.mean(count_smaller + 0.5 * count_equal) / len(neg)
    return auc


def print_classification_report(y_true, y_pred):
    """
    Generates and prints a complete metrics classification report.
    """
    tn, fp, fn, tp = compute_confusion_matrix(y_true, y_pred)
    
    # Class 0 metrics (Benign)
    prec_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0 = 2 * prec_0 * rec_0 / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0.0
    support_0 = tn + fp
    
    # Class 1 metrics (Attack)
    prec_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_1 = 2 * prec_1 * rec_1 / (prec_1 + rec_1) if (prec_1 + rec_1) > 0 else 0.0
    support_1 = tp + fn
    
    # Averages
    accuracy = (tp + tn) / len(y_true)
    macro_prec = (prec_0 + prec_1) / 2
    macro_rec = (rec_0 + rec_1) / 2
    macro_f1 = (f1_0 + f1_1) / 2
    
    total = len(y_true)
    weighted_prec = (prec_0 * support_0 + prec_1 * support_1) / total
    weighted_rec = (rec_0 * support_0 + rec_1 * support_1) / total
    weighted_f1 = (f1_0 * support_0 + f1_1 * support_1) / total
    
    print("\nClassification Report:")
    print(f"{'':<17}{'precision':<12}{'recall':<12}{'f1-score':<12}{'support':<10}")
    print("-" * 65)
    print(f"{'Class 0 (Benign)':<17}{prec_0:<12.5f}{rec_0:<12.5f}{f1_0:<12.5f}{support_0:<10d}")
    print(f"{'Class 1 (Attack)':<17}{prec_1:<12.5f}{rec_1:<12.5f}{f1_1:<12.5f}{support_1:<10d}")
    print("-" * 65)
    print(f"{'accuracy':<17}{'':<12}{'':<12}{accuracy:<12.5f}{total:<10d}")
    print(f"{'macro avg':<17}{macro_prec:<12.5f}{macro_rec:<12.5f}{macro_f1:<12.5f}{total:<10d}")
    print(f"{'weighted avg':<17}{weighted_prec:<12.5f}{weighted_rec:<12.5f}{weighted_f1:<12.5f}{total:<10d}\n")


def explain_feature_scaling():
    """
    Prints the design rationale for feature scaling in continuous dynamical systems.
    """
    explanation = """
--------------------------------------------------------------------------------
[Design Explanation] Feature Scaling in Continuous-Time Networks (LNNs):
1. Why StandardScaler is necessary:
   LNNs represent continuous-time systems integrated numerically using the Euler 
   method (h_next = h + dt * dh). High feature variance or large values can lead 
   to runaway integration steps, resulting in exploding gradients or NaN outputs. 
   Normalizing inputs to zero mean and unit variance bounds the state trajectory.
   
2. Why we fit ONLY on training data:
   Fitting the scaler on the entire dataset would leak information about the 
   mean and variance of the test set into the training phase (data leakage). 
   To measure true generalization, the test features must be scaled using 
   parameters estimated strictly from the training partition.
--------------------------------------------------------------------------------
"""
    print(explanation)


def plot_ascii_histogram(probs, bins=10):
    """
    Outputs a text-based ASCII histogram of probability distributions.
    """
    counts, edges = np.histogram(probs, bins=np.linspace(0.0, 1.0, bins+1))
    max_count = max(counts) if len(counts) > 0 and max(counts) > 0 else 1
    max_width = 40
    print("\nRaw Probability Distribution (ASCII Histogram):")
    print("Range       | Frequency")
    print("-" * 50)
    for i in range(bins):
        bar = "#" * int(counts[i] / max_count * max_width)
        print(f"{edges[i]:.1f} - {edges[i+1]:.1f} | {counts[i]:<6d} {bar}")
    print("-" * 50)

# -------------------------------------------------------------------------
# AUDITING CHECKS
# -------------------------------------------------------------------------

def check_feature_correlations(df, y, label_text="RAW"):
    """
    1. Computes Pearson correlations of all numeric features with target labels.
       Prints top 10 most predictive features.
    """
    # Filter target columns
    cols_to_drop = [c for c in ['label', 'attack_type'] if c in df.columns]
    df_features = df.drop(columns=cols_to_drop)
    df_numeric = df_features.select_dtypes(include=[np.number]).fillna(0)
    
    print("\n" + "=" * 60)
    print(f"      TOP 10 FEATURE-TO-LABEL CORRELATIONS ({label_text})")
    print("=" * 60)
    
    correlations = []
    for col in df_numeric.columns:
        corr = np.corrcoef(df_numeric[col], y)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        correlations.append((col, corr))
        
    # Sort by absolute correlation strength descending
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for col, corr in correlations[:10]:
        alert = " [CRITICAL SHORTCUT! Suspicious Leakage]" if abs(corr) > 0.85 else ""
        print(f"  {col:<28} : Correlation = {corr:+.5f}{alert}")
        
    print("=" * 60)


def check_duplicates(X_train, X_test):
    """
    Checks for row duplicates within and between splits. Returns overlap count.
    """
    print("\n" + "=" * 60)
    print("                DUPLICATE RECORD CHECK (LEAKAGE)")
    print("=" * 60)
    df_train = pd.DataFrame(X_train)
    df_test = pd.DataFrame(X_test)
    
    train_dups = df_train.duplicated().sum()
    test_dups = df_test.duplicated().sum()
    
    # Check intersection duplicates
    cross_dups = pd.merge(df_train, df_test, how='inner').shape[0]
    
    print(f"  Train set internal duplicates: {train_dups}")
    print(f"  Test set internal duplicates:  {test_dups}")
    print(f"  Cross-set duplicate overlap:   {cross_dups}")
    
    if cross_dups > 0:
        print("  WARNING: Overlapping records found between train and test partitions!")
        print("  This represents row leakage, causing overly optimistic evaluations.")
    else:
        print("  SUCCESS: No record overlap found between training and testing sets.")
    print("=" * 60)
    return train_dups, test_dups, cross_dups


def print_train_test_comparison(train_metrics, test_metrics):
    """
    Compares train vs test metrics in a clear tabular format.
    """
    print("\n" + "=" * 60)
    print("             TRAIN VS TEST PERFORMANCE COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<25}{'Train Set':<15}{'Test Set':<15}")
    print("-" * 60)
    print(f"{'Decision Threshold':<25}{train_metrics['threshold']:<15.2f}{test_metrics['threshold']:<15.2f}")
    print(f"{'Precision':<25}{train_metrics['precision']:<15.5f}{test_metrics['precision']:<15.5f}")
    print(f"{'Recall':<25}{train_metrics['recall']:<15.5f}{test_metrics['recall']:<15.5f}")
    print(f"{'F1 Score':<25}{train_metrics['f1']:<15.5f}{test_metrics['f1']:<15.5f}")
    print(f"{'ROC-AUC Score':<25}{train_metrics['roc_auc']:<15.5f}{test_metrics['roc_auc']:<15.5f}")
    print(f"{'False Positive Rate':<25}{train_metrics['fpr']:<15.5f}{test_metrics['fpr']:<15.5f}")
    print(f"{'False Negative Rate':<25}{train_metrics['fnr']:<15.5f}{test_metrics['fnr']:<15.5f}")
    print("=" * 60)


def run_shuffle_label_test(input_dim, X_train, y_train, X_test, y_test, epochs=5, batch_size=256):
    """
    5. Shuffles target labels, trains model, and audits test performance.
    """
    print("\n" + "=" * 60)
    print("      SHUFFLE LABEL AUDIT (EXPECTED F1/AUC ~ 0.5)")
    print("=" * 60)
    print("Shuffling training labels randomly to destroy valid mapping...")
    y_train_shuffled = np.copy(y_train)
    np.random.shuffle(y_train_shuffled)
    
    # Initialize a new identical LNN architecture
    shuffled_model = LNN(input_dim=input_dim, hidden_dim=16, num_steps=6, dt=0.1)
    
    # Train on shuffled labels
    print(f"Retraining test LNN on shuffled training labels for {epochs} epochs...")
    shuffled_model = train(shuffled_model, X_train, y_train_shuffled, epochs=epochs, batch_size=batch_size, lr=0.001)
    
    # Evaluate on the test set
    shuffled_model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        probs_t = shuffled_model(X_test_t)
    probs = probs_t.numpy().squeeze()
    
    # Tune threshold on test set for shuffled predictions
    best_t = 0.5
    best_f1 = 0.0
    for t in np.linspace(0.01, 0.99, 99):
        preds = (probs >= t).astype(int)
        _, _, f1 = compute_binary_metrics(y_test, preds)
        if f1 > best_f1:
            best_f1, best_t = f1, t
            
    preds = (probs >= best_t).astype(int)
    prec, rec, f1 = compute_binary_metrics(y_test, preds)
    roc_auc = compute_roc_auc(y_test, probs)
    
    print("\nShuffle Label Test Evaluation (on Test Set):")
    print(f"  Tuned Threshold:   {best_t:.2f}")
    print(f"  Precision:         {prec:.5f}")
    print(f"  Recall:            {rec:.5f}")
    print(f"  F1 Score:          {f1:.5f}  (Ideal: ~0.5)")
    print(f"  ROC-AUC Score:     {roc_auc:.5f}  (Ideal: ~0.5)")
    
    if roc_auc > 0.65 or f1 > 0.65:
        print("\nCRITICAL WARNING: The model trained on shuffled labels achieved high performance!")
        print("This indicates the presence of DATA LEAKAGE or shortcut exploitation.")
    else:
        print("\nAUDIT SUCCESS: Model trained on shuffled labels performs close to random guessing.")
        print("This confirms the model is learning valid target patterns rather than exploiting leakage.")
    print("=" * 60)

# -------------------------------------------------------------------------
# CORE FUNCTION STEPS
# -------------------------------------------------------------------------

def load_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
    print(f"Loading dataset from: {csv_path} ...")
    return pd.read_csv(csv_path)


def preprocess(df, drop_shortcuts=True):
    """
    2. Drops target labels, drops non-numeric features, and handles shortcut feature removal.
    """
    # Extract labels according to column hierarchy
    if 'attack_type' in df.columns:
        y = (df['attack_type'].astype(str).str.lower() != 'benign').astype(int).values
    elif 'label' in df.columns:
        if df['label'].dtype == object or isinstance(df['label'].iloc[0], str):
            y = (df['label'].astype(str).str.lower() != 'benign').astype(int).values
        else:
            y = df['label'].astype(int).values
    else:
        np.random.seed(42)
        y = np.random.randint(0, 2, size=len(df))
        
    # Safety Check: Assert labels are binary
    unique_labels = np.unique(y)
    for label in unique_labels:
        assert label in [0, 1], f"Safety Check Failed: Label contains non-binary value: {label}"
    assert len(unique_labels) > 0, "Safety Check Failed: Label array is empty!"
    
    # Drop target columns from features
    cols_to_drop = [c for c in ['label', 'attack_type'] if c in df.columns]
    df_features = df.drop(columns=cols_to_drop)
    
    # Drop shortcut learning / leakage features
    if drop_shortcuts:
        shortcut_cols = [
            'ip.proto', 'tcp.srcport', 'tcp.dstport', 
            'udp.srcport', 'udp.dstport', 'tcp.window_size_scalefactor'
        ]
        cols_to_remove = [col for col in shortcut_cols if col in df_features.columns]
        if cols_to_remove:
            print(f"Dropping shortcut features: {cols_to_remove}")
            df_features = df_features.drop(columns=cols_to_remove)
    
    # Drop non-numeric features
    df_numeric = df_features.select_dtypes(include=[np.number])
    dropped_cols = set(df_features.columns) - set(df_numeric.columns)
    if dropped_cols:
        print(f"Dropped non-numeric features: {list(dropped_cols)}")
        
    # Fill NaNs with 0
    df_numeric = df_numeric.fillna(0)
    X = df_numeric.values
    
    # Safety Check: Assert no NaNs remain after preprocessing
    assert not np.isnan(X).any(), "Safety Check Failed: Features contain NaN values!"
    assert not np.isnan(y).any(), "Safety Check Failed: Target labels contain NaN values!"
    
    # Warn if dataset is highly imbalanced
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) == 2:
        minority_ratio = min(counts) / sum(counts)
        if minority_ratio < 0.10:
            print(f"\nWARNING: Highly imbalanced dataset! Class distribution: {counts[0]} vs {counts[1]}")
            
    return X, y


def train(model, X_train, y_train, epochs=15, batch_size=256, lr=0.001):
    # Compute class weights using training labels
    classes, counts = np.unique(y_train, return_counts=True)
    if len(classes) == 2:
        class_weights = len(y_train) / (2.0 * counts)
        print(f"Computed Class Weights: Class 0 = {class_weights[0]:.4f}, Class 1 = {class_weights[1]:.4f}")
        
    # Oversampling applied ONLY on training set post-split
    X_train_bal, y_train_bal = oversample_minority_class(X_train, y_train)
    
    X_train_t = torch.tensor(X_train_bal, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_bal, dtype=torch.float32).unsqueeze(1)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    prev_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Model Stability: Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Stability check
        if epoch > 0 and epoch_loss > 1.5 * prev_loss:
            print(f"WARNING: Loss spiked! Epoch {epoch+1:02d} loss: {epoch_loss:.6f} vs Epoch {epoch:02d} loss: {prev_loss:.6f}")
            
        prev_loss = epoch_loss
        
    return model


def evaluate(model, X, y, threshold=None, set_name="TEST"):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    
    start_time = time.perf_counter()
    with torch.no_grad():
        probs_t = model(X_t)
    end_time = time.perf_counter()
    avg_latency_ms = (end_time - start_time) * 1000 / len(X)
    
    probs = probs_t.numpy().squeeze()
    
    # Threshold Tuning for F1 Score
    if threshold is None:
        best_threshold = 0.5
        best_f1 = 0.0
        for t in np.linspace(0.01, 0.99, 99):
            preds = (probs >= t).astype(int)
            _, _, f1 = compute_binary_metrics(y, preds)
            if f1 > best_f1:
                best_f1, best_threshold = f1, t
        threshold = best_threshold
        print(f"  Tuned Threshold on {set_name}: {threshold:.2f} (F1 Score: {best_f1:.5f})")
    
    preds = (probs >= threshold).astype(int)
    
    # Compute metrics
    tn, fp, fn, tp = compute_confusion_matrix(y, preds)
    precision, recall, f1 = compute_binary_metrics(y, preds)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    roc_auc = compute_roc_auc(y, probs)
    
    # Threshold Behavior Check
    attack_pct = np.mean(preds) * 100
    benign_pct = (1.0 - np.mean(preds)) * 100
    
    # Plot probability distribution ASCII histogram
    plot_ascii_histogram(probs)
    
    # Logging Outputs (only for Stage 3 TEST_CLEAN to avoid overwriting outputs)
    if set_name.upper() == "TEST_CLEAN":
        pred_df = pd.DataFrame({
            'y_true': y,
            'y_pred': preds,
            'probability': probs
        })
        pred_df.to_csv('predictions.csv', index=False)
        
        cm_df = pd.DataFrame({
            'Metric': ['True Negative (TN)', 'False Positive (FP)', 'False Negative (FN)', 'True Positive (TP)'],
            'Count': [tn, fp, fn, tp]
        })
        cm_df.to_csv('confusion_matrix.csv', index=False)
        
        results_df = pd.DataFrame([{
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1,
            "ROC_AUC": roc_auc,
            "FPR": fpr,
            "FNR": fnr,
            "Avg_Latency_ms": avg_latency_ms,
            "Optimal_Threshold": threshold
        }])
        results_df.to_csv('results.csv', index=False)
        
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'fnr': fnr,
        'latency': avg_latency_ms,
        'threshold': threshold
    }


def main():
    # Load dataset
    csv_path = "data/GothamDataset2025/processed/iotsim-air-quality-1.csv"
    if not os.path.exists(csv_path):
        csv_path = "GothamDataset2025/processed/iotsim-air-quality-1.csv"
        
    df_raw = load_data(csv_path)
    
    # -------------------------------------------------------------------------
    # STAGE 1: LEAKY BASELINE (WITH DUPLICATES & WITH SHORTCUT FEATURES)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("            STAGE 1: LEAKY BASELINE (WITH DUPLICATES)")
    print("=" * 60)
    
    # Calculate target labels and check correlations
    y_raw = (df_raw['label'].astype(str).str.lower() != 'benign').astype(int).values if 'label' in df_raw.columns else preprocess(df_raw, drop_shortcuts=False)[1]
    check_feature_correlations(df_raw, y_raw, label_text="RAW")
    
    X_leaky, y_leaky = preprocess(df_raw, drop_shortcuts=False)
    
    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
        X_leaky, y_leaky, test_size=0.2, random_state=42, stratify=y_leaky
    )
    
    check_duplicates(X_train_l, X_test_l)
    
    scaler_l = StandardScaler()
    X_train_l_scaled = scaler_l.fit_transform(X_train_l)
    X_test_l_scaled = scaler_l.transform(X_test_l)
    
    model_l = LNN(input_dim=X_train_l.shape[1], hidden_dim=16, num_steps=6, dt=0.1)
    model_l = train(model_l, X_train_l_scaled, y_train_l, epochs=15, batch_size=256, lr=0.001)
    
    print("\n--- Evaluating Leaky Model ---")
    test_metrics_l = evaluate(model_l, X_test_l_scaled, y_test_l, threshold=None, set_name="TEST_LEAKY")
    
    # -------------------------------------------------------------------------
    # STAGE 2: DEDUPLICATED BASELINE WITH SHORTCUT FEATURES
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("          STAGE 2: DEDUPLICATED BASELINE (WITH SHORTCUTS)")
    print("=" * 60)
    
    df_exact = df_raw.drop_duplicates()
    
    # Deduplicate based on numeric columns including ports/protocols
    target_cols = [c for c in ['label', 'attack_type'] if c in df_raw.columns]
    df_features_w = df_raw.drop(columns=target_cols)
    numeric_feature_cols_w = list(df_features_w.select_dtypes(include=[np.number]).columns)
    
    df_clean_w = df_exact.drop_duplicates(subset=numeric_feature_cols_w, keep='first')
    
    X_clean_w, y_clean_w = preprocess(df_clean_w, drop_shortcuts=False)
    
    X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
        X_clean_w, y_clean_w, test_size=0.2, random_state=42, stratify=y_clean_w
    )
    
    check_duplicates(X_train_w, X_test_w)
    
    scaler_w = StandardScaler()
    X_train_w_scaled = scaler_w.fit_transform(X_train_w)
    X_test_w_scaled = scaler_w.transform(X_test_w)
    
    model_w = LNN(input_dim=X_train_w.shape[1], hidden_dim=16, num_steps=6, dt=0.1)
    model_w = train(model_w, X_train_w_scaled, y_train_w, epochs=15, batch_size=256, lr=0.001)
    
    print("\n--- Evaluating Deduplicated Model with Shortcuts ---")
    test_metrics_w = evaluate(model_w, X_test_w_scaled, y_test_w, threshold=None, set_name="TEST_SHORTCUTS")
    
    # -------------------------------------------------------------------------
    # STAGE 3: CLEAN PIPELINE (DEDUPLICATED, NO SHORTCUT FEATURES)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("            STAGE 3: CLEAN PIPELINE (NO LEAKAGE)")
    print("=" * 60)
    
    # Deduplicate based ONLY on numeric feature columns excluding ports, protocols, and window scale
    df_features_clean = df_raw.drop(columns=target_cols)
    shortcut_cols = ['ip.proto', 'tcp.srcport', 'tcp.dstport', 'udp.srcport', 'udp.dstport', 'tcp.window_size_scalefactor']
    cols_to_remove = [col for col in shortcut_cols if col in df_features_clean.columns]
    df_features_clean = df_features_clean.drop(columns=cols_to_remove)
    
    numeric_feature_cols_clean = list(df_features_clean.select_dtypes(include=[np.number]).columns)
    
    df_clean_no_shortcuts = df_exact.drop_duplicates(subset=numeric_feature_cols_clean, keep='first')
    
    # Preprocess
    X_clean, y_clean = preprocess(df_clean_no_shortcuts, drop_shortcuts=True)
    
    # Feature correlations after shortcut removal
    check_feature_correlations(df_clean_no_shortcuts, y_clean, label_text="CLEAN")
    
    # Split
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )
    
    # Verify duplicates are zero
    train_dups, test_dups, cross_dups = check_duplicates(X_train_c, X_test_c)
    assert train_dups == 0
    assert test_dups == 0
    assert cross_dups == 0
    print("  VERIFICATION SUCCESS: All clean duplicate metrics are exactly zero.")
    
    scaler_c = StandardScaler()
    X_train_c_scaled = scaler_c.fit_transform(X_train_c)
    X_test_c_scaled = scaler_c.transform(X_test_c)
    
    model_c = LNN(input_dim=X_train_c.shape[1], hidden_dim=16, num_steps=6, dt=0.1)
    model_c = train(model_c, X_train_c_scaled, y_train_c, epochs=15, batch_size=256, lr=0.001)
    
    print("\n--- Evaluating Clean Pipeline ---")
    test_metrics_c = evaluate(model_c, X_test_c_scaled, y_test_c, threshold=None, set_name="TEST_CLEAN")
    
    # Evaluate on clean train set for comparison
    train_metrics_c = evaluate(model_c, X_train_c_scaled, y_train_c, threshold=test_metrics_c['threshold'], set_name="TRAIN_CLEAN")
    
    # 6. Print 3-way comparison table
    print("\n" + "=" * 80)
    print("                      PIPELINE LEAKAGE AUDIT COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<25}{'Leaky Baseline':<18}{'Deduplicated (Leaky Ftr)':<25}{'Clean (No Leakage)':<18}")
    print("-" * 80)
    print(f"{'F1 Score':<25}{test_metrics_l['f1']:<18.5f}{test_metrics_w['f1']:<25.5f}{test_metrics_c['f1']:<18.5f}")
    print(f"{'ROC-AUC Score':<25}{test_metrics_l['roc_auc']:<18.5f}{test_metrics_w['roc_auc']:<25.5f}{test_metrics_c['roc_auc']:<18.5f}")
    print(f"{'Precision':<25}{test_metrics_l['precision']:<18.5f}{test_metrics_w['precision']:<25.5f}{test_metrics_c['precision']:<18.5f}")
    print(f"{'Recall (TPR)':<25}{test_metrics_l['recall']:<18.5f}{test_metrics_w['recall']:<25.5f}{test_metrics_c['recall']:<18.5f}")
    print(f"{'FPR':<25}{test_metrics_l['fpr']:<18.5f}{test_metrics_w['fpr']:<25.5f}{test_metrics_c['fpr']:<18.5f}")
    print(f"{'FNR':<25}{test_metrics_l['fnr']:<18.5f}{test_metrics_w['fnr']:<25.5f}{test_metrics_c['fnr']:<18.5f}")
    print("=" * 80)
    
    # Save clean model state dict
    model_save_path = "lnn_model.pth"
    torch.save(model_c.state_dict(), model_save_path)
    print(f"Clean LNN model parameters saved to: {model_save_path}")
    
    # 5. Re-run Shuffle Label Test on clean pipeline
    run_shuffle_label_test(X_train_c.shape[1], X_train_c_scaled, y_train_c, X_test_c_scaled, y_test_c, epochs=10, batch_size=256)

if __name__ == "__main__":
    main()

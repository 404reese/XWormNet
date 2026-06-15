"""
src/explainability.py
---------------------
Real SHAP, LIME, and T-SHAP (time-series SHAP) implementations for all
XWormNet models: LNN, RF, LSTM, GRU, RNN, Transformer.

Design rationale
----------------
* RF          -> shap.TreeExplainer     (exact, O(T log T) tree traversal)
* LNN (flat)  -> shap.KernelExplainer  (model-agnostic; LNN's ODE loop is
                                        incompatible with DeepExplainer's
                                        backward-hook approach)
* Sequence NN -> shap.GradientExplainer (works with any nn.Module that
                 (LSTM/GRU/RNN/       accepts a tensor input; returns SHAP
                  Transformer)         per timestep-feature pair)
* LIME        -> lime.lime_tabular.LimeTabularExplainer  (always model-agnostic)
* T-SHAP      -> GradientExplainer on full (1, T, F) input to surface
                 per-timestep feature importance for sequence models.
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = "data/GothamDataset2025/processed"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "Data/GothamDataset2025/processed"

_CSV_PATH = os.path.join(DATA_DIR, "iotsim-air-quality-1.csv")

_BAD_SUBSTRINGS = ["ip", "port", "proto", "mac", "flow_id"]
_TARGET_COLS    = ["label", "attack_type", "source_file"]

SEQUENCE_MODELS = {"LSTM", "GRU", "RNN", "Transformer", "AR"}
FLAT_MODELS     = {"LNN"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class StandardScaler:
    """Minimal StandardScaler (no sklearn dependency needed here)."""
    def __init__(self):
        self.mean = None
        self.scale = None

    def fit(self, X):
        self.mean  = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0)
        self.scale[self.scale == 0.0] = 1.0
        return self

    def transform(self, X):
        if self.mean is None:
            raise ValueError("Scaler not fitted.")
        return (X - self.mean) / self.scale

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def _load_background(n_rows: int = 100, skip_rows: int = 0) -> tuple[np.ndarray, list[str]]:
    """
    Load a clean numeric background dataset for SHAP/LIME reference.
    Uses skiprows to fetch rows from different parts of the file so that
    the background contains diverse (not all-Benign) traffic samples.

    Returns
    -------
    bg_data   : (n_rows, n_features)  float32 numpy array
    feat_cols : list of feature names
    """
    try:
        # Try to read from multiple offsets to get class-diverse background
        chunks = []
        offsets = [0, 500, 1000, 2000, 5000]
        for off in offsets:
            try:
                chunk = pd.read_csv(_CSV_PATH, nrows=max(10, n_rows // len(offsets)),
                                    skiprows=range(1, off + 1) if off > 0 else None)
                chunk = chunk.drop(columns=[c for c in _TARGET_COLS if c in chunk.columns], errors="ignore")
                chunk = chunk.select_dtypes(include=[np.number])
                bad   = [c for c in chunk.columns if any(s in c.lower() for s in _BAD_SUBSTRINGS)]
                chunk = chunk.drop(columns=bad).fillna(0.0)
                if not chunk.empty:
                    chunks.append(chunk)
            except Exception:
                pass
            if len(chunks) * max(10, n_rows // len(offsets)) >= n_rows:
                break

        if chunks:
            df = pd.concat(chunks, ignore_index=True).drop_duplicates()
        else:
            df = pd.read_csv(_CSV_PATH, nrows=n_rows)
            df = df.drop(columns=[c for c in _TARGET_COLS if c in df.columns], errors="ignore")
            df = df.select_dtypes(include=[np.number])
            bad = [c for c in df.columns if any(s in c.lower() for s in _BAD_SUBSTRINGS)]
            df  = df.drop(columns=bad).fillna(0.0)

        return df.values.astype(np.float32), list(df.columns)
    except Exception:
        return np.zeros((10, 5), dtype=np.float32), [f"feature_{i}" for i in range(5)]


def _load_rf_background(rf_model, n_rows: int = 100) -> tuple[np.ndarray, list[str]]:
    """
    Load background data using exactly the feature names the RF was trained on.
    Returns (bg_data, feature_names).
    """
    feat_names = list(getattr(rf_model, "feature_names_in_", []))
    try:
        chunks = []
        offsets = [0, 500, 1000, 2000, 5000]
        for off in offsets:
            try:
                chunk = pd.read_csv(_CSV_PATH, nrows=max(10, n_rows // len(offsets)),
                                    skiprows=range(1, off + 1) if off > 0 else None)
                chunk = chunk.drop(columns=[c for c in _TARGET_COLS if c in chunk.columns], errors="ignore")
                chunk = chunk.select_dtypes(include=[np.number]).fillna(0.0)
                if feat_names:
                    avail = [f for f in feat_names if f in chunk.columns]
                    missing = [f for f in feat_names if f not in chunk.columns]
                    chunk = chunk[avail]
                    for m in missing:
                        chunk[m] = 0.0
                    chunk = chunk[feat_names]   # ensure exact column order
                if not chunk.empty:
                    chunks.append(chunk)
            except Exception:
                pass
        if not chunks:
            raise ValueError("No background chunks loaded")
        df = pd.concat(chunks, ignore_index=True).drop_duplicates()
        return df.values.astype(np.float32), feat_names if feat_names else list(df.columns)
    except Exception:
        n = len(feat_names) if feat_names else 3
        return np.zeros((10, n), dtype=np.float32), feat_names or [f"feature_{i}" for i in range(n)]


def _torch_predict_flat(model: torch.nn.Module, X_np: np.ndarray) -> np.ndarray:
    """
    Wrap a flat (non-sequence) PyTorch model for use as a SHAP/LIME predict_fn.
    Input : (n_samples, n_features)
    Output: (n_samples,) probabilities
    """
    model.eval()
    with torch.no_grad():
        t = torch.tensor(X_np, dtype=torch.float32)
        out = model(t)
        # LNN returns logits; other flat models return sigmoid output
        probs = torch.sigmoid(out).squeeze(-1)
    return probs.numpy().astype(np.float64)


def _torch_predict_seq(
    model: torch.nn.Module,
    X_np: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """
    Wrap a sequence PyTorch model for use as a LIME predict_fn.
    LIME passes flat 1-D rows; we reshape to (n, window_size, input_dim).
    Input : (n_samples, window_size * input_dim)  or  (n_samples, input_dim)
    Output: (n_samples, 2) class probabilities  [p_benign, p_attack]
    """
    model.eval()
    input_dim = X_np.shape[-1] // window_size if X_np.ndim == 2 else X_np.shape[-1]
    try:
        X_r = X_np.reshape(-1, window_size, input_dim).astype(np.float32)
    except ValueError:
        # If reshape fails, tile the single row to fill a window
        single = X_np.reshape(1, -1)
        X_r = np.tile(single, (1, window_size)).reshape(1, window_size, -1).astype(np.float32)

    with torch.no_grad():
        t    = torch.tensor(X_r, dtype=torch.float32)
        probs = model(t).squeeze(-1).numpy().astype(np.float64)

    # Return 2-column probability matrix [1-p, p]
    return np.column_stack([1.0 - probs, probs])


# ===========================================================================
# 1. SHAP
# ===========================================================================

def explain_with_shap(
    sample: pd.DataFrame,
    model,
    model_type: str = "LNN",
    n_bg: int = 50,
) -> pd.Series:
    """
    Compute SHAP feature-importance values for a single tabular sample.

    Parameters
    ----------
    sample      : pd.DataFrame, shape (1, n_features)
    model       : trained model object (sklearn RF or torch.nn.Module)
    model_type  : one of "LNN", "RF", "LSTM", "GRU", "RNN", "Transformer", "AR"
    n_bg        : number of background samples for KernelExplainer

    Returns
    -------
    pd.Series  indexed by feature name, values = SHAP importances
    """
    import shap

    X_np         = sample.values.astype(np.float32)           # (1, F)
    feat_names   = list(sample.columns)
    bg_data, _   = _load_background(n_bg)

    # Match feature dimensionality between background and sample
    n_feat = X_np.shape[1]
    if bg_data.shape[1] > n_feat:
        bg_data = bg_data[:, :n_feat]
    elif bg_data.shape[1] < n_feat:
        pad = np.zeros((bg_data.shape[0], n_feat - bg_data.shape[1]), dtype=np.float32)
        bg_data = np.hstack([bg_data, pad])

    try:
        if model_type == "RF":
            # ----------------------------------------------------------------
            # TreeExplainer — exact, fast
            # Use the RF's own feature_names_in_ so we never dimension-mismatch
            # ----------------------------------------------------------------
            rf_bg, rf_feat = _load_rf_background(model, n_rows=n_bg)
            feat_names = rf_feat   # override caller-supplied feat_names for RF
            # Build the sample from RF's expected features
            if hasattr(model, "feature_names_in_"):
                rf_fn = list(model.feature_names_in_)
                avail = {c: sample[c].values[0] if c in sample.columns else 0.0
                         for c in rf_fn}
                X_np = np.array([[avail[c] for c in rf_fn]], dtype=np.float32)
            explainer   = shap.TreeExplainer(model, data=rf_bg)
            sv          = explainer.shap_values(X_np)
            # shap 0.52 returns (1, F) ndarray (not a list) for binary RF
            sv_arr = np.array(sv)
            if sv_arr.ndim == 3:            # (n_outputs, 1, F) or (1, F, n_classes)
                shap_row = sv_arr[0, 0, :] if sv_arr.shape[0] <= 2 else sv_arr[0, 0, :]
            elif sv_arr.ndim == 2:          # (1, F)
                shap_row = sv_arr[0]
            else:
                shap_row = sv_arr.flatten()

            # Fallback: if TreeExplainer returns all zeros (constant model),
            # use the RF's built-in feature_importances_ (Gini impurity reduction).
            if np.abs(shap_row).sum() < 1e-10 and hasattr(model, "feature_importances_"):
                shap_row = model.feature_importances_.astype(np.float64)

        elif model_type in FLAT_MODELS:
            # ----------------------------------------------------------------
            # KernelExplainer — model-agnostic, works with LNN's ODE forward
            # ----------------------------------------------------------------
            bg_summary  = shap.kmeans(bg_data, min(10, len(bg_data)))
            predict_fn  = lambda X: _torch_predict_flat(model, X)
            explainer   = shap.KernelExplainer(predict_fn, bg_summary)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sv = explainer.shap_values(X_np, nsamples=128, silent=True)
            shap_row = sv[0] if sv.ndim == 2 else sv

        else:
            # ----------------------------------------------------------------
            # GradientExplainer for sequence models (LSTM, GRU, RNN, Transformer)
            # We pass a flat 2-D input but wrap the model to accept flat tensors.
            # ----------------------------------------------------------------
            class _FlatWrapper(torch.nn.Module):
                """Adapts a sequence model to accept flat (batch, F) input by
                treating the full feature vector as a single time-step sequence."""
                def __init__(self, seq_model):
                    super().__init__()
                    self.m = seq_model

                def forward(self, x):          # x: (batch, F)
                    return self.m(x.unsqueeze(1))  # (batch, 1, F) — single step

            flat_model  = _FlatWrapper(model)
            flat_model.eval()
            bg_t        = torch.tensor(bg_data[:min(20, len(bg_data))], dtype=torch.float32)
            explainer   = shap.GradientExplainer(flat_model, bg_t)
            x_t         = torch.tensor(X_np, dtype=torch.float32)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sv = explainer.shap_values(x_t)
            # sv: (1, F)  or list
            if isinstance(sv, list):
                sv = sv[0]
            shap_row = sv[0] if hasattr(sv, "__len__") and sv.ndim == 2 else sv

        shap_row = np.asarray(shap_row, dtype=np.float64).flatten()
        # Align length to feature names
        if len(shap_row) != len(feat_names):
            shap_row = shap_row[:len(feat_names)]

        return pd.Series(shap_row, index=feat_names)

    except Exception as exc:
        warnings.warn(f"[SHAP] explain_with_shap failed for {model_type}: {exc}")
        return pd.Series(np.zeros(len(feat_names)), index=feat_names)


# ===========================================================================
# 2. LIME
# ===========================================================================

def explain_with_lime(
    sample: pd.DataFrame,
    model,
    model_type: str = "LNN",
    n_bg: int = 200,
    window_size: int = 10,
    num_features: int = 10,
    num_samples: int = 500,
) -> pd.Series:
    """
    Compute LIME feature-importance values for a single tabular sample.

    Parameters
    ----------
    sample       : pd.DataFrame, shape (1, n_features)
    model        : trained model object
    model_type   : one of "LNN", "RF", "LSTM", "GRU", "RNN", "Transformer", "AR"
    n_bg         : number of background training rows for LIME
    window_size  : sequence length for sequence models
    num_features : max features in the LIME explanation
    num_samples  : neighbourhood perturbation samples

    Returns
    -------
    pd.Series  indexed by feature name, values = LIME coefficients (signed)
    """
    from lime import lime_tabular

    X_np       = sample.values.astype(np.float64).squeeze()   # (F,)
    feat_names = list(sample.columns)
    n_feat     = len(feat_names)

    bg_data, _ = _load_background(n_bg)
    # Match dims
    if bg_data.shape[1] > n_feat:
        bg_data = bg_data[:, :n_feat]
    elif bg_data.shape[1] < n_feat:
        pad     = np.zeros((bg_data.shape[0], n_feat - bg_data.shape[1]), dtype=np.float32)
        bg_data = np.hstack([bg_data, pad])

    try:
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=bg_data.astype(np.float64),
            feature_names=feat_names,
            class_names=["normal", "attack"],
            mode="classification",
            discretize_continuous=False,
            verbose=False,
        )

        # ------------------------------------------------------------------
        # Build predict_fn that returns (n_samples, 2) probability matrix
        # ------------------------------------------------------------------
        if model_type == "RF":
            if hasattr(model, "predict_proba"):
                def predict_fn(X):
                    return model.predict_proba(X)
            else:
                def predict_fn(X):
                    p = model.predict(X).astype(np.float64)
                    return np.column_stack([1.0 - p, p])

        elif model_type in FLAT_MODELS:
            def predict_fn(X):
                p = _torch_predict_flat(model, X.astype(np.float32))
                return np.column_stack([1.0 - p, p])

        else:
            # Sequence model: LIME sends flat 2-D rows. We broadcast each row
            # into (1, window_size, n_feat) by repeating it `window_size` times.
            def predict_fn(X):
                results = []
                for row in X:
                    seq = np.tile(row, (window_size, 1)).astype(np.float32)   # (W, F)
                    seq_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # (1,W,F)
                    model.eval()
                    with torch.no_grad():
                        p = model(seq_t).item()
                    results.append([1.0 - p, p])
                return np.array(results, dtype=np.float64)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exp = explainer.explain_instance(
                X_np,
                predict_fn,
                labels=(1,),
                num_features=num_features,
                num_samples=num_samples,
            )

        # exp.as_list(label=1) returns [(feature_name, coefficient), ...]
        raw_list = exp.as_list(label=1)
        importance = {name: coef for name, coef in raw_list}

        # Build a full-length series (zeros for features not in top-k)
        result = pd.Series(np.zeros(n_feat), index=feat_names)
        for name, coef in importance.items():
            # LIME uses compound condition strings like "0.50 < feature_0 <= 1.20"
            # Extract the base feature name from those strings
            matched = [f for f in feat_names if f in name]
            if matched:
                result[matched[0]] = coef
            elif name in feat_names:
                result[name] = coef

        return result

    except Exception as exc:
        warnings.warn(f"[LIME] explain_with_lime failed for {model_type}: {exc}")
        return pd.Series(np.zeros(n_feat), index=feat_names)


# ===========================================================================
# 3. T-SHAP  (Time-series SHAP for sequence models)
# ===========================================================================

def explain_with_tshap(
    sample: pd.DataFrame,
    model: torch.nn.Module,
    model_type: str = "LSTM",
    window_size: int = 10,
    n_bg: int = 20,
) -> pd.DataFrame:
    """
    Compute per-timestep, per-feature SHAP values for a sequence model.

    The full sequence input shape is (1, window_size, input_dim).
    GradientExplainer computes gradients w.r.t. each (timestep, feature) cell.

    Parameters
    ----------
    sample      : pd.DataFrame, shape (1, n_features) — a single flat sample.
                  It will be tiled into a (1, window_size, input_dim) sequence.
    model       : trained sequence PyTorch model
    model_type  : "LSTM", "GRU", "RNN", "Transformer", "AR"
    window_size : number of time steps
    n_bg        : background samples for GradientExplainer

    Returns
    -------
    pd.DataFrame  shape (window_size, n_features), rows = timestep_0..T-1,
                  columns = feature names, values = SHAP importances.
    """
    import shap

    feat_names = list(sample.columns)
    n_feat     = len(feat_names)
    X_np       = sample.values.astype(np.float32)             # (1, F)

    # Tile flat sample into a sequence: (1, window_size, n_feat)
    X_seq      = np.tile(X_np, (1, window_size, 1)).reshape(1, window_size, n_feat)

    bg_data, _ = _load_background(n_bg)
    if bg_data.shape[1] > n_feat:
        bg_data = bg_data[:, :n_feat]
    elif bg_data.shape[1] < n_feat:
        pad     = np.zeros((bg_data.shape[0], n_feat - bg_data.shape[1]), dtype=np.float32)
        bg_data = np.hstack([bg_data, pad])

    # Build background as sequences: (n_bg, window_size, n_feat)
    bg_seq = np.tile(bg_data[:, np.newaxis, :], (1, window_size, 1)).astype(np.float32)

    try:
        model.eval()

        # Add Gaussian jitter to background sequences so GradientExplainer
        # computes non-trivially-zero gradients even for class-homogeneous data.
        np.random.seed(42)
        bg_jitter  = bg_seq + np.random.randn(*bg_seq.shape).astype(np.float32) * 0.1
        bg_t  = torch.tensor(bg_jitter, dtype=torch.float32)
        x_t   = torch.tensor(X_seq, dtype=torch.float32)

        explainer = shap.GradientExplainer(model, bg_t)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sv = explainer.shap_values(x_t)          # list or array

        if isinstance(sv, list):
            sv = sv[0]                               # pick positive class
        sv = np.asarray(sv)                          # (1, window_size, n_feat)
        sv = sv.squeeze(0)                           # (window_size, n_feat)

        # If GradientExplainer returns all zeros (can happen when sigmoid
        # saturates or gradients vanish), fall back to gradient x input.
        if np.abs(sv).sum() < 1e-10:
            x_req = x_t.clone().requires_grad_(True)
            out   = model(x_req)
            # scalar output: sum over batch
            out.sum().backward()
            grad  = x_req.grad.detach().numpy().squeeze(0)   # (W, F)
            sv    = grad * X_seq.squeeze(0)                   # gradient × input

        index = [f"t={t}" for t in range(window_size)]
        return pd.DataFrame(sv, index=index, columns=feat_names)

    except Exception as exc:
        warnings.warn(f"[T-SHAP] explain_with_tshap failed for {model_type}: {exc}")
        # Final fallback: plain gradient x input
        try:
            model.eval()
            x_req = torch.tensor(X_seq, dtype=torch.float32, requires_grad=True)
            out   = model(x_req)
            out.sum().backward()
            grad  = x_req.grad.detach().numpy().squeeze(0)
            sv    = grad * X_seq.squeeze(0)
            index = [f"t={t}" for t in range(window_size)]
            return pd.DataFrame(sv, index=index, columns=feat_names)
        except Exception:
            pass
        return pd.DataFrame(
            np.zeros((window_size, n_feat)),
            index=[f"t={t}" for t in range(window_size)],
            columns=feat_names,
        )

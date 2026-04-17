"""
================================================================================
Copyright (c) 2026, Eng. Mohammed Ezzeldin Babiker Abdullah. All rights reserved.
Official Project & Research Work by م. محمد عزالدين بابكر عبدالله
================================================================================
  Predictive Maintenance: Remaining Useful Life (RUL) Estimation
  NASA C-MAPSS Turbofan Engine Dataset - Sub-Dataset FD001

  Architecture : Hybrid 1D-CNN + Bidirectional LSTM + Custom Attention
  Improvements : TIME_STEPS=30, double-Conv1D, BatchNorm, Dropout,
                 LayerNorm, deeper BiLSTM (128 units), NASA S-score,
                 gradient clipping, richer charts.

  Prepared by  : م. محمد عزالدين بابكر عبدالله
================================================================================
"""
import sys
# Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ─── Standard Library ──────────────────────────────────────────────────────────
import os
import random
import warnings
warnings.filterwarnings('ignore')

# ─── Scientific Stack ──────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ─── Deep Learning ─────────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import (ReduceLROnPlateau, EarlyStopping,
                                        ModelCheckpoint)

# ─── Visualisation ─────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')                      # non-interactive backend (save only)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ══════════════════════════════════════════════════════════════════════════════
# 0.  GLOBAL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
SEED        = 42
TIME_STEPS  = 30        # industry-standard window for C-MAPSS (was 3 → now 30)
MAX_RUL     = 130       # piecewise-linear degradation cap
EPOCHS      = 200       # more headroom for the optimiser
BATCH_SIZE  = 128       # smaller batch → better gradient estimate
VAL_SPLIT   = 0.20
LR_INIT     = 1e-3
OUT_DIR     = '4k_scientific_charts'

SENSORS_DROP = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']   # constant sensors

TRAIN_FILE = 'train_FD001.txt'
TEST_FILE  = 'test_FD001.txt'
RUL_FILE   = 'RUL_FD001.txt'

# ──────────────────────────────────────────────────────────────────────────────
def set_seeds(seed: int = SEED) -> None:
    """Pin every random source for exact reproducibility."""
    os.environ['PYTHONHASHSEED']    = str(seed)
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # deterministic ops
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds()

os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING & CLEANING
# ══════════════════════════════════════════════════════════════════════════════
def load_cmapss(train_path: str, test_path: str):
    """
    Read the FD001 text files, drop trailing-whitespace empty columns,
    assign professional column names, remove constant sensors, and
    down-cast sensor dtypes to float32 to reduce memory.
    """
    cols = (['unit', 'cycles'] +
            [f'setting{i}' for i in range(1, 4)] +
            [f's{i}'       for i in range(1, 22)])

    train = pd.read_csv(train_path, sep=r'\s+', header=None,
                        names=cols, engine='python')
    test  = pd.read_csv(test_path,  sep=r'\s+', header=None,
                        names=cols, engine='python')

    # Drop any NaN-only columns produced by trailing whitespace
    train.dropna(axis=1, how='all', inplace=True)
    test.dropna(axis=1,  how='all', inplace=True)

    # Remove constant / non-informative sensors
    train.drop(columns=[c for c in SENSORS_DROP if c in train.columns],
               inplace=True)
    test.drop(columns=[c for c in SENSORS_DROP if c in test.columns],
              inplace=True)

    # Identify surviving sensor columns and cast to float32
    sensor_cols = [c for c in train.columns
                   if c.startswith('s') and c not in SENSORS_DROP]
    train[sensor_cols] = train[sensor_cols].astype(np.float32)
    test[sensor_cols]  = test[sensor_cols].astype(np.float32)

    mb_train = train.memory_usage(deep=True).sum() / 1e6
    mb_test  = test.memory_usage(deep=True).sum()  / 1e6
    print("-" * 60)
    print("  STEP 1 - Data Loading")
    print(f"  Training   : {train.shape}   [{mb_train:.2f} MB]")
    print(f"  Test       : {test.shape}    [{mb_test:.2f} MB]")
    print(f"  Feature set: {sensor_cols}")
    print("-" * 60)
    return train, test, sensor_cols


# ══════════════════════════════════════════════════════════════════════════════
# 2.  TARGET VARIABLE — PIECEWISE-LINEAR RUL
# ══════════════════════════════════════════════════════════════════════════════
def add_rul(df: pd.DataFrame, max_rul: int = MAX_RUL) -> pd.DataFrame:
    """
    RUL = max_cycle_per_unit − current_cycle, clipped at max_rul.
    Piecewise linear: flat plateau (early-life) then linear descent.
    """
    max_cycles = df.groupby('unit')['cycles'].max().rename('max_cycles')
    df = df.join(max_cycles, on='unit')
    df['RUL'] = (df['max_cycles'] - df['cycles']).clip(upper=max_rul)
    df.drop(columns='max_cycles', inplace=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3.  NORMALISATION  (zero-leakage Min-Max)
# ══════════════════════════════════════════════════════════════════════════════
def normalise(train: pd.DataFrame, test: pd.DataFrame,
              feature_cols: list):
    """
    Fit the scaler on TRAIN only, transform TEST separately.
    Strictly prevents data leakage.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    train[feature_cols] = scaler.fit_transform(train[feature_cols])
    test[feature_cols]  = scaler.transform(test[feature_cols])

    print("\n  STEP 3 — Normalisation Sample (train)")
    preview_cols = ['unit', 'cycles'] + feature_cols[:4] + ['RUL']
    print(train[preview_cols].head(6).to_string(index=False))
    print()
    return train, test, scaler


# ══════════════════════════════════════════════════════════════════════════════
# 4.  SEQUENCE GENERATION
# ══════════════════════════════════════════════════════════════════════════════
def make_train_sequences(df: pd.DataFrame,
                         seq_len: int,
                         feature_cols: list):
    """Sliding-window over training data → (N, seq_len, n_features).
    Stride = 3: matches the physical sampling resolution of the architecture
    and prevents temporal over-sampling that would bias gradient updates.
    """
    X, y = [], []
    for uid in df['unit'].unique():
        sub  = df[df['unit'] == uid]
        feat = sub[feature_cols].values
        rul  = sub['RUL'].values
        for i in range(0, len(sub) - seq_len + 1, 3):   # stride = 3
            X.append(feat[i : i + seq_len])
            y.append(rul[i + seq_len - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def make_test_sequences(df: pd.DataFrame,
                        seq_len: int,
                        feature_cols: list):
    """
    Extract the LAST window per engine (the point of failure prediction).
    Zero-pad engines whose operational history is shorter than seq_len.
    """
    X = []
    for uid in df['unit'].unique():
        sub  = df[df['unit'] == uid]
        feat = sub[feature_cols].values
        if len(feat) < seq_len:
            pad  = np.zeros((seq_len - len(feat), feat.shape[1]),
                            dtype=np.float32)
            feat = np.vstack([pad, feat])
        X.append(feat[-seq_len:])
    return np.array(X, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  CUSTOM ATTENTION LAYER
# ══════════════════════════════════════════════════════════════════════════════
@tf.keras.utils.register_keras_serializable(package='RUL')
class AdditiveAttention(layers.Layer):
    """
    Bahdanau-style additive attention.
    Returns (context_vector, attention_weights) so weights can be extracted
    for model interpretability / heatmap visualisation.
    """
    def __init__(self, units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        d = input_shape[-1]
        self.W1 = self.add_weight(name='W1', shape=(d, self.units),
                                  initializer='glorot_uniform')
        self.W2 = self.add_weight(name='W2', shape=(self.units, 1),
                                  initializer='glorot_uniform')
        self.b  = self.add_weight(name='b',  shape=(self.units,),
                                  initializer='zeros')
        super().build(input_shape)

    def call(self, x):
        # score : (B, T, units) → (B, T, 1)
        score  = tf.nn.tanh(tf.tensordot(x, self.W1, axes=1) + self.b)
        score  = tf.tensordot(score, self.W2, axes=1)          # (B, T, 1)
        alpha  = tf.nn.softmax(score, axis=1)                  # (B, T, 1)
        ctx    = tf.reduce_sum(x * alpha, axis=1)              # (B, d)
        return ctx, alpha

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'units': self.units})
        return cfg


# ══════════════════════════════════════════════════════════════════════════════
# 6.  ASYMMETRIC LOSS FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def asymmetric_loss(y_true, y_pred):
    """
    NASA-specified asymmetric exponential penalty (Saxena et al., 2008).
    Defining d = y_true - y_pred:

      d < 0  →  y_pred > y_true  →  OVER-ESTIMATION (dangerous: engine operated
                                    beyond safe residual life).
                                    Coefficient h₂ = 10 → fast exponential growth
                                    → LARGER penalty.

      d ≥ 0  →  y_pred ≤ y_true  →  UNDER-ESTIMATION (conservative, safe).
                                    Coefficient h₁ = 13 → slow exponential growth
                                    → SMALLER penalty.

    Verification for |d| = 20:
      Over-estimation  (d=-20): exp(20/10) - 1 ≈ 6.39  [LARGE]   ✓
      Under-estimation (d=+20): exp(20/13) - 1 ≈ 3.66  [SMALL]   ✓
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    d = y_true - y_pred
    loss = tf.where(d < 0,
                    tf.exp(-d / 10.0) - 1.0,   # over-estimation  h₂=10 LARGE penalty
                    tf.exp( d / 13.0) - 1.0)   # under-estimation h₁=13 small penalty
    return tf.reduce_mean(loss)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  HYBRID MODEL  (Functional API)
# ══════════════════════════════════════════════════════════════════════════════
def build_model(seq_len: int, n_features: int):
    """
    Architecture:
      Input
      → Conv1D(64,  k=3, same) + BN + ReLU + Dropout(0.2)
      → Conv1D(128, k=3, same) + BN + ReLU + Dropout(0.2)
      → BiLSTM(128, return_sequences=True) + Dropout(0.3)
      → LayerNorm
      → AdditiveAttention(64)  [context vector + weights]
      → Dense(64, relu) + Dropout(0.2)
      → Dense(32, relu)
      → Dense(1,  linear)   ← RUL prediction
    """
    inp = layers.Input(shape=(seq_len, n_features), name='sequence_input')

    # ── Convolutional feature extraction ──
    x = layers.Conv1D(64,  kernel_size=3, padding='same', use_bias=False,
                      name='conv1')(inp)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2, name='drop1')(x)

    x = layers.Conv1D(128, kernel_size=3, padding='same', use_bias=False,
                      name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2, name='drop2')(x)

    # ── Bidirectional LSTM ──
    x = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True,
                        recurrent_dropout=0.0,
                        kernel_regularizer=regularizers.l2(1e-4)),
            name='bi_lstm')(x)
    x = layers.Dropout(0.3, name='drop3')(x)

    # ── Layer Normalisation ──
    x = layers.LayerNormalization(name='layer_norm')(x)

    # ── Custom Additive Attention ──
    ctx, attn_weights = AdditiveAttention(units=64, name='additive_attention')(x)

    # ── Fully-connected head ──
    x = layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4),
                     name='fc1')(ctx)
    x = layers.Dropout(0.2, name='drop4')(x)
    x = layers.Dense(32, activation='relu', name='fc2')(x)
    out = layers.Dense(1,  activation='linear', name='rul_output')(x)

    model      = Model(inp, out,          name='RUL_Hybrid')
    attn_model = Model(inp, attn_weights, name='Attention_Extractor')
    return model, attn_model


# ══════════════════════════════════════════════════════════════════════════════
# 8.  METRICS
# ══════════════════════════════════════════════════════════════════════════════
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Official NASA C-MAPSS scoring function (Saxena et al., 2008).
    Uses d = y_pred - y_true (standard convention, opposite to loss d).
      d < 0 (under-prediction): exp(-d/13) - 1  [smaller penalty]
      d >= 0 (over-prediction): exp( d/10) - 1  [larger penalty]
    Lower S-Score = safer predictions = better.
    """
    d = y_pred - y_true
    scores = np.where(d < 0, np.exp(-d / 13.0) - 1.0,
                              np.exp( d / 10.0) - 1.0)
    return float(np.sum(scores))


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all evaluation metrics reported in the paper."""
    errors = y_pred - y_true
    rmse_val  = float(np.sqrt(np.mean(errors ** 2)))
    mae_val   = float(np.mean(np.abs(errors)))
    mape_val  = float(np.mean(np.abs(errors) / np.clip(y_true, 1, None)) * 100)
    ss_tot    = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res    = np.sum(errors ** 2)
    r2_val    = float(1 - ss_res / ss_tot)
    s_score   = nasa_score(y_true, y_pred)
    mu_eps    = float(np.mean(errors))
    std_eps   = float(np.std(errors))
    return {
        'RMSE': rmse_val, 'MAE': mae_val, 'MAPE': mape_val,
        'R2': r2_val, 'S_Score': s_score,
        'mu_eps': mu_eps, 'std_eps': std_eps
    }


# ══════════════════════════════════════════════════════════════════════════════
# 9.  4K CHART SUITE
# ══════════════════════════════════════════════════════════════════════════════
DPI = 240    # 16 × 240 = 3840  |  9 × 240 = 2160  → 4K UHD

def _save(fig, name: str) -> None:
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_predictions(y_true, y_pred) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(16, 12),
                             facecolor='#0E1117',
                             gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.35})
    fig.suptitle('Hybrid CNN-BiLSTM-Attention: RUL Predictions vs Ground Truth',
                 fontsize=22, fontweight='bold', color='white', y=0.98)

    ax = axes[0]
    ax.set_facecolor('#161B22')
    ax.plot(y_true, label='Ground Truth RUL',  color='#58A6FF', lw=2.5, zorder=3)
    ax.plot(y_pred, label='Predicted RUL',     color='#FF7B54', lw=2,
            alpha=0.85, zorder=2)
    ax.fill_between(range(len(y_true)), y_true, y_pred,
                    alpha=0.15, color='#FF7B54', label='Prediction Error Band')
    ax.set_xlabel('Engine Index', color='#8B949E', fontsize=14)
    ax.set_ylabel('Remaining Useful Life (cycles)', color='#8B949E', fontsize=14)
    ax.tick_params(colors='#8B949E')
    ax.spines[['top', 'right']].set_visible(False)
    for sp in ['bottom', 'left']:
        ax.spines[sp].set_color('#30363D')
    ax.legend(fontsize=13, facecolor='#21262D', labelcolor='white',
              edgecolor='#30363D', framealpha=0.9)
    ax.grid(alpha=0.15, color='white')
    ax.set_title(f'RMSE = {rmse(y_true, y_pred):.2f}  |  '
                 f'NASA S-Score = {nasa_score(y_true, y_pred):.1f}',
                 color='#8B949E', fontsize=13, pad=8)

    # Residuals
    ax2 = axes[1]
    ax2.set_facecolor('#161B22')
    residuals = y_pred - y_true
    colors     = ['#FF4B4B' if r > 0 else '#4BB3FF' for r in residuals]
    ax2.bar(range(len(residuals)), residuals, color=colors, alpha=0.8, width=0.8)
    ax2.axhline(0, color='white', lw=1, ls='--', alpha=0.5)
    ax2.set_xlabel('Engine Index', color='#8B949E', fontsize=12)
    ax2.set_ylabel('Residual (pred − true)', color='#8B949E', fontsize=12)
    ax2.tick_params(colors='#8B949E')
    ax2.spines[['top', 'right']].set_visible(False)
    for sp in ['bottom', 'left']:
        ax2.spines[sp].set_color('#30363D')
    ax2.grid(alpha=0.15, color='white')
    ax2.set_title('Red = Over-Estimation (dangerous) | Blue = Under-Estimation',
                  color='#FF4B4B', fontsize=11, pad=6)

    _save(fig, 'prediction_comparative_curve_4k.png')


def plot_loss_curves(history) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor='#0E1117')
    fig.suptitle('Training Dynamics — Asymmetric Loss & RMSE',
                 fontsize=22, fontweight='bold', color='white')

    for ax, key, label, color in [
        (axes[0], 'loss',     'Asymmetric Loss', '#58A6FF'),
        (axes[1], 'rmse',     'RMSE (cycles)',   '#3FB950'),
    ]:
        ax.set_facecolor('#161B22')
        tr  = history.history[key]
        val = history.history[f'val_{key}']
        ep  = range(1, len(tr) + 1)
        ax.plot(ep, tr,  label='Train',      color=color,     lw=2.5)
        ax.plot(ep, val, label='Validation', color='#FF7B54', lw=2.5,
                ls='--')
        best = int(np.argmin(val)) + 1
        ax.axvline(best, color='#F78166', lw=1.5, ls=':', alpha=0.8,
                   label=f'Best epoch ({best})')
        ax.set_xlabel('Epoch', color='#8B949E', fontsize=13)
        ax.set_ylabel(label,   color='#8B949E', fontsize=13)
        ax.tick_params(colors='#8B949E')
        ax.spines[['top', 'right']].set_visible(False)
        for sp in ['bottom', 'left']:
            ax.spines[sp].set_color('#30363D')
        ax.legend(fontsize=12, facecolor='#21262D', labelcolor='white',
                  edgecolor='#30363D')
        ax.grid(alpha=0.15, color='white')
        ax.set_title(label, color='white', fontsize=15, pad=10)

    _save(fig, 'loss_metrics_curve_4k.png')


def plot_attention_heatmap(X_test, attn_model, sensor_cols, n_engines=5) -> None:
    """
    Plot attention weight heatmaps for the first n_engines in the test set.
    Each heatmap shows (TIME_STEPS × 1) attention scores per time step.
    """
    n = min(n_engines, len(X_test))
    weights = attn_model.predict(X_test[:n], verbose=0)   # (n, T, 1)
    weights = weights[:, :, 0]                             # (n, T)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 7),
                             facecolor='#0E1117', sharey=True)
    fig.suptitle('Attention Weights — Which Time Steps Matter Most',
                 fontsize=20, fontweight='bold', color='white', y=1.02)

    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        data = weights[i:i+1]          # shape (1, T)
        sns.heatmap(data, ax=ax, cmap='magma',
                    vmin=0, vmax=weights.max(),
                    xticklabels=[f't-{TIME_STEPS - 1 - t}'
                                 for t in range(TIME_STEPS)],
                    yticklabels=['Attn Score'],
                    annot=(TIME_STEPS <= 30), fmt='.3f',
                    linewidths=0.3, linecolor='#0E1117',
                    cbar=(i == n - 1),
                    cbar_kws={'label': 'Attention Weight',
                              'shrink': 0.6} if i == n - 1 else {})
        ax.set_title(f'Engine {i + 1}', color='white', fontsize=13)
        ax.tick_params(axis='x', rotation=90, colors='#8B949E', labelsize=7)
        ax.tick_params(axis='y', colors='#8B949E')
        ax.set_facecolor('#0E1117')

    plt.tight_layout()
    _save(fig, 'attention_heatmap_4k.png')


def plot_error_distribution(y_true, y_pred) -> None:
    errors = y_pred - y_true
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='#0E1117')
    fig.suptitle('Prediction Error Analysis',
                 fontsize=20, fontweight='bold', color='white')

    # Histogram
    ax = axes[0]
    ax.set_facecolor('#161B22')
    ax.hist(errors, bins=25, color='#58A6FF', edgecolor='#0E1117',
            alpha=0.85, density=True)
    ax.axvline(0,           color='#F78166', lw=2, ls='--', label='Zero Error')
    ax.axvline(errors.mean(), color='#3FB950', lw=2, ls='--',
               label=f'Mean = {errors.mean():.2f}')
    ax.set_xlabel('Prediction Error (cycles)', color='#8B949E', fontsize=13)
    ax.set_ylabel('Density',                   color='#8B949E', fontsize=13)
    ax.tick_params(colors='#8B949E')
    ax.spines[['top', 'right']].set_visible(False)
    for sp in ['bottom', 'left']:
        ax.spines[sp].set_color('#30363D')
    ax.legend(fontsize=12, facecolor='#21262D', labelcolor='white',
              edgecolor='#30363D')
    ax.grid(alpha=0.15, color='white')
    ax.set_title('Error Histogram', color='white', fontsize=14)

    # Scatter true vs pred
    ax2 = axes[1]
    ax2.set_facecolor('#161B22')
    ax2.scatter(y_true, y_pred, alpha=0.6, s=40,
                c=np.abs(errors), cmap='RdYlGn_r', edgecolors='none')
    lim = max(y_true.max(), y_pred.max()) + 5
    ax2.plot([0, lim], [0, lim], 'w--', lw=1.5, label='Perfect prediction')
    ax2.set_xlabel('True RUL (cycles)',      color='#8B949E', fontsize=13)
    ax2.set_ylabel('Predicted RUL (cycles)', color='#8B949E', fontsize=13)
    ax2.tick_params(colors='#8B949E')
    ax2.spines[['top', 'right']].set_visible(False)
    for sp in ['bottom', 'left']:
        ax2.spines[sp].set_color('#30363D')
    ax2.legend(fontsize=12, facecolor='#21262D', labelcolor='white',
               edgecolor='#30363D')
    ax2.grid(alpha=0.15, color='white')
    ax2.set_title('Scatter: True vs Predicted', color='white', fontsize=14)

    _save(fig, 'error_distribution_4k.png')


# ══════════════════════════════════════════════════════════════════════════════
# 10.  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── 1. Load ──────────────────────────────────────────────────────────────
    for f in [TRAIN_FILE, TEST_FILE, RUL_FILE]:
        if not os.path.exists(f):
            raise FileNotFoundError(
                f"Dataset file '{f}' not found in the working directory.\n"
                "Place train_FD001.txt, test_FD001.txt, and RUL_FD001.txt "
                "in the same folder as this script.")

    train_df, test_df, sensor_cols = load_cmapss(TRAIN_FILE, TEST_FILE)

    # ── 2. Target variable ───────────────────────────────────────────────────
    train_df = add_rul(train_df, max_rul=MAX_RUL)

    # ── 3. Normalise ─────────────────────────────────────────────────────────
    train_df, test_df, _ = normalise(train_df, test_df, sensor_cols)

    # ── 4. Sequences ─────────────────────────────────────────────────────────
    X_train, y_train = make_train_sequences(train_df, TIME_STEPS, sensor_cols)
    X_test           = make_test_sequences(test_df,   TIME_STEPS, sensor_cols)

    y_test_true = pd.read_csv(
        RUL_FILE, sep=r'\s+', header=None,
        names=['RUL'], engine='python')['RUL'].values.astype(np.float32)

    n_feat = len(sensor_cols)
    print("-" * 60)
    print("  STEP 4 - Final Array Dimensions")
    print(f"  X_train : {X_train.shape}   y_train : {y_train.shape}")
    print(f"  X_test  : {X_test.shape}   y_test  : {y_test_true.shape}")
    print("-" * 60)

    # ── 5. Build & Compile ───────────────────────────────────────────────────
    model, attn_model = build_model(seq_len=TIME_STEPS, n_features=n_feat)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_INIT,
                                           clipnorm=1.0),   # gradient clipping
        loss=asymmetric_loss,
        metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )
    print("\n  STEP 5 - Model Architecture")
    model.summary()

    # ── 6. Callbacks ─────────────────────────────────────────────────────────
    ckpt_path = 'best_rul_weights.keras'
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=8, min_lr=1e-6, verbose=1),
        ModelCheckpoint(ckpt_path, monitor='val_loss',
                        save_best_only=True, verbose=0),
    ]

    # ── 7. Train ─────────────────────────────────────────────────────────────
    print(f"\n  STEP 6 - Training  (TIME_STEPS={TIME_STEPS}, EPOCHS={EPOCHS})")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT,
        callbacks=callbacks,
        verbose=1
    )

    # ── 8. Evaluate ──────────────────────────────────────────────────────────
    print("\n  STEP 7 - Evaluation on Test Set")
    y_pred       = model.predict(X_test, verbose=0).flatten()
    final_rmse   = rmse(y_test_true, y_pred)
    final_score  = nasa_score(y_test_true, y_pred)
    best_val_rmse = min(history.history['val_rmse'])

    print("-" * 60)
    print(f"  RMSE (test)          : {final_rmse:.4f} cycles")
    print(f"  NASA S-Score (test)  : {final_score:.2f}")
    print(f"  Best Val RMSE        : {best_val_rmse:.4f}")
    print("-" * 60)

    # ── 9. Visualise ────────────────────────────────────────────────────────
    print("\n  STEP 8 - Generating 4K Charts")
    plot_predictions(y_test_true, y_pred)
    plot_loss_curves(history)
    plot_attention_heatmap(X_test, attn_model, sensor_cols, n_engines=5)
    plot_error_distribution(y_test_true, y_pred)
    print(f"\n  All charts saved in '{OUT_DIR}/'")

    # ── 10. Save model ───────────────────────────────────────────────────────
    model.save('nasa_rul_hybrid_architecture.h5')
    print("\n  STEP 9 - Model saved -> nasa_rul_hybrid_architecture.h5")
    print("-" * 60)
    print("  [OK] Pipeline complete. Ready for production deployment.")
    print("-" * 60)


if __name__ == '__main__':
    main()

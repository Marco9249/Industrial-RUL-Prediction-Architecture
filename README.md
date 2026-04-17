<div align="center">

# ⚙️ Asymmetric-Loss-Guided Hybrid CNN-BiLSTM-Attention
### *Industrial RUL Prediction with Interpretable Failure Heatmaps*

[![arXiv](https://img.shields.io/badge/arXiv-2604.13459-b31b1b?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2604.13459)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Author](https://img.shields.io/badge/Author-Mohammed%20E.%20B.%20Abdullah-blue?style=for-the-badge)](https://github.com/Marco9249)
[![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow)](https://tensorflow.org/)
[![Dataset](https://img.shields.io/badge/Dataset-NASA%20C--MAPSS-005A9C?style=for-the-badge)](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)

</div>

---

## 📄 Abstract

Turbofan engine degradation under sustained operational stress requires prognostic systems that can **simultaneously** capture multi-sensor spatial correlations and long-range temporal dependencies — while prioritizing **industrial safety** through asymmetric penalization.

This study proposes a hybrid architecture combining:
- **Twin-Stage 1D-CNN** for multi-sensor spatial correlation extraction
- **Bidirectional LSTM** for long-range temporal dependency modeling
- **Bahdanau Additive Attention** (placed after BiLSTM) for interpretable temporal weighting
- **NASA Asymmetric Exponential Loss** that disproportionately penalizes over-estimation

### Results on NASA C-MAPSS FD001 (100 Test Engines):

| Metric | Value |
|--------|-------|
| **RMSE** | **17.52 cycles** |
| **NASA S-Score** | **922.06** |
| **Loss Asymmetry** | h₂=10 (over-est.) vs h₁=13 (under-est.) |

---

## 🏗️ Model Architecture

```
NASA C-MAPSS Sensor Input (14 sensors × 30 cycles)
                │
   ┌────────────▼───────────────┐
   │  Conv1D (64 filters, k=3)  │  ← Stage 1: Spatial correlation
   │  BatchNorm + ReLU + Drop   │
   └────────────┬───────────────┘
                │
   ┌────────────▼───────────────┐
   │  Conv1D (128 filters, k=3) │  ← Stage 2: Deep feature refinement
   │  BatchNorm + ReLU + Drop   │
   └────────────┬───────────────┘
                │
   ┌────────────▼──────────────────────────┐
   │  BiLSTM (128 units, return_seq=True)  │  ← Long-range temporal context
   │  + Dropout(0.3) + LayerNorm           │
   └────────────┬──────────────────────────┘
                │
   ┌────────────▼──────────────────────────┐
   │  Bahdanau Additive Attention (64u)    │  ← Positioned AFTER BiLSTM
   │  → Context vector + Attention weights │    Generates interpretable heatmaps
   └────────────┬──────────────────────────┘
                │
   ┌────────────▼───────────┐
   │  Dense(64) → Dense(32) │  ← Prediction head
   │  → Dense(1, linear)    │
   └────────────┬───────────┘
                │
          [RUL Prediction — cycles]
```

### 🔬 Key Design Elements

| Element | Value | Description |
|---------|-------|-------------|
| **Sliding Window** | **3-step stride** | Matches physical sampling resolution |
| Over-estimation penalty | h₂ = 10 | Fast exponential growth → LARGE safety penalty |
| Under-estimation penalty | h₁ = 13 | Slow exponential growth → smaller penalty |
| RUL cap | 130 cycles | Piecewise-linear plateau labeling |
| Preprocessing | Zero-leakage | Scaler fit on train only |

---

## 📂 Project Structure

```
📁 Industrial-RUL-Prediction-Architecture/
├── 📁 كود التدريب/                       # Training pipeline
│   └── nasa_rul_prediction.py            # Full architecture + Asymmetric loss
├── 📁 بيانات التدريب والاختبار/           # NASA C-MAPSS FD001
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

```bash
# Clone & install
git clone https://github.com/Marco9249/Industrial-RUL-Prediction-Architecture.git
cd Industrial-RUL-Prediction-Architecture
pip install -r requirements.txt

# Data: download NASA C-MAPSS FD001 from:
# https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
# Place train_FD001.txt, test_FD001.txt, RUL_FD001.txt in root directory

# Run full pipeline (training + evaluation + 4K charts)
python "كود التدريب/nasa_rul_prediction.py"
```

### Requirements
```
numpy
pandas
scikit-learn
tensorflow
matplotlib
seaborn
```

### 📊 Outputs Generated
The pipeline automatically generates:
- `prediction_comparative_curve_4k.png` — Predictions vs ground truth
- `loss_metrics_curve_4k.png` — Training convergence curves
- `attention_heatmap_4k.png` — Interpretable temporal attention maps per engine
- `error_distribution_4k.png` — Error histogram & scatter plot

---

## 🔗 Related Research by the Same Author

| # | Paper | Repository | arXiv |
|---|-------|------------|-------|
| 1 | Physics-Guided CNN-BiLSTM Solar Forecast | [Physics-Guided-CNN-BiLSTM-Solar](https://github.com/Marco9249/Physics-Guided-CNN-BiLSTM-Solar) | [2604.13455](https://arxiv.org/abs/2604.13455) |
| 2 | Physics-Informed State Space Models (PISSM) | [PISSM-Solar-Forecasting](https://github.com/Marco9249/PISSM-Solar-Forecasting) | [2604.11807](https://arxiv.org/abs/2604.11807) |
| 3 | Thermodynamic Liquid Manifold Networks (TLMN) | [TLMN-Thermodynamic-Solar-Microgrids](https://github.com/Marco9249/TLMN-Thermodynamic-Solar-Microgrids) | [2604.11909](https://arxiv.org/abs/2604.11909) |
| 4 | **Asymmetric-Loss RUL Prediction** *(this repo)* | [Here](https://github.com/Marco9249/Industrial-RUL-Prediction-Architecture) | [2604.13459](https://arxiv.org/abs/2604.13459) |

---

## 📖 Citation

```bibtex
@misc{abdullah2026rul,
  title   = {Asymmetric-Loss-Guided Hybrid CNN-BiLSTM-Attention Model
             for Industrial RUL Prediction with Interpretable
             Failure Heatmaps},
  author  = {Mohammed Ezzeldin Babiker Abdullah},
  year    = {2026},
  eprint  = {2604.13459},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url     = {https://arxiv.org/abs/2604.13459}
}
```

**APA 7th Edition:**
> Abdullah, M. E. B. (2026). *Asymmetric-Loss-Guided Hybrid CNN-BiLSTM-Attention Model for Industrial RUL Prediction with Interpretable Failure Heatmaps*. arXiv. https://arxiv.org/abs/2604.13459

---

## 👤 Author

**Mohammed Ezzeldin Babiker Abdullah**
*Researcher in Physics-Informed Deep Learning & Industrial Prognostics*

[![GitHub](https://img.shields.io/badge/GitHub-Marco9249-black?style=flat-square&logo=github)](https://github.com/Marco9249)

---

<div align="center">

© 2026 Mohammed Ezzeldin Babiker Abdullah. All rights reserved.

</div>

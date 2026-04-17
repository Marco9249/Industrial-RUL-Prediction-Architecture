<div align="center">

<img src="https://img.shields.io/badge/%E2%9A%99%EF%B8%8F-Industrial%20AI-FF8C00?style=for-the-badge&labelColor=1a1a2e" alt="Industrial AI"/>

# Asymmetric-Loss-Guided Hybrid CNN-BiLSTM-Attention

### ⚙️ *Industrial RUL Prediction with Interpretable Failure Heatmaps* ⚙️

<br/>

[![arXiv](https://img.shields.io/badge/arXiv-2604.13459-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.13459)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![NASA C-MAPSS](https://img.shields.io/badge/Data-NASA%20C--MAPSS-005288?style=for-the-badge&logo=nasa&logoColor=white)](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)

<br/>

<img src="https://img.shields.io/badge/Author-Mohammed%20Ezzeldin%20Babiker%20Abdullah-4A90D9?style=flat-square&logo=google-scholar&logoColor=white" alt="Author"/>

---

*"Over-estimation kills engines. Our asymmetric loss knows that."*

</div>

---

## 🎯 Why Asymmetric Loss Matters

> In predictive maintenance, **not all errors are equal**. Over-estimating RUL means operating an engine beyond its safe life — a potentially catastrophic failure.

<div align="center">

| Error Type | Real-World Impact | Loss Penalty |
|:----------:|:-----------------:|:------------:|
| ⚠️ **Over-estimation** (d < 0) | Engine failure risk | h₂ = 10 (HEAVY) |
| ✅ **Under-estimation** (d ≥ 0) | Conservative, safe | h₁ = 13 (lighter) |

</div>

### 🏆 Results on NASA C-MAPSS FD001 (100 Test Engines)

<div align="center">

| Metric | Value |
|:------:|:-----:|
| 📉 **RMSE** | **17.52 cycles** |
| 🎯 **NASA S-Score** | **922.06** |
| 📊 **MAE** | Competitive |
| 🔍 **Interpretability** | Attention heatmaps per engine |

</div>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   ⚙️  NASA C-MAPSS Input (14 sensors × 30 cycles)  │
│                        │                            │
│         ┌──────────────▼──────────────┐             │
│         │  Conv1D (64, k=3, same)     │  Stage 1    │
│         │  BatchNorm → ReLU → Drop    │  spatial    │
│         └──────────────┬──────────────┘             │
│                        │                            │
│         ┌──────────────▼──────────────┐             │
│         │  Conv1D (128, k=3, same)    │  Stage 2    │
│         │  BatchNorm → ReLU → Drop    │  deeper     │
│         └──────────────┬──────────────┘             │
│                        │                            │
│      ┌─────────────────▼─────────────────┐          │
│      │  BiLSTM (128 units)               │  Long-   │
│      │  return_sequences=True            │  range    │
│      │  + Dropout(0.3) + LayerNorm       │  context  │
│      └─────────────────┬─────────────────┘          │
│                        │                            │
│      ┌─────────────────▼─────────────────┐          │
│      │  🎯 Bahdanau Additive Attention   │  Placed  │
│      │  (64 units)                       │  AFTER   │
│      │  → Context vector                 │  BiLSTM  │
│      │  → Attention weights (heatmaps)   │          │
│      └─────────────────┬─────────────────┘          │
│                        │                            │
│            ┌───────────▼───────────┐                │
│            │  Dense(64) → Dense(32)│  Prediction    │
│            │  → Dense(1, linear)   │  head          │
│            └───────────┬───────────┘                │
│                        │                            │
│    ⚙️ RUL Prediction (cycles) — Asymmetric Loss     │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 🔬 Key Technical Specifications

| Specification | Value |
|:-------------:|:-----:|
| 🪟 **Sliding Window** | 30 cycles, **3-step stride** |
| 📏 **RUL Cap** | 130 cycles (piecewise-linear) |
| 🔒 **Zero Leakage** | Scaler fit on train only |
| 📐 **Regularization** | L2 (1e-4) + Gradient clipping (1.0) |
| 🎯 **Attention** | Bahdanau Additive — **after BiLSTM** |
| 📊 **Output Charts** | 4K resolution (3840×2160) |

---

## 📂 Repository Structure

```
📦 Industrial-RUL-Prediction-Architecture/
│
├── 📁 training_code/
│   └── 🧠 nasa_rul_prediction.py         # Full pipeline: train + eval + charts
│
├── 📁 dataset/
│   ├── 📊 train_FD001.txt                # Training trajectories (100 engines)
│   ├── 📊 test_FD001.txt                 # Test trajectories
│   └── 📊 RUL_FD001.txt                  # Ground truth RUL values
│
├── 📄 RUL_Prediction_Paper.pdf            # Published paper
├── 📄 RUL_Prediction_Paper.docx
├── 📋 requirements.txt
└── 📖 README.md
```

---

## 🚀 Quick Start

```bash
# Clone & setup
git clone https://github.com/Marco9249/Industrial-RUL-Prediction-Architecture.git
cd Industrial-RUL-Prediction-Architecture
pip install -r requirements.txt

# Run the full pipeline (training + evaluation + 4K charts)
python training_code/nasa_rul_prediction.py
```

### 📊 Auto-Generated Outputs

The pipeline automatically produces:
- `prediction_comparative_curve_4k.png` — Predictions vs ground truth
- `loss_metrics_curve_4k.png` — Training convergence curves  
- `attention_heatmap_4k.png` — Interpretable attention maps per engine
- `error_distribution_4k.png` — Error histogram & scatter analysis

---

## 📚 Related Research Papers

<div align="center">

| # | Paper | Repository | arXiv |
|:-:|:------|:----------:|:-----:|
| 1 | Physics-Guided CNN-BiLSTM Solar Forecast | [![Repo](https://img.shields.io/badge/-Repo-181717?style=flat-square&logo=github)](https://github.com/Marco9249/Physics-Guided-CNN-BiLSTM-Solar) | [![arXiv](https://img.shields.io/badge/-2604.13455-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2604.13455) |
| 2 | Physics-Informed State Space Model (PISSM) | [![Repo](https://img.shields.io/badge/-Repo-181717?style=flat-square&logo=github)](https://github.com/Marco9249/PISSM-Solar-Forecasting) | [![arXiv](https://img.shields.io/badge/-2604.11807-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2604.11807) |
| 3 | Thermodynamic Liquid Manifold Networks | [![Repo](https://img.shields.io/badge/-Repo-181717?style=flat-square&logo=github)](https://github.com/Marco9249/TLMN-Thermodynamic-Solar-Microgrids) | [![arXiv](https://img.shields.io/badge/-2604.11909-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2604.11909) |
| **4** | **Asymmetric-Loss RUL Prediction** *(this repo)* 🌟 | [![Repo](https://img.shields.io/badge/-Repo-181717?style=flat-square&logo=github)](https://github.com/Marco9249/Industrial-RUL-Prediction-Architecture) | [![arXiv](https://img.shields.io/badge/-2604.13459-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2604.13459) |
| 🎮 | Interactive 3D Architecture Visualization | [![Repo](https://img.shields.io/badge/-Repo-181717?style=flat-square&logo=github)](https://github.com/Marco9249/PI-Hybrid-3D-Viz) | — |

</div>

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

> **APA 7th Edition:**
> Abdullah, M. E. B. (2026). *Asymmetric-Loss-Guided Hybrid CNN-BiLSTM-Attention Model for Industrial RUL Prediction with Interpretable Failure Heatmaps*. arXiv. https://arxiv.org/abs/2604.13459

---

<div align="center">

### 👤 Author

**Mohammed Ezzeldin Babiker Abdullah**

[![GitHub](https://img.shields.io/badge/GitHub-Marco9249-181717?style=for-the-badge&logo=github)](https://github.com/Marco9249)

---

© 2026 Mohammed Ezzeldin Babiker Abdullah — All rights reserved.

</div>

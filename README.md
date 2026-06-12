# Multimodal Driver Fatigue Monitor

A real-time driver drowsiness detection system that fuses geometric facial features, head pose estimation, and deep learning eye-state classification into a single fatigue label. Built with MediaPipe, PyTorch, and XGBoost.

---

## Architecture

The pipeline has three stacked layers:

```
Webcam Frame
     │
     ▼
┌─────────────────────────────────┐
│  Layer 1 — Feature Extractor   │
│  MediaPipe Face Mesh + OpenCV  │
│  EAR, MAR, head pose (pitch,   │
│  roll), blink metrics          │
└───────────────┬─────────────────┘
                │
     ┌──────────▼──────────┐
     │  Layer 2 — Eye CNN  │
     │  PyTorch (grayscale │
     │  64×64, 3 conv blocks│
     │  → cnn_p_sleepy)    │
     └──────────┬──────────┘
                │
     ┌──────────▼──────────────┐
     │  Layer 3 — XGBoost      │
     │  Feature concat (9      │
     │  features) → awake /    │
     │  drowsy classification  │
     └─────────────────────────┘
```

**Fusion strategy:** Late feature concatenation. The CNN outputs a single probability score (`cnn_p_sleepy`) which is concatenated with the 8 geometric/temporal features into a 9-feature vector. XGBoost makes the final binary classification.

---

## Features

| Feature | Source | Description |
|---|---|---|
| `EAR_min` | MediaPipe | Minimum eye aspect ratio over 10s window |
| `EAR_var` | MediaPipe | Variance in eye aspect ratio |
| `MAR_mean` | MediaPipe | Mean mouth aspect ratio (yawn proxy) |
| `MAR_var` | MediaPipe | Variance in mouth aspect ratio |
| `blink_count_10s` | MediaPipe | Blink count per 10s window |
| `avg_blink_duration_10s` | MediaPipe | Average blink duration |
| `pitch` | PnP solver | Head pitch angle (nodding) |
| `roll` | PnP solver | Head roll angle (tilting) |
| `cnn_p_sleepy` | PyTorch CNN | CNN probability of closed/drowsy eye state |

> **Note:** `pitch` and `roll` were the dominant XGBoost features by permutation importance, outperforming EAR and CNN score. Head pose is a surprisingly strong fatigue signal.

---

## Model Performance

### CNN — Eye State Classifier (`mrl_best_model.pth`)
Trained on a custom webcam eye dataset (grayscale 64×64).

| | Precision | Recall | F1 |
|---|---|---|---|
| closed | 0.81 | 0.76 | 0.78 |
| open | 0.77 | 0.82 | 0.80 |
| **overall** | | | **0.79** |

Train: 1,454 images · Val: 180 images · 4 subjects

### XGBoost — Fatigue Classifier (`fatigue_xgb.pkl`)
Trained on 10s-windowed CSV logs from self-recorded driving simulator videos.

| Metric | Score |
|---|---|
| Test Accuracy | **89%** |
| ROC-AUC (test) | **0.933** |
| 5-fold CV ROC-AUC | **0.899 ± 0.068** |

Train samples: 186 (96 awake, 90 drowsy)

---

## Dataset

### Eye State Dataset
Custom webcam dataset collected across 4 subjects in multiple gaze directions (straight, left, right, up, down), converted to grayscale for privacy.

- 1,454 train / 180 val images
- Classes: `open`, `closed`
- Published on Kaggle under **CC BY 4.0**: [Eye State Dataset Open Closed Webcam](https://www.kaggle.com/)

> The MRL benchmark eye dataset was tested first but caused distribution mismatch and glasses-related failures. Custom collection resolved this.

### Fatigue Labels
Self-recorded driving simulator videos (F1 game) labeled as `awake` (0) or `drowsy` (1). Feature extraction run at 10s windows using `feature_extractor_with_cnn.py`.

---

## Repo Structure

```
multimodal_driver_fatigue_monitor/
├── src/
│   ├── feature_extractor_with_cnn.py  # Main pipeline: MediaPipe + CNN → CSV logger
│   ├── FaceMeshModule.py              # MediaPipe face mesh wrapper
│   ├── cnn_for_eyes.ipynb             # CNN training notebook
│   ├── fatigue_xgboost.ipynb          # XGBoost training + evaluation notebook
│   ├── mrl_eye_cnn.ipynb              # Earlier MRL dataset experiments
│   ├── dataset_collector.ipynb        # Webcam dataset collection tool
│   └── feature_names.json             # Feature order for XGBoost inference
└── data/
    └── custom_eye_dataset/            # See Kaggle link above
```

> Model weights (`mrl_best_model.pth`, `fatigue_xgb.pkl`) are not tracked in git due to size. Download from the releases page or retrain using the notebooks.

---

## Stack

- **MediaPipe** — face mesh landmark detection (478 landmarks)
- **OpenCV** — video capture, frame preprocessing, PnP head pose solver
- **PyTorch** — custom CNN for eye state classification
- **XGBoost** — final fatigue classification
- **scikit-learn** — evaluation, cross-validation
- **FastAPI** — deployment endpoint (in progress)
- **Streamlit** — live webcam frontend (in progress)

---

## Setup

```bash
conda activate ml_env
pip install mediapipe opencv-python torch torchvision xgboost scikit-learn
```

Download model weights and place in `src/`:
- `mrl_best_model.pth`
- `fatigue_xgb.pkl`

Run the feature extractor on a video:
```bash
python src/feature_extractor_with_cnn.py --input your_video.mp4
```

---

## Status

- [x] MediaPipe feature extraction
- [x] CNN eye state classifier
- [x] XGBoost fatigue classifier
- [x] 10s windowed CSV logger
- [ ] FastAPI `/predict` endpoint
- [ ] Streamlit live webcam UI
- [ ] End-to-end real-time integration

# 🤟 ASL Gesture Recognition — Comparative ML/DL Study
 
> A comparative study of **Random Forest**, **LSTM**, **CNN**, and **CNN+LSTM** for recognizing 10 American Sign Language words from a **custom video dataset**, using MediaPipe hand landmarks as features.
> https://docs.google.com/document/d/1RJCy1rqzqFcpuMXfVCxcRIk5Gi2vTHDe2H2r5EHUz4k/edit?usp=sharing
 
---
 
## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Custom Dataset](#custom-dataset)
4. [Pipeline Overview](#pipeline-overview)
5. [Models Compared](#models-compared)
6. [Results](#results)
7. [Getting Started](#getting-started)
8. [Running Each Model](#running-each-model)
9. [EDA](#eda)
10. [Report](#report)
11. [Team](#team)
 
---
 
## Project Overview
 
This project builds an automated ASL gesture recognition system that classifies **10 predefined ASL words** from video input. Rather than using a public dataset, we **recorded our own dataset** under controlled conditions — consistent lighting, background, and camera angle — to reduce noise and improve reliability.
 
**The 10 gesture classes are:** Airplane, Bad, Child, Drink, Drop, Sit, Stop, Teacher, Why, Yes
 
Four models were trained and compared on accuracy, precision, recall, and F1-score using **k-fold cross-validation**:
 
| Rank | Model | Accuracy | F1-Score |
|------|-------|:--------:|:--------:|
| 🥇 1st | CNN (1D Conv on landmarks) | **97%** | **0.97** |
| 🥈 2nd | CNN + LSTM (Hybrid) | 95% | 0.95 |
| 🥉 3rd | LSTM (Bidirectional + Attention) | 89.9% | 0.896 |
| 4th | Random Forest | 88.1% | 0.965* |
 
*Random Forest had high precision (0.966) but lower overall accuracy due to limited temporal modeling.
 
**Key finding:** CNN outperformed all models because ASL gestures in this dataset are differentiated more by *spatial* hand shape than by motion over time — which is exactly what CNNs capture best.
 
---
 
## Repository Structure
 
```
ASL_detection/
│
├── EDA/
│   ├── extract_eda_data.py       # Extracts per-video stats using MediaPipe (brightness, duration, wrist coords)
│   └── generate_plots.py         # Generates 4 EDA visualizations from the extracted CSVs
│
├── cnn/
│   └── cnn.py                    # 1D CNN on MediaPipe landmark sequences — best performing model (97%)
│
├── lstm/
│   └── lstm.py                   # Bidirectional LSTM with attention mechanism (89.9%)
│
├── cnn+lstm/
│   └── cnn_lstm.py               # Hybrid: linear feature extractor → LSTM classifier (95%)
│
├── asl_randomforest/
│   └── rf.py                     # Random Forest on landmark features — classical ML baseline (88.1%)
│
├── .gitignore
└── README.md
```
 
Each folder is **fully self-contained** — you can run any model independently.
 
---
 
## Custom Dataset
 
> 📁 **Dataset link:** https://drive.google.com/drive/folders/1JXVaQd0Z8qY6DWtPOhDN3PChPmSnN2zq
 
We built a **custom dataset** of ASL gesture videos — not a public dataset.
 
| Property | Detail |
|----------|--------|
| Classes | 10 ASL words (Airplane, Bad, Child, Drink, Drop, Sit, Stop, Teacher, Why, Yes) |
| Videos per class | 10 |
| Total videos | 100 |
| Format | Colour video (MP4/AVI/MOV) |
| Recording conditions | Controlled — consistent background, lighting, camera angle |
| Train / Test split | 80% / 20% |
 
### Preprocessing pipeline
 
1. **Frame extraction** — frames pulled from each video using `ffmpeg`
2. **Hand landmark detection** — MediaPipe Hand Landmarker extracts 21 3D keypoints per frame (126 values per frame for both hands)
3. **Normalization** — pixel values normalized; landmarks scaled to frame dimensions
4. **Feature vectors** — landmark coordinates used as input to all models (not raw pixels)
 
### Setup
 
Download the dataset from the Google Drive link above and place it as:
 
```
ASL_detection/
└── Dataset_backup/
    ├── Airplane/
    │   ├── video1.mp4
    │   └── ...
    ├── Bad/
    └── ... (10 class folders, 10 videos each)
```
 
---
 
## Pipeline Overview
 
```
Video files → MediaPipe hand landmark extraction → Landmark sequences → Model → Predicted ASL word
```
 
All models use **MediaPipe hand landmark coordinates** (21 keypoints × 3D = 63 values per hand, 126 total) as input — not raw image pixels. This keeps models lightweight and robust to lighting/background variation.
 
---
 
## Models Compared
 
### 1. Random Forest
Classical ML baseline. Uses scikit-learn `RandomForestClassifier` on landmark features. No temporal modeling. Best tree count found to be ~100 (performance stabilizes beyond that).
 
### 2. LSTM (Bidirectional + Attention)
`input(126) → BiLSTM(hidden=256, layers=2) → Attention → Linear(256→256→10) → 10 classes`
 
Processes the landmark sequence frame-by-frame. Bidirectional means it reads the gesture both forward and backward. Attention lets the model focus on the most important frames. Xavier weight initialization for training stability.
 
### 3. CNN (1D Convolution) — Best Model
`Conv1d(64) → BN → Pool → Conv1d(128) → BN → Pool → Conv1d(256) → GlobalAvgPool → Linear(256→128→10)`
 
Treats the per-frame landmark sequence as a 1D signal. Three convolutional blocks with batch normalization, max-pooling, and dropout (0.3–0.4). Captures local shape patterns across the gesture sequence.
 
### 4. CNN + LSTM Hybrid
`Linear(126→256→128) → LSTM(hidden=256, layers=2) → last hidden state → Linear(256→128→10)`
 
A feature extractor first compresses each frame's landmarks into a compact 128-d vector, then LSTM models the sequence of those vectors. Scored 95% — strong, but temporal modeling did not add value over CNN alone for this dataset.
 
---
 
## Results
 
### Full metrics
 
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|:--------:|:---------:|:------:|:--------:|
| Random Forest | 0.881 | 0.966 | 0.965 | 0.965 |
| LSTM | 0.899 | 0.910 | 0.849 | 0.896 |
| CNN | **0.970** | **0.973** | **0.972** | **0.970** |
| CNN+LSTM | 0.950 | 0.950 | 0.950 | 0.950 |
 
### Why CNN won
ASL gestures in this dataset are primarily distinguished by **hand shape** (spatial features), not motion over time. CNN extracts spatial patterns from landmark sequences extremely well. LSTM adds temporal modeling that, for a small controlled dataset with consistent recordings, introduces complexity without meaningful benefit — consistent with the finding by Sundar et al. that hybrid/temporal models show clearer advantages on larger, more varied datasets.
 
### Hardest classes to classify
- **Airplane** — confused most often across all models (visually complex gesture)
- **Child / Yes** — similar hand shapes caused minor misclassification in CNN
- **Teacher** — highest kinematic complexity (most wrist movement), yet well-classified by CNN
 
---
 
## Getting Started
 
### Requirements
 
- Python 3.8+
- pip
 
### 1. Clone the repo
 
```bash
git clone https://github.com/IppiliSahasra08/ASL_detection.git
cd ASL_detection
```
 
### 2. Create a virtual environment (recommended)
 
```bash
python -m venv asl_env
 
# Windows
asl_env\Scripts\activate
 
# Mac / Linux
source asl_env/bin/activate
```
 
### 3. Install dependencies
 
```bash
pip install torch torchvision mediapipe opencv-python numpy pandas matplotlib seaborn scikit-learn tqdm
```
 
### 4. Download the MediaPipe hand landmark model file
 
The EDA script requires this file in the repo root:
 
```bash
wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```
 
Place `hand_landmarker.task` in `ASL_detection/` (the repo root).
 
### 5. Download the dataset
 
Get it from the [Google Drive link](#custom-dataset) and place it as `Dataset_backup/` in the repo root (folder structure described above).
 
---
 
## Running Each Model
 
### Step 1 — EDA (understand the data first)
 
```bash
cd EDA
 
# Extract video stats and landmark coords from all 100 videos
python extract_eda_data.py
# Produces: video_stats.csv, wrist_coords.csv
 
# Generate all 4 plots
python generate_plots.py
# Produces: 1_temporal_durations.png, 2_kinematic_complexity.png,
#           3_environmental_brightness.png, 4_spatial_heatmap.png
```
 
---
 
### Step 2 — Run any model
 
**Random Forest** (no GPU needed, fastest to run):
```bash
cd asl_randomforest
python rf.py
```
 
**LSTM:**
```bash
cd lstm
python lstm.py
```
 
**CNN** (best model):
```bash
cd cnn
python cnn.py
```
 
**CNN + LSTM Hybrid:**
```bash
cd "cnn+lstm"
python cnn_lstm.py
```
 
All models output a **confusion matrix** and **per-class metrics** (precision, recall, F1-score).
 
---
 
## EDA
 
Four analyses were performed to validate dataset quality before training:
 
| Analysis | Tool | What was confirmed |
|----------|------|--------------------|
| Video duration distribution (violin plot) | Seaborn | Most gestures recorded at 1.2–2 sec; consistent across classes |
| Kinematic complexity (bar chart) | Seaborn | "Teacher" most dynamic; "Drop"/"Yes" simplest — useful for understanding which classes may be harder |
| Lighting quality (brightness histogram) | Seaborn | Tight brightness range (95–110 px intensity); controlled recording confirmed |
| Spatial signing space (wrist heatmap) | Seaborn KDE | Signing concentrated in centre-frame; spatially stable dataset |
 
EDA conclusion: dataset is consistent, well-lit, and spatially stable — suitable for model training.
 
---
 
## Report
 
The report covers: custom dataset design, EDA methodology, all 4 model architectures with code, results with confusion matrices, comparative analysis vs 16 published papers, and a discussion of why CNN outperformed the hybrid on this dataset.
 
---
 
## Troubleshooting
 
| Problem | Fix |
|---------|-----|
| `hand_landmarker.task not found` | Run the `wget` command in step 4 above, place file in repo root |
| `Dataset_backup/ not found` | Download from the Google Drive link in the Dataset section |
| `ModuleNotFoundError: mediapipe` | `pip install mediapipe` |
| `ModuleNotFoundError: torch` | `pip install torch` |
| Slow training | Open in Google Colab: Runtime → Change runtime type → T4 GPU |
 
---
 
## Team
 
| Name | Registration Number |
|------|-------------|
| Ippili Sahasra | 24BCE5035 |
| Deeksha Chowdary  | 24BCE1571 |
| Adonia Alphonsa Romal | 24BCE5009 |
| Suyesha Saha | 24BCE1962 |
 
**Course:** Foundations of Data Science  
**Institution:** Vellore Institute of Technology 
**Year:** 2024–25

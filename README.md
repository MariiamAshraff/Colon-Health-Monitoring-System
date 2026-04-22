# 🦠 Colon Disease & Cancer Detection Using Deep Learning

> An end-to-end AI diagnostic assistant for automated detection and multi-class classification of colon abnormalities from medical imaging data.

## 📖 Problem & Use Case
Manual endoscopic and histopathological screening for colon diseases is labor-intensive, subjective, and prone to inter-observer variability. Early and accurate detection is critical for timely treatment planning, but clinicians face high workloads and delayed diagnoses due to the sheer volume of medical imagery. This project aims to provide a reliable, automated second opinion to streamline screening workflows and reduce diagnostic delays.

## 💡 Solution & Core Idea
A modular Deep Learning pipeline that automates the detection and classification of colon conditions. The system leverages **custom CNN architectures** and **state-of-the-art Transfer Learning (EfficientNetB3)** to extract complex pathological features from endoscopic, histopathological, and CT-scan imagery. The pipeline is optimized for high sensitivity, real-time inference, and seamless deployment as an interactive diagnostic web application.

## 🛠️ Tech Stack
| Category          | Tools & Frameworks                                                                 |
|-------------------|------------------------------------------------------------------------------------|
| **Core**          | Python, TensorFlow / Keras, NumPy, Pandas                                          |
| **Computer Vision**| OpenCV, ImageDataGenerator, EfficientNetB3, Custom CNN                             |
| **Evaluation**    | Scikit-learn, Matplotlib, Seaborn, Confusion Matrices, Classification Reports      |
| **Deployment**    | Streamlit, FastAPI (optional), Pickle / HDF5 (`.h5` model serialization)           |
| **Environment**   | Google Colab, Jupyter Notebooks, Virtualenv / Conda                                |

## 📊 Dataset & Preprocessing
- **Datasets Used:** 
  - `curated-colon-dataset-for-deep-learning` (Endoscopic imagery)
  - `lung-and-colon-cancer-histopathological-images` (Histology slides)
  - `ColonCancerCT-2025` (Abdominal CT scans)
- **Preprocessing Pipeline:**
  - Image resizing to `224x224` & pixel normalization (`1./255`)
  - Stratified Train/Val/Test splits (`60% / 20% / 20%`) preserving class distribution
  - Real-time Data Augmentation: rotation, zoom, horizontal/vertical flips, and shifting to mitigate overfitting
  - Class-imbalance handling via weighted sampling & aggressive augmentation

## 🧠 Model Architecture
The project implements two complementary pipelines:

### 1️⃣ Multi-Class Endoscopic Classifier (Custom CNN)
- **Architecture:** 4-Block CNN with `BatchNormalization`, `MaxPooling2D`, and `GlobalAveragePooling2D`
- **Regularization:** `Dropout(0.5)` + Adaptive Learning Rate (`ReduceLROnPlateau`)
- **Task:** Classifies `Normal`, `Ulcerative Colitis`, `Polyps`, and `Esophagitis`
- **Optimizer:** Adam (`lr=1e-4`) with `categorical_crossentropy` loss

### 2️⃣ Binary Cancer Detector (Transfer Learning)
- **Base Model:** `EfficientNetB3` (pre-trained on ImageNet)
- **Fine-Tuning:** Frozen base layers initially, followed by gradual unfreezing
- **Callbacks:** `EarlyStopping(patience=5, restore_best_weights=True)` + `ReduceLROnPlateau`
- **Task:** Binary classification of `Cancer` vs `Non_Cancer` on CT scans

## 📈 Results & Metrics
| Pipeline                  | Test Accuracy | Test Loss | Key Strengths                                  |
|---------------------------|:-------------:|:---------:|------------------------------------------------|
| Custom CNN (4-Class)      | **98.12%**    | 0.0762    | High macro F1 (~0.98), robust to image noise   |
| EfficientNetB3 (Binary)   | **99.69%**    | 0.0184    | Near-perfect sensitivity, minimal false negatives |

📊 *Visualizations include training curves, confusion matrices, and Grad-CAM activation maps (planned for v2.0) for clinical interpretability.*

## 🚀 How to Run & Deployment

### 📦 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/MariiamAshraff/colon-disease-classifier.git
cd colon-disease-classifier

# Install dependencies
pip install -r requirements.txt

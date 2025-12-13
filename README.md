# Early Detection of Diabetes & Prediabetes Using Machine Learning

This project builds an end-to-end machine learning pipeline to detect **Prediabetes** and **Diabetes** early using the **CDC BRFSS 2015** health indicators dataset.  
The focus is on maximizing **recall for minority classes** (Prediabetes & Diabetes), where early intervention has the highest clinical value.

The project implements:
- Feature engineering
- Advanced resampling (ADASYN / BorderlineSMOTE)
- Optimized SVM (Calibrated RBF)
- XGBoost with custom class weights + threshold tuning
- Deep Neural Networks (DNN) with focal loss + moderate class weights
- Unified reproducible pipeline runnable locally via scripts **or in Google Colab**

---

## 📁 Project Structure
```text
Early-Detection-of-Diabetes-Using-ML/
│
├── data/
│   ├── raw/
│   │   └── diabetes_012_health_indicators_BRFSS2015.csv                 # Original dataset
│   └── processed/
│       └── df_engineered.csv            # Saved after feature engineering
│
├── notebooks/
│   └── EDA.ipynb                        # Exploratory Data Analysis notebook
│
├── src/
│   ├── __init__.py
│   ├── config.py                        # Paths for dataset & processed files
│   ├── data_pipeline.py                 # Load → Feature engineer → Split → Scale
│   └── models/
│       ├── __init__.py
│       ├── svm_model.py                 # Calibrated RBF SVM (best variant)
│       ├── xgboost_model.py             # Weighted XGBoost + threshold sweep
│       └── dnn_model.py                 # DNN with focal loss + class weights
│
├── scripts/
│   ├── run_preprocessing.py             # Runs only data pipeline
│   ├── run_svm.py                       # Train + evaluate SVM model
│   ├── run_xgboost.py                   # Train + evaluate XGBoost model
│   ├── run_dnn.py                       # Train + evaluate DNN
│   └── run_all.py                       # Full pipeline: preprocess + all models
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🚀 Getting Started

## Choose Your Environment:

### 🌟 **Option 1: Google Colab (Recommended - No Setup Required!)**
[Jump to Colab Instructions →](#-running-in-google-colab)

### 💻 **Option 2: Local Machine**
[Jump to Local Setup Instructions →](#-running-locally)

---

# ☁️ Running in Google Colab

## Why Colab?
✅ **No installation required** - Everything runs in the cloud  
✅ **Free GPU access** - Faster training for DNN  
✅ **Pre-installed libraries** - TensorFlow and most dependencies ready  
✅ **Works on any device** - Just need a browser  
✅ **No TensorFlow DLL issues** - Common Windows problem avoided  

---

## ⚡ Quick Start (5 Minutes)

1. **Open Google Colab:** [colab.research.google.com](https://colab.research.google.com/)
2. **Create a new notebook:** `File` → `New notebook`
3. **Copy and paste this code** into the first cell:
```python
# Complete Setup & Execution
!git clone https://github.com/99anjalipai/Early-Detection-of-Diabetes-Using-ML.git
%cd Early-Detection-of-Diabetes-Using-ML
!pip install -q -r requirements.txt
!python scripts/run_preprocessing.py
!python scripts/run_all.py
```

4. **Run the cell:** Press `Shift+Enter` or click ▶️
5. **Wait 10-15 minutes** for training to complete
6. **View results** directly in the output!

---

## 📖 Step-by-Step Colab Guide

### **Step 1: Clone the Repository**
```python
# Clone repository
!git clone https://github.com/99anjalipai/Early-Detection-of-Diabetes-Using-ML.git
%cd Early-Detection-of-Diabetes-Using-ML

# Verify files
!ls
```

### **Step 2: Install Dependencies**
```python
# Install required packages (takes 2-3 minutes)
!pip install -q -r requirements.txt
```

### **Step 3: Verify Dataset**
```python
# Check if dataset is present
!ls data/raw/
```

Expected output: `diabetes_012_health_indicators_BRFSS2015.csv`

### **Step 4: Run Data Preprocessing**
```python
!python scripts/run_preprocessing.py
```

**Expected Output:**
- ✅ Data loaded successfully
- ✅ Feature engineering completed
- ✅ Train/test split created
- ✅ Data scaled and saved

### **Step 5: Train Models**

**Run all models together:**
```python
!python scripts/run_all.py
```

**Or run individually:**
```python
# XGBoost (Best Model)
!python scripts/run_xgboost.py

# SVM Model
!python scripts/run_svm.py

# Deep Neural Network
!python scripts/run_dnn.py
```

### **Step 6: View Results**

Results appear directly in the notebook output, including:
- Model accuracy and F1 scores
- Classification reports
- Confusion matrices
- Recall metrics for each class

---

## 🔧 Troubleshooting in Colab

### **Issue: Convergence Warnings (SVM)**

You may see warnings like:
```
ConvergenceWarning: Solver terminated early (max_iter=1000)
```

**Solution:** These are normal and don't affect results. To suppress them:
```python
import warnings
warnings.filterwarnings('ignore')
!python scripts/run_svm.py
```

### **Issue: Dataset Not Found**

If the dataset is missing from the repository:
```python
from google.colab import files
import shutil
import os

# Create directory
os.makedirs('data/raw', exist_ok=True)

# Upload your CSV file
print("📤 Please upload diabetes_012_health_indicators_BRFSS2015.csv:")
uploaded = files.upload()

# Move to correct location
for filename in uploaded.keys():
    shutil.move(filename, 'data/raw/diabetes_012_health_indicators_BRFSS2015.csv')
    print("✅ Dataset uploaded!")
```

### **Issue: Session Timeout**

Colab sessions disconnect after 90 minutes of inactivity.

**Prevention:**
- Keep the browser tab active
- Save work periodically to Google Drive

---

## 💾 Saving Your Work in Colab

### **Save to Google Drive**
```python
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Copy project to Drive
!cp -r /content/Early-Detection-of-Diabetes-Using-ML /content/drive/MyDrive/

print("✅ Project saved to Google Drive!")
```

### **Download Files**
```python
from google.colab import files

# Download processed data
files.download('data/processed/df_engineered.csv')
```

---

## 🚀 Enable GPU (Optional - Faster Training)

For faster DNN training:

1. Go to `Runtime` → `Change runtime type`
2. Select `GPU` under Hardware accelerator
3. Click `Save`

Verify GPU is available:
```python
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
```

---

## ⏱️ Expected Runtime in Colab

| Task | Time |
|------|------|
| Clone Repository | 10-20 seconds |
| Install Dependencies | 2-3 minutes |
| Preprocessing | 30-60 seconds |
| XGBoost Training | 1-3 minutes |
| SVM Training | 2-5 minutes |
| DNN Training | 3-10 minutes |
| **Total** | **~10-20 minutes** |

---

## 📝 Complete Colab Template

Copy this for a complete ready-to-run setup:
```python
# ========================================
# DIABETES DETECTION ML - COMPLETE SETUP
# ========================================

import warnings
warnings.filterwarnings('ignore')

print("📥 Step 1: Cloning repository...")
!git clone https://github.com/99anjalipai/Early-Detection-of-Diabetes-Using-ML.git
%cd Early-Detection-of-Diabetes-Using-ML

print("\n📦 Step 2: Installing dependencies...")
!pip install -q -r requirements.txt

print("\n📊 Step 3: Verifying dataset...")
!ls data/raw/

print("\n⚙️ Step 4: Running preprocessing...")
!python scripts/run_preprocessing.py

print("\n🚀 Step 5: Training XGBoost (Best Model)...")
!python scripts/run_xgboost.py

print("\n🔬 Step 6: Training SVM...")
!python scripts/run_svm.py

print("\n🧠 Step 7: Training DNN...")
!python scripts/run_dnn.py

print("\n✅ All models completed successfully!")
```

---

# 💻 Running Locally

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (optional, for cloning)

---

## **1️⃣ Install Dependencies**

From the project root, run:
```bash
pip install -r requirements.txt
```

This installs scikit-learn, XGBoost, TensorFlow (CPU), imbalanced-learn, pandas, numpy, seaborn, matplotlib, etc.

---

## **2️⃣ Verify Dataset Location**

Ensure your dataset CSV is placed at:
```bash
data/raw/diabetes_012_health_indicators_BRFSS2015.csv
```

If your file name is different, update the path in: `src/config.py`

---

## **3️⃣ Run Preprocessing Pipeline**

This loads the raw dataset, performs feature engineering, splits data, scales features, and saves processed outputs.
```bash
python scripts/run_preprocessing.py
```

You should see logs confirming:
- `df_engineered.csv` created under `data/processed/`
- Train/test split sizes
- Scaling completed successfully

---

## **4️⃣ Run XGBoost Model (Best Model)**

Trains weighted XGBoost, performs threshold sweep, and prints optimized metrics.
```bash
python scripts/run_xgboost.py
```

**Expected outputs:**
- Selected best threshold
- Accuracy & Macro F1
- Prediabetes recall
- Diabetes recall
- Full classification report

---

## **5️⃣ Run SVM (Calibrated RBF Kernel)**
```bash
python scripts/run_svm.py
```

**Outputs:**
- Best parameters found via GridSearch
- Calibrated probability model
- Accuracy, Macro F1
- Classification report + confusion matrix

---

## **6️⃣ Run Deep Neural Network (DNN)**

Uses focal loss + moderate class weights + ADASYN.
```bash
python scripts/run_dnn.py
```

**Outputs:**
- Training & validation logs
- Final accuracy & macro F1
- Classification report

---

## **7️⃣ Run the Entire Pipeline (All Models)**

To execute preprocessing + SVM + XGBoost + DNN together:
```bash
python scripts/run_all.py
```

This provides a consolidated comparison of all models.

---

## **8️⃣ (Optional) Explore EDA Notebook**

Open the EDA notebook for visual insights:
```bash
jupyter notebook notebooks/EDA.ipynb
```

---

## 🐛 Local Troubleshooting

### **Windows TensorFlow DLL Error**

If you encounter DLL errors with TensorFlow on Windows:

**Option 1:** Comment out DNN import in `src/models/__init__.py`
```python
from . import svm_model
from . import xgboost_model
# from . import dnn_model  # Comment this out
```

**Option 2:** Install TensorFlow CPU version
```bash
pip uninstall tensorflow
pip install tensorflow-cpu==2.13.0
```

**Option 3:** Install Microsoft Visual C++ Redistributable
- Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
- Install and restart your computer

### **Virtual Environment Setup (Recommended)**
```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📊 Dataset Information

**Source:** CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015

**Dataset:** `diabetes_012_health_indicators_BRFSS2015.csv`

**Download:** [Kaggle - Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

**Classes:**
- 0: No Diabetes
- 1: Prediabetes
- 2: Diabetes

**Features:** 21 health indicators including BMI, blood pressure, cholesterol, age, physical activity, and more.

---


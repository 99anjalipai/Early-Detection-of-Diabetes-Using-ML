# Early Detection of Diabetes & Prediabetes Using Machine Learning

This project builds an end-to-end machine learning pipeline to detect **Prediabetes** and **Diabetes** early using the **CDC BRFSS 2015** health indicators dataset.  
The focus is on maximizing **recall for minority classes** (Prediabetes & Diabetes), where early intervention has the highest clinical value.

The project implements:
- Feature engineering
- Advanced resampling (ADASYN / BorderlineSMOTE)
- Optimized SVM (Calibrated RBF)
- XGBoost with custom class weights + threshold tuning
- Deep Neural Networks (DNN) with focal loss + moderate class weights
- Unified reproducible pipeline runnable locally via scripts

---

## 📁 Project Structure

```text
Early-Detection-of-Diabetes-Using-ML/
│
├── data/
│   ├── raw/
│   │   └── diabetes.csv                 # Original dataset
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
│   ├── evaluation.py                    # Common evaluation helpers
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



# 🚀 Steps to Execute the Project

Follow the steps below to run the full machine learning pipeline end-to-end on your local system.

---

## **1️⃣ Install Dependencies**

From the project root, run:

```bash
pip install -r requirements.txt
```

This installs scikit-learn, XGBoost, TensorFlow (CPU), imbalanced-learn, pandas, numpy, seaborn, matplotlib, etc.

## **2️⃣ Verify Dataset Location**

Ensure your dataset CSV is placed at:

```bash
data/raw/diabetes.csv
```

If your file name is different, update the path in:

src/config.py

## **3️⃣ Run Preprocessing Pipeline**

This loads the raw dataset, performs feature engineering, splits data, scales features, and saves processed outputs.

Run:

py scripts/run_preprocessing.py


You should see logs confirming:

df_engineered.csv created under data/processed/

Train/test split sizes

Scaling completed successfully

## **4️⃣ Run XGBoost Model (Best Model)**

Trains weighted XGBoost, performs threshold sweep, and prints optimized metrics.

Run:

py scripts/run_xgboost.py


Expected outputs:

Selected best threshold

Accuracy & Macro F1

Prediabetes recall

Diabetes recall

Full classification report

## **5️⃣ Run SVM (Calibrated RBF Kernel)**

Run:

py scripts/run_svm.py


Outputs:

Best parameters found via GridSearch

Calibrated probability model

Accuracy, Macro F1

Classification report + confusion matrix

## **6️⃣ Run Deep Neural Network (DNN)**

Uses focal loss + moderate class weights + ADASYN.

Run:

py scripts/run_dnn.py


Outputs:

Training & validation logs

Final accuracy & macro F1

Classification report

## **7️⃣ Run the Entire Pipeline (All Models)**

To execute preprocessing + SVM + XGBoost + DNN together:
```bash
py scripts/run_all.py
```

This provides a consolidated comparison of all models.

## **8️⃣ (Optional) Explore EDA Notebook**

Open the EDA notebook for visual insights:

```bash
notebooks/EDA.ipynb
```
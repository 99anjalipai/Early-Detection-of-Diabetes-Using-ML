import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import BorderlineSMOTE, ADASYN

from src.config import TARGET_COL, RANDOM_STATE


# ----------------------------
# 1. Load Data
# ----------------------------
def load_data(path):
    df = pd.read_csv(path)
    print(f"[load_data] Loaded dataset with shape {df.shape}")
    return df


# ----------------------------
# 2. Feature Engineering
# ----------------------------
def feature_engineering(df):
    df = df.copy()

    # Composite risk scores
    df["Cardiovascular_Risk"] = df["HighBP"] + df["HighChol"] + df["Stroke"] + df["HeartDiseaseorAttack"]
    df["Lifestyle_Risk"] = df["Smoker"] + (1 - df["PhysActivity"]) + (1 - df["Fruits"]) + (1 - df["Veggies"])

    # Interaction terms
    df["Age_BMI_Interaction"] = df["Age"] * df["BMI"]
    df["BMI_PhysActivity"] = df["BMI"] * (1 - df["PhysActivity"])
    df["Age_GenHlth"] = df["Age"] * df["GenHlth"]

    # BMI categories
    def categorize_bmi(bmi):
        if bmi < 18.5: return 0
        if bmi < 25: return 1
        if bmi < 30: return 2
        return 3

    df["BMI_Category"] = df["BMI"].apply(categorize_bmi)

    # Age groups
    def categorize_age(age):
        if age <= 4: return 0
        if age <= 8: return 1
        if age <= 11: return 2
        return 3

    df["Age_Group"] = df["Age"].apply(categorize_age)

    # Comorbidity count
    comorbidity_feats = ["HighBP", "HighChol", "Stroke", "HeartDiseaseorAttack", "DiffWalk"]
    df["Comorbidity_Count"] = df[comorbidity_feats].sum(axis=1)

    print(f"[feature_engineering] Final feature count: {df.shape[1] - 1}")
    return df


# ----------------------------
# 3. Train/Test Split + Scaling
# ----------------------------
def split_and_scale(df):
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    print(f"[split_and_scale] Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train_scaled, X_test_scaled, y_train, y_test


# ----------------------------
# 4. BorderlineSMOTE for SVM
# ----------------------------
def svm_sampling(X_train_scaled, y_train, subset_size=40000):
    sm = BorderlineSMOTE(
        random_state=RANDOM_STATE,
        k_neighbors=5,
        sampling_strategy="not majority"
    )
    X_res, y_res = sm.fit_resample(X_train_scaled, y_train)

    # Stratified subset
    if len(X_res) > subset_size:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=subset_size, random_state=RANDOM_STATE)
        for idx, _ in sss.split(X_res, y_res):
            X_res, y_res = X_res.iloc[idx], y_res.iloc[idx]

    return np.array(X_res), np.array(y_res)


# ----------------------------
# 5. ADASYN for XGB & DNN
# ----------------------------
def adasyn_sampling(X_train_scaled, y_train):
    ad = ADASYN(random_state=RANDOM_STATE, n_neighbors=5, sampling_strategy="not majority")
    X_res, y_res = ad.fit_resample(X_train_scaled, y_train)
    return np.array(X_res), np.array(y_res)

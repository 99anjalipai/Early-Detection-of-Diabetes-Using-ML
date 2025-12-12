import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from src.config import RAW_DATA_FILE
from src.data_pipeline import (
    load_data,
    feature_engineering,
    split_and_scale,
    svm_sampling,
)
from src.models.svm_model import train_svm_calibrated, evaluate_svm


def main():
    print("=" * 80)
    print("[RUN] SVM: Calibrated RBF SVM (Best Variant)")
    print("=" * 80)

    # 1) Load and engineer features
    df = load_data(RAW_DATA_FILE)
    df_eng = feature_engineering(df)
    X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale(df_eng)

    # 2) SVM-specific sampling (BorderlineSMOTE + stratified subset)
    X_train_svm, y_train_svm = svm_sampling(X_train_scaled, y_train, subset_size=40000)

    # 3) Train calibrated SVM
    svm_model, best_params = train_svm_calibrated(X_train_svm, y_train_svm)

    # 4) Evaluate
    metrics, y_pred, y_proba, cm = evaluate_svm(svm_model, X_test_scaled, y_test)

    print("\n[RUN] SVM finished.")
    print(f"[RUN] Best params: {best_params}")
    print(f"[RUN] Final Accuracy: {metrics['accuracy']:.4f}")
    print(f"[RUN] Final Macro F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()

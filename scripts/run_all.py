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
    adasyn_sampling,
)
from src.models.svm_model import train_svm_calibrated, evaluate_svm
from src.models.xgboost_model import train_xgb, threshold_sweep, evaluate_xgb
from src.models.dnn_model import train_dnn_focal_moderate


def main():
    print("=" * 80)
    print("[RUN ALL] Full Pipeline: SVM + XGBoost + DNN")
    print("=" * 80)

    # Common: load + feature engineering + split
    df = load_data(RAW_DATA_FILE)
    df_eng = feature_engineering(df)
    X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale(df_eng)

    # -------------------------------------------------
    # 1) SVM
    # -------------------------------------------------
    print("\n" + "-" * 80)
    print("[RUN ALL] 1) SVM (Calibrated RBF)")
    print("-" * 80)

    X_train_svm, y_train_svm = svm_sampling(X_train_scaled, y_train, subset_size=40000)
    svm_model, svm_params = train_svm_calibrated(X_train_svm, y_train_svm)
    svm_metrics, _, _, _ = evaluate_svm(svm_model, X_test_scaled, y_test)

    # -------------------------------------------------
    # 2) XGBoost
    # -------------------------------------------------
    print("\n" + "-" * 80)
    print("[RUN ALL] 2) XGBoost (Weighted + Softprob + Threshold Tuning)")
    print("-" * 80)

    X_train_resampled, y_train_resampled = adasyn_sampling(X_train_scaled, y_train)
    xgb_model = train_xgb(
        X_train=X_train_resampled,
        y_train=y_train_resampled,
        class_weights={0: 1.0, 1: 14.0, 2: 2.0},
    )
    best_row, xgb_results_df = threshold_sweep(
        model=xgb_model,
        X_test=X_test_scaled,
        y_test=y_test,
        min_accuracy=0.58,
        min_diabetes_recall=0.25,
    )
    _ = evaluate_xgb(best_row, y_test)

    # -------------------------------------------------
    # 3) DNN
    # -------------------------------------------------
    print("\n" + "-" * 80)
    print("[RUN ALL] 3) DNN (Focal + Moderate Class Weights)")
    print("-" * 80)

    # reuse ADASYN resampled train
    dnn_model, dnn_metrics, _, _ = train_dnn_focal_moderate(
        X_train_resampled=X_train_resampled,
        y_train_resampled=y_train_resampled,
        X_test_scaled=X_test_scaled,
        y_test=y_test,
    )

    print("\n" + "=" * 80)
    print("[RUN ALL] Summary")
    print("=" * 80)
    print(f"SVM   → Acc: {svm_metrics['accuracy']:.4f}, Macro F1: {svm_metrics['macro_f1']:.4f}")
    print(f"XGB   → Best threshold: {best_row['threshold']:.3f}, "
          f"Acc: {best_row['accuracy']:.4f}, Macro F1: {best_row['macro_f1']:.4f}")
    print(f"DNN   → Acc: {dnn_metrics['accuracy']:.4f}, Macro F1: {dnn_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()

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
    adasyn_sampling,
)
from src.models.xgboost_model import train_xgb, threshold_sweep, evaluate_xgb


def main():
    print("=" * 80)
    print("[RUN] XGBoost: Weighted + Softprob + Threshold Tuning")
    print("=" * 80)

    # 1) Load and engineer features
    df = load_data(RAW_DATA_FILE)
    df_eng = feature_engineering(df)
    X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale(df_eng)

    # 2) ADASYN resampling
    X_train_resampled, y_train_resampled = adasyn_sampling(X_train_scaled, y_train)

    # 3) Train weighted XGBoost
    xgb_model = train_xgb(
        X_train=X_train_resampled,
        y_train=y_train_resampled,
        class_weights={0: 1.0, 1: 14.0, 2: 2.0},
    )

    # 4) Threshold sweep for Prediabetes
    best_row, results_df = threshold_sweep(
        model=xgb_model,
        X_test=X_test_scaled,
        y_test=y_test,
        min_accuracy=0.58,
        min_diabetes_recall=0.25,
    )

    print("\n[RUN] Best threshold row:")
    print(best_row)

    # 5) Final evaluation using best threshold
    preds = evaluate_xgb(best_row, y_test)

    print("\n[RUN] XGBoost finished.")
    print(f"[RUN] Final Accuracy: {results_df[results_df['threshold'] == best_row['threshold']]['accuracy'].values[0]:.4f}")
    print(f"[RUN] Final Macro F1: {results_df[results_df['threshold'] == best_row['threshold']]['macro_f1'].values[0]:.4f}")


if __name__ == "__main__":
    main()

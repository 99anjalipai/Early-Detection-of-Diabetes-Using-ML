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
from src.models.dnn_model import train_dnn_focal_moderate


def main():
    print("=" * 80)
    print("[RUN] DNN: Focal Loss + Moderate Class Weights (with ADASYN)")
    print("=" * 80)

    # 1) Load + feature engineering
    df = load_data(RAW_DATA_FILE)
    df_eng = feature_engineering(df)
    X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale(df_eng)

    # 2) ADASYN resampling
    X_train_resampled, y_train_resampled = adasyn_sampling(X_train_scaled, y_train)

    # 3) Train DNN
    dnn_model, metrics, y_pred, y_pred_proba = train_dnn_focal_moderate(
        X_train_resampled=X_train_resampled,
        y_train_resampled=y_train_resampled,
        X_test_scaled=X_test_scaled,
        y_test=y_test,
    )

    print("\n[RUN] DNN finished.")
    print(f"[RUN] Final Accuracy: {metrics['accuracy']:.4f}")
    print(f"[RUN] Final Macro F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()

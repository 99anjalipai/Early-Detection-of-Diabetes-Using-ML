import os
import sys

# Ensure project root is on sys.path
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from src.config import RAW_DATA_FILE, DATA_PROCESSED
from src.data_pipeline import load_data, feature_engineering, split_and_scale


def main():
    print("=" * 80)
    print("[RUN] Preprocessing: Load → Feature Engineering → Split & Scale")
    print("=" * 80)

    # 1) Load raw data
    df = load_data(RAW_DATA_FILE)

    # 2) Feature engineering
    df_eng = feature_engineering(df)

    # 3) Train-test split + scaling
    X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale(df_eng)

    # 4) Optionally save engineered full dataset
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    out_path = os.path.join(DATA_PROCESSED, "df_engineered.csv")
    df_eng.to_csv(out_path, index=False)
    print(f"\n[RUN] Saved engineered dataset to: {out_path}")

    print(f"[RUN] X_train_scaled shape: {X_train_scaled.shape}")
    print(f"[RUN] X_test_scaled shape:  {X_test_scaled.shape}")
    print(f"[RUN] y_train size: {y_train.shape[0]}")
    print(f"[RUN] y_test size:  {y_test.shape[0]}")


if __name__ == "__main__":
    main()

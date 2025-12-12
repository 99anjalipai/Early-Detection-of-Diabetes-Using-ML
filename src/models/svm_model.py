import numpy as np
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


# -------------------------------------------------------------
# 1) Compute custom SVM class weights
# -------------------------------------------------------------
def compute_svm_class_weights(y_train_svm):
    """
    Compute balanced class weights and then boost:
      - Prediabetes (class 1) × 2.0
      - Diabetes (class 2) × 1.5
    """
    base_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train_svm),
        y=y_train_svm,
    )
    # base_weights is ordered by sorted class labels [0,1,2]
    base_weights = np.array(base_weights, dtype=float)
    base_weights[1] *= 2.0   # 2x boost for Prediabetes
    base_weights[2] *= 1.5   # 1.5x boost for Diabetes

    class_weight_dict = {int(i): float(w) for i, w in enumerate(base_weights)}
    print("\n[SVM] Custom class weights:", class_weight_dict)
    return class_weight_dict


# -------------------------------------------------------------
# 2) Train calibrated RBF SVM (best-performing variant)
# -------------------------------------------------------------
def train_svm_calibrated(X_train_svm, y_train_svm, random_state=42):
    """
    Train the best SVM variant:
      - RBF kernel SVM
      - Small GridSearch for C and gamma
      - Calibrated probabilities via CalibratedClassifierCV (sigmoid)

    Returns:
        svm_calibrated (fitted model), best_params (dict)
    """
    # Compute custom class weights
    class_weight_dict = compute_svm_class_weights(y_train_svm)

    print("\n" + "=" * 80)
    print("[SVM] Training Variant: Calibrated RBF SVM (with small GridSearch)")
    print("=" * 80)

    # Base RBF SVM
    svm_rbf_base = SVC(
        kernel="rbf",
        class_weight=class_weight_dict,
        random_state=random_state,
        probability=True,
        cache_size=2000,
        max_iter=1000,   # limit iterations for speed
    )

    # Small grid (same spirit as your notebook)
    param_grid_rbf = {
        "C": [0.5, 1.0],
        "gamma": ["scale", 0.1],
    }

    grid_search_rbf = GridSearchCV(
        estimator=svm_rbf_base,
        param_grid=param_grid_rbf,
        cv=2,                  # reduced folds for speed
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
    )

    grid_search_rbf.fit(X_train_svm, y_train_svm)
    best_rbf = grid_search_rbf.best_estimator_
    print(f"\n[SVM] Best RBF params: {grid_search_rbf.best_params_}")
    print(f"[SVM] Best CV macro F1: {grid_search_rbf.best_score_:.4f}")

    # Calibrated SVM on top of best RBF model
    print("\n[SVM] Fitting CalibratedClassifierCV (sigmoid)...")
    svm_calibrated = CalibratedClassifierCV(
        estimator=best_rbf,
        cv=2,
        method="sigmoid",
    )
    svm_calibrated.fit(X_train_svm, y_train_svm)
    print("✓ Calibrated SVM trained.")

    return svm_calibrated, grid_search_rbf.best_params_


# -------------------------------------------------------------
# 3) Evaluate SVM on test set
# -------------------------------------------------------------
def evaluate_svm(model, X_test_scaled, y_test):
    """
    Evaluate a fitted SVM model on the test set.

    Returns:
        metrics_dict, y_pred, y_pred_proba, confusion_mat
    """
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("\n--- Calibrated SVM Performance ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["No Diabetes", "Prediabetes", "Diabetes"],
        )
    )

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
    }

    return metrics, y_pred, y_proba, cm

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report
import xgboost as xgb


# -------------------------------------------------------------
# Train XGBoost (weighted + softprob)
# -------------------------------------------------------------
def train_xgb(X_train, y_train, class_weights=None):
    """
    Train XGBoost classifier with softprob and optional class weights.
    """
    if class_weights is None:
        class_weights = {0: 1.0, 1: 14.0, 2: 2.0}

    # Convert class weights to per-sample weights
    sample_weight = np.vectorize(class_weights.get)(y_train)

    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='mlogloss',
        tree_method='hist',
        n_jobs=-1
    )

    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


# -------------------------------------------------------------
# Threshold sweep for optimized Prediabetes recall
# -------------------------------------------------------------
def threshold_sweep(model, X_test, y_test, 
                    min_accuracy=0.60,
                    min_diabetes_recall=0.25):
    """
    Sweep thresholds for class 1 (Prediabetes) using predicted probabilities.
    Returns the best threshold row.
    """

    proba = model.predict_proba(X_test)

    thresholds = np.linspace(0.03, 0.40, 15)
    results = []

    for thr in thresholds:
        preds = []
        for p in proba:
            if p[1] > thr:
                preds.append(1)
            else:
                preds.append(np.argmax(p))
        preds = np.array(preds)

        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro")
        recall_vals = recall_score(y_test, preds, labels=[0, 1, 2], average=None)

        results.append({
            "threshold": thr,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "recall_no_diabetes": recall_vals[0],
            "recall_prediabetes": recall_vals[1],
            "recall_diabetes": recall_vals[2],
            "preds": preds
        })

    df = pd.DataFrame(results)

    # Apply constraints
    candidates = df[
        (df["accuracy"] >= min_accuracy) &
        (df["recall_diabetes"] >= min_diabetes_recall)
    ]

    if len(candidates) == 0:
        print("[XGBoost] No threshold meets constraints → using max Prediabetes recall")
        best = df.sort_values(
            by=["recall_prediabetes", "macro_f1"],
            ascending=[False, False]
        ).iloc[0]
    else:
        best = candidates.sort_values(
            by=["recall_prediabetes", "macro_f1"],
            ascending=[False, False]
        ).iloc[0]

    return best, df

def evaluate_xgb(best_row, y_test):
    """
    Evaluate XGBoost using the best_row returned from threshold_sweep.

    best_row must contain:
      - 'preds'   → final predicted labels (np.array-like)
      - 'threshold', 'accuracy', 'macro_f1', 'recall_*' for logging

    Parameters
    ----------
    best_row : pandas.Series or dict-like
        Row from results_df returned by threshold_sweep (with 'preds').

    y_test : array-like
        True labels for the test set.

    Returns
    -------
    preds : np.ndarray
        Final predictions used for evaluation.
    """

    preds = np.array(best_row["preds"])

    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")

    print("\n====================== XGBoost Final Evaluation ======================")
    print(f"Selected Threshold: {best_row['threshold']:.3f}")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Macro F1-Score:    {macro_f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            preds,
            target_names=["No Diabetes", "Prediabetes", "Diabetes"],
        )
    )

    return preds
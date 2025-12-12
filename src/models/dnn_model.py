import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ----------------------------
# Focal loss for multi-class
# ----------------------------
def focal_loss_multi(gamma=1.5, alpha=None):
    """
    Focal loss for multi-class with optional alpha weighting.
    y_true: one-hot
    y_pred: softmax probabilities
    """
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # standard cross-entropy
        ce = -y_true * tf.math.log(y_pred)

        if alpha is not None:
            alpha_tensor = tf.cast(alpha, tf.float32)
            alpha_factor = y_true * alpha_tensor  # per-class alpha
            ce = alpha_factor * ce

        weight = tf.pow(1.0 - y_pred, gamma)
        fl = weight * ce
        return tf.reduce_sum(fl, axis=-1)

    return loss_fn


# ----------------------------
# DNN architecture
# ----------------------------
def build_dnn_focal_moderate(input_dim, num_classes=3):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(32, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(num_classes, activation="softmax"),
    ])
    return model


# ----------------------------
# Train + evaluate best DNN
# ----------------------------
def train_dnn_focal_moderate(
    X_train_resampled,
    y_train_resampled,
    X_test_scaled,
    y_test,
    random_state=42,
):
    """
    Train your best-performing DNN:
      - ADASYN-resampled data (passed in)
      - Focal loss with moderate alpha/gamma
      - Moderate class weights

    Returns:
        model, metrics_dict, y_pred, y_pred_proba
    """
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    num_classes = 3
    input_dim = X_train_resampled.shape[1]

    # One-hot labels
    y_train_res_cat = keras.utils.to_categorical(
        y_train_resampled, num_classes=num_classes
    )

    # -------- Class weights (moderate) --------
    classes = np.unique(y_train_resampled)
    base_class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train_resampled,
    )
    base_class_weights_arr = np.array(base_class_weights_arr, dtype=float)

    # small bumps for Prediabetes (1) and Diabetes (2)
    base_class_weights_arr[1] *= 1.2
    base_class_weights_arr[2] *= 1.1

    class_weight_dict = {
        int(c): float(w) for c, w in zip(classes, base_class_weights_arr)
    }
    print("\n[DNN] Moderate class weights:", class_weight_dict)

    # -------- Focal loss config --------
    alpha_moderate = tf.constant([1.0, 1.2, 1.05], dtype=tf.float32)
    gamma_moderate = 1.5
    focal_loss_fn = focal_loss_multi(
        gamma=gamma_moderate,
        alpha=alpha_moderate,
    )

    # -------- Build + compile model --------
    model = build_dnn_focal_moderate(input_dim, num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=focal_loss_fn,
        metrics=["accuracy"],
    )

    # -------- Callbacks --------
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=6,
        min_lr=1e-6,
    )

    # -------- Train --------
    print("\n[DNN] Training (Focal + Moderate Class Weights, with ADASYN)...")
    history = model.fit(
        X_train_resampled,
        y_train_res_cat,
        validation_split=0.2,
        epochs=60,
        batch_size=512,
        class_weight=class_weight_dict,
        callbacks=[early_stop, reduce_lr],
        verbose=0,
    )
    print("✓ DNN training complete.")

    # -------- Evaluation (argmax baseline) --------
    y_pred_proba = model.predict(X_test_scaled, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("\n--- DNN Focal+Moderate (with ADASYN) – Baseline Argmax ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["No Diabetes", "Prediabetes", "Diabetes"],
        )
    )

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
    }

    return model, metrics, y_pred, y_pred_proba

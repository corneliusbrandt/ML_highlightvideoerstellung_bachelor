import numpy as np
import torch
from tsai.all import *
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from helper import load_data
from focal_loss import FocalLoss
from helper import calculate_class_weights


# -----------------------------
# Daten laden
# -----------------------------
X_train, y_train = load_data("datasets_output/train_dataset_augmented_binary.npz")
X_val, y_val = load_data("datasets_output/val_dataset_binary.npz")


# -----------------------------
# Shape anpassen
# -----------------------------
# Von: (samples, time_steps, channels)
# Zu:  (samples, channels, time_steps)
X_train = np.transpose(X_train, (0, 2, 1))
X_val = np.transpose(X_val, (0, 2, 1))


# -----------------------------
# Daten bereinigen
# -----------------------------
# X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
# X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

# X_train = np.ascontiguousarray(X_train, dtype=np.float32)
# X_val = np.ascontiguousarray(X_val, dtype=np.float32)

# y_train = np.asarray(y_train, dtype=np.int64)
# y_val = np.asarray(y_val, dtype=np.int64)


print("X_train:", X_train.shape, X_train.dtype, X_train.flags["C_CONTIGUOUS"])
print("y_train:", y_train.shape, y_train.dtype)
print("Klassen Train:", np.unique(y_train, return_counts=True))

print("X_val:", X_val.shape, X_val.dtype, X_val.flags["C_CONTIGUOUS"])
print("y_val:", y_val.shape, y_val.dtype)
print("Klassen Val:", np.unique(y_val, return_counts=True))


# -----------------------------
# Train/Val zusammenführen
# -----------------------------
X = np.concatenate([X_train, X_val], axis=0)
y = np.concatenate([y_train, y_val], axis=0)

train_idxs = np.arange(len(X_train))
val_idxs = np.arange(len(X_train), len(X_train) + len(X_val))

splits = (train_idxs, val_idxs)


# -----------------------------
# DataLoaders erstellen
# -----------------------------
batch_size = 32

dls = get_ts_dls(
    X,
    y,
    splits=splits,
    bs=batch_size,
    tfms=[None, TSCategorize()]
)


# -----------------------------
# Modell erstellen
# -----------------------------
n_channels = X_train.shape[1]
n_classes = len(np.unique(y_train))

class_weights = calculate_class_weights(y_train, scaling_factor=2.5, min_weight=1.5)
loss_function = FocalLoss(gamma=3, alpha=class_weights, task_type='multi-class', num_classes=n_classes)

model = InceptionTime(
    c_in=n_channels,
    c_out=n_classes
)

print("n_channels:", n_channels)
print("n_classes:", n_classes)


# -----------------------------
# Learner erstellen
# -----------------------------
learn = Learner(
    dls,
    model,
    loss_func=loss_function,
    metrics=accuracy
)


# -----------------------------
# Training
# -----------------------------
learn.fit_one_cycle(
    n_epoch=30,
    lr_max=1e-3
)


# -----------------------------
# Vorhersage auf Validation Set
# -----------------------------
preds, targets = learn.get_preds(ds_idx=1)

probs = preds.numpy()
y_true = targets.numpy()

y_pred = np.argmax(probs, axis=1)


# -----------------------------
# Auswertung mit argmax
# -----------------------------
print("\n--- Auswertung mit argmax ---")
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, zero_division=0))
print("F1:", f1_score(y_true, y_pred, zero_division=0))


# -----------------------------
# Auswertung mit Threshold
# -----------------------------
threshold = 0.8

event_probs = probs[:, 1]
y_pred_threshold = (event_probs > threshold).astype(int)

print("\n--- Auswertung mit Threshold ---")
print("Threshold:", threshold)
print(confusion_matrix(y_true, y_pred_threshold))
print(classification_report(y_true, y_pred_threshold, zero_division=0))
print("F1:", f1_score(y_true, y_pred_threshold, zero_division=0))


# -----------------------------
# Modell speichern
# -----------------------------
learn.export("inception_time_tsai.pkl")
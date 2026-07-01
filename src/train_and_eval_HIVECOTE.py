from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from aeon.classification.hybrid import HIVECOTEV2
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from src.helper import load_data

'''
Warning: This script is not integrated into the full pipline yet. It is a standalone script for training and evaluating the HIVE-COTE model.
'''

X_train, y_train = load_data("datasets_output/train_dataset_augmented_binary.npz")
X_val, y_val = load_data("datasets_output/val_dataset_binary.npz")

# Falls X die Form (samples, time_steps, channels) hat:
X_train = np.transpose(X_train, (0, 2, 1))
X_val = np.transpose(X_val, (0, 2, 1))

# Wichtig für aeon/numba:
# NICHT float32, sondern float64 + zusammenhängender Speicher
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

X_train = np.ascontiguousarray(X_train, dtype=np.float64)
X_val = np.ascontiguousarray(X_val, dtype=np.float64)

y_train = np.asarray(y_train, dtype=np.int64)
y_val = np.asarray(y_val, dtype=np.int64)

print("X_train:", X_train.shape, X_train.dtype, X_train.flags["C_CONTIGUOUS"])
print("y_train:", y_train.shape, y_train.dtype)
print("Klassen:", np.unique(y_train, return_counts=True))

# Balanciertes Subset
idx_0 = np.where(y_train == 0)[0]
idx_1 = np.where(y_train == 1)[0]

n = min(len(idx_0), len(idx_1), 500)

rng = np.random.default_rng(42)

idx_balanced = np.concatenate([
    rng.choice(idx_0, n, replace=False),
    rng.choice(idx_1, n, replace=False)
])

rng.shuffle(idx_balanced)

X_train_small = X_train[idx_balanced]
y_train_small = y_train[idx_balanced]

# Auch das Subset nochmal explizit contiguous machen
X_train_small = np.ascontiguousarray(X_train_small, dtype=np.float64)
y_train_small = np.asarray(y_train_small, dtype=np.int64)

print("X_train_small:", X_train_small.shape, X_train_small.dtype, X_train_small.flags["C_CONTIGUOUS"])
print("y_train_small:", np.unique(y_train_small, return_counts=True))

model = HIVECOTEV2(
    stc_params={
        "n_shapelet_samples": 500,
        "max_shapelets": 50,
        "max_shapelet_length": 60,
        "batch_size": 50,
    },
    drcif_params={
        "n_estimators": 100,
    },
    arsenal_params={
        "n_estimators": 10,
        "n_kernels": 500,
    },
    tde_params={
        "n_parameter_samples": 50,
        "max_ensemble_size": 10,
        "randomly_selected_params": 20,
    },
    n_jobs=-1,
    random_state=42,
    verbose=1
)

model.fit(X_train_small, y_train_small)

y_pred = model.predict(X_val)

print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred, zero_division=0))
print("F1:", f1_score(y_val, y_pred, zero_division=0))
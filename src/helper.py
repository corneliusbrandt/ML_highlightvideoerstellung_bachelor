import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle
from matplotlib.widgets import Slider


# Useful functions can be defined here so that they can be imported into other files
# -> better code organization and reusability

def load_data(dataset_path):
    data = np.load(dataset_path)

    X = data['X']
    y = data['y']

    return X, y


def format_data_for_pytorch(X, y):
    # reshape X to (num_samples, num_channels, num_time_steps)
    # because PyTorch expects (batch_size, num_channels, num_time_steps)
    X = np.transpose(X, (0, 2, 1))
    return X, y


# function to calculate the class weights for weighted cross entropy loss
def calculate_class_weights(y, num_classes, scaling_factor=1.0, min_weight=1.0):
    y = np.asarray(y)
    counts = np.bincount(y, minlength=num_classes)
    total = counts.sum()
    weights = np.ones(num_classes, dtype=np.float32)

    for cls in range(num_classes):
        if counts[cls] > 0:
            weights[cls] = total / (num_classes * counts[cls])
        else:
            weights[cls] = 0.0 # if class is not present in the specific training set the weight will be 0

    weights = weights * scaling_factor
    weights = np.maximum(weights, min_weight)
    return torch.tensor(weights, dtype=torch.float32)


def plot_loss_history(train_loss_history, val_loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss History')
    plt.legend()
    plt.grid()
    plt.show()


def plot_all_features_with_pca(all_features, all_labels, n_components=3):
    
    pca = PCA(n_components=n_components)
    features_2d = pca.fit_transform(all_features)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
    features_2d[:, 0],
    features_2d[:, 1],
    features_2d[:, 2],
    c=all_labels,
    cmap='viridis',
    alpha=0.7
)

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('PCA of Extracted Features')
    ax.legend(*scatter.legend_elements(), title="Classes")

    plt.show()


def save_and_load_pickle(data, path, mode='save'):
    if mode == 'save':
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    elif mode == 'load':
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError("Mode must be either 'save' or 'load'")



def _prepare_window_slider(fig, axes, y_true, window):
    max_start = max(0, len(y_true) - window)

    plt.subplots_adjust(bottom=0.14)

    slider_ax = fig.add_axes([0.15, 0.04, 0.7, 0.03])
    slider = Slider(
        slider_ax,
        "Start",
        0,
        max_start,
        valinit=0,
        valstep=1
    )

    return slider


def plot_onehot_predictions_with_slider(y_true, y_pred, num_classes, window=200):
    """
    Pro Klasse ein eigener Linienplot:
    Ground Truth = 1, wenn Klasse aktiv ist, sonst 0
    Prediction = 1, wenn Klasse vorhergesagt wurde, sonst 0
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    fig, axes = plt.subplots(
        num_classes,
        1,
        figsize=(15, 2 * num_classes),
        sharex=True
    )

    if num_classes == 1:
        axes = [axes]

    start = 0
    end = min(window, len(y_true))
    x = np.arange(start, end)

    gt_lines = []
    pred_lines = []

    for cls, ax in enumerate(axes):
        gt = (y_true[start:end] == cls).astype(int)
        pred = (y_pred[start:end] == cls).astype(int)

        gt_line, = ax.step(
            x,
            gt,
            where="post",
            linewidth=2,
            label="Ground Truth"
        )

        pred_line, = ax.step(
            x,
            pred,
            where="post",
            linewidth=2,
            linestyle="--",
            label="Prediction"
        )

        gt_lines.append(gt_line)
        pred_lines.append(pred_line)

        ax.set_ylim(-0.2, 1.2)
        ax.set_yticks([0, 1])
        ax.set_ylabel(f"Class {cls}")
        ax.grid(True, axis="x", alpha=0.3)

    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Sample Index")
    fig.suptitle("One-Hot Predictions vs Ground Truth")

    slider = _prepare_window_slider(fig, axes, y_true, window)

    def update(val):
        start = int(slider.val)
        end = min(start + window, len(y_true))
        x = np.arange(start, end)

        for cls, ax in enumerate(axes):
            gt = (y_true[start:end] == cls).astype(int)
            pred = (y_pred[start:end] == cls).astype(int)

            gt_lines[cls].set_data(x, gt)
            pred_lines[cls].set_data(x, pred)
            ax.set_xlim(start, start + window)

        fig.canvas.draw_idle()

    slider.on_changed(update)

    for ax in axes:
        ax.set_xlim(0, window)

    plt.show()


def plot_softmax_with_slider(softmax_outputs, y_true=None, num_classes=None, window=200):
    """
    Plottet die Softmax-Wahrscheinlichkeiten pro Klasse
    plus passende Ground-Truth-One-Hot-Linie pro Klasse.

    softmax_outputs shape:
    (num_samples, num_classes)
    """

    softmax_outputs = np.asarray(softmax_outputs)

    if num_classes is None:
        num_classes = softmax_outputs.shape[1]

    if y_true is None:
        y_true = np.zeros(len(softmax_outputs))
    else:
        y_true = np.asarray(y_true)

    fig, axes = plt.subplots(
        num_classes,
        1,
        figsize=(15, 2 * num_classes),
        sharex=True
    )

    if num_classes == 1:
        axes = [axes]

    start = 0
    end = min(window, len(softmax_outputs))
    x = np.arange(start, end)

    softmax_lines = []
    gt_lines = []

    for cls, ax in enumerate(axes):
        softmax_line, = ax.plot(
            x,
            softmax_outputs[start:end, cls],
            linewidth=2,
            label=f"Softmax Class {cls}"
        )

        gt = (y_true[start:end] == cls).astype(int)

        gt_line, = ax.step(
            x,
            gt,
            where="post",
            linewidth=2,
            linestyle="--",
            label=f"Ground Truth Class {cls}"
        )

        softmax_lines.append(softmax_line)
        gt_lines.append(gt_line)

        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel(f"Class {cls}")
        ax.grid(True, axis="x", alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Sample Index")
    fig.suptitle("Softmax Outputs per Class with Ground Truth")

    slider = _prepare_window_slider(fig, axes, y_true, window)

    def update(val):
        start = int(slider.val)
        end = min(start + window, len(softmax_outputs))
        x = np.arange(start, end)

        for cls, ax in enumerate(axes):
            softmax_lines[cls].set_data(
                x,
                softmax_outputs[start:end, cls]
            )

            gt = (y_true[start:end] == cls).astype(int)
            gt_lines[cls].set_data(x, gt)

            ax.set_xlim(start, start + window)

        fig.canvas.draw_idle()

    slider.on_changed(update)

    for ax in axes:
        ax.set_xlim(0, window)

    plt.show()


def plot_confidence_with_slider(softmax_outputs, y_true=None, y_pred=None, window=200):
    """
    Plottet die Modell-Confidence:
    max(softmax_outputs) pro Sample.
    
    Optional:
    - y_true und y_pred können übergeben werden.
    - Falsch klassifizierte Bereiche werden dann zusätzlich markiert.
    """

    softmax_outputs = np.asarray(softmax_outputs)
    confidence = np.max(softmax_outputs, axis=1)

    if y_true is not None:
        y_true = np.asarray(y_true)

    if y_pred is None:
        y_pred = np.argmax(softmax_outputs, axis=1)
    else:
        y_pred = np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(15, 4))

    start = 0
    end = min(window, len(confidence))
    x = np.arange(start, end)

    conf_line, = ax.plot(
        x,
        confidence[start:end],
        linewidth=2,
        label="Confidence"
    )

    error_scatter = None

    if y_true is not None:
        errors = y_true[start:end] != y_pred[start:end]

        error_scatter = ax.scatter(
            x[errors],
            confidence[start:end][errors],
            marker="x",
            s=40,
            label="Wrong prediction"
        )

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, window)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Confidence")
    ax.set_title("Prediction Confidence")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="upper right")

    slider = _prepare_window_slider(fig, [ax], np.arange(len(confidence)), window)

    def update(val):
        nonlocal error_scatter

        start = int(slider.val)
        end = min(start + window, len(confidence))
        x = np.arange(start, end)

        conf_line.set_data(x, confidence[start:end])
        ax.set_xlim(start, start + window)

        if y_true is not None:
            if error_scatter is not None:
                error_scatter.remove()

            errors = y_true[start:end] != y_pred[start:end]

            error_scatter = ax.scatter(
                x[errors],
                confidence[start:end][errors],
                marker="x",
                s=40,
                label="Wrong prediction"
            )

        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()
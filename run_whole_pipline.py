import time
import nbformat
import numpy as np

from nbclient import NotebookClient
from datetime import datetime
from pathlib import Path
from sklearn.metrics import classification_report

from src.pipeline_config import RUNS


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def mean_std(values):
    return np.mean(values), np.std(values)


def format_mean_std(values):
    mean, std = mean_std(values)
    return f"{mean:.4f} ± {std:.4f}"


def format_confusion_matrix(cm):
    return "\n".join([
        " ".join([f"{int(value):6d}" for value in row])
        for row in cm
    ])


def format_classification_report(all_labels, all_preds):
    return classification_report(
        all_labels,
        all_preds,
        zero_division=0,
        digits=4
    )


# ------------------------------------------------------------
# Setup parameters
# ------------------------------------------------------------
notebook_path = "dataset_builder.ipynb"

n_iterations = 10

models = [
    "CNN1D_binary", "CNN1D_multiclass",
    "CNN1D_RF_binary", "CNN1D_RF_multiclass",
    "ResNet_binary", "ResNet_multiclass",
    "InceptionTime_binary", "InceptionTime_multiclass",
    "FCN_binary", "FCN_multiclass",
    "OmniScaleCNN_binary", "OmniScaleCNN_multiclass",
]


# ------------------------------------------------------------
# Containers for metrics
# ------------------------------------------------------------
results = {
    model_name: {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "macro_f1": [],
        "confusion_matrices": [],
        "classification_reports": [],
        "training_time": [],
        "inference_time": [],
    }
    for model_name in models
}


# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------
for i in range(n_iterations):
    print(f"\nRunning iteration {i + 1}/{n_iterations}")
    print("=" * 60)

    # Load notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Execute dataset builder
    client = NotebookClient(nb, timeout=None, kernel_name="python3")
    client.execute()

    # Save executed notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print("Executed notebook successfully")

    # Train and test all selected models
    for model_name in models:
        if model_name not in RUNS:
            raise ValueError(f"Unknown model name: {model_name}")

        config = RUNS[model_name]

        print(f"\nTraining {model_name}...")

        train_start = time.time()
        config["train_func"](**config["train_kwargs"])
        train_end = time.time()

        print(f"Testing {model_name}...")

        test_start = time.time()
        acc, prec, rec, f1, cm, all_labels, all_preds, softmax_outputs = (
            config["test_func"](**config["test_kwargs"])
        )
        test_end = time.time()


        results[model_name]["accuracy"].append(acc)
        results[model_name]["precision"].append(prec)
        results[model_name]["recall"].append(rec)
        results[model_name]["f1"].append(f1)
        results[model_name]["confusion_matrices"].append(cm)
        results[model_name]["classification_reports"].append(
            format_classification_report(all_labels, all_preds)
        )
        results[model_name]["training_time"].append(train_end - train_start)
        results[model_name]["inference_time"].append(test_end - test_start)

        print(f"{model_name} finished")
        print("-" * 60)


# ------------------------------------------------------------
# Print average results and testing log
# ------------------------------------------------------------
print("\nAverage results")
print("=" * 60)

Path("Testing_log").mkdir(exist_ok=True)
log_path = "Testing_log/results_log.txt"

with open(log_path, "a", encoding="utf-8") as log_file:
    log_file.write("\n" + "#" * 90 + "\n")
    log_file.write(f"PIPELINE RUN: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    log_file.write(f"Iterations: {n_iterations}\n")
    log_file.write("#" * 90 + "\n\n")

    for model_name, metrics in results.items():

        # Keep console output as before
        print(f"\n{model_name}")
        print(f"Accuracy:  {format_mean_std(metrics['accuracy'])}")
        print(f"Precision: {format_mean_std(metrics['precision'])}")
        print(f"Recall:    {format_mean_std(metrics['recall'])}")
        print(f"F1-Score:  {format_mean_std(metrics['f1'])}")

        mean_cm = np.mean(metrics["confusion_matrices"], axis=0)

        log_file.write("=" * 90 + "\n")
        log_file.write(f"MODEL: {model_name}\n")
        log_file.write("=" * 90 + "\n\n")

        log_file.write("[SUMMARY METRICS]\n")
        log_file.write(f"Accuracy:       {format_mean_std(metrics['accuracy'])}\n")
        log_file.write(f"Precision:      {format_mean_std(metrics['precision'])}\n")
        log_file.write(f"Recall:         {format_mean_std(metrics['recall'])}\n")
        log_file.write(f"F1-Score:       {format_mean_std(metrics['f1'])}\n")
        log_file.write(f"Training time:  {format_mean_std(metrics['training_time'])} s\n")
        log_file.write(f"Inference time: {format_mean_std(metrics['inference_time'])} s\n\n")

        log_file.write("[CONFUSION MATRIX - MEAN OVER ITERATIONS]\n")
        log_file.write(format_confusion_matrix(mean_cm))
        log_file.write("\n\n")

        log_file.write("[PER-CLASS PRECISION / RECALL / F1]\n")
        log_file.write("Last iteration report:\n")
        log_file.write(metrics["classification_reports"][-1])
        log_file.write("\n\n")
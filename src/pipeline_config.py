from src.train_and_eval_binary import run_train_and_eval_binary
from src.train_and_eval_multiclass import run_train_and_eval_multiclass
from src.train_and_eval_RF import run_train_and_eval_RF_binary
from src.train_and_eval_RF_multiclass import run_train_and_eval_RF_multiclass
from src.train_and_eval_tsai import run_train_and_eval_tsai
from src.test_models import run_test_models


RUNS = {
    # ------------------------------------------------------------
    # CNN1D
    # ------------------------------------------------------------
    "CNN1D_binary": {
        "train_func": run_train_and_eval_binary,
        "train_kwargs": {
            "debug": False,
        },
        "test_func": run_test_models,
        "test_kwargs": {
            "rf_combination": False,
            "evaluation_mode": "binary",
            "num_classes": 2,
            "cnn_model_path": r"src\Models\CNN_binary.pth",
            "test_dataset_path": "datasets_output/test_dataset_binary.npz",
            "model_architecture": "CNN1D_V3",
            "debug": False,
        },
    },

    "CNN1D_multiclass": {
        "train_func": run_train_and_eval_multiclass,
        "train_kwargs": {
            "debug": False,
        },
        "test_func": run_test_models,
        "test_kwargs": {
            "rf_combination": False,
            "evaluation_mode": "multiclass",
            "num_classes": 6,
            "cnn_model_path": r"src\Models\CNN_multiclass.pth",
            "test_dataset_path": "datasets_output/test_dataset.npz",
            "model_architecture": "CNN1D_V3",
            "debug": False,
        },
    },

    # ------------------------------------------------------------
    # CNN1D + Random Forest
    # ------------------------------------------------------------
    "CNN1D_RF_binary": {
        "train_func": run_train_and_eval_RF_binary,
        "train_kwargs": {
            "debug": False,
        },
        "test_func": run_test_models,
        "test_kwargs": {
            "rf_combination": True,
            "evaluation_mode": "binary",
            "num_classes": 2,
            "cnn_model_path": r"src\Models\CNN_RF\CNN_RF_binary.pth",
            "rf_model_path": r"src\Models\CNN_RF\RF_binary.pkl",
            "test_dataset_path": "datasets_output/test_dataset_binary.npz",
            "model_architecture": "CNN1D_V3",
            "debug": False,
        },
    },

    "CNN1D_RF_multiclass": {
        "train_func": run_train_and_eval_RF_multiclass,
        "train_kwargs": {
            "debug": False,
        },
        "test_func": run_test_models,
        "test_kwargs": {
            "rf_combination": True,
            "evaluation_mode": "multiclass",
            "num_classes": 6,
            "cnn_model_path": r"src\Models\CNN_RF\CNN_RF_multiclass.pth",
            "rf_model_path": r"src\Models\CNN_RF\RF_multiclass.pkl",
            "test_dataset_path": "datasets_output/test_dataset.npz",
            "model_architecture": "CNN1D_V3",
            "debug": False,
        },
    },

    # ------------------------------------------------------------
    # tsai Modelle binary
    # ------------------------------------------------------------
    "ResNet_binary": {
        "train_func": run_train_and_eval_tsai,
        "train_kwargs": {
            "debug": False,
            "metric_calculation": "binary",
            "mode": "binary",
            "train_dataset_path": "datasets_output/train_dataset_augmented_binary.npz",
            "val_dataset_path": "datasets_output/val_dataset_augmented_binary.npz",
            "models_to_train": ["ResNet"],
        },
        "test_func": run_test_models,
        "test_kwargs": {
            "rf_combination": False,
            "evaluation_mode": "binary",
            "num_classes": 2,
            "cnn_model_path": r"src\Models\models\ResNet_binary.pth",
            "test_dataset_path": "datasets_output/test_dataset_binary.npz",
            "model_architecture": "ResNet",
            "debug": False,
        },
    },

    "InceptionTime_binary": {
        "train_func": run_train_and_eval_tsai,
        "train_kwargs": {
            "debug": False,
            "metric_calculation": "binary",
            "mode": "binary",
            "train_dataset_path": "datasets_output/train_dataset_augmented_binary.npz",
            "val_dataset_path": "datasets_output/val_dataset_augmented_binary.npz",
            "models_to_train": ["InceptionTime"],
        },
        "test_func": run_test_models,
        "test_kwargs": {
            "rf_combination": False,
            "evaluation_mode": "binary",
            "num_classes": 2,
            "cnn_model_path": r"src\Models\models\InceptionTime_binary.pth",
            "test_dataset_path": "datasets_output/test_dataset_binary.npz",
            "model_architecture": "InceptionTime",
            "debug": False,
        },
    },

    "FCN_binary": {
        "train_func": run_train_and_eval_tsai,
        "train_kwargs": {
            "debug": False,
            "metric_calculation": "binary",
            "mode": "binary",
            "train_dataset_path": "datasets_output/train_dataset_augmented_binary.npz",
            "val_dataset_path": "datasets_output/val_dataset_augmented_binary.npz",
            "models_to_train": ["FCN"],
        },
        "test_func": run_test_models,
        "test_kwargs": {
            "rf_combination": False,
            "evaluation_mode": "binary",
            "num_classes": 2,
            "cnn_model_path": r"src\Models\models\FCN_binary.pth",
            "test_dataset_path": "datasets_output/test_dataset_binary.npz",
            "model_architecture": "FCN",
            "debug": False,
        },
    },

    "OmniScaleCNN_binary": {
        "train_func": run_train_and_eval_tsai,
        "train_kwargs": {
            "debug": False,
            "metric_calculation": "binary",
            "mode": "binary",
            "train_dataset_path": "datasets_output/train_dataset_augmented_binary.npz",
            "val_dataset_path": "datasets_output/val_dataset_augmented_binary.npz",
            "models_to_train": ["OmniScaleCNN"],
        },
        "test_func": run_test_models,
        "test_kwargs": {
            "rf_combination": False,
            "evaluation_mode": "binary",
            "num_classes": 2,
            "cnn_model_path": r"src\Models\models\OmniScaleCNN_binary.pth",
            "test_dataset_path": "datasets_output/test_dataset_binary.npz",
            "model_architecture": "OmniScaleCNN",
            "debug": False,
        },
    },

    # ------------------------------------------------------------
    # tsai Modelle multiclass
    # ------------------------------------------------------------
    "ResNet_multiclass": {
        "train_func": run_train_and_eval_tsai,
        "train_kwargs": {
            "debug": False,
            "metric_calculation": "macro",
            "mode": "multiclass",
            "train_dataset_path": "datasets_output/train_dataset_augmented.npz",
            "val_dataset_path": "datasets_output/val_dataset_augmented.npz",
            "models_to_train": ["ResNet"],
        },
        "test_func": run_test_models,
        "test_kwargs": {
            "rf_combination": False,
            "evaluation_mode": "multiclass",
            "num_classes": 6,
            "cnn_model_path": r"src\Models\models\ResNet_multiclass.pth",
            "test_dataset_path": "datasets_output/test_dataset.npz",
            "model_architecture": "ResNet",
            "debug": False,
        },
    },

    "InceptionTime_multiclass": {
        "train_func": run_train_and_eval_tsai,
        "train_kwargs": {
            "debug": False,
            "metric_calculation": "macro",
            "mode": "multiclass",
            "train_dataset_path": "datasets_output/train_dataset_augmented.npz",
            "val_dataset_path": "datasets_output/val_dataset_augmented.npz",
            "models_to_train": ["InceptionTime"],
        },
        "test_func": run_test_models,
        "test_kwargs": {
            "rf_combination": False,
            "evaluation_mode": "multiclass",
            "num_classes": 6,
            "cnn_model_path": r"src\Models\models\InceptionTime_multiclass.pth",
            "test_dataset_path": "datasets_output/test_dataset.npz",
            "model_architecture": "InceptionTime",
            "debug": False,
        },
    },

    "FCN_multiclass": {
        "train_func": run_train_and_eval_tsai,
        "train_kwargs": {
            "debug": False,
            "metric_calculation": "macro",
            "mode": "multiclass",
            "train_dataset_path": "datasets_output/train_dataset_augmented.npz",
            "val_dataset_path": "datasets_output/val_dataset_augmented.npz",
            "models_to_train": ["FCN"],
        },
        "test_func": run_test_models,
        "test_kwargs": {
            "rf_combination": False,
            "evaluation_mode": "multiclass",
            "num_classes": 6,
            "cnn_model_path": r"src\Models\models\FCN_multiclass.pth",
            "test_dataset_path": "datasets_output/test_dataset.npz",
            "model_architecture": "FCN",
            "debug": False,
        },
    },

    "OmniScaleCNN_multiclass": {
        "train_func": run_train_and_eval_tsai,
        "train_kwargs": {
            "debug": False,
            "metric_calculation": "macro",
            "mode": "multiclass",
            "train_dataset_path": "datasets_output/train_dataset_augmented.npz",
            "val_dataset_path": "datasets_output/val_dataset_augmented.npz",
            "models_to_train": ["OmniScaleCNN"],
        },
        "test_func": run_test_models,
        "test_kwargs": {
            "rf_combination": False,
            "evaluation_mode": "multiclass",
            "num_classes": 6,
            "cnn_model_path": r"src\Models\models\OmniScaleCNN_multiclass.pth",
            "test_dataset_path": "datasets_output/test_dataset.npz",
            "model_architecture": "OmniScaleCNN",
            "debug": False,
        },
    },
}
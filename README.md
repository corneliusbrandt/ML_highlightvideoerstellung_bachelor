# Automated Mountain Bike Highlight Video Generation

This repository contains the source code developed as part of my Bachelor's thesis.

The objective of this project is the automatic detection of relevant downhill mountain bike riding events from wearable IMU sensor data using machine learning. The detected events are intended to serve as the basis for an automated highlight video generation pipeline.

---

# Project Overview

Creating highlight videos from long downhill mountain bike recordings is a time-consuming task that usually requires manually reviewing the complete video footage.

This project investigates whether riding events can be detected solely from inertial sensor data collected by three wearable IMUs.

The complete software pipeline includes

- automatic dataset generation
- preprocessing and synchronization of sensor data
- optional data augmentation
- model training
- validation and test evaluation
- automated experiment logging

Several custom and literature-based time series classification models are implemented and compared.

---

# Repository Structure

```
.
├── Labeled/
│   └── Label files (not included)
│
├── sensor_and_video_data/
│   └── Raw sensor recordings and videos (not included)
│
├── src/
│   ├── Dataset generation
│   ├── Data preprocessing
│   ├── Data augmentation
│   ├── Model architectures
│   ├── Training scripts
│   ├── Evaluation scripts
│   ├── Helper functions
│   └── Saved model checkpoints
│
├── Testing_log/
│   └── Automatically generated experiment logs
│
├── run_whole_pipeline.py
│
└── requirements.txt
```

---

# Dataset

The raw dataset and the label files are **not included** in this repository, since they are not publicly available. However they sould be included in the repository like this:

Create the following folder structure:

```
sensor_and_video_data/
Labeled/
```

Place all run folders and label files inside these folder. It should look like this:

```
sensor_and_video_data/
    ├── 0713_0833
    ├── 0713_0903
    ├── 0713_0932
    ├── 0717_0717
    └── ...
Labeled/
    ├── 0713_0833_hot.json
    ├── 0713_0833_sequences.json
    ├── 0713_0903_hot.json
    ├── 0713_0903_sequences.json
    └── ...
```

---

# Installation

The project was developed using

- Python 3.13.7
- Windows

Install all required packages using

```bash
pip install -r requirements.txt
```

---

# Complete Pipeline

The entire workflow can be executed using

```bash
python run_whole_pipeline.py
```

The pipeline automatically performs the following steps:

1. Dataset generation
2. Data preprocessing
3. Optional data augmentation depending on which dataset you load
4. Model training
5. Validation evaluation
6. Test evaluation
7. Logging of all results

---

# Implemented Models

## Custom Models

- CNN1D_V1
- CNN1D_V2
- CNN1D_V3
- CNN Feature Extractor + Random Forest (Binary)
- CNN Feature Extractor + Random Forest (Multiclass)

---

## Literature Models

The following state-of-the-art Time Series Classification models are included.

- Fully Convolutional Network (FCN)
- ResNet
- InceptionTime
- Omni-Scale CNN (OSCNN)
- HIVE-COTE

---

# Dataset Generation

The dataset generation pipeline automatically performs

- synchronization of sensor and video data
- label generation
- signal preprocessing
- sliding window generation
- train/validation/test split
- optional data augmentation
- normalization
- dataset export

The generated datasets are then used automatically by the training pipeline.

---

# Training

Each configured model is automatically

- initialized
- trained
- validated
- saved
- evaluated on the test dataset

No manual intervention is required during execution.

---

# Configuration

Most experiment parameters can be configured inside the configuration file (```pipeline_config.py```), the specific model training files (e.g. ```train_and_eval_RF.py```) or inside the data pipeline (```dataset_builder.jpynb```)

Typical parameters include

- model architecture
- binary / multiclass classification
- sequence width
- step size
- window size
- learning rate
- batch size
- number of epochs
- focal loss parameters
- class weight scaling
- data augmentation
- evaluation mode (binary or multiclass)

---

# Output

Depending on the selected configuration, the pipeline automatically generates

- trained model checkpoints
- confusion matrices
- evaluation metrics
- precision, recall and F1-score
- experiment logs
- datasets

---

# Experiment Logs

Every execution of the pipeline automatically creates a log file containing

- training configuration
- evaluation metrics
- confusion matrices
- classification reports
- execution times
- average performance over multiple runs

This enables reproducible experiments and simplifies model comparison.


---

# Notes

This repository accompanies the Bachelor's thesis

> **Automatische Highlight-Videoerstellung
beim Downhill-Mountainbiking**

The current focus of the project is automatic event detection.

The final automatic video generation component has not yet been implemented due to the achieved classification performance.

Future work may include

- larger datasets
- additional model architectures
- improved data augmentation
- multimodal sensor fusion
- automatic highlight video generation
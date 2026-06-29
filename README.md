# Automated Mountain Bike Highlight Video Generation

This repository contains the source code developed as part of my Bachelor's thesis. The project aims to automatically generate highlight videos of downhill mountain bike rides by detecting relevant riding events from inertial sensor data using machine learning.

---

## Project Overview

Creating highlight videos from downhill mountain bike footage is a time-consuming process that typically requires manually reviewing long video recordings and selecting interesting sections.

This project aims to autmomate this workflow by combining wearable IMU sensor data with machine learning models to detect relevant riding events such as:

* Jumps
* Left turns
* Right turns
* Rear wheel blocks
* Other riding events

The detected events can subsequently be used to generate compact highlight videos. However this functionality is not yet finished.

---

## Repository Structure

```
.
├── Labeled/
│   └── Label files used for dataset generation
│
├── src/
│   ├── Training and evaluation scripts
│   ├── Saved model checkpoints
│   ├── Helper functions
|   └── Dataset building scripts
│
└── Testing_log/
    └── Test log files
```

---

## Dataset

The raw sensor recordings are **not included** in this repository.

The dataset must be downloaded separately.

**Dataset download:**

> **TODO:** Insert download link

After downloading, place the dataset in the samr directory as all the other folders before running the dataset generation pipeline. All the data should be located in a folder called ```sensor_and_video_data```. 

---

## Installation

The project was developed using:

* Python 3.13.7
* Windows

Install all required dependencies via

```bash
pip install -r requirements.txt
```

---

## Supported Models

The repository contains implementations and evaluation scripts for the following models:

### Custom Models

* CNN1D_V1
* CNN1D_V2
* CNN1D_V3
* CNN Feature Extractor + Random Forest Classifier

Trained and evaluated on the validation set using the following scripts:

```
train_and_eval_binary.py
train_and_eval_multiclass.py
train_and_eval_RF.py
train_and_eval_RF_multiclass.py
```

### Existing Time Series Classification Models

* Fully Convolutional Network (FCN)
* ResNet
* InceptionTime
* Omni-Scale CNN (OS-CNN)
* HIVE-COTE

Trained and evaluated on the validation set using the following scripts:

```
train_and_eval_tsai.py
train_and_eval_HIVECOTE.py
```

---

## Workflow

The complete workflow consists of the following steps.

### 1. Dataset Generation

```
dataset_builder.ipynb
```

This notebook

* loads the raw sensor recordings
* synchronizes sensor and label data
* preprocesses the signals
* generates sliding windows
* performs optional data augmentation
* exports the final datasets.

---

### 2. Model Training

Run one of the training scripts inside the `src` directory.

Depending on the selected script, different model architectures are trained.

---

### 3. Model Evaluation on Test Dataset

After training, the models can be evaluated on the testdataset using ```test_models.py```. Make sure to adjust the following parameter in the test script depending on which model you want to run:
* num_classes (2 or 6 for binary or multiclass)
* rf_combiation (wether the model you want to test is a combination of CNN and Random Forest or not)
* evaluation_mode (multiclass or binary)
* cnn_model_patch and rf_model_path
* Path to the dataset
* used model

---

## Configuration

Training parameters are configured directly inside the respective training scripts.

Typical adjustable parameters include:

* model architecture
* number of classes
* learning rate
* batch size
* number of epochs
* optimizer
* loss function
* focal loss parameters
* class weights
* early stopping settings
* decision threshold

This allows every experiment to be configured independently.

---

## Output

Depending on the executed script, the repository generates

* trained model checkpoints
* evaluation metrics
* confusion matrices
* generated datasets

---

## Notes

The repository is intended for research and educational purposes as part of a Bachelor's thesis on automatic event detection in downhill mountain biking using inertial sensor data.

Future work may extend the pipeline with additional datasets, model architectures, or improved highlight video generation methods.


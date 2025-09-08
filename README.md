# CognitionForge 🧠

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end framework for training and deploying regression models on tabular data. CognitionForge provides a complete pipeline using PyTorch and Scikit-learn to predict teen phone addiction levels from a complex dataset of usage and behavioral features.

This project demonstrates a full machine learning lifecycle, from data preprocessing to final evaluation and single-point prediction.

---

## 📋 Core Features

* **Complete ML Pipeline:** Handles the entire workflow: data loading, preprocessing, training, validation, evaluation, and prediction.
* **Robust Preprocessing:** Utilizes a `scikit-learn` pipeline to handle mixed data types, automatically applying `StandardScaler` to numerical features and `OneHotEncoder` to categorical features.
* **Custom Neural Network:** Implements a custom feedforward neural network in PyTorch, designed specifically for this regression task.
* **Three Operational Modes:** The script can be run in three modes via command-line arguments:
    * `--mode train`: Trains the model from scratch and saves the best checkpoint and preprocessor.
    * `--mode evaluate`: Loads the saved model to evaluate its performance on the test set.
    * `--mode predict`: Loads the model to make a prediction on a sample data point.
* **Multi-GPU Support:** Automatically leverages `torch.nn.DataParallel` for accelerated training.

---

## 🚀 Performance

After training, the model achieves a high level of accuracy on the held-out test set, demonstrating its predictive power.

* **Final Test Mean Absolute Error (MAE): 0.1342**
    * This means the model's predictions are, on average, off by only 0.13 points on the addiction level scale.

---

## 🛠️ Getting Started

Follow these instructions to set up and run the CognitionForge framework.

> **Note**: This project was developed and tested on **Ubuntu 22.04 LTS**. While it is expected to be cross-platform, minor adjustments might be needed for other operating systems.

### 1. Installation

First, clone the repository and navigate into the project directory:
```bash
git clone [https://github.com/TheVictor777/CognitionForge.git](https://github.com/TheVictor777/CognitionForge.git)
cd CognitionForge
```

Next, create and activate a Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

Then, install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation

The dataset is not included in this repository.

1.  Create a folder named `Datasets` in the project's root directory.
2.  **Download the dataset** from [Kaggle: Teen Phone Addiction Dataset](https://www.kaggle.com/datasets/sumedh1507/teen-phone-addiction).
3.  Unzip the file and ensure the final path to the data is `Datasets/Teen Smartphone Usage and Addiction Impact Dataset/teen_phone_addiction_dataset.csv`.

### 3. Running the Framework

The entire pipeline is controlled via command-line arguments.

* **To train the model from scratch:**
    ```bash
    python3 CognitionForge.py --mode train
    ```

* **To evaluate the final performance of a trained model:**
    ```bash
    python3 CognitionForge.py --mode evaluate
    ```

* **To make a prediction on a sample data point:**
    ```bash
    python3 CognitionForge.py --mode predict
    ```

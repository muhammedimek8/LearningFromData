# LearningFromData
# Sentiment Classification of IMDb Movie Reviews

This repository contains a complete machine learning pipeline for binary sentiment classification of movie reviews.  
The goal of the project is to classify reviews as **positive** or **negative** using traditional machine learning methods.

---

## Project Overview

In this project, we:
- Process raw movie review data
- Apply text preprocessing techniques
- Extract numerical features using TF-IDF
- Train and evaluate multiple classification models
- Compare model performances using standard evaluation metrics

---

## Project Structure

PythonProject/
├── aclImdb/ # Raw IMDb dataset (optional, already processed)
├── data/
│ └── dataset.csv # Final processed dataset (included)
├── preprocessing.py # Text preprocessing functions
├── features.py # Feature extraction (TF-IDF + custom features)
├── models.py # Model definitions
├── prepare_imdb_dataset.py # Script to generate dataset.csv
├── train_evaluate.py # Training and evaluation pipeline
├── requirements.txt # Project dependencies
└── README.md

yaml
Kodu kopyala

---

## Dataset

The dataset used in this project is based on the **IMDb Large Movie Review Dataset**.

- Total samples: **25,000**
- Positive reviews: **12,500**
- Negative reviews: **12,500**
- Class distribution: **Balanced**

The processed dataset (`data/dataset.csv`) is already included in this repository.

---

## Setup Instructions

### 1. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
2. Install dependencies
bash
Kodu kopyala
pip install -r requirements.txt
Running the Project
Step 1: (Optional) Re-create the dataset
This step is optional since dataset.csv is already provided.

bash
Kodu kopyala
python prepare_imdb_dataset.py
Step 2: Train and evaluate models
bash
Kodu kopyala
python train_evaluate.py
This script performs the following steps:

Text preprocessing

Feature extraction using TF-IDF

Addition of handcrafted features

Model training

5-fold cross-validation

Performance evaluation using classification reports and confusion matrices

Models Used
Logistic Regression

Linear Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Random Forest

Multilayer Perceptron (MLP)

Evaluation Metrics
Accuracy

Precision

Recall

F1-Score

Confusion Matrix

Notes
The full pipeline is implemented using clean and modular Python scripts.

A Jupyter Notebook is not required, as the entire workflow is reproducible via Python files.

The dataset is balanced, so no additional resampling techniques were applied.

References
Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011).
Learning word vectors for sentiment analysis. Proceedings of the ACL.

IMDb Large Movie Review Dataset:
https://ai.stanford.edu/~amaas/data/sentiment/

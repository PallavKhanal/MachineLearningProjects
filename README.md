# Machine Learning Projects

A curated collection of machine learning projects spanning **regression**, **classification**, **tree-based models**, **deep learning**, and **neural networks built from scratch**. Each project is self-contained with clear explanations, exploratory analysis, and production-style workflows.

---

## Table of Contents

- [Overview](#overview)
- [Projects](#projects)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)

---

## Overview

This repository showcases end-to-end ML workflows—from foundational algorithms (linear and logistic regression, decision trees) to modern deep learning (CNNs on CIFAR-10) and educational implementations (autograd and feedforward nets from scratch). Projects use real or canonical datasets and emphasize interpretability, reproducibility, and clean code.

| Project | Type | Key Techniques |
|--------|------|----------------|
| [Ames Housing](#1-ames-housing) | Regression | Linear regression, OLS, gradient descent, feature scaling |
| [Titanic Survival](#2-titanic-survival-prediction) | Classification | Logistic regression, EDA, imputation, class balance |
| [Loan Approval](#3-loan-approval-predictions) | Classification | Decision trees, ensemble methods, Gini/entropy |
| [Australian Open](#4-australian-open-predictor) | Classification | Random forest, time-series features, sports analytics |
| [CIFAR-10](#5-cifar-10) | Image classification | CNNs, TensorFlow/Keras, data augmentation |
| [FNN from scratch](#6-fnn-from-scratch) | Educational | Autograd, backprop, MLP, XOR & two-moons |

---

## Projects

### 1. Ames Housing

**Notebook:** [`Ames_Housing.ipynb`](Ames_Housing.ipynb)

Predict **house prices** using the Ames Housing dataset. Introduces **linear regression** in depth: the normal equation (closed-form OLS), gradient descent, and the role of **feature scaling**. Covers the math (MSE, derivatives, matrix form) and practical considerations for regression in production.

- **Goal:** Predict continuous sale price from features (size, bedrooms, condition, etc.).
- **Highlights:** OLS derivation, gradient descent from first principles, scaling (e.g. StandardScaler), train/validation split, and evaluation metrics.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PallavKhanal/MachineLearningProjects/blob/main/Ames_Housing.ipynb)

---

### 2. Titanic Survival Prediction

**Notebook:** [`Titanic_Survival_Prediction.ipynb`](Titanic_Survival_Prediction.ipynb)

**Binary classification:** predict whether a passenger **survived** (1) or **did not survive** (0) the Titanic disaster. Built with **logistic regression** and careful EDA. Demonstrates handling class imbalance, missing data (e.g. age, embarked), and feature engineering (e.g. group-wise imputation).

- **Goal:** Classify survival from demographics and ticket information.
- **Highlights:** Sigmoid and decision boundary, class balance and metrics, missing-value strategy, and visual EDA (e.g. survival by class and sex).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PallavKhanal/MachineLearningProjects/blob/main/Titanic_Survival_Prediction.ipynb)

---

### 3. Loan Approval Predictions

**Notebook:** [`Loan_Approval_Predictions.ipynb`](Loan_Approval_Predictions.ipynb)

Predict **loan approval** using **decision trees** and related ensemble methods. Moves beyond linear boundaries: the model learns if-then rules (e.g. income and employment thresholds). Covers split criteria (Gini impurity, entropy), pruning, and the transition from a single tree to more robust ensembles.

- **Goal:** Classify approval/rejection from applicant and loan features.
- **Highlights:** How trees choose splits, impurity measures, and interpretable rule-based predictions.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PallavKhanal/MachineLearningProjects/blob/main/Loan_Approval_Predictions.ipynb)

---

### 4. Australian Open Predictor

**Notebook:** [`Australian_Open_Predictor.ipynb`](Australian_Open_Predictor.ipynb)

**Sports analytics:** predict **match outcomes** on the ATP tour (e.g. who wins a given match). Uses ATP match data (2000–2024), restricted to **hard courts** to align with the Australian Open. A **Random Forest** is trained on **rolling, pre-match statistics** so the model uses only information available before each match—no leakage from the outcome.

- **Goal:** Simulate the Australian Open bracket and evaluate predictions (e.g. tournament winner).
- **Highlights:** Time-aware feature engineering, train/test split by time, and model interpretation (e.g. feature importance).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PallavKhanal/MachineLearningProjects/blob/main/Australian_Open_Predictor.ipynb)

---

### 5. CIFAR-10

**Notebook:** [`CIFAR10.ipynb`](CIFAR10.ipynb)

**Image classification** on the **CIFAR-10** dataset (60k 32×32 color images in 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). Implemented with **TensorFlow/Keras**: convolutional layers, pooling, dropout, and optional data augmentation.

- **Goal:** Train a CNN to classify small natural images.
- **Highlights:** Data loading and preprocessing, CNN architecture design, training loop, and evaluation on the test set.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PallavKhanal/MachineLearningProjects/blob/main/CIFAR10.ipynb)

---

### 6. FNN from Scratch

**Folder:** [`fnn/`](fnn/)

A **feedforward neural network** implemented **from scratch** (no PyTorch/TensorFlow): custom **autograd** (micrograd-style), **backpropagation**, and **MLP** with tanh activations. Used for two minimal benchmarks: **XOR** (non-linear separability) and **two-moons** (sklearn), with training curves and decision-boundary visualizations.

- **Goal:** Understand gradients, backprop, and multi-layer nets by building them step by step.
- **Contents:**
  - `fnn.py` — `Value` class (differentiable scalars), `Neuron`, `Layer`, `MLP`, and backward pass.
  - `fnn.ipynb` — Interactive notebook with computation-graph visualization and training.
  - `xor_train.py` — Train MLP on the XOR truth table; plot loss and predictions.
  - `train_moons.py` — Train on `make_moons`; plot loss, decision boundary, and predictions.
- **Highlights:** Autograd design, chain rule in code, and minimal dependencies (NumPy/sklearn only for data and plotting).

**Run locally:**

```bash
cd fnn
pip install numpy matplotlib scikit-learn   # if needed
python xor_train.py
python train_moons.py
```

---

## Tech Stack

| Category | Tools |
|----------|--------|
| **Language** | Python 3 |
| **Numerical / ML** | NumPy, pandas, scikit-learn |
| **Deep learning** | TensorFlow, Keras |
| **Visualization** | Matplotlib, Seaborn |
| **Notebooks** | Jupyter, Google Colab |

---

## Repository Structure

```
MachineLearningProjects/
├── README.md
├── Ames_Housing.ipynb
├── Australian_Open_Predictor.ipynb
├── CIFAR10.ipynb
├── Loan_Approval_Predictions.ipynb
├── Titanic_Survival_Prediction.ipynb
└── fnn/
    ├── fnn.py           # Autograd + MLP implementation
    ├── fnn.ipynb        # Interactive notebook
    ├── xor_train.py     # XOR training script
    ├── train_moons.py   # Two-moons training script
    ├── training_results.png
    └── moons_results.png
```

---

## Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/PallavKhanal/MachineLearningProjects.git
   cd MachineLearningProjects
   ```

2. **Run notebooks**  
   Open any `.ipynb` in Jupyter or use the “Open in Colab” links above. Install dependencies as needed (e.g. `pip install numpy pandas matplotlib scikit-learn tensorflow`).

3. **Run the FNN scripts**  
   From the repo root: `cd fnn && python xor_train.py` or `python train_moons.py`.

---

## License

This repository is for educational and portfolio use. Dataset and third-party asset licenses may vary; see individual notebooks and sources for details.

---

*Maintained by [Pallav Khanal](https://github.com/PallavKhanal).*

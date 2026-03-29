# Medical Insurance Cost Prediction — ML Regression

**Project completed:** January 2026  
**Tools:** Python · Scikit-learn · Pandas · Matplotlib · Seaborn  
**Domain:** Machine Learning · Healthcare · Regression

---

## Overview

An end-to-end machine learning regression pipeline to predict individual medical insurance  
charges based on patient demographics and lifestyle factors.

The project identifies and resolves a critical non-linearity problem caused by a  
**bimodal target distribution** (smokers vs non-smokers behave as two completely  
different populations), leading to a final test **R² of 0.86**.

**Key result:** Random Forest achieved **R² = 0.86** on the test set after resolving  
the non-linearity issue that caused linear models to underperform significantly.

---

## Problem Statement

Predicting insurance costs is not straightforward — the target variable (charges) is  
heavily influenced by smoking status, which creates two distinct cost distributions  
in the data. A single linear model cannot capture both populations simultaneously,  
which is why naive regression fails on this dataset without proper diagnosis.

---

## Methodology

### 1. Exploratory Data Analysis
- Identified **bimodal distribution** in insurance charges caused by smoker vs non-smoker split
- Visualized feature correlations — smoking status emerged as the dominant driver
- Detected skewness in the target variable requiring special handling

### 2. Preprocessing Pipeline
- Built **ColumnTransformer** to handle mixed feature types cleanly
- Applied OneHotEncoding for categorical features (sex, smoker, region)
- Applied StandardScaler for numerical features (age, BMI, children)
- All transformations fitted on training set only — no data leakage

### 3. Models Compared

| Model | Test R² | Notes |
|---|---|---|
| Linear Regression | Low | Fails due to non-linearity |
| ElasticNet | Moderate | Better regularization, still limited |
| Random Forest | **0.86** | Handles non-linearity naturally |

### 4. Non-Linearity Resolution
- Diagnosed why linear models failed using residual plots
- Random Forest naturally handles the smoker/non-smoker split  
  by learning separate decision paths
- Confirmed with residual diagnostics — errors random and evenly distributed

### 5. Feature Importance Analysis
- **Smoking status: ~70% influence** on predicted charges
- Age and BMI as secondary drivers
- Quantified using Random Forest feature importance scores

### 6. Validation
- **5-fold cross-validation** for generalization check
- Bias-variance diagnostics to ensure model is not overfitting

---

## Key Findings

- Smoking status alone explains ~70% of variance in insurance charges
- Linear models are fundamentally wrong for this dataset without transformation
- Residual diagnostics are essential — good R² alone does not confirm a good model
- ColumnTransformer is the correct way to handle mixed-type features in production pipelines

---

## Repository Structure

```
medical-insurance-cost-prediction/
├── medical_insurance_cost_prediction.ipynb   # Full analysis notebook
├── README.md
```

> Dataset sourced from Kaggle — download and place in the root folder before running.

---

## How to Run

1. Clone the repository
```bash
git clone https://github.com/Dev-2004-DA/medical-insurance-cost-prediction.git
```

2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Download the dataset from Kaggle and place it in the project folder

4. Open the notebook
```bash
jupyter notebook medical_insurance_cost_prediction.ipynb
```

---

## Skills Demonstrated

`Regression` `Random Forest` `ElasticNet` `Linear Regression`  
`ColumnTransformer` `Feature Importance` `Residual Diagnostics`  
`Bias-Variance Tradeoff` `Cross-Validation` `Scikit-learn` `Python`

---

*Part of my Data Analytics portfolio — [github.com/Dev-2004-DA](https://github.com/Dev-2004-DA)*

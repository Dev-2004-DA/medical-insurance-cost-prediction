# 🏥 Medical Insurance Cost Prediction  
### Predictive Modeling of Bimodal Healthcare Costs

This project builds a machine learning system to predict individual medical insurance charges using demographic and lifestyle data.  

The most important discovery of this project is that the target variable **charges is bimodal**, meaning the dataset actually contains **two different populations**:

- **Non-smokers** → low cost, low variance  
- **Smokers** → high cost, high variance  

This explains why traditional linear regression models fail and why residuals are never normally distributed.

The final selected model is a **Random Forest Regressor**, which naturally handles this multi-population structure.

---

## 📊 Dataset
The dataset contains **1338 rows** with the following features:

| Column | Description |
|--------|------------|
| age | Age of primary beneficiary |
| sex | Gender (male, female) |
| bmi | Body Mass Index |
| children | Number of dependents |
| smoker | Smoking status |
| region | Residential region |
| charges | Individual medical insurance cost |

---

## 🔍 Key Insights

### 1. Bimodal Target Distribution
KDE and boxplots showed that **charges** come from two different distributions:

- Non-smokers → mostly below 15,000  
- Smokers → heavy right tail reaching above 60,000  

Because of this mixture, the residuals **cannot be normally distributed**.  
This is not a modeling error — it is a **data property**.

---

### 2. Outliers Are Business Signals
Using IQR and 3σ rules:

- 169 values were detected as outliers  
- **157 (≈92%) belong to smokers**

These points represent high-risk customers and must be kept.  
Removing them would make the model unrealistic.

---

### 3. Model Comparison

| Model | Test R² | Test RMSE |
|--------|----------|-------------|
| Linear Regression | ~0.746 | ~5751 |
| ElasticNet | ~0.746 | ~5750 |
| ElasticNet + Polynomial | ~0.849 | ~4437 |
| Decision Tree | ~0.854 | ~4364 |
| **Random Forest (Final)** | **0.859** | **~4277** |
| RF + Polynomial | ~0.86 | ~4214 |

---

### 4. Feature Importance (Random Forest)

| Feature | Importance |
|---------|------------|
| Smoker | ~70% |
| Age | ~10% |
| BMI | ~15% |
| Others | <5% |

Smoking is the dominant factor affecting medical insurance cost.

---

## 🧠 Final Conclusion
- The data is **non-linear and bimodal**.
- Linear models fail due to violated assumptions.
- Tree-based ensemble models handle this structure naturally.
- **Random Forest is the best model** for this problem.

Residual non-normality is not a flaw — it is a reflection of real-world population differences.

---

## 🛠️ Tech Stack
- Python  
- Pandas, NumPy  
- Seaborn, Matplotlib  
- Scikit-learn  
- SciPy  

---

## 📈 Workflow
1. Exploratory Data Analysis  
2. Outlier investigation (retain real extremes)  
3. Categorical encoding  
4. Polynomial feature engineering  
5. Model training & tuning (GridSearchCV)  
6. Residual diagnostics  
7. Feature importance analysis  

---

##
> “The residuals are non-normal due to the bimodal nature of the target variable.  
> 92% of the outliers belong to smokers. These are valid observations, and the Random Forest model successfully captures this variance.”

---


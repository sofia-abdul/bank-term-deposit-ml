# Bank Term Deposit Subscription Prediction

## Overview

This project develops an end-to-end supervised machine learning pipeline to predict whether a bank customer will subscribe to a term deposit product using the UCI Bank Marketing dataset (4,521 records, 17 features).

The task presents a highly imbalanced binary classification problem (~11.5% positive class), requiring careful evaluation beyond simple accuracy metrics.

---

## Business Context

Banks incur substantial costs from inefficient outbound marketing campaigns.

The objective of this project is to:

- Identify high-probability customers before outreach  
- Improve marketing efficiency  
- Reduce wasted campaign expenditure  
- Support data-driven targeting decisions  

---

## Dataset

- **Source:** UCI Bank Marketing Dataset  
- **Records:** 4,521  
- **Features:** 17  
- **Target Variable:** Term deposit subscription (Yes / No)  
- **Class Imbalance:** ~11.5% positive class  

---

## Methodology

### Data Preprocessing

- One-hot encoding for categorical variables  
- Standard scaling for numerical features  
- Class imbalance handling using SMOTE  
- Unified preprocessing using `ColumnTransformer` inside a scikit-learn `Pipeline`  

This ensures reproducibility and prevents data leakage during cross-validation.

---

### Models Implemented

1. **Logistic Regression**
   - Interpretable linear baseline
   - Useful for understanding feature directionality

2. **Random Forest Classifier**
   - Non-linear ensemble model
   - Captures complex feature interactions

---

### Model Selection & Validation

- Stratified train/test split  
- 5-fold cross-validation  
- Hyperparameter tuning using `GridSearchCV`  
- Evaluation metrics:
  - Precision  
  - Recall  
  - F1-score  
  - ROC-AUC  
  - Confusion Matrix  

Performance comparison was conducted to assess trade-offs between interpretability and predictive power.

---

## Results

| Model               | ROC-AUC | Key Observation |
|--------------------|---------|----------------|
| Logistic Regression | ~0.89   | Strong separability but limited minority recall |
| Random Forest       | ~0.90   | Improved performance via non-linear modelling |

Random Forest demonstrated slightly superior discrimination performance.

---

## Feature Importance

Feature analysis revealed:

- **Call duration** as the strongest predictor  
- **Age** and **account balance** as secondary contributors  

Note: Call duration represents post-contact information and may introduce deployment-time data leakage if used in pre-call prediction scenarios.

---

## Key Insights

- Severe class imbalance significantly impacts minority class recall.
- Ensemble methods outperform linear models on structured tabular data.
- Precision–recall trade-offs are crucial when modelling marketing response.
- Pipeline-based preprocessing ensures consistent transformation across folds.

---

## Future Improvements

- Cost-sensitive learning  
- Gradient boosting models (XGBoost / LightGBM)  
- SHAP-based explainability  
- Deployment simulation excluding post-contact variables  

---

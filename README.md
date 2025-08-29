# Fraud Detection on Large-Scale Transaction Data

## Overview
This repository demonstrates proactive fraud detection using simulated transaction data, focusing on end-to-end EDA, targeted feature engineering, and a supervised classification setup to identify fraudulent transactions.  
The notebook indicates a large dataset workflow and derives transaction-consistency features to improve separability between legitimate and fraudulent activity.

## Data
The project uses a CSV dataset named **Fraud.csv** with shape `6,362,620 × 11`, consistent with PaySim-style simulated mobile money transactions.  
Core columns include:
- `step`
- `type`
- `amount`
- `nameOrig`
- `oldbalanceOrg`
- `newbalanceOrig`
- `nameDest`
- `oldbalanceDest`
- `newbalanceDest`
- `isFraud`
- `isFlaggedFraud`

In PaySim, `isFraud` marks ground-truth fraud, while `isFlaggedFraud` is a simple rule-based flag triggered by high transfer amounts (e.g., >200,000), and only **CASH_OUT** and **TRANSFER** can be fraudulent types.

## EDA
Exploratory analysis is performed using **pandas**, **NumPy**, **Seaborn**, and **Matplotlib** to inspect schema, preview data, and understand class and transaction-type distributions.  
The dataset exhibits severe class imbalance typical of fraud problems, with frauds being a tiny fraction of all transactions in PaySim-like data. This motivates careful metric selection and resampling/weighting strategies during modeling.

## Feature Engineering
Derived features include:
- `deltaOrig`, `deltaDest` – balance change amounts  
- `errOrig`, `errDest` – reconciliation errors  
- `destIsMerchant` – recipient type flag  
- `log_amount` – scaled transaction amount  

These features supplement raw fields to highlight patterns consistent with fraudulent behavior while preserving explainability for model diagnostics.

## Modeling
The task is framed as **binary classification** to predict `isFraud`, emphasizing recall/precision trade-offs due to class imbalance and the high cost of missed fraud.  
Model evaluation should report:
- Precision  
- Recall  
- F1-score  
- ROC-AUC  
- PR-AUC  
- Confusion Matrix  

Thresholds should be tuned for business objectives. If transaction-type filtering is applied, prioritize **CASH_OUT** and **TRANSFER** transactions.

## Results
_Add finalized results here after model training._  
Example metrics:
- Validation/Test ROC-AUC  
- PR-AUC  
- Recall at selected precision  
- Confusion matrix  
- Thresholding strategy aligned to fraud-detection goals  

If applicable, include ablation notes showing the impact of engineered features such as `log_amount` and consistency deltas on performance.

## How to Run
1. Open `notebooks/Fraud-Detection-Model.ipynb` and run cells sequentially.  
2. Ensure the dataset path points to `data/Fraud.csv`.  
3. Required libraries: **pandas**, **numpy**, **seaborn**, **matplotlib** (add training libraries if extending to model building).

## Repository Structure


## Notes
- The dataset scale (~6.36M rows) requires efficient data loading and memory-aware EDA; consider chunked reads or dtype optimization in constrained environments.  
- In PaySim-style data, only **CASH_OUT** and **TRANSFER** can be fraudulent.  
- `isFlaggedFraud` is a naive rule — do not use it as a feature for training, but it can serve as a baseline.

## Author
**Uday Sadvik Kothapalli** 


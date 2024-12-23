![banner](https://github.com/PJURC-data-science/fraudulent-transactions/blob/main/media/banner.png)

# Bank Transaction Fraud Detection: Real-Time Risk Assessment
[View Notebook](https://github.com/PJURC-data-science/fraudulent-transactions/blob/master/notebook.ipynb)

A machine learning solution to identify fraudulent bank card transactions in real-time. This study develops a model to support a fraud analyst team in reviewing high-risk transactions, optimizing fraud prevention while maintaining operational efficiency.

## Overview

### Business Question 
How can we effectively identify fraudulent transactions for a 400-transaction monthly review capacity while maximizing fraud prevention?

### Key Findings
- 0.8% fraud rate in transactions
- Model captures 88% of fraud cases
- 95 daily alerts optimal
- 91% fraud capture rate
- Threshold set at 0.45

### Impact/Results
- AUPRC: 0.23
- AUROC: 0.90
- False Positives: 27.64%
- True Positives: 0.62%
- Business case validated

## Data

### Source Information
- Dataset: Bank Transaction Records
- Period: 1 year
- Size: ~118,000 transactions
- Fraud Rate: <0.8%
- Source: EU Banking Data

### Variables Analyzed
- Transaction details
- Merchant information
- Account data
- Entry modes
- Time patterns
- Geographic indicators
- Financial metrics

## Methods

### Analysis Approach
1. Feature Engineering
   - Time-based features
   - Rolling aggregations
   - Custom cross-validation
2. Model Development
   - Multiple base models
   - Ensemble methods
   - Class imbalance handling
3. Business Implementation
   - Threshold optimization
   - Alert system design
   - Performance monitoring

### Tools Used
- Python (Data Science)
  - Base Models:
    - LogisticRegression
    - MLPClassifier
    - LightGBM
    - CatBoostClassifier
    - BalancedRandomForestClassifier
  - Ensemble Methods:
    - VotingClassifier (primary)
    - StackingClassifier
  - Imbalance Handling:
    - RandomUnderSampler
    - RandomOverSampler
    - Class weights
  - Optimization:
    - Optuna tuning
    - Custom time-series CV
  - Performance Metrics:
    - AUPRC/AUROC
    - F1/Fbeta scores
    - Precision at recall
    - Custom business metrics

## Getting Started

### Prerequisites
```python
catboost==1.2.7
category_encoders==2.6.4
imbalanced_learn==0.12.4
imblearn==0.0
joblib==1.4.2
lightgbm==4.5.0
matplotlib==3.9.2
numpy==1.26.4
optuna==4.0.0
pandas==2.2.3
phik==0.12.4
plotly==5.24.1
scikit_learn==1.5.2
scipy==1.14.1
seaborn==0.13.2
shap==0.46.0
xgboost==2.1.2
```

### Installation & Usage
```bash
git clone git@github.com:PJURC-data-science/fraudulent-transactions.git
cd fraudulent-transactions
pip install -r requirements.txt
jupyter notebook "Fraudulent Transactions.ipynb"
```

## Project Structure
```
fraudulent-transactions/
│   README.md
│   requirements.txt
│   Fraudulent Transactions.ipynb
│   business case.pptx
|   utils.py
|   styles.css
└── data/
    └── Data Dictionary.xlsx
    └── labels_obf.csv
    └── transactions_obf.csv
    └── stores.csv
└── exports/
    └── auprc.csv
    └── model_AUPRC_LightGBM.joblib
    └── model_AUPRC_Logistic Regression.joblib
    └── model_AUPRC_Neural Network.joblib
    └── model_Balanced Random Forest.joblib
    └── model_LightGBM.joblib
    └── model_Logistic Regression.joblib
    └── model_Neural Network.joblib
```

## Strategic Recommendations
1. **Alert System**
   - Review 95 alerts daily
   - Set 0.45 threshold
   - Monitor fraud patterns
   - Track capture rates

2. **Model Management**
   - Regular retraining
   - Pattern monitoring
   - Performance tracking
   - Threshold adjustment

3. **Operational Integration**
   - Analyst team workflow
   - Alert prioritization
   - Performance reporting
   - Cost monitoring

## Future Improvements
- Test advanced models (HMM, LSTM)
- Implement Graph Neural Networks
- Expand Optuna trials
- Test dimensionality reduction
- Improve encoding methods
- Add XGBoost/Isolation Forest
- Update data recency
- Expand geographic coverage
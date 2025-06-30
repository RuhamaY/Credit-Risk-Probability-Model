# Credit Risk Probability Model for Alternative Data

## Overview
An end-to-end implementation for building, deploying, and automating a credit risk model using eCommerce behavioral data to power a buy-now-pay-later (BNPL) service.

## Business Context
Bati Bank is partnering with an eCommerce platform to offer BNPL services. This project transforms customer transaction data into credit risk assessments by:
1. Creating a proxy for default risk using behavioral patterns
2. Developing a predictive model for risk probability
3. Building a scoring system for loan decisions
4. Determining optimal loan amounts and durations

## Project Structure

credit-risk-model/
├── .github/workflows/ci.yml   # CI/CD pipeline
├── data/                     # Data storage (gitignored)
│   ├── raw/                  # Original dataset
│   └── processed/            # Processed features
├── notebooks/
│   └── 1.0-eda.ipynb         # Exploratory analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py    # Feature engineering
│   ├── train.py              # Model training
│   ├── predict.py            # Inference logic
│   └── api/
│       ├── main.py           # FastAPI app
│       └── pydantic_models.py # Data schemas
├── tests/
│   └── test_data_processing.py # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md

## Credit Scoring Business Understanding

### 1. Basel II Accord's Impact on Model Design
The Basel II framework mandates:
- **Risk-sensitive capital requirements** that demand precise probability of default (PD) estimates
- **Model validation** processes requiring full documentation of methodology
- **Transparency** in risk assessment for regulatory compliance
- **Consistency** with the 90-day past due default definition

This necessitates:
- Clear audit trails for all modeling decisions
- Documentation of proxy variable rationale
- Explainable model architectures
- Robust performance validation

### 2. Proxy Variable Strategy
**Proxy Definition Approach:**
```python
# Pseudocode for risk proxy creation
def create_risk_proxy(df):
    # RFM features
    recency = days_since_last_transaction
    frequency = transactions_per_month
    monetary = average_transaction_value
    
    # Behavioral flags
    has_fraud_history = max(fraud_result)
    chargeback_rate = negative_transactions / total_transactions
    
    # Combine into risk score
    risk_proxy = (0.3*recency_score + 
                  0.2*frequency_score + 
                  0.3*monetary_score + 
                  0.2*fraud_score)
    
    return (risk_proxy > threshold)
```

**Business Risks and Mitigations:**

| Risk | Impact | Mitigation |
|------|--------|------------|
| Proxy Accuracy | 15-20% potential misclassification | Validate against multiple definitions |
| Regulatory Alignment | May not match Basel PD | Document mapping to 90-day default |
| Feature Drift | Behavioral changes over time | Quarterly model recalibration |
| Bias | Disparate impact risks | Fairness testing by demographic |


## Model Development

**Training Framework:**
```python
import mlflow
from sklearn.ensemble import GradientBoostingClassifier

with mlflow.start_run():
    model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.1
    )
    
    model.fit(X_train, y_train)
    
    # Log metrics
    mlflow.log_metric("auc", roc_auc_score(y_val, model.predict_proba(X_val)[:,1]))
    mlflow.sklearn.log_model(model, "model")
```

**Evaluation Metrics:**
1. Primary: AUC-ROC (>0.8 target)
2. Secondary: Precision at 90% recall
3. Business: Expected loss calculation


## Getting Started

1. **Setup**:
```bash
git clone https://github.com/RuhamaY/Credit-Risk-Probability-Model.git
cd credit-risk-model
pip install -r requirements.txt
```

2. **Run Pipeline**:
```bash
# Process data
python src/data_processing.py

# Train model
python src/train.py

# Start API
uvicorn src.api.main:app --reload
```

3. **API Endpoints**:
- `POST /predict` - Risk scoring
- `GET /metrics` - Model performance
- `POST /batch_predict` - Bulk processing

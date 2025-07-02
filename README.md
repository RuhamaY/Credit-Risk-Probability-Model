# Credit Risk Probability Model for Alternative Data

## Overview

An end-to-end implementation for building, deploying, and automating a credit risk model using eCommerce behavioral data to power a Buy-Now-Pay-Later (BNPL) service.

## Business Context

Bati Bank is partnering with an eCommerce platform to offer BNPL services. This project transforms customer transaction data into credit risk assessments by:

1. Creating a proxy for default risk using behavioral patterns.
2. Developing a predictive model for risk probability.
3. Building a scoring system for loan decisions.
4. Determining optimal loan amounts and durations.

## Project Structure

```
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
```

## Credit Scoring Business Understanding

### 1. Basel II Accord's Impact on Model Design

The Basel II framework mandates:

* **Risk-sensitive capital requirements** requiring precise Probability of Default (PD) estimates.
* **Model validation** with full methodology documentation.
* **Transparency** in risk assessment for regulatory compliance.
* **Alignment** with the 90-day past due default definition.

This necessitates:

* Clear audit trails for all modeling decisions.
* Documented proxy variable rationale.
* Explainable, interpretable model architectures.
* Robust performance validation and monitoring.

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

| Risk                 | Impact                             | Mitigation                         |
| -------------------- | ---------------------------------- | ---------------------------------- |
| Proxy Accuracy       | 15-20% potential misclassification | Validate with multiple definitions |
| Regulatory Alignment | May not match Basel PD             | Document mapping to 90-day default |
| Feature Drift        | Behavioral changes over time       | Quarterly model recalibration      |
| Bias                 | Potential disparate impact         | Fairness testing by demographic    |

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

* Primary: AUC-ROC (target > 0.8)
* Secondary: Precision at 90% recall
* Business: Expected loss calculations

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Shegaw-21hub/credit-risk-model.git
cd credit-risk-model
```

### 2. Local Python Virtual Environment (Recommended)

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Data Acquisition

Place your raw dataset files in the `data/raw/` directory.

### 5. Run the Data Processing Pipeline

```bash
python src/data_processing.py
```

This script performs cleaning, feature engineering (including RFM and WoE), and generates `model_ready_data.csv`.

### 6. Run Unit Tests

```bash
python tests/test_data_processing.py
```

### 7. Model Training

```bash
python src/train.py
```

This trains models, performs hyperparameter tuning, evaluates performance, and logs results with MLflow.

## Docker Compose Setup for Local Development

**Prerequisites:**

* Docker Desktop or Docker Engine
* Git
* (Optional) Local Python for script execution

### 1. Build and Run with Docker Compose

```bash
docker-compose up --build
```

This will:

* Build the FastAPI service container
* Start both FastAPI and MLflow Tracking Server
* Mount the project directory for live code updates
* Persist MLflow experiments locally in `mlruns/`

### 2. Accessing Services

* MLflow UI (Experiment Tracking): [http://localhost:5000](http://localhost:5000)
* FastAPI Application with docs: [http://localhost:8000/docs](http://localhost:8000/docs)

## API Endpoints

* `POST /predict` - Risk scoring for a new customer
* `GET /metrics` - View current model performance
* `POST /batch_predict` - Bulk risk predictions

---

# Credit Risk Probability Model for Alternative Data

An end-to-end credit risk model built on eCommerce behavioural data to power a buy-now-pay-later (BNPL) service for Bati Bank. The project covers exploratory analysis, an automated feature engineering pipeline, proxy target creation, model training with experiment tracking, and a containerised scoring API with CI.

## Business Context

Bati Bank is partnering with an eCommerce platform to offer BNPL services. The transaction dataset contains no repayment history, so the project:

1. Creates a proxy for default risk from behavioural patterns
2. Trains a predictive model for risk probability
3. Converts probabilities into credit scores and lending recommendations
4. Serves the model behind a REST API

---

## Credit Scoring Business Understanding

### 1. How Basel II's emphasis on risk measurement shapes the need for an interpretable, well-documented model

The Basel II Accord ties a bank's regulatory capital directly to its own measured risk. Under the Internal Ratings-Based approach, the capital a bank must hold is computed from parameters it estimates itself — chiefly the Probability of Default (PD), alongside Loss Given Default and Exposure at Default. Because the model's output changes how much capital the bank is legally required to hold, the model stops being an internal analytics tool and becomes part of the bank's regulatory reporting.

Three consequences follow directly:

- **Supervisory review requires explanation, not just accuracy.** Pillar 2 gives regulators the authority to examine and challenge the bank's internal estimates. A model whose decisions cannot be explained in business terms cannot survive that review, no matter how strong its validation metrics are.
- **Pillar 3 market discipline requires disclosure.** The bank must publicly describe its rating system, its methodology, and its risk parameters. This is impossible without documentation written as the model is built rather than reconstructed afterwards.
- **Every judgement call must be auditable.** Basel II expects the definition of default to be applied consistently — conventionally 90 days past due. In this project there is no repayment data at all, so the gap between our proxy and the regulatory definition is itself a modelling assumption that has to be written down, justified, and revisited.

The practical effect is that a marginally less accurate model with clear reasoning is often worth more than a marginally better black box, because only the former can be validated, defended, and approved.

### 2. Why a proxy variable is necessary, and the business risks it creates

**Why it is necessary.** The Xente dataset records transactions, not loans. There is no repayment column, no delinquency flag, and no 90-days-past-due indicator — the analysis in `notebooks/1.0-eda.ipynb` confirms this. Supervised learning needs a target, so one has to be constructed. The only signal available is behavioural: customers who have gone quiet, transact rarely, and spend little are the least engaged, and disengagement is a widely used stand-in for credit risk. We therefore label the least engaged RFM segment as `is_high_risk = 1`.

`FraudResult` is the only label-like column in the data, but it is unusable as a target: it flags just **0.202%** of transactions, and fraud is a fundamentally different event from failing to repay a loan.

**The business risks.** The proxy measures disengagement, and disengagement is not default. That gap creates concrete exposure:

| Risk | What goes wrong in practice | Mitigation |
| --- | --- | --- |
| **Label bias** | The model learns "who stopped shopping", not "who fails to repay". A dormant customer may simply have switched platforms while remaining perfectly creditworthy. | Validate the proxy against real repayment outcomes as soon as the first loan cohort matures. |
| **Lost revenue (false positives)** | Creditworthy but low-activity customers are denied credit, shrinking the addressable market and pushing good customers to competitors. | Track approval rates and override rates by segment; treat the model as a ranking tool with a tunable cut-off, not an automatic decline. |
| **Credit losses (false negatives)** | An engaged, high-spending customer can still be over-extended. High activity may even indicate borrowing distress rather than health. | Set conservative initial limits, and combine the score with affordability and exposure caps. |
| **Regulatory challenge** | A supervisor may reject capital estimates derived from a target that has no demonstrated link to the 90-day default definition. | Document the proxy explicitly, treat early PD estimates as provisional, and hold additional capital buffers until validated. |
| **Fairness** | Engagement correlates with income, geography, and access to connectivity, so the proxy can encode disparate impact. | Test outcomes across demographic segments and monitor for adverse impact. |
| **Drift** | Clusters are fitted to one snapshot; seasonality or a platform change can shift what "disengaged" means. | Recalibrate on a schedule and monitor the high-risk rate for unexpected movement. |

The honest position is that this proxy makes the problem *tractable*, not *solved*. It is a starting point that must be replaced with observed repayment behaviour once that data exists.

### 3. Trade-offs: simple interpretable models vs. complex high-performance models

| Dimension | Logistic Regression (+ WoE) | Gradient Boosting / Ensembles |
| --- | --- | --- |
| **Interpretability** | High — each coefficient is a signed, monotonic contribution that maps onto a scorecard and can be explained to a credit committee. | Low — hundreds of interacting trees; explanations require post-hoc tools like SHAP, which approximate rather than describe the model. |
| **Regulatory acceptance** | Strong and well-precedented; validators know how to review a scorecard. | Achievable but expensive, requiring extra explainability evidence and validation. |
| **Predictive performance** | Moderate; linear in the WoE space and can miss genuine interactions. | Typically stronger on complex, non-linear behavioural data. |
| **Stability and monitoring** | Coefficients are stable and drift is easy to detect per feature. | More sensitive to distribution shift; failures are harder to localise. |
| **Cost to operate** | Low — quick to retrain, document, and hand over. | Higher — tuning, explainability tooling, and governance overhead. |

**Where the value actually lies.** In credit scoring the marginal AUC gain from an ensemble is usually modest — often a couple of points — while the cost is a model the bank may not be able to defend. That trade is only worth taking when the performance gain is large enough to move real money.

**Recommendation for this project:** run Logistic Regression as the regulatory-facing model and Gradient Boosting as a challenger. Promote the challenger only if it materially outperforms *and* ships with documented explainability and monitoring. The training script in `src/train.py` implements exactly this comparison, evaluating Logistic Regression, Random Forest, and Gradient Boosting side by side.

---

## EDA Key Insights

Full analysis with plots in [`notebooks/1.0-eda.ipynb`](notebooks/1.0-eda.ipynb). The dataset holds **95,662 transactions across 16 columns for 3,742 customers**, with **no missing values**.

1. **Extreme right skew in transaction amounts.** `Value` has a skewness of **51.3** — median 1,000 against a maximum of 9,880,000. Unscaled, a handful of transactions would dominate any distance-based model, so features are standardised and customers are modelled through RFM aggregates rather than raw transactions.

2. **Severe customer concentration.** The **top 1% of customers account for 55.8% of total transaction value**. Engagement, not demographics, is what separates customers here — which is what makes RFM segmentation a credible basis for the proxy target.

3. **Redundant and constant columns.** `Amount` and `Value` correlate at **r = 0.990** (`Value` is the unsigned magnitude of `Amount`), so only `Value` feeds the monetary aggregates. `CountryCode` and `CurrencyCode` each have a single distinct value and carry no information, so they are dropped.

4. **`FraudResult` cannot serve as the target.** Only **0.202%** of transactions are flagged, and fraud is not default. Together with the absence of any repayment column, this confirms the proxy label is unavoidable.

5. **Heavily imbalanced categoricals.** `financial_services` (47.5%) and `airtime` (47.1%) cover **94.5%** of `ProductCategory`; two channels cover 98.4% of `ChannelId`. Unseen categories are therefore expected in production, so `OneHotEncoder` uses `handle_unknown="ignore"`.

6. **Almost 40% of transactions have a negative `Amount`**, indicating refunds and reversals. Summing signed amounts would understate activity, so monetary aggregates use `Value`.

---

## Proxy Target: RFM + K-Means

`src/data_processing.py` builds the `is_high_risk` label in four steps:

1. **Calculate RFM per customer** against a snapshot date (the latest transaction timestamp, so Recency is consistent).
2. **Scale and cluster** the RFM profile into 3 segments with `KMeans(random_state=42)` for reproducibility.
3. **Identify the least engaged cluster** by an engagement score combining low frequency, low monetary value, and high recency; that cluster is labelled `1`.
4. **Merge the label back** into the customer-level dataset for training.

The resulting clusters are clearly separated, and the flagged segment is dormant on every dimension at once:

| Cluster | Customers | Avg recency (days) | Avg frequency | Avg monetary | Label |
| --- | --- | --- | --- | --- | --- |
| **0** | 1,426 (38%) | **60.9** | **7.7** | **89,738** | **High risk (1)** |
| 1 | 2,312 (62%) | 11.7 | 34.7 | 224,757 | Low risk (0) |
| 2 | 4 (0.1%) | 22.3 | 1,104.5 | 74,876,587 | Low risk (0) |

Cluster 0 shows roughly **5x more days since last activity, 4.5x fewer transactions, and 2.5x lower spend** than cluster 1 — the disengaged profile the proxy is meant to capture. Cluster 2 isolates four extreme-volume outliers, keeping them from distorting the other centroids. The overall high-risk rate is **38.1%**.

> **Known limitation — the target is derived from the features.** `is_high_risk` is a deterministic K-Means function of Recency, Frequency, and MonetarySum, and those same aggregates are model inputs. The classifiers therefore reach a near-perfect ROC-AUC (~0.9999) because they are recovering the cluster boundary, not predicting genuine default. These metrics validate that the pipeline is wired correctly; they are **not** an estimate of real-world credit performance. Honest performance figures require observed repayment outcomes.

---

## Feature Engineering Pipeline

All logic lives in `src/data_processing.py` and is composed with `sklearn.pipeline.Pipeline`.

| Step | Implementation |
| --- | --- |
| **Aggregate features** | `RFMTransformer` — per customer: `MonetarySum` (total), `MonetaryMean` (average), `Frequency` (count), `MonetaryStd` (variability, 0 for single-transaction customers) |
| **Datetime extraction** | `TransactionHour`, `TransactionDay`, `TransactionMonth` from `TransactionStartTime` |
| **Proxy target** | `HighRiskLabelGenerator` — scaled K-Means on RFM |
| **Missing values** | `SimpleImputer` — median for numeric, most frequent for categorical |
| **Categorical encoding** | `OneHotEncoder(handle_unknown="ignore")`; `LabelEncoder` inside the WoE reporting utility |
| **Scaling** | `StandardScaler` (zero mean, unit variance) |

**One pipeline for training and serving.** `build_model_pipeline()` bundles the `ColumnTransformer` together with the estimator, so the persisted model performs its own imputation, encoding, and scaling. The API can therefore post raw, human-readable values (`"ProductCategory": "airtime"`) and the identical transformations are applied at inference. `data/processed/model_ready_data.csv` is likewise stored in raw form, which keeps it inspectable.

Weight of Evidence encoding is produced as a **reporting artefact** (`data/processed/woe_report.csv`) rather than a model input, since fitting WoE on the full target would leak the label into training.

---

## Model Training and Tracking

`src/train.py` trains three candidates — **Logistic Regression, Random Forest, and Gradient Boosting** — tuning each with `GridSearchCV` (5-fold, scored on ROC-AUC) *inside* the preprocessing pipeline, so scaling statistics never leak across folds.

Each model is evaluated on a stratified hold-out set using **accuracy, precision, recall, F1, and ROC-AUC**, with every run logged to MLflow. The best model by ROC-AUC is saved to `models/best_model.pkl` and registered in the MLflow Model Registry as **`CreditRiskClassifier`**.

See the limitation note above before interpreting these scores.

---

## Project Structure

```
credit-risk-model/
├── .github/workflows/ci.yml     # Lint + test pipeline
├── data/                        # Gitignored
│   ├── raw/                     # Original dataset
│   └── processed/               # Model-ready features
├── notebooks/
│   └── 1.0-eda.ipynb            # Exploratory analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py       # Feature engineering + proxy target
│   ├── train.py                 # Training, tuning, MLflow registry
│   ├── predict.py               # Model loading and inference
│   └── api/
│       ├── main.py              # FastAPI application
│       └── pydantic_models.py   # Request/response schemas
├── tests/
│   ├── test_data_processing.py  # Pipeline unit tests
│   └── test_api.py              # API tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── setup.cfg                    # flake8 + pytest config
└── README.md
```

---

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/RuhamaY/Credit-Risk-Probability-Model.git
cd Credit-Risk-Probability-Model

python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### 2. Add the data

Place the Xente dataset at `data/raw/data.csv`. The `data/` directory is gitignored.

### 3. Run the processing pipeline

```bash
python -m src.data_processing
```

Builds RFM features, assigns the `is_high_risk` proxy, and writes `data/processed/model_ready_data.csv`.

### 4. Train the models

```bash
python -m src.train
```

Trains and tunes all three models, logs runs to `./mlruns`, saves `models/best_model.pkl`, and registers the best model. Inspect results with `mlflow ui`.

### 5. Run the tests

```bash
pytest
```

Configuration lives in `setup.cfg`, so no `PYTHONPATH` export is needed.

### 6. Start the API

```bash
uvicorn src.api.main:app --reload
```

Interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## Docker

```bash
docker-compose up --build
```

This starts the FastAPI service on port 8000 and an MLflow tracking UI on port 5000, mounting `models/` and `mlruns/` so the container serves the model you trained locally.

- API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- MLflow UI: [http://localhost:5000](http://localhost:5000)

Train a model before starting the container; otherwise `/predict` returns `503` until `models/best_model.pkl` exists.

---

## API Reference

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | Health check; reports whether the model is loaded |
| `POST` | `/predict` | Risk probability, score, and lending recommendation for one customer |
| `POST` | `/batch_predict` | The same scoring for a list of customers |

### Example

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "customer_id": "CustomerId_1",
    "Recency": 83, "Frequency": 1,
    "MonetarySum": 10000, "MonetaryMean": 10000, "MonetaryStd": 0,
    "TransactionHour": 16, "TransactionDay": 21, "TransactionMonth": 11,
    "ProductCategory": "airtime", "ChannelId": "ChannelId_2", "PricingStrategy": "4"
  }'
```

```json
{
  "customer_id": "CustomerId_1",
  "risk_probability": 1.0,
  "risk_category": "High",
  "credit_score": 300,
  "recommended_limit": 500.0,
  "recommended_term": 12
}
```

Risk probability is mapped onto a 300–850 credit score, banded into Low / Medium / High, and translated into an indicative limit and term.

---

## CI/CD

`.github/workflows/ci.yml` runs on every push and pull request to `main`:

1. **`flake8 src tests`** — the build fails on any style violation.
2. **`pytest`** — the build fails on any test failure.

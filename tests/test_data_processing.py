"""Unit tests for feature engineering, proxy labelling, and the model pipeline."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src.data_processing import (
    CATEGORICAL_FEATURE_COLS,
    NUMERIC_FEATURE_COLS,
    HighRiskLabelGenerator,
    RFMTransformer,
    aggregate_to_customer_level,
    build_feature_pipeline,
    build_model_pipeline,
    coerce_categoricals,
)


@pytest.fixture
def fixed_now():
    """Single fixed timestamp so Recency assertions stay deterministic."""
    return datetime(2023, 1, 15, 12, 30, 45)


@pytest.fixture
def sample_data(fixed_now):
    """Synthetic transactions with varied timestamps and three customers."""
    data = {
        'CustomerId': [1, 1, 2, 2, 3],
        'TransactionId': [101, 102, 201, 202, 301],
        'TransactionStartTime': [
            fixed_now - timedelta(days=5, hours=3),     # Jan 10, 09:30:45
            fixed_now - timedelta(days=2, minutes=15),  # Jan 13, 12:15:45
            fixed_now - timedelta(days=10, seconds=1),  # Jan 5, 12:30:44
            fixed_now - timedelta(hours=18),            # Jan 14, 18:30:45
            fixed_now - timedelta(days=20),             # Dec 26, 12:30:45
        ],
        'Value': [100, 200, 50, 150, 300],
        'ProductCategory': ['A', 'B', 'A', 'C', 'B'],
        'ChannelId': ['Web', 'App', 'App', 'Web', 'App'],
        'PricingStrategy': ['Standard', 'Premium', 'Standard', 'Discount', 'Premium'],
    }
    df = pd.DataFrame(data)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    return df


def test_rfm_transformer_columns(sample_data, fixed_now):
    """All RFM and temporal features are created without dropping rows."""
    transformer = RFMTransformer(snapshot_date=fixed_now)
    transformed = transformer.transform(sample_data.copy())

    expected_columns = {
        'Recency', 'Frequency', 'MonetarySum',
        'MonetaryMean', 'MonetaryStd',
        'TransactionHour', 'TransactionDay', 'TransactionMonth',
    }
    assert expected_columns.issubset(transformed.columns)
    assert transformed.shape[0] == sample_data.shape[0]


def test_rfm_calculations(sample_data, fixed_now):
    """Verify RFM aggregates for multi- and single-transaction customers."""
    transformer = RFMTransformer(snapshot_date=fixed_now)
    transformed = transformer.transform(sample_data.copy())

    cust1_data = sample_data[sample_data['CustomerId'] == 1]
    cust1 = transformed[transformed['CustomerId'] == 1].iloc[0]
    assert len(transformed[transformed['CustomerId'] == 1]) == 2
    assert np.allclose(cust1['MonetarySum'], 300)
    assert np.allclose(cust1['MonetaryMean'], 150)
    assert np.allclose(cust1['MonetaryStd'], np.std([100, 200], ddof=1))
    assert np.allclose(cust1['Recency'], (fixed_now - cust1_data['TransactionStartTime'].max()).days)
    assert np.allclose(cust1['Frequency'], 2)

    cust2_data = sample_data[sample_data['CustomerId'] == 2]
    cust2 = transformed[transformed['CustomerId'] == 2].iloc[0]
    assert len(transformed[transformed['CustomerId'] == 2]) == 2
    assert np.allclose(cust2['MonetarySum'], 200)
    assert np.allclose(cust2['MonetaryMean'], 100)
    assert np.allclose(cust2['MonetaryStd'], np.std([50, 150], ddof=1))
    assert np.allclose(cust2['Recency'], (fixed_now - cust2_data['TransactionStartTime'].max()).days)
    assert np.allclose(cust2['Frequency'], 2)

    # A single transaction has no dispersion, so std must fall back to 0 rather than NaN.
    cust3_data = sample_data[sample_data['CustomerId'] == 3]
    cust3 = transformed[transformed['CustomerId'] == 3].iloc[0]
    assert len(transformed[transformed['CustomerId'] == 3]) == 1
    assert np.allclose(cust3['MonetarySum'], 300)
    assert np.allclose(cust3['MonetaryMean'], 300)
    assert np.allclose(cust3['MonetaryStd'], 0)
    assert np.allclose(cust3['Recency'], (fixed_now - cust3_data['TransactionStartTime'].max()).days)
    assert np.allclose(cust3['Frequency'], 1)


def test_temporal_features(sample_data, fixed_now):
    """Hour, day, and month are extracted from each transaction timestamp."""
    transformer = RFMTransformer(snapshot_date=fixed_now)
    transformed = transformer.transform(sample_data.copy())

    expected_values = [
        (9, 10, 1),    # Jan 10, 09:30:45
        (12, 13, 1),   # Jan 13, 12:15:45
        (12, 5, 1),    # Jan 5, 12:30:44
        (18, 14, 1),   # Jan 14, 18:30:45
        (12, 26, 12),  # Dec 26, 12:30:45 (previous year)
    ]

    for idx, (exp_hour, exp_day, exp_month) in enumerate(expected_values):
        assert transformed.loc[idx, 'TransactionHour'] == exp_hour
        assert transformed.loc[idx, 'TransactionDay'] == exp_day
        assert transformed.loc[idx, 'TransactionMonth'] == exp_month


def test_high_risk_labeler(sample_data, fixed_now):
    """The labeller fits, exposes its chosen cluster, and emits a binary target."""
    rfm_data = RFMTransformer(snapshot_date=fixed_now).transform(sample_data.copy())

    labeler = HighRiskLabelGenerator(n_clusters=2, random_state=42)
    labeled_data = labeler.fit_transform(rfm_data)

    check_is_fitted(labeler, 'kmeans')
    assert hasattr(labeler, 'high_risk_cluster_')
    assert 'is_high_risk' in labeled_data.columns
    assert set(labeled_data['is_high_risk'].unique()) == {0, 1}
    assert 0.2 <= labeled_data['is_high_risk'].mean() < 0.8


def test_high_risk_cluster_is_least_engaged(fixed_now):
    """The flagged cluster must be the dormant, low-value one, not an arbitrary group."""
    rows = []
    # Engaged customers: recent, frequent, high value.
    for cid in range(10):
        for txn in range(8):
            rows.append({
                'CustomerId': f'engaged_{cid}',
                'TransactionId': f'e{cid}_{txn}',
                'TransactionStartTime': fixed_now - timedelta(days=txn),
                'Value': 5000,
                'ProductCategory': 'A',
                'ChannelId': 'Web',
                'PricingStrategy': 'Standard',
            })
    # Disengaged customers: dormant, single small transaction.
    for cid in range(10):
        rows.append({
            'CustomerId': f'dormant_{cid}',
            'TransactionId': f'd{cid}',
            'TransactionStartTime': fixed_now - timedelta(days=300),
            'Value': 10,
            'ProductCategory': 'A',
            'ChannelId': 'Web',
            'PricingStrategy': 'Standard',
        })

    df = pd.DataFrame(rows)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

    rfm = RFMTransformer(snapshot_date=fixed_now).transform(df)
    labeled = HighRiskLabelGenerator(n_clusters=2, random_state=42).fit_transform(rfm)

    per_customer = labeled.groupby('CustomerId')['is_high_risk'].first()
    assert per_customer.filter(like='dormant').eq(1).all()
    assert per_customer.filter(like='engaged').eq(0).all()


def test_high_risk_label_is_reproducible(sample_data, fixed_now):
    """Fixing random_state must yield identical labels across runs."""
    rfm_data = RFMTransformer(snapshot_date=fixed_now).transform(sample_data.copy())

    first = HighRiskLabelGenerator(n_clusters=2, random_state=42).fit_transform(rfm_data)
    second = HighRiskLabelGenerator(n_clusters=2, random_state=42).fit_transform(rfm_data)

    pd.testing.assert_series_equal(first['is_high_risk'], second['is_high_risk'])


def test_aggregate_to_customer_level(sample_data, fixed_now):
    """Aggregation collapses transactions to exactly one row per customer."""
    rfm = RFMTransformer(snapshot_date=fixed_now).transform(sample_data.copy())
    labeled = HighRiskLabelGenerator(n_clusters=2, random_state=42).fit_transform(rfm)
    customers = aggregate_to_customer_level(labeled)

    assert len(customers) == sample_data['CustomerId'].nunique()
    assert customers['CustomerId'].is_unique
    assert 'is_high_risk' in customers.columns
    for col in NUMERIC_FEATURE_COLS + CATEGORICAL_FEATURE_COLS:
        assert col in customers.columns


def test_coerce_categoricals_normalises_types():
    """Integer-valued categoricals become strings while NaN stays missing."""
    df = pd.DataFrame({
        'ProductCategory': ['airtime', np.nan],
        'ChannelId': ['ChannelId_3', 'ChannelId_2'],
        'PricingStrategy': [2, 4],
        'Recency': [1, 2],
    })
    result = coerce_categoricals(df)

    assert list(result['PricingStrategy']) == ['2', '4']
    assert result['ProductCategory'].iloc[0] == 'airtime'
    assert pd.isna(result['ProductCategory'].iloc[1])
    # Numeric columns must be left untouched.
    assert result['Recency'].tolist() == [1, 2]


def test_full_pipeline_output(sample_data):
    """The feature pipeline returns a dense numeric matrix of the expected width."""
    pipeline = build_feature_pipeline()
    processed = pipeline.fit_transform(sample_data.copy())
    assert isinstance(processed, np.ndarray)
    assert processed.shape == (5, 16)  # 8 numeric + 8 one-hot columns


def test_pipeline_with_missing_data(fixed_now):
    """Missing values are imputed rather than propagated."""
    data = pd.DataFrame({
        'CustomerId': [1, 1, 2],
        'TransactionId': [101, 102, 201],
        'TransactionStartTime': [fixed_now] * 3,
        'Value': [100, np.nan, 200],
        'ProductCategory': ['A', np.nan, 'B'],
        'ChannelId': ['Web', 'App', np.nan],
        'PricingStrategy': ['Standard', 'Premium', 'Standard'],
    })
    data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])

    pipeline = build_feature_pipeline()
    processed = pipeline.fit_transform(data)
    assert not np.isnan(processed).any()


def test_pipeline_structure():
    """Verify pipeline composition."""
    pipeline = build_feature_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == [
        'rfm_features',
        'risk_labels',
        'preprocessing',
    ]


@pytest.fixture
def customer_frame():
    """Customer-level features in the raw form the API submits."""
    rng = np.random.default_rng(0)
    size = 40
    return pd.DataFrame({
        'Recency': rng.integers(1, 90, size),
        'Frequency': rng.integers(1, 40, size),
        'MonetarySum': rng.integers(100, 90000, size),
        'MonetaryMean': rng.integers(100, 9000, size),
        'MonetaryStd': rng.integers(0, 500, size),
        'TransactionHour': rng.integers(0, 24, size),
        'TransactionDay': rng.integers(1, 28, size),
        'TransactionMonth': rng.integers(1, 13, size),
        'ProductCategory': rng.choice(['airtime', 'financial_services'], size),
        'ChannelId': rng.choice(['ChannelId_2', 'ChannelId_3'], size),
        'PricingStrategy': rng.choice(['2', '4'], size),
    })


def test_model_pipeline_accepts_raw_features(customer_frame):
    """The model pipeline must train and score on raw, unencoded customer rows.

    This guards the training/serving contract: the API posts human-readable
    values, so the fitted estimator has to own its own encoding and scaling.
    """
    y = (customer_frame['Recency'] > 45).astype(int)

    pipeline = build_model_pipeline(LogisticRegression(max_iter=1000, random_state=42))
    pipeline.fit(customer_frame, y)
    probabilities = pipeline.predict_proba(customer_frame.head(3))[:, 1]

    assert probabilities.shape == (3,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_model_pipeline_handles_unseen_category(customer_frame):
    """Categories absent from training must not raise at inference time."""
    y = (customer_frame['Recency'] > 45).astype(int)
    pipeline = build_model_pipeline(LogisticRegression(max_iter=1000, random_state=42))
    pipeline.fit(customer_frame, y)

    unseen = customer_frame.head(1).copy()
    unseen['ProductCategory'] = 'a_brand_new_category'

    probabilities = pipeline.predict_proba(unseen)[:, 1]
    assert probabilities.shape == (1,)
    assert 0.0 <= probabilities[0] <= 1.0
